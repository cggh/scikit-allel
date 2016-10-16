# -*- coding: utf-8 -*-
"""This module provides alternative implementations of array and table
classes defined in the :mod:`allel.model.ndarray` module, using
chunked arrays for data storage. Chunked arrays can be compressed and
optionally stored on disk, providing a means for working with data too
large to fit uncompressed in main memory.

Either `Zarr <http://zarr.readthedocs.io>`_, HDF5 (via `h5py <http://www.h5py.org/>`_)
or `bcolz <http://bcolz.blosc.org>`_ can be used as the underlying storage
layer. Choice of storage layer can be made via the `storage` keyword
argument which all class methods accept.  This argument can either be
a string identifying one of the predefined storage layer
configurations, or an object implementing the chunked storage API. For more
information about controlling storage see the :mod:`allel.chunked` module.

"""
from __future__ import absolute_import, print_function, division
import itertools


import numpy as np


from allel.compat import copy_method_doc, string_types
from allel import chunked as _chunked
from allel.chunked import ChunkedArrayWrapper, ChunkedTableWrapper
from allel.io import write_vcf_header, write_vcf_data, iter_gff3
from allel.util import check_ndim, check_integer_dtype
from allel.abc import DisplayAs2D
from .ndarray import GenotypeVector, GenotypeArray, HaplotypeArray, AlleleCountsArray, \
    VariantTable, FeatureTable, SortedIndex, SortedMultiIndex, GenotypeAlleleCountsArray, \
    GenotypeAlleleCountsVector
from .generic import compress_genotypes, \
    take_genotypes, concatenate_genotypes, index_genotype_array, subset_genotype_array, \
    index_haplotype_array, compress_haplotype_array, take_haplotype_array, \
    subset_haplotype_array, concatenate_haplotype_array, index_allele_counts_array, \
    compress_allele_counts_array, take_allele_counts_array, concatenate_allele_counts_array,\
    compress_genotype_ac, take_genotype_ac, concatenate_genotype_ac, \
    subset_genotype_ac_array, index_genotype_ac_array


__all__ = ['GenotypeChunkedArray', 'HaplotypeChunkedArray',
           'AlleleCountsChunkedArray', 'VariantChunkedTable',
           'FeatureChunkedTable', 'AlleleCountsChunkedTable',
           'GenotypeAlleleCountsChunkedArray']


class GenotypeChunkedArray(ChunkedArrayWrapper, DisplayAs2D):
    """Alternative implementation of the
    :class:`allel.model.ndarray.GenotypeArray` class, wrapping a
    chunked array as the backing store.

    Parameters
    ----------
    data : array_like, int, shape (n_variants, n_samples, ploidy)
        Genotype data to be wrapped. May be a bcolz carray, h5py dataset, or
        anything providing a similar interface.

    Examples
    --------

    Wrap an HDF5 dataset::

        >>> import h5py
        >>> with h5py.File('callset.h5', mode='w') as h5f:
        ...     h5g = h5f.create_group('/3L/calldata')
        ...     h5g.create_dataset('genotype',
        ...                        data=[[[0, 0], [0, 1]],
        ...                              [[0, 1], [1, 1]],
        ...                              [[0, 2], [-1, -1]]],
        ...                        dtype='i1', chunks=(2, 2, 2),
        ...                        compression='gzip', compression_opts=1)
        ...
        <HDF5 dataset "genotype": shape (3, 2, 2), type "|i1">
        >>> import allel
        >>> callset = h5py.File('callset.h5', mode='r')
        >>> g = allel.GenotypeChunkedArray(callset['/3L/calldata/genotype'])
        >>> g
        <GenotypeChunkedArray shape=(3, 2, 2) dtype=int8 chunks=(2, 2, 2)
           nbytes=12 cbytes=30 cratio=0.4
           compression=gzip compression_opts=1
           values=h5py._hl.dataset.Dataset>
        >>> g.values
        <HDF5 dataset "genotype": shape (3, 2, 2), type "|i1">

    Obtain a numpy array by slicing, e.g.::

        >>> g[:]
        <GenotypeArray shape=(3, 2, 2) dtype=int8>
        0/0 0/1
        0/1 1/1
        0/2 ./.

    Note that most methods will return a chunked array, using whatever
    chunked storage is set as default (bcolz carray) or specified
    directly via the `storage` keyword argument. E.g.::

        >>> g.copy()
        <GenotypeChunkedArray shape=(3, 2, 2) dtype=int8 chunks=(3, 2, 2)
           nbytes=12 cbytes=359 cratio=0.0
           compression=blosc compression_opts={'shuffle': 1, 'cname': 'lz4', 'clevel': 5}
           values=zarr.core.Array>
        >>> g.copy(storage='bcolzmem')  # doctest: +ELLIPSIS
        <GenotypeChunkedArray shape=(3, 2, 2) dtype=int8 chunks=(4096, 2, 2)
           nbytes=12 cbytes=16.0K cratio=0.0
           compression=blosc compression_opts=cparams(clevel=5, shuffle=1, cname='blosclz')
           values=bcolz.carray_ext.carray>
        >>> g.copy(storage='hdf5mem_zlib1')
        <GenotypeChunkedArray shape=(3, 2, 2) dtype=int8 chunks=(3, 2, 2)
           nbytes=12 cbytes=20 cratio=0.6
           compression=gzip compression_opts=1
           values=h5py._hl.dataset.Dataset>

    """  # flake8: noqa

    def __init__(self, data):
        super(GenotypeChunkedArray, self).__init__(data)
        check_ndim(self.values, 3)
        check_integer_dtype(self.values)
        self._mask = None
        self._is_phased = None

    def __getitem__(self, item):
        return index_genotype_array(self, item, array_cls=GenotypeArray,
                                    vector_cls=GenotypeVector)

    @property
    def n_variants(self):
        """Number of variants."""
        return self.shape[0]

    @property
    def n_samples(self):
        """Number of samples."""
        return self.shape[1]

    @property
    def ploidy(self):
        """Sample ploidy."""
        return self.shape[-1]

    @property
    def n_allele_calls(self):
        """Total number of allele calls."""
        return np.prod(self.shape)

    @property
    def n_calls(self):
        """Total number of genotype calls."""
        return self.n_allele_calls // self.ploidy

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask):
        if not hasattr(mask, 'shape'):
            mask = np.asarray(mask, dtype=bool)
        if mask.shape != self.shape[:2]:
            raise ValueError('mask has incorrect shape')
        self._mask = mask

    @property
    def is_phased(self):
        return self._is_phased

    @is_phased.setter
    def is_phased(self, is_phased):
        if not hasattr(is_phased, 'shape'):
            is_phased = np.asarray(is_phased, dtype=bool)
        if is_phased.shape != self.shape[:2]:
            raise ValueError('is_phased has incorrect shape')
        self._is_phased = is_phased

    def fill_masked(self, value=-1, **kwargs):
        out = self.map_blocks_method('fill_masked', kwargs=dict(value=value),
                                **kwargs)
        return GenotypeChunkedArray(out)

    def is_called(self, **kwargs):
        return self.map_blocks_method('is_called', **kwargs)

    def is_missing(self, **kwargs):
        return self.map_blocks_method('is_missing', **kwargs)

    def is_hom(self, allele=None, **kwargs):
        return self.map_blocks_method('is_hom', kwargs=dict(allele=allele),
                                 **kwargs)

    def is_hom_ref(self, **kwargs):
        return self.map_blocks_method('is_hom_ref', **kwargs)

    def is_hom_alt(self, **kwargs):
        return self.map_blocks_method('is_hom_alt', **kwargs)

    def is_het(self, allele=None, **kwargs):
        return self.map_blocks_method('is_het', kwargs=dict(allele=allele),
                                 **kwargs)

    def is_call(self, call, **kwargs):
        return self.map_blocks_method('is_call', kwargs=dict(call=call),
                                 **kwargs)

    def _count(self, method_name, axis, kwargs=None, **storage_kwargs):
        if kwargs is None:
            kwargs = dict()

        def mapper(block):
            method = getattr(block, method_name)
            return method(**kwargs)
        out = self.sum(axis=axis, mapper=mapper, **storage_kwargs)
        return out

    def count_called(self, axis=None, **kwargs):
        return self._count('is_called', axis, **kwargs)

    def count_missing(self, axis=None, **kwargs):
        return self._count('is_missing', axis, **kwargs)

    def count_hom(self, allele=None, axis=None, **kwargs):
        return self._count('is_hom', axis, kwargs=dict(allele=allele),
                           **kwargs)

    def count_hom_ref(self, axis=None, **kwargs):
        return self._count('is_hom_ref', axis, **kwargs)

    def count_hom_alt(self, axis=None, **kwargs):
        return self._count('is_hom_alt', axis, **kwargs)

    def count_het(self, allele=None, axis=None, **kwargs):
        return self._count('is_het', axis, kwargs=dict(allele=allele),
                           **kwargs)

    def count_call(self, call, axis=None, **kwargs):
        return self._count('is_call', axis, kwargs=dict(call=call),
                           **kwargs)

    def to_haplotypes(self, **kwargs):
        out = self.map_blocks_method('to_haplotypes', **kwargs)
        return HaplotypeChunkedArray(out)

    def to_n_ref(self, fill=0, dtype='i1', **kwargs):
        out = self.map_blocks_method('to_n_ref', kwargs=dict(fill=fill,
                                                        dtype=dtype),
                                **kwargs)
        return out

    def to_n_alt(self, fill=0, dtype='i1', **kwargs):
        out = self.map_blocks_method('to_n_alt', kwargs=dict(fill=fill,
                                                        dtype=dtype),
                                **kwargs)
        return out

    def to_allele_counts(self, max_allele=None, **kwargs):
        # determine alleles to count
        if max_allele is None:
            max_allele = self.max()

        out = self.map_blocks_method('to_allele_counts',
                                kwargs=dict(max_allele=max_allele),
                                **kwargs)
        out = GenotypeAlleleCountsChunkedArray(out)
        return out

    def to_packed(self, boundscheck=True, **kwargs):
        out = self.map_blocks_method('to_packed',
                                kwargs=dict(boundscheck=boundscheck),
                                **kwargs)
        return out

    @classmethod
    def from_packed(cls, packed, **kwargs):
        def f(block):
            return GenotypeArray.from_packed(block)
        out = _chunked.map_blocks(packed, f, **kwargs)
        return cls(out)

    def count_alleles(self, max_allele=None, subpop=None, **kwargs):
        if max_allele is None:
            max_allele = self.max()
        out = self.map_blocks_method('count_alleles',
                                     kwargs=dict(max_allele=max_allele, subpop=subpop),
                                     **kwargs)
        return AlleleCountsChunkedArray(out)

    def count_alleles_subpops(self, subpops, max_allele=None,
                              **kwargs):
        if max_allele is None:
            max_allele = self.max()

        def f(block):
            return block.count_alleles_subpops(subpops, max_allele=max_allele)

        out = _chunked.map_blocks(self, f, create='table', **kwargs)
        return AlleleCountsChunkedTable(out)

    def to_gt(self, max_allele=None, **kwargs):
        out = self.map_blocks_method('to_gt', kwargs=dict(max_allele=max_allele), **kwargs)
        return out

    def map_alleles(self, mapping, **kwargs):
        def f(block, bmapping):
            return block.map_alleles(bmapping, copy=False)
        domain = (self, mapping)
        out = _chunked.map_blocks(domain, f, **kwargs)
        return GenotypeChunkedArray(out)

    def compress(self, condition, axis=0, **kwargs):
        return compress_genotypes(self, condition, axis=axis, wrap_axes={0, 1},
                                  cls=type(self), compress=_chunked.compress, **kwargs)

    def take(self, indices, axis=0, **kwargs):
        return take_genotypes(self, indices, axis=axis, wrap_axes={0, 1}, cls=type(self),
                              take=_chunked.take, **kwargs)

    def subset(self, sel0=None, sel1=None, **kwargs):
        return subset_genotype_array(self, sel0, sel1, cls=type(self),
                                     subset=_chunked.subset, **kwargs)

    def concatenate(self, others, axis=0, **kwargs):
        return concatenate_genotypes(self, others, axis=axis, wrap_axes={0, 1},
                                     cls=type(self), concatenate=_chunked.concatenate,
                                     **kwargs)


# copy docstrings
copy_method_doc(GenotypeChunkedArray.fill_masked, GenotypeArray.fill_masked)
copy_method_doc(GenotypeChunkedArray.subset, GenotypeArray.subset)
copy_method_doc(GenotypeChunkedArray.is_called, GenotypeArray.is_called)
copy_method_doc(GenotypeChunkedArray.is_missing, GenotypeArray.is_missing)
copy_method_doc(GenotypeChunkedArray.is_hom, GenotypeArray.is_hom)
copy_method_doc(GenotypeChunkedArray.is_hom_ref, GenotypeArray.is_hom_ref)
copy_method_doc(GenotypeChunkedArray.is_hom_alt, GenotypeArray.is_hom_alt)
copy_method_doc(GenotypeChunkedArray.is_het, GenotypeArray.is_het)
copy_method_doc(GenotypeChunkedArray.is_call, GenotypeArray.is_call)
copy_method_doc(GenotypeChunkedArray.to_haplotypes, GenotypeArray.to_haplotypes)
copy_method_doc(GenotypeChunkedArray.to_n_ref, GenotypeArray.to_n_ref)
copy_method_doc(GenotypeChunkedArray.to_n_alt, GenotypeArray.to_n_alt)
copy_method_doc(GenotypeChunkedArray.to_allele_counts, GenotypeArray.to_allele_counts)
copy_method_doc(GenotypeChunkedArray.to_packed, GenotypeArray.to_packed)
# TODO
# copy_method_doc(GenotypeChunkedArray.from_packed, GenotypeArray.from_packed)
copy_method_doc(GenotypeChunkedArray.count_alleles, GenotypeArray.count_alleles)
copy_method_doc(GenotypeChunkedArray.count_alleles_subpops,
                GenotypeArray.count_alleles_subpops)
copy_method_doc(GenotypeChunkedArray.to_gt, GenotypeArray.to_gt)
copy_method_doc(GenotypeChunkedArray.map_alleles, GenotypeArray.map_alleles)
copy_method_doc(GenotypeChunkedArray.concatenate, GenotypeArray.concatenate)


class HaplotypeChunkedArray(ChunkedArrayWrapper, DisplayAs2D):
    """Alternative implementation of the
    :class:`allel.model.ndarray.HaplotypeArray` class, using a chunked array as
    the backing store.

    Parameters
    ----------
    data : array_like, int, shape (n_variants, n_haplotypes)
        Haplotype data to be wrapped. May be a bcolz carray, h5py dataset, or
        anything providing a similar interface.

    """

    def __init__(self, data):
        super(HaplotypeChunkedArray, self).__init__(data)
        check_ndim(self.values, 2)
        check_integer_dtype(self.values)

    def __getitem__(self, item):
        return index_haplotype_array(self, item, cls=HaplotypeArray)

    @property
    def n_variants(self):
        """Number of variants."""
        return self.shape[0]

    @property
    def n_haplotypes(self):
        """Number of haplotypes."""
        return self.shape[1]

    def to_genotypes(self, ploidy=2, **kwargs):

        # check ploidy is compatible
        if (self.n_haplotypes % ploidy) > 0:
            raise ValueError('incompatible ploidy')

        # build output
        def f(block):
            return block.to_genotypes(ploidy)

        out = self.map_blocks(f, **kwargs)
        return GenotypeChunkedArray(out)

    def is_called(self, **kwargs):
        return self.__ge__(0, **kwargs)

    def is_missing(self, **kwargs):
        return self.__lt__(0, **kwargs)

    def is_ref(self, **kwargs):
        return self.__eq__(0, **kwargs)

    def is_alt(self, **kwargs):
        return self.__gt__(0, **kwargs)

    def is_call(self, allele, **kwargs):
        return self.__eq__(allele, **kwargs)

    def _count(self, method_name, axis, kwargs=None, **storage_kwargs):
        if kwargs is None:
            kwargs = dict()

        def mapper(block):
            method = getattr(block, method_name)
            return method(**kwargs)
        out = self.sum(axis=axis, mapper=mapper, **storage_kwargs)
        return out

    def count_called(self, axis=None, **kwargs):
        return self._count('is_called', axis=axis, **kwargs)

    def count_missing(self, axis=None, **kwargs):
        return self._count('is_missing', axis=axis, **kwargs)

    def count_ref(self, axis=None, **kwargs):
        return self._count('is_ref', axis=axis, **kwargs)

    def count_alt(self, axis=None, **kwargs):
        return self._count('is_alt', axis=axis, **kwargs)

    def count_call(self, allele, axis=None, **kwargs):
        return self._count('is_call', axis=axis,
                           kwargs=dict(allele=allele),
                           **kwargs)

    def count_alleles(self, max_allele=None, subpop=None, **kwargs):
        # if max_allele not specified, count all alleles
        if max_allele is None:
            max_allele = self.max()

        def f(block):
            return block.count_alleles(max_allele=max_allele, subpop=subpop)
        out = self.map_blocks(f, **kwargs)
        return AlleleCountsChunkedArray(out)

    def count_alleles_subpops(self, subpops, max_allele=None,
                              **kwargs):
        if max_allele is None:
            max_allele = self.max()

        def f(block):
            return block.count_alleles_subpops(subpops, max_allele=max_allele)
        out = _chunked.map_blocks(self, f, create='table', **kwargs)
        return AlleleCountsChunkedTable(out)

    def map_alleles(self, mapping, **kwargs):
        def f(block, bmapping):
            return block.map_alleles(bmapping, copy=False)
        domain = (self, mapping)
        out = _chunked.map_blocks(domain, f, **kwargs)
        return HaplotypeChunkedArray(out)

    def compress(self, condition, axis=0, **kwargs):
        return compress_haplotype_array(self, condition, axis=axis, cls=type(self),
                                        compress=_chunked.compress, **kwargs)

    def take(self, indices, axis=0, **kwargs):
        return take_haplotype_array(self, indices, axis=axis, cls=type(self),
                                    take=_chunked.take, **kwargs)

    def subset(self, sel0=None, sel1=None, **kwargs):
        return subset_haplotype_array(self, sel0, sel1, cls=type(self),
                                      subset=_chunked.subset, **kwargs)

    def concatenate(self, others, axis=0, **kwargs):
        return concatenate_haplotype_array(self, others, axis=axis, cls=type(self),
                                           concatenate=_chunked.concatenate, **kwargs)


# copy docstrings
copy_method_doc(HaplotypeChunkedArray.to_genotypes, HaplotypeArray.to_genotypes)
copy_method_doc(HaplotypeChunkedArray.count_alleles, HaplotypeArray.count_alleles)
copy_method_doc(HaplotypeChunkedArray.count_alleles_subpops,
                HaplotypeArray.count_alleles_subpops)
copy_method_doc(HaplotypeChunkedArray.map_alleles, HaplotypeArray.map_alleles)
copy_method_doc(HaplotypeChunkedArray.subset, HaplotypeArray.subset)


class AlleleCountsChunkedArray(ChunkedArrayWrapper, DisplayAs2D):
    """Alternative implementation of the
    :class:`allel.model.ndarray.AlleleCountsArray` class, using a chunked
    array as the backing store.

    Parameters
    ----------
    data : array_like, int, shape (n_variants, n_alleles)
        Allele counts data to be wrapped. May be a bcolz carray,
        h5py dataset, or anything providing a similar interface.

    """

    def __init__(self, data):
        super(AlleleCountsChunkedArray, self).__init__(data)
        check_ndim(self.values, 2)
        check_integer_dtype(self.values)

    def __getitem__(self, item):
        return index_allele_counts_array(self, item, cls=AlleleCountsArray)

    def __add__(self, other, **kwargs):
        ret = super(AlleleCountsChunkedArray, self).__add__(other, **kwargs)
        if hasattr(ret, 'shape') and ret.shape == self.shape:
            ret = AlleleCountsChunkedArray(ret)
        return ret

    def __sub__(self, other, **kwargs):
        ret = super(AlleleCountsChunkedArray, self).__sub__(other, **kwargs)
        if hasattr(ret, 'shape') and ret.shape == self.shape:
            ret = AlleleCountsChunkedArray(ret)
        return ret

    @property
    def n_variants(self):
        """Number of variants (length of first array dimension)."""
        return self.shape[0]

    @property
    def n_alleles(self):
        """Number of alleles (length of second array dimension)."""
        return self.shape[1]

    def to_frequencies(self, fill=np.nan, **kwargs):
        out = self.map_blocks_method('to_frequencies', kwargs=dict(fill=fill), **kwargs)
        return ChunkedArrayWrapper(out)

    def allelism(self, **kwargs):
        out = self.map_blocks_method('allelism', **kwargs)
        return ChunkedArrayWrapper(out)

    def max_allele(self, **kwargs):
        out = self.map_blocks_method('max_allele', **kwargs)
        return ChunkedArrayWrapper(out)

    def is_variant(self, **kwargs):
        out = self.map_blocks_method('is_variant', **kwargs)
        return ChunkedArrayWrapper(out)

    def is_non_variant(self, **kwargs):
        out = self.map_blocks_method('is_non_variant', **kwargs)
        return ChunkedArrayWrapper(out)

    def is_segregating(self, **kwargs):
        out = self.map_blocks_method('is_segregating', **kwargs)
        return ChunkedArrayWrapper(out)

    def is_non_segregating(self, allele=None, **kwargs):
        out = self.map_blocks_method('is_non_segregating',
                                kwargs=dict(allele=allele),
                                **kwargs)
        return ChunkedArrayWrapper(out)

    def is_singleton(self, allele=1, **kwargs):
        out = self.map_blocks_method('is_singleton',
                                kwargs=dict(allele=allele),
                                **kwargs)
        return ChunkedArrayWrapper(out)

    def is_doubleton(self, allele=1, **kwargs):
        out = self.map_blocks_method('is_doubleton',
                                kwargs=dict(allele=allele),
                                **kwargs)
        return ChunkedArrayWrapper(out)

    def is_biallelic(self, **kwargs):
        out = self.map_blocks_method('is_biallelic', **kwargs)
        return ChunkedArrayWrapper(out)

    def is_biallelic_01(self, min_mac=None, **kwargs):
        out = self.map_blocks_method('is_biallelic_01',
                                kwargs=dict(min_mac=min_mac),
                                **kwargs)
        return ChunkedArrayWrapper(out)

    def _count(self, method_name, kwargs=None, **storage_kwargs):
        if kwargs is None:
            kwargs = dict()

        def mapper(block):
            method = getattr(block, method_name)
            return method(**kwargs)
        out = self.sum(mapper=mapper, **storage_kwargs)
        return out

    def count_variant(self, **kwargs):
        return self._count('is_variant', **kwargs)

    def count_non_variant(self, **kwargs):
        return self._count('is_non_variant', **kwargs)

    def count_segregating(self, **kwargs):
        return self._count('is_segregating', **kwargs)

    def count_non_segregating(self, allele=None, **kwargs):
        return self._count('is_non_segregating', kwargs=dict(allele=allele),
                           **kwargs)

    def count_singleton(self, allele=1, **kwargs):
        return self._count('is_singleton', kwargs=dict(allele=allele),
                           **kwargs)

    def count_doubleton(self, allele=1, **kwargs):
        return self._count('is_doubleton', kwargs=dict(allele=allele),
                           **kwargs)

    def map_alleles(self, mapping, **kwargs):
        def f(block, bmapping):
            return block.map_alleles(bmapping)
        domain = (self, mapping)
        out = _chunked.map_blocks(domain, f, **kwargs)
        return AlleleCountsChunkedArray(out)

    def compress(self, condition, axis=0, **kwargs):
        return compress_allele_counts_array(self, condition, axis=axis, cls=type(self),
                                            compress=_chunked.compress, **kwargs)

    def take(self, indices, axis=0, **kwargs):
        return take_allele_counts_array(self, indices, axis=axis, cls=type(self),
                                        take=_chunked.take, **kwargs)

    def concatenate(self, others, axis=0, **kwargs):
        return concatenate_allele_counts_array(self, others, axis=axis, cls=type(self),
                                               concatenate=_chunked.concatenate, **kwargs)


copy_method_doc(AlleleCountsChunkedArray.allelism, AlleleCountsArray.allelism)
copy_method_doc(AlleleCountsChunkedArray.max_allele, AlleleCountsArray.max_allele)
copy_method_doc(AlleleCountsChunkedArray.map_alleles, AlleleCountsArray.map_alleles)
copy_method_doc(AlleleCountsChunkedArray.to_frequencies, AlleleCountsArray.to_frequencies)
copy_method_doc(AlleleCountsChunkedArray.is_variant, AlleleCountsArray.is_variant)
copy_method_doc(AlleleCountsChunkedArray.is_non_variant, AlleleCountsArray.is_non_variant)
copy_method_doc(AlleleCountsChunkedArray.is_segregating, AlleleCountsArray.is_segregating)
copy_method_doc(AlleleCountsChunkedArray.is_non_segregating,
                AlleleCountsArray.is_non_segregating)
copy_method_doc(AlleleCountsChunkedArray.is_singleton, AlleleCountsArray.is_singleton)
copy_method_doc(AlleleCountsChunkedArray.is_doubleton, AlleleCountsArray.is_doubleton)
copy_method_doc(AlleleCountsChunkedArray.is_biallelic, AlleleCountsArray.is_biallelic)
copy_method_doc(AlleleCountsChunkedArray.is_biallelic_01, AlleleCountsArray.is_biallelic_01)


class GenotypeAlleleCountsChunkedArray(ChunkedArrayWrapper, DisplayAs2D):

    def __init__(self, data):
        super(GenotypeAlleleCountsChunkedArray, self).__init__(data)
        check_ndim(self.values, 3)
        check_integer_dtype(self.values)

    def __getitem__(self, item):
        return index_genotype_ac_array(self, item, array_cls=GenotypeAlleleCountsArray,
                                       vector_cls=GenotypeAlleleCountsVector)

    @property
    def n_variants(self):
        """Number of variants."""
        return self.shape[0]

    @property
    def n_samples(self):
        """Number of samples."""
        return self.shape[1]

    @property
    def n_alleles(self):
        """Number of alleles."""
        return self.shape[2]

    def is_called(self, **kwargs):
        return self.map_blocks_method('is_called', **kwargs)

    def is_missing(self, **kwargs):
        return self.map_blocks_method('is_missing', **kwargs)

    def is_hom(self, allele=None, **kwargs):
        return self.map_blocks_method('is_hom', kwargs=dict(allele=allele),
                                 **kwargs)

    def is_hom_ref(self, **kwargs):
        return self.map_blocks_method('is_hom_ref', **kwargs)

    def is_hom_alt(self, **kwargs):
        return self.map_blocks_method('is_hom_alt', **kwargs)

    def is_het(self, allele=None, **kwargs):
        return self.map_blocks_method('is_het', kwargs=dict(allele=allele),
                                 **kwargs)

    def count_alleles(self, subpop=None, **kwargs):
        out = self.map_blocks_method('count_alleles', kwargs=dict(subpop=subpop), **kwargs)
        return AlleleCountsChunkedArray(out)

    def to_gt(self, max_allele=None, **kwargs):
        out = self.map_blocks_method('to_gt', kwargs=dict(max_allele=max_allele), **kwargs)
        return out

    def compress(self, condition, axis=0):
        return compress_genotype_ac(self, condition=condition, axis=axis, wrap_axes={0, 1},
                                    cls=type(self), compress=_chunked.compress)

    def take(self, indices, axis=0):
        return take_genotype_ac(self, indices=indices, axis=axis, wrap_axes={0, 1},
                                cls=type(self), take=_chunked.take)

    def concatenate(self, others, axis=0):
        return concatenate_genotype_ac(self, others=others, axis=axis, wrap_axes={0, 1},
                                       cls=type(self), concatenate=_chunked.concatenate)

    def subset(self, sel0=None, sel1=None):
        return subset_genotype_ac_array(self, sel0, sel1, cls=type(self),
                                        subset=_chunked.subset)


class VariantChunkedTable(ChunkedTableWrapper):
    """Alternative implementation of the
    :class:`allel.model.ndarray.VariantTable` class, using a chunked table as
    the backing store.

    Parameters
    ----------
    data: table_like
        Data to be wrapped. May be a tuple or list of columns (array-like),
        a dict mapping names to columns, a bcolz ctable, h5py group,
        numpy recarray, or anything providing a similar interface.
    names : sequence of strings
        Column names.

    Examples
    --------

    Wrap columns stored as datasets within an HDF5 group::

        >>> import h5py
        >>> chrom = [b'chr1', b'chr1', b'chr2', b'chr2', b'chr3']
        >>> pos = [2, 7, 3, 9, 6]
        >>> dp = [35, 12, 78, 22, 99]
        >>> qd = [4.5, 6.7, 1.2, 4.4, 2.8]
        >>> ac = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]
        >>> with h5py.File('callset.h5', mode='w') as h5f:
        ...     h5g = h5f.create_group('/3L/variants')
        ...     h5g.create_dataset('CHROM', data=chrom, chunks=True)
        ...     h5g.create_dataset('POS', data=pos, chunks=True)
        ...     h5g.create_dataset('DP', data=dp, chunks=True)
        ...     h5g.create_dataset('QD', data=qd, chunks=True)
        ...     h5g.create_dataset('AC', data=ac, chunks=True)
        ...
        <HDF5 dataset "CHROM": shape (5,), type "|S4">
        <HDF5 dataset "POS": shape (5,), type "<i8">
        <HDF5 dataset "DP": shape (5,), type "<i8">
        <HDF5 dataset "QD": shape (5,), type "<f8">
        <HDF5 dataset "AC": shape (5, 2), type "<i8">
        >>> import allel
        >>> callset = h5py.File('callset.h5', mode='r')
        >>> vt = allel.VariantChunkedTable(callset['/3L/variants'],
        ...                                names=['CHROM', 'POS', 'AC', 'QD', 'DP'])
        >>> vt  # doctest: +ELLIPSIS
        <VariantChunkedTable shape=(5,) dtype=[('CHROM', 'S4'), ('POS', '<i8'), ('AC', ...
           nbytes=220 cbytes=220 cratio=1.0
           values=h5py._hl.group.Group>

    Obtain a single row::

        >>> vt[0]
        row(CHROM=b'chr1', POS=2, AC=array([1, 2]), QD=4.5, DP=35)

    Obtain a numpy array by slicing::

        >>> vt[:]  # doctest: +ELLIPSIS
        <VariantTable shape=(5,) dtype=(numpy.record, [('CHROM', 'S4'), ('POS', '<i8'), ...
        [(b'chr1', 2, array([1, 2]), 4.5, 35) (b'chr1', 7, array([3, 4]), 6.7, 12)
         (b'chr2', 3, array([5, 6]), 1.2, 78) (b'chr2', 9, array([7, 8]), 4.4, 22)
         (b'chr3', 6, array([ 9, 10]), 2.8, 99)]

    Access a subset of columns::

        >>> vt[['CHROM', 'POS']]
        <VariantChunkedTable shape=(5,) dtype=[('CHROM', 'S4'), ('POS', '<i8')]
           nbytes=60 cbytes=60 cratio=1.0
           values=builtins.list>

    Note that most methods will return a chunked table, using whatever
    chunked storage is set as default (bcolz ctable) or specified
    directly via the `storage` keyword argument. E.g.::

        >>> vt.copy()  # doctest: +ELLIPSIS
        <VariantChunkedTable shape=(5,) dtype=[('CHROM', 'S4'), ('POS', '<i8'), ('AC', ...
           nbytes=220 cbytes=1.7K cratio=0.1
           values=allel.chunked.storage_zarr.ZarrTable>
        >>> vt.copy(storage='bcolzmem')  # doctest: +ELLIPSIS
        <VariantChunkedTable shape=(5,) dtype=[('CHROM', 'S4'), ('POS', '<i8'), ('AC', ...
           nbytes=220 cbytes=80.0K cratio=0.0
           values=bcolz.ctable.ctable>
        >>> vt.copy(storage='hdf5mem_zlib1')  # doctest: +ELLIPSIS
        <VariantChunkedTable shape=(5,) dtype=[('CHROM', 'S4'), ('POS', '<i8'), ('AC', ...
           nbytes=220 cbytes=131 cratio=1.7
           values=h5py._hl.files.File>

    """

    array_cls = VariantTable

    def __init__(self, data, names=None, index=None):
        super(VariantChunkedTable, self).__init__(data, names=names)
        self.index = None
        if index is not None:
            self.set_index(index)

    @property
    def n_variants(self):
        return len(self)

    def set_index(self, spec):
        if isinstance(spec, string_types):
            self.index = SortedIndex(self[spec][:], copy=False)
        elif isinstance(spec, (tuple, list)) and len(spec) == 2:
            self.index = SortedMultiIndex(self[spec[0]][:], self[spec[1]][:], copy=False)
        else:
            raise ValueError('invalid index argument, expected string or '
                             'pair of strings, found %s' % repr(spec))

    def to_vcf(self, path, rename=None, number=None, description=None,
               fill=None, blen=None, write_header=True):
        with open(path, 'w') as vcf_file:
            if write_header:
                write_vcf_header(vcf_file, self, rename=rename, number=number,
                                 description=description)
            blen = _chunked.get_blen_table(self, blen)
            for i in range(0, len(self), blen):
                j = min(i+blen, len(self))
                block = self[i:j]
                write_vcf_data(vcf_file, block, rename=rename, fill=fill)


class FeatureChunkedTable(ChunkedTableWrapper):
    """Alternative implementation of the
    :class:`allel.model.ndarray.FeatureTable` class, using a chunked table as
    the backing store.

    Parameters
    ----------
    data: table_like
        Data to be wrapped. May be a tuple or list of columns (array-like),
        a dict mapping names to columns, a bcolz ctable, h5py group,
        numpy recarray, or anything providing a similar interface.
    names : sequence of strings
        Column names.

    """

    array_cls = FeatureTable

    def __init__(self, data, names=None):
        super(FeatureChunkedTable, self).__init__(data, names=names)

    @property
    def n_features(self):
        return len(self)

    def to_mask(self, size, start_name='start', stop_name='end'):
        m = np.zeros(size, dtype=bool)
        start = self[start_name]
        stop = self[stop_name]
        for i, j in zip(start, stop):
            # assume 1-based inclusive coords
            m[i-1:j] = True
        return m

    @staticmethod
    def from_gff3(path, attributes=None, region=None, score_fill=-1,
                  phase_fill=-1, attributes_fill=b'.', dtype=None,
                  blen=None, storage=None, create='table', expectedlen=200000,
                  **kwargs):

        # setup iterator
        recs = iter_gff3(path, attributes=attributes, region=region,
                         score_fill=score_fill, phase_fill=phase_fill,
                         attributes_fill=attributes_fill)

        # read a sample to determine dtype, blen
        recs_sample = list(itertools.islice(recs, 1000))
        if not recs_sample:
            raise ValueError('no records found')
        names = 'seqid', 'source', 'type', 'start', 'end', 'score', 'strand', \
                'phase'
        if attributes:
            names += tuple(attributes)
        ra = np.rec.array(recs_sample, names=names, dtype=dtype)
        dtype = ra.dtype

        # setup output
        storage = _chunked.get_storage(storage)
        out = getattr(storage, create)(ra, expectedlen=expectedlen,
                                       **kwargs)
        blen = _chunked.get_blen_table(out, blen=blen)

        # read block-wise
        block = list(itertools.islice(recs, 0, blen))
        while block:
            a = np.asarray(block, dtype=dtype)
            out.append(a)
            block = list(itertools.islice(recs, 0, blen))

        out = FeatureChunkedTable(out)
        return out


class AlleleCountsChunkedTable(ChunkedTableWrapper):

    def __getitem__(self, item):
        out = super(AlleleCountsChunkedTable, self).__getitem__(item)
        if isinstance(item, string_types):
            # rewrap
            out = AlleleCountsChunkedArray(out.values)
        return out

    @property
    def n_variants(self):
        return len(self)
