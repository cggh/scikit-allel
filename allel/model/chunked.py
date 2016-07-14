# -*- coding: utf-8 -*-
"""This module provides alternative implementations of array and table
classes defined in the :mod:`allel.model.ndarray` module, using
chunked arrays for data storage. Chunked arrays can be compressed and
optionally stored on disk, providing a means for working with data too
large to fit uncompressed in main memory.

Either HDF5 (via `h5py <http://www.h5py.org/>`_) or `bcolz
<http://bcolz.blosc.org>`_ can be used as the underlying storage
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
from allel.model import ndarray as _ndarray
from allel import chunked as _chunked
from allel.io import write_vcf_header, write_vcf_data, iter_gff3


__all__ = ['GenotypeChunkedArray', 'HaplotypeChunkedArray',
           'AlleleCountsChunkedArray', 'VariantChunkedTable',
           'FeatureChunkedTable', 'AlleleCountsChunkedTable']


class GenotypeChunkedArray(_chunked.ChunkedArray):
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
        GenotypeChunkedArray((3, 2, 2), int8, chunks=(2, 2, 2))
          nbytes: 12; cbytes: 30; cratio: 0.4;
          compression: gzip; compression_opts: 1;
          data: h5py._hl.dataset.Dataset
        >>> g.data
        <HDF5 dataset "genotype": shape (3, 2, 2), type "|i1">

    Obtain a numpy array by slicing, e.g.::

        >>> g[:]
        GenotypeArray((3, 2, 2), dtype=int8)
        [[[ 0  0]
          [ 0  1]]
         [[ 0  1]
          [ 1  1]]
         [[ 0  2]
          [-1 -1]]]

    Note that most methods will return a chunked array, using whatever
    chunked storage is set as default (bcolz carray) or specified
    directly via the `storage` keyword argument. E.g.::

        >>> g.copy()
        GenotypeChunkedArray((3, 2, 2), int8, chunks=(4096, 2, 2))
          nbytes: 12; cbytes: 16.0K; cratio: 0.0;
          compression: blosc; compression_opts: cparams(clevel=5, shuffle=1, cname='lz4', quantize=0);
          data: bcolz.carray_ext.carray
        >>> g.copy(storage='zarrmem')
        GenotypeChunkedArray((3, 2, 2), int8, chunks=(262144, 2, 2))
          nbytes: 12; cbytes: 4.9K; cratio: 0.0;
          compression: blosc; compression_opts: {'clevel': 5, 'cname': 'blosclz', 'shuffle': 1};
          data: zarr.core.Array
        >>> g.copy(storage='hdf5mem_zlib1')
        GenotypeChunkedArray((3, 2, 2), int8, chunks=(262144, 2, 2))
          nbytes: 12; cbytes: 4.5K; cratio: 0.0;
          compression: gzip; compression_opts: 1;
          data: h5py._hl.dataset.Dataset

    """  # flake8: noqa

    def __init__(self, data):
        super(GenotypeChunkedArray, self).__init__(data)
        self._check_input_data(self.data)
        self._mask = None

    @staticmethod
    def _check_input_data(data):
        if len(data.shape) != 3:
            raise ValueError('expected 3 dimensions')
        if data.dtype.kind not in 'ui':
            raise TypeError('expected integer dtype')

    def __getitem__(self, *args):
        out = super(GenotypeChunkedArray, self).__getitem__(*args)
        if hasattr(out, 'shape') \
                and len(self.shape) == len(out.shape) \
                and self.shape[2] == out.shape[2]:
            # dimensionality and ploidy preserved
            out = _ndarray.GenotypeArray(out)
            if self.mask is not None:
                # attempt to slice mask too
                m = self.mask.__getitem__(*args)
                out.mask = m
        return out

    def _repr_html_(self):
        return self[:6].to_html_str(caption=repr(self))

    @property
    def n_variants(self):
        return self.shape[0]

    @property
    def n_samples(self):
        return self.shape[1]

    @property
    def ploidy(self):
        return self.shape[2]

    @property
    def n_calls(self):
        return self.shape[0] * self.shape[1]

    @property
    def n_allele_calls(self):
        return self.shape[0] * self.shape[1] * self.shape[2]

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask):

        # check input
        if not hasattr(mask, 'shape'):
            mask = np.asarray(mask)
        if mask.shape != self.shape[:2]:
            raise ValueError('mask has incorrect shape')

        # store
        self._mask = _chunked.ChunkedArray(mask)

    def compress(self, condition, axis=0, **storage_kwargs):
        out = super(GenotypeChunkedArray, self).compress(condition, axis=axis,
                                                         **storage_kwargs)
        if self.mask is not None:
            out.mask = self.mask.compress(condition, axis=axis,
                                          **storage_kwargs)
        return out

    def take(self, indices, axis=0, **storage_kwargs):
        out = super(GenotypeChunkedArray, self).take(indices, axis=axis,
                                                     **storage_kwargs)
        if self.mask is not None:
            out.mask = self.mask.take(indices, axis=axis, **storage_kwargs)
        return out

    def subset(self, sel0=None, sel1=None, **storage_kwargs):
        out = super(GenotypeChunkedArray, self).subset(sel0, sel1,
                                                       **storage_kwargs)
        if self.mask is not None:
            out.mask = self.mask.subset(sel0, sel1, **storage_kwargs)
        return out

    def fill_masked(self, value=-1, **storage_kwargs):
        out = self.apply_method('fill_masked', kwargs=dict(value=value),
                                **storage_kwargs)
        return GenotypeChunkedArray(out)

    def is_called(self, **storage_kwargs):
        return self.apply_method('is_called', **storage_kwargs)

    def is_missing(self, **storage_kwargs):
        return self.apply_method('is_missing', **storage_kwargs)

    def is_hom(self, allele=None, **storage_kwargs):
        return self.apply_method('is_hom', kwargs=dict(allele=allele),
                                 **storage_kwargs)

    def is_hom_ref(self, **storage_kwargs):
        return self.apply_method('is_hom_ref', **storage_kwargs)

    def is_hom_alt(self, **storage_kwargs):
        return self.apply_method('is_hom_alt', **storage_kwargs)

    def is_het(self, allele=None, **storage_kwargs):
        return self.apply_method('is_het', kwargs=dict(allele=allele),
                                 **storage_kwargs)

    def is_call(self, call, **storage_kwargs):
        return self.apply_method('is_call', kwargs=dict(call=call),
                                 **storage_kwargs)

    def _count(self, method_name, axis, kwargs=None, **storage_kwargs):
        if kwargs is None:
            kwargs = dict()

        def mapper(block):
            method = getattr(block, method_name)
            return method(**kwargs)
        out = self.sum(axis=axis, mapper=mapper, **storage_kwargs)
        return out

    def count_called(self, axis=None, **storage_kwargs):
        return self._count('is_called', axis, **storage_kwargs)

    def count_missing(self, axis=None, **storage_kwargs):
        return self._count('is_missing', axis, **storage_kwargs)

    def count_hom(self, allele=None, axis=None, **storage_kwargs):
        return self._count('is_hom', axis, kwargs=dict(allele=allele),
                           **storage_kwargs)

    def count_hom_ref(self, axis=None, **storage_kwargs):
        return self._count('is_hom_ref', axis, **storage_kwargs)

    def count_hom_alt(self, axis=None, **storage_kwargs):
        return self._count('is_hom_alt', axis, **storage_kwargs)

    def count_het(self, allele=None, axis=None, **storage_kwargs):
        return self._count('is_het', axis, kwargs=dict(allele=allele),
                           **storage_kwargs)

    def count_call(self, call, axis=None, **storage_kwargs):
        return self._count('is_call', axis, kwargs=dict(call=call),
                           **storage_kwargs)

    def to_haplotypes(self, **storage_kwargs):
        out = self.apply_method('to_haplotypes', **storage_kwargs)
        return HaplotypeChunkedArray(out)

    def to_n_ref(self, fill=0, dtype='i1', **storage_kwargs):
        out = self.apply_method('to_n_ref', kwargs=dict(fill=fill,
                                                        dtype=dtype),
                                **storage_kwargs)
        return _chunked.ChunkedArray(out)

    def to_n_alt(self, fill=0, dtype='i1', **storage_kwargs):
        out = self.apply_method('to_n_alt', kwargs=dict(fill=fill,
                                                        dtype=dtype),
                                **storage_kwargs)
        return _chunked.ChunkedArray(out)

    def to_allele_counts(self, alleles=None, **storage_kwargs):
        # determine alleles to count
        if alleles is None:
            m = self.max()
            alleles = list(range(m+1))

        out = self.apply_method('to_allele_counts',
                                kwargs=dict(alleles=alleles),
                                **storage_kwargs)
        return _chunked.ChunkedArray(out)

    def to_packed(self, boundscheck=True, **storage_kwargs):
        out = self.apply_method('to_packed',
                                kwargs=dict(boundscheck=boundscheck),
                                **storage_kwargs)
        return _chunked.ChunkedArray(out)

    @staticmethod
    def from_packed(packed, **storage_kwargs):
        def f(block):
            return _ndarray.GenotypeArray.from_packed(block)
        out = _chunked.apply(packed, f, **storage_kwargs)
        return GenotypeChunkedArray(out)

    def count_alleles(self, max_allele=None, subpop=None, **storage_kwargs):
        # if max_allele not specified, count all alleles
        if max_allele is None:
            max_allele = self.max()

        out = self.apply_method('count_alleles',
                                kwargs=dict(max_allele=max_allele,
                                            subpop=subpop),
                                **storage_kwargs)
        return AlleleCountsChunkedArray(out)

    def count_alleles_subpops(self, subpops, max_allele=None,
                              **storage_kwargs):
        if max_allele is None:
            max_allele = self.max()

        def f(block):
            return block.count_alleles_subpops(subpops, max_allele=max_allele)
        out = _chunked.apply(self, f, create='table', **storage_kwargs)
        return AlleleCountsChunkedTable(out)

    def to_gt(self, phased=False, max_allele=None, **storage_kwargs):
        out = self.apply_method('to_gt',
                                kwargs=dict(phased=phased,
                                            max_allele=max_allele),
                                **storage_kwargs)
        return _chunked.ChunkedArray(out)

    def map_alleles(self, mapping, **storage_kwargs):
        def f(block, bmapping):
            return block.map_alleles(bmapping, copy=False)
        domain = (self, mapping)
        out = _chunked.apply(domain, f, **storage_kwargs)
        return GenotypeChunkedArray(out)


# copy docstrings
copy_method_doc(GenotypeChunkedArray.fill_masked,
                _ndarray.GenotypeArray.fill_masked)
copy_method_doc(GenotypeChunkedArray.subset,
                _ndarray.GenotypeArray.subset)
copy_method_doc(GenotypeChunkedArray.is_called,
                _ndarray.GenotypeArray.is_called)
copy_method_doc(GenotypeChunkedArray.is_missing,
                _ndarray.GenotypeArray.is_missing)
copy_method_doc(GenotypeChunkedArray.is_hom,
                _ndarray.GenotypeArray.is_hom)
copy_method_doc(GenotypeChunkedArray.is_hom_ref,
                _ndarray.GenotypeArray.is_hom_ref)
copy_method_doc(GenotypeChunkedArray.is_hom_alt,
                _ndarray.GenotypeArray.is_hom_alt)
copy_method_doc(GenotypeChunkedArray.is_het,
                _ndarray.GenotypeArray.is_het)
copy_method_doc(GenotypeChunkedArray.is_call,
                _ndarray.GenotypeArray.is_call)
copy_method_doc(GenotypeChunkedArray.to_haplotypes,
                _ndarray.GenotypeArray.to_haplotypes)
copy_method_doc(GenotypeChunkedArray.to_n_ref,
                _ndarray.GenotypeArray.to_n_ref)
copy_method_doc(GenotypeChunkedArray.to_n_alt,
                _ndarray.GenotypeArray.to_n_alt)
copy_method_doc(GenotypeChunkedArray.to_allele_counts,
                _ndarray.GenotypeArray.to_allele_counts)
copy_method_doc(GenotypeChunkedArray.to_packed,
                _ndarray.GenotypeArray.to_packed)
GenotypeChunkedArray.from_packed.__doc__ = \
    _ndarray.GenotypeArray.from_packed.__doc__
copy_method_doc(GenotypeChunkedArray.count_alleles,
                _ndarray.GenotypeArray.count_alleles)
copy_method_doc(GenotypeChunkedArray.count_alleles_subpops,
                _ndarray.GenotypeArray.count_alleles_subpops)
copy_method_doc(GenotypeChunkedArray.to_gt, _ndarray.GenotypeArray.to_gt)
copy_method_doc(GenotypeChunkedArray.map_alleles,
                _ndarray.GenotypeArray.map_alleles)
copy_method_doc(GenotypeChunkedArray.hstack, _ndarray.GenotypeArray.hstack)
copy_method_doc(GenotypeChunkedArray.vstack, _ndarray.GenotypeArray.vstack)


class HaplotypeChunkedArray(_chunked.ChunkedArray):
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
        self._check_input_data(self.data)

    @staticmethod
    def _check_input_data(data):
        if len(data.shape) != 2:
            raise ValueError('expected 2 dimensions')
        if data.dtype.kind not in 'ui':
            raise TypeError('expected integer dtype')

    def __getitem__(self, *args):
        out = super(HaplotypeChunkedArray, self).__getitem__(*args)
        if hasattr(out, 'shape') and len(self.shape) == len(out.shape):
            # dimensionality preserved
            out = _ndarray.HaplotypeArray(out)
        return out

    def _repr_html_(self):
        return self[:6].to_html_str(caption=repr(self))

    @property
    def n_variants(self):
        return self.shape[0]

    @property
    def n_haplotypes(self):
        return self.shape[1]

    def to_genotypes(self, ploidy=2, **storage_kwargs):

        # check ploidy is compatible
        if (self.n_haplotypes % ploidy) > 0:
            raise ValueError('incompatible ploidy')

        # build output
        def f(block):
            return block.to_genotypes(ploidy)

        out = self.apply(f, **storage_kwargs)
        return GenotypeChunkedArray(out)

    def is_called(self, **storage_kwargs):
        return self.__ge__(0, **storage_kwargs)

    def is_missing(self, **storage_kwargs):
        return self.__lt__(0, **storage_kwargs)

    def is_ref(self, **storage_kwargs):
        return self.__eq__(0, **storage_kwargs)

    def is_alt(self, **storage_kwargs):
        return self.__gt__(0, **storage_kwargs)

    def is_call(self, allele, **storage_kwargs):
        return self.__eq__(allele, **storage_kwargs)

    def _count(self, method_name, axis, kwargs=None, **storage_kwargs):
        if kwargs is None:
            kwargs = dict()

        def mapper(block):
            method = getattr(block, method_name)
            return method(**kwargs)
        out = self.sum(axis=axis, mapper=mapper, **storage_kwargs)
        return out

    def count_called(self, axis=None, **storage_kwargs):
        return self._count('is_called', axis=axis, **storage_kwargs)

    def count_missing(self, axis=None, **storage_kwargs):
        return self._count('is_missing', axis=axis, **storage_kwargs)

    def count_ref(self, axis=None, **storage_kwargs):
        return self._count('is_ref', axis=axis, **storage_kwargs)

    def count_alt(self, axis=None, **storage_kwargs):
        return self._count('is_alt', axis=axis, **storage_kwargs)

    def count_call(self, allele, axis=None, **storage_kwargs):
        return self._count('is_call', axis=axis,
                           kwargs=dict(allele=allele),
                           **storage_kwargs)

    def count_alleles(self, max_allele=None, subpop=None, **storage_kwargs):
        # if max_allele not specified, count all alleles
        if max_allele is None:
            max_allele = self.max()

        def f(block):
            return block.count_alleles(max_allele=max_allele, subpop=subpop)
        out = self.apply(f, **storage_kwargs)
        return AlleleCountsChunkedArray(out)

    def count_alleles_subpops(self, subpops, max_allele=None,
                              **storage_kwargs):
        if max_allele is None:
            max_allele = self.max()

        def f(block):
            return block.count_alleles_subpops(subpops, max_allele=max_allele)
        out = _chunked.apply(self, f, create='table', **storage_kwargs)
        return AlleleCountsChunkedTable(out)

    def map_alleles(self, mapping, **storage_kwargs):
        def f(block, bmapping):
            return block.map_alleles(bmapping, copy=False)
        domain = (self, mapping)
        out = _chunked.apply(domain, f, **storage_kwargs)
        return HaplotypeChunkedArray(out)

    def subset(self, sel0=None, sel1=None, **storage_kwargs):
        out = super(HaplotypeChunkedArray, self).subset(sel0, sel1,
                                                        **storage_kwargs)
        return out


# copy docstrings
copy_method_doc(HaplotypeChunkedArray.to_genotypes,
                _ndarray.HaplotypeArray.to_genotypes)
copy_method_doc(HaplotypeChunkedArray.count_alleles,
                _ndarray.HaplotypeArray.count_alleles)
copy_method_doc(HaplotypeChunkedArray.count_alleles_subpops,
                _ndarray.HaplotypeArray.count_alleles_subpops)
copy_method_doc(HaplotypeChunkedArray.map_alleles,
                _ndarray.HaplotypeArray.map_alleles)
copy_method_doc(HaplotypeChunkedArray.subset,
                _ndarray.HaplotypeArray.subset)


class AlleleCountsChunkedArray(_chunked.ChunkedArray):
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
        self._check_input_data(self.data)

    @staticmethod
    def _check_input_data(data):
        if len(data.shape) != 2:
            raise ValueError('expected 2 dimensions')
        if data.dtype.kind not in 'ui':
            raise TypeError('expected integer dtype')

    def __getitem__(self, *args):
        out = super(AlleleCountsChunkedArray, self).__getitem__(*args)
        if hasattr(out, 'shape') and len(self.shape) == len(out.shape) and \
                out.shape[1] == self.shape[1]:
            # dimensionality and allele indices preserved
            out = _ndarray.AlleleCountsArray(out)
        return out

    def __add__(self, other):
        ret = super(AlleleCountsChunkedArray, self).__add__(other)
        if hasattr(other, 'shape') and other.shape == self.shape:
            ret = AlleleCountsChunkedArray(ret.data)
        return ret

    def __sub__(self, other):
        ret = super(AlleleCountsChunkedArray, self).__sub__(other)
        if hasattr(other, 'shape') and other.shape == self.shape:
            ret = AlleleCountsChunkedArray(ret.data)
        return ret

    def _repr_html_(self):
        return self[:6].to_html_str(caption=repr(self))

    @property
    def n_variants(self):
        return self.shape[0]

    @property
    def n_alleles(self):
        return self.shape[1]

    def to_frequencies(self, fill=np.nan, **storage_kwargs):
        out = self.apply_method('to_frequencies',
                                kwargs=dict(fill=fill),
                                **storage_kwargs)
        return _chunked.ChunkedArray(out)

    def allelism(self, **storage_kwargs):
        out = self.apply_method('allelism', **storage_kwargs)
        return _chunked.ChunkedArray(out)

    def max_allele(self, **storage_kwargs):
        out = self.apply_method('max_allele', **storage_kwargs)
        return _chunked.ChunkedArray(out)

    def is_variant(self, **storage_kwargs):
        out = self.apply_method('is_variant', **storage_kwargs)
        return _chunked.ChunkedArray(out)

    def is_non_variant(self, **storage_kwargs):
        out = self.apply_method('is_non_variant', **storage_kwargs)
        return _chunked.ChunkedArray(out)

    def is_segregating(self, **storage_kwargs):
        out = self.apply_method('is_segregating', **storage_kwargs)
        return _chunked.ChunkedArray(out)

    def is_non_segregating(self, allele=None, **storage_kwargs):
        out = self.apply_method('is_non_segregating',
                                kwargs=dict(allele=allele),
                                **storage_kwargs)
        return _chunked.ChunkedArray(out)

    def is_singleton(self, allele=1, **storage_kwargs):
        out = self.apply_method('is_singleton',
                                kwargs=dict(allele=allele),
                                **storage_kwargs)
        return _chunked.ChunkedArray(out)

    def is_doubleton(self, allele=1, **storage_kwargs):
        out = self.apply_method('is_doubleton',
                                kwargs=dict(allele=allele),
                                **storage_kwargs)
        return _chunked.ChunkedArray(out)

    def is_biallelic(self, **storage_kwargs):
        out = self.apply_method('is_biallelic', **storage_kwargs)
        return _chunked.ChunkedArray(out)

    def is_biallelic_01(self, min_mac=None, **storage_kwargs):
        out = self.apply_method('is_biallelic_01',
                                kwargs=dict(min_mac=min_mac),
                                **storage_kwargs)
        return _chunked.ChunkedArray(out)

    def _count(self, method_name, kwargs=None, **storage_kwargs):
        if kwargs is None:
            kwargs = dict()

        def mapper(block):
            method = getattr(block, method_name)
            return method(**kwargs)
        out = self.sum(mapper=mapper, **storage_kwargs)
        return out

    def count_variant(self, **storage_kwargs):
        return self._count('is_variant', **storage_kwargs)

    def count_non_variant(self, **storage_kwargs):
        return self._count('is_non_variant', **storage_kwargs)

    def count_segregating(self, **storage_kwargs):
        return self._count('is_segregating', **storage_kwargs)

    def count_non_segregating(self, allele=None, **storage_kwargs):
        return self._count('is_non_segregating', kwargs=dict(allele=allele),
                           **storage_kwargs)

    def count_singleton(self, allele=1, **storage_kwargs):
        return self._count('is_singleton', kwargs=dict(allele=allele),
                           **storage_kwargs)

    def count_doubleton(self, allele=1, **storage_kwargs):
        return self._count('is_doubleton', kwargs=dict(allele=allele),
                           **storage_kwargs)

    def map_alleles(self, mapping, **storage_kwargs):
        def f(block, bmapping):
            return block.map_alleles(bmapping)
        domain = (self, mapping)
        out = _chunked.apply(domain, f, **storage_kwargs)
        return AlleleCountsChunkedArray(out)


copy_method_doc(AlleleCountsChunkedArray.allelism,
                _ndarray.AlleleCountsArray.allelism)
copy_method_doc(AlleleCountsChunkedArray.max_allele,
                _ndarray.AlleleCountsArray.max_allele)
copy_method_doc(AlleleCountsChunkedArray.map_alleles,
                _ndarray.AlleleCountsArray.map_alleles)
copy_method_doc(AlleleCountsChunkedArray.to_frequencies,
                _ndarray.AlleleCountsArray.to_frequencies)
copy_method_doc(AlleleCountsChunkedArray.is_variant, 
                _ndarray.AlleleCountsArray.is_variant)
copy_method_doc(AlleleCountsChunkedArray.is_non_variant,
                _ndarray.AlleleCountsArray.is_non_variant)
copy_method_doc(AlleleCountsChunkedArray.is_segregating,
                _ndarray.AlleleCountsArray.is_segregating)
copy_method_doc(AlleleCountsChunkedArray.is_non_segregating,
                _ndarray.AlleleCountsArray.is_non_segregating)
copy_method_doc(AlleleCountsChunkedArray.is_singleton,
                _ndarray.AlleleCountsArray.is_singleton)
copy_method_doc(AlleleCountsChunkedArray.is_doubleton,
                _ndarray.AlleleCountsArray.is_doubleton)
copy_method_doc(AlleleCountsChunkedArray.is_biallelic,
                _ndarray.AlleleCountsArray.is_biallelic)
copy_method_doc(AlleleCountsChunkedArray.is_biallelic_01,
                _ndarray.AlleleCountsArray.is_biallelic_01)


class VariantChunkedTable(_chunked.ChunkedTable):
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
        >>> vt
        VariantChunkedTable(5)
          nbytes: 220; cbytes: 220; cratio: 1.0;
          data: h5py._hl.group.Group

    Obtain a single row::

        >>> vt[0]
        row(CHROM=b'chr1', POS=2, AC=array([1, 2]), QD=4.5, DP=35)

    Obtain a numpy array by slicing::

        >>> vt[:] # doctest: +ELLIPSIS
        VariantTable((5,), dtype=[('CHROM', 'S4'), ('POS', '<i8'), ('AC', ...
        [(b'chr1', 2, [1, 2], 4.5, 35) (b'chr1', 7, [3, 4], 6.7, 12)
         (b'chr2', 3, [5, 6], 1.2, 78) (b'chr2', 9, [7, 8], 4.4, 22)
         (b'chr3', 6, [9, 10], 2.8, 99)]

    Access a subset of columns::

        >>> vt[['CHROM', 'POS']]
        VariantChunkedTable(5)
          nbytes: 60; cbytes: 60; cratio: 1.0;
          data: builtins.list

    Note that most methods will return a chunked table, using whatever
    chunked storage is set as default (bcolz ctable) or specified
    directly via the `storage` keyword argument. E.g.::

        >>> vt.copy()
        VariantChunkedTable(5)
          nbytes: 220; cbytes: 80.0K; cratio: 0.0;
          data: bcolz.ctable.ctable
        >>> vt.copy(storage='zarr')
        VariantChunkedTable(5)
          nbytes: 220; cbytes: 39.1K; cratio: 0.0;
          data: allel.chunked.storage_zarr.ZarrTable
        >>> vt.copy(storage='hdf5mem_zlib1')
        VariantChunkedTable(5)
          nbytes: 220; cbytes: 22.5K; cratio: 0.0;
          data: h5py._hl.files.File

    """  # flake8: noqa

    view_cls = _ndarray.VariantTable

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
            self.index = _ndarray.SortedIndex(self[spec][:], copy=False)
        elif isinstance(spec, (tuple, list)) and len(spec) == 2:
            self.index = _ndarray.SortedMultiIndex(self[spec[0]][:],
                                                   self[spec[1]][:],
                                                   copy=False)
        else:
            raise ValueError('invalid index argument, expected string or '
                             'pair of strings, found %s' % repr(spec))

    def to_vcf(self, path, rename=None, number=None, description=None,
               fill=None, blen=None, write_header=True):
        with open(path, 'w') as vcf_file:
            if write_header:
                write_vcf_header(vcf_file, self, rename=rename, number=number,
                                 description=description)
            blen = _chunked.get_blen_table(self)
            for i in range(0, len(self), blen):
                j = min(i+blen, len(self))
                block = self[i:j]
                write_vcf_data(vcf_file, block, rename=rename, fill=fill)


class FeatureChunkedTable(_chunked.ChunkedTable):
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

    view_cls = _ndarray.FeatureTable

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
                  **storage_kwargs):

        # setup iterator
        recs = iter_gff3(path, attributes=attributes, region=region,
                         score_fill=score_fill, phase_fill=phase_fill,
                         attributes_fill=attributes_fill)

        # read a sample to determine dtype, blen
        recs_sample = list(itertools.islice(recs, 1000))
        names = 'seqid', 'source', 'type', 'start', 'end', 'score', 'strand', \
                'phase'
        if attributes:
            names += tuple(attributes)
        ra = np.rec.array(recs_sample, names=names, dtype=dtype)
        dtype = ra.dtype

        # setup output
        storage = _chunked.get_storage(storage)
        out = getattr(storage, create)(ra, expectedlen=expectedlen,
                                       **storage_kwargs)
        blen = _chunked.get_blen_table(out, blen=blen)

        # read block-wise
        block = list(itertools.islice(recs, 0, blen))
        while block:
            a = np.asarray(block, dtype=dtype)
            out.append(a)
            block = list(itertools.islice(recs, 0, blen))

        out = FeatureChunkedTable(out)
        return out


class AlleleCountsChunkedTable(_chunked.ChunkedTable):

    def __getitem__(self, item):
        out = super(AlleleCountsChunkedTable, self).__getitem__(item)
        if isinstance(item, string_types):
            # rewrap
            out = AlleleCountsChunkedArray(out.data)
        return out
