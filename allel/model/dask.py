# -*- coding: utf-8 -*-
"""This module provides alternative implementations of array
classes defined in the :mod:`allel.model.ndarray` module, using
`dask.array <http://dask.pydata.org/en/latest/array.html>`_ as the
computational engine.

Dask uses blocked algorithms and task scheduling to break up work into
smaller pieces, allowing computation over large datasets. It also uses
lazy evaluation, meaning that multiple operations can be chained together
into a task graph, reducing total memory requirements for intermediate
results, and only the tasks required to generate the requested
part of the final data set will be executed.

This module is experimental, if you find a bug please `raise an issue on GitHub
<https://github.com/cggh/scikit-allel/issues/new>`_.

This module requires Dask >= 0.7.6.

"""
from __future__ import absolute_import, print_function, division


import numpy as np
import dask.array as da


from allel.util import check_shape, check_dtype, check_ndim, check_integer_dtype
from allel.abc import ArrayWrapper, DisplayAs2D, DisplayAs1D
from allel.compat import copy_method_doc
from .ndarray import GenotypeArray, HaplotypeArray, AlleleCountsArray, GenotypeVector
from .generic import *


__all__ = ['GenotypeDaskArray', 'HaplotypeDaskArray', 'AlleleCountsDaskArray']


def get_chunks(data, chunks=None):
    """Try to guess a reasonable chunk shape to use for block-wise
    algorithms operating over `data`."""

    if chunks is None:

        if hasattr(data, 'chunklen') and hasattr(data, 'shape'):
            # bcolz carray, chunk first dimension only
            return (data.chunklen,) + data.shape[1:]

        elif hasattr(data, 'chunks') and hasattr(data, 'shape') and \
                len(data.chunks) == len(data.shape):
            # h5py dataset or zarr array
            return data.chunks

        else:
            # fall back to something simple, ~4Mb chunks of first dimension
            row = np.asarray(data[0])
            chunklen = max(1, (2**22) // row.nbytes)
            if row.shape:
                chunks = (chunklen,) + row.shape
            else:
                chunks = (chunklen,)
            return chunks

    else:

        return chunks
    
    
def ensure_dask_array(data, chunks=None):
    if isinstance(data, da.Array):
        return data
    else:
        chunks = get_chunks(data, chunks)
        return da.from_array(data, chunks=chunks)


def da_subset(d, sel0, sel1):
    if sel0 is None and sel1 is None:
        out = d
    elif sel1 is None:
        out = d[sel0]
    elif sel0 is None:
        out = d[:, sel1]
    else:
        out = d[sel0][:, sel1]
    return out


class DaskArrayWrapper(ArrayWrapper):

    def __init__(self, data, chunks=None, name=None, lock=False):
        if not isinstance(data, da.Array):
            chunks = get_chunks(data, chunks)
            data = da.from_array(data, chunks=chunks, name=name, lock=lock)
        super(DaskArrayWrapper, self).__init__(data)

    def __repr__(self):
        return self.caption


class GenotypesDask(DaskArrayWrapper):

    array_cls = None

    def __init__(self, data, chunks=None, name=None, lock=False):
        super(GenotypesDask, self).__init__(data, chunks=chunks, name=name, lock=lock)
        self._mask = None
        self._is_phased = None

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
        if mask is not None:
            mask = ensure_dask_array(mask, self.chunks)
            check_shape(mask, self.shape[:-1])
            check_dtype(mask, bool)
        self._mask = mask

    @property
    def is_phased(self):
        return self._is_phased

    @is_phased.setter
    def is_phased(self, is_phased):
        if is_phased is not None:
            is_phased = ensure_dask_array(is_phased, self.chunks)
            check_shape(is_phased, self.shape[:-1])
            check_dtype(is_phased, bool)
        self._is_phased = is_phased

    def compute(self, *args, **kwargs):
        arrays = [self.values]
        if self.mask is not None:
            arrays.append(self.mask)
        if self.is_phased is not None:
            arrays.append(self.is_phased)
        arrays = list(da.compute(arrays, *args, **kwargs))
        out = self.array_cls(arrays.pop())
        if self.mask is not None:
            out.mask = arrays.pop()
        if self.is_phased is not None:
            out.is_phased = arrays.pop()

    def rechunk(self, *args, **kwargs):
        values = self.values.rechunk(*args, **kwargs)
        out = type(self)(values)
        if self.mask is not None:
            out.mask = self.mask.rechunk(*args, **kwargs)
        if self.is_phased is not None:
            out.is_phased = self.is_phased.rechunk(*args, **kwargs)
        return out

    def _method(self, method_name, chunks=None, drop_axis=None, **kwargs):
        if chunks is None:
            # no shape change
            chunks = self.chunks
        array_cls = self.array_cls

        if self.mask is None:
            # simple case, no mask
            def f(block):
                g = array_cls(block)
                method = getattr(g, method_name)
                return method(**kwargs)
            out = da.map_blocks(f, self.values, chunks=chunks, drop_axis=drop_axis)

        else:
            # map with mask
            def f(block, bmask):
                g = array_cls(block)
                g.mask = bmask[..., 0]
                method = getattr(g, method_name)
                return method(**kwargs)
            m = self.mask[..., np.newaxis]
            out = da.map_blocks(f, self.values, m, chunks=chunks, drop_axis=drop_axis)

        return out

    def _method_drop_ploidy(self, method_name, **kwargs):
        chunks = self.chunks[:-1]
        drop_axis = len(self.shape) - 1  # TODO does -1 work?
        return self._method(method_name, chunks=chunks, drop_axis=drop_axis, **kwargs)

    def fill_masked(self, value=-1):
        out = self._method('fill_masked', value=value)
        return type(self)(out)

    def is_called(self):
        return self._method_drop_ploidy('is_called')

    def is_missing(self):
        return self._method_drop_ploidy('is_missing')

    def is_hom(self, allele=None):
        return self._method_drop_ploidy('is_hom', allele=allele)

    def is_hom_ref(self):
        return self._method_drop_ploidy('is_hom_ref')

    def is_hom_alt(self):
        return self._method_drop_ploidy('is_hom_alt')

    def is_het(self, allele=None):
        return self._method_drop_ploidy('is_het', allele=allele)

    def is_call(self, call):
        return self._method_drop_ploidy('is_call', call=call)

    def _count(self, method_name, axis, **kwargs):
        method = getattr(self, method_name)
        out = method(**kwargs).sum(axis=axis)
        if axis is None:
            # result is scalar, might as well compute now (also helps tests)
            return out.compute()[()]
        else:
            return out

    def count_called(self, axis=None):
        return self._count('is_called', axis)

    def count_missing(self, axis=None):
        return self._count('is_missing', axis)

    def count_hom(self, allele=None, axis=None):
        return self._count('is_hom', axis, allele=allele)

    def count_hom_ref(self, axis=None):
        return self._count('is_hom_ref', axis)

    def count_hom_alt(self, axis=None):
        return self._count('is_hom_alt', axis)

    def count_het(self, allele=None, axis=None):
        return self._count('is_het', axis, allele=allele)

    def count_call(self, call, axis=None):
        return self._count('is_call', axis, call=call)

    def str_items(self):
        self.compute().str_items()


class GenotypeDaskVector(GenotypesDask, DisplayAs1D):

    array_cls = GenotypeVector

    @classmethod
    def check_values(cls, data):
        check_ndim(data, 2)
        check_integer_dtype(data)

    def __init__(self, data, chunks=None, name=None, lock=False):
        super(GenotypeDaskVector, self).__init__(data, chunks=chunks, name=name, lock=lock)

    def __getitem__(self, item):
        return index_genotype_vector(self, item, cls=type(self))

    def compress(self, condition, axis=0, **kwargs):
        return compress_genotypes(self, condition, axis=axis, wrap_axes={0},
                                  cls=type(self), compress=da.compress, **kwargs)

    def take(self, indices, axis=0, **kwargs):
        return take_genotypes(self, indices, axis=axis, wrap_axes={0}, cls=type(self),
                              take=da.take, **kwargs)

    def concatenate(self, others, axis=0, **kwargs):
        return concatenate_genotypes(self, others, axis=axis, wrap_axes={0},
                                     cls=type(self), concatenate=da.concatenate,
                                     **kwargs)


class GenotypeDaskArray(GenotypesDask, DisplayAs2D):

    array_cls = GenotypeArray

    @classmethod
    def check_values(cls, data):
        check_ndim(data, 3)
        check_integer_dtype(data)

    def __init__(self, data, chunks=None, name=None, lock=False):
        super(GenotypeDaskArray, self).__init__(data, chunks=chunks, name=name, lock=lock)

    def __getitem__(self, item):
        return index_genotype_array(self, item, array_cls=type(self),
                                    vector_cls=GenotypeDaskVector)

    @property
    def n_variants(self):
        """Number of variants."""
        return self.shape[0]

    @property
    def n_samples(self):
        """Number of samples."""
        return self.shape[1]

    def count_alleles(self, max_allele=None, subpop=None):

        # if max_allele not specified, count all alleles
        if max_allele is None:
            max_allele = self.max().compute()[()]

        # deal with subpop
        if subpop:
            gd = self.take(subpop, axis=1)
        else:
            gd = self

        # determine output chunks - preserve axis0; change axis1, axis2
        chunks = (gd.chunks[0], (1,)*len(gd.chunks[1]), (max_allele+1,))

        if self.mask is None:

            # simple case, no mask
            def f(block):
                gb = GenotypeArray(block)
                return gb.count_alleles(max_allele=max_allele)[:, None, :]

            # map blocks and reduce
            out = gd.map_blocks(f, chunks=chunks).sum(axis=1)

        else:

            # map with mask
            def f(block, bmask):
                g = GenotypeArray(block)
                g.mask = bmask[:, :, 0]
                return g.count_alleles(max_allele=max_allele)[:, None, :]

            md = self.mask[:, :, None]
            out = da.map_blocks(f, gd, md, chunks=chunks).sum(axis=1)

        return AlleleCountsDaskArray(out)

    def count_alleles_subpops(self, subpops, max_allele=None):

        # if max_allele not specified, count all alleles
        if max_allele is None:
            max_allele = self.max().compute()[()]

        return {k: self.count_alleles(max_allele=max_allele, subpop=v)
                for k, v in subpops.items()}

    def to_packed(self, boundscheck=True):
        return self._method_drop_ploidy('to_packed', boundscheck=boundscheck)

    @classmethod
    def from_packed(cls, packed, chunks=None):
        def f(block):
            return GenotypeArray.from_packed(block)
        packed = ensure_dask_array(packed, chunks)
        chunks = (packed.chunks[0], packed.chunks[1], (2,))
        out = da.map_blocks(f, packed, chunks=chunks, new_axis=2)
        return cls(out)

    def map_alleles(self, mapping, **kwargs):

        def f(block, bmapping):
            g = GenotypeArray(block)
            m = bmapping[:, 0, :]
            return g.map_alleles(m)

        # obtain dask array
        mapping = da.from_array(mapping, chunks=(self.chunks[0], None))

        # map blocks
        out = da.map_blocks(f, self, mapping[:, None, :], chunks=self.chunks)
        return type(self)(out)

    def to_allele_counts(self, alleles=None):

        # determine alleles to count
        if alleles is None:
            m = self.max().compute()[()]
            alleles = list(range(m+1))

        chunks = (self.chunks[0], self.chunks[1], (len(alleles),))
        return self._method('to_allele_counts', chunks=chunks, alleles=alleles)

    def to_gt(self, phased=False, max_allele=None):
        return self._method_drop_ploidy('to_gt', phased=phased, max_allele=max_allele)

    def to_haplotypes(self):
        out = self.reshape(self.shape[0], -1)
        return HaplotypeDaskArray(out)

    def to_n_ref(self, fill=0, dtype='i1'):
        return self._method_drop_ploidy('to_n_ref', fill=fill, dtype=dtype)

    def to_n_alt(self, fill=0, dtype='i1'):
        return self._method_drop_ploidy('to_n_alt', fill=fill, dtype=dtype)

    def compress(self, condition, axis=0, **kwargs):
        return compress_genotypes(self, condition, axis=axis, wrap_axes={0, 1},
                                  cls=type(self), compress=da.compress, **kwargs)

    def take(self, indices, axis=0, **kwargs):
        return take_genotypes(self, indices, axis=axis, wrap_axes={0, 1}, cls=type(self),
                              take=da.take, **kwargs)

    def subset(self, sel0=None, sel1=None, **kwargs):
        return subset_genotype_array(self, sel0, sel1, cls=type(self),
                                     subset=da_subset, **kwargs)

    def concatenate(self, others, axis=0, **kwargs):
        return concatenate_genotypes(self, others, axis=axis, wrap_axes={0, 1},
                                     cls=type(self), concatenate=da.concatenate,
                                     **kwargs)


# copy docstrings
copy_method_doc(GenotypeDaskArray.fill_masked, GenotypeArray.fill_masked)
copy_method_doc(GenotypeDaskArray.subset, GenotypeArray.subset)
copy_method_doc(GenotypeDaskArray.is_called, GenotypeArray.is_called)
copy_method_doc(GenotypeDaskArray.is_missing, GenotypeArray.is_missing)
copy_method_doc(GenotypeDaskArray.is_hom, GenotypeArray.is_hom)
copy_method_doc(GenotypeDaskArray.is_hom_ref, GenotypeArray.is_hom_ref)
copy_method_doc(GenotypeDaskArray.is_hom_alt, GenotypeArray.is_hom_alt)
copy_method_doc(GenotypeDaskArray.is_het, GenotypeArray.is_het)
copy_method_doc(GenotypeDaskArray.is_call, GenotypeArray.is_call)
copy_method_doc(GenotypeDaskArray.to_haplotypes, GenotypeArray.to_haplotypes)
copy_method_doc(GenotypeDaskArray.to_n_ref, GenotypeArray.to_n_ref)
copy_method_doc(GenotypeDaskArray.to_n_alt, GenotypeArray.to_n_alt)
copy_method_doc(GenotypeDaskArray.to_allele_counts, GenotypeArray.to_allele_counts)
copy_method_doc(GenotypeDaskArray.to_packed, GenotypeArray.to_packed)
GenotypeDaskArray.from_packed.__doc__ = GenotypeArray.from_packed.__doc__
copy_method_doc(GenotypeDaskArray.count_alleles, GenotypeArray.count_alleles)
copy_method_doc(GenotypeDaskArray.count_alleles_subpops,
                GenotypeArray.count_alleles_subpops)
copy_method_doc(GenotypeDaskArray.to_gt, GenotypeArray.to_gt)
copy_method_doc(GenotypeDaskArray.map_alleles, GenotypeArray.map_alleles)
copy_method_doc(GenotypeDaskArray.concatenate, GenotypeArray.concatenate)


class HaplotypeDaskArray(DaskArrayWrapper, DisplayAs2D):

    @classmethod
    def check_values(cls, data):
        check_ndim(data, 2)
        check_integer_dtype(data)

    def __init__(self, data, chunks=None, name=None, lock=False):
        super(HaplotypeDaskArray, self).__init__(data, chunks=chunks, name=name, lock=lock)

    def __getitem__(self, item):
        return index_haplotype_array(self, item, cls=type(self))

    @property
    def n_variants(self):
        return self.shape[0]

    @property
    def n_haplotypes(self):
        return self.shape[1]

    def compute(self, *args, **kwargs):
        out = super(HaplotypeDaskArray, self).compute(*args, **kwargs)
        return HaplotypeArray(out)

    def to_genotypes(self, ploidy=2):

        # check ploidy is compatible
        if (self.n_haplotypes % ploidy) > 0:
            raise ValueError('incompatible ploidy')

        # mapper function
        def f(block):
            h = HaplotypeArray(block)
            return h.to_genotypes(ploidy)

        # rechunk across all columns to ensure chunk boundaries don't break
        # individuals
        hd = self.rechunk(chunks={1: self.n_haplotypes})

        # determine output chunks
        chunks = (hd.chunks[0], hd.chunks[1], (ploidy,))

        # map blocks
        out = hd.map_blocks(f, chunks=chunks, new_axis=2)
        return GenotypeDaskArray(out)

    def is_called(self):
        return self >= 0

    def is_missing(self):
        return self < 0

    def is_ref(self):
        return self == 0

    def is_alt(self):
        return self > 0

    def is_call(self, allele):
        return self == allele

    def count_called(self, axis=None):
        return self.is_called().sum(axis=axis)

    def count_missing(self, axis=None):
        return self.is_missing().sum(axis=axis)

    def count_ref(self, axis=None):
        return self.is_ref().sum(axis=axis)

    def count_alt(self, axis=None):
        return self.is_alt().sum(axis=axis)

    def count_call(self, allele, axis=None):
        return self.is_call(allele).sum(axis=axis)

    def count_alleles(self, max_allele=None, subpop=None):

        # if max_allele not specified, count all alleles
        if max_allele is None:
            max_allele = self.max().compute()[()]

        # deal with subpop
        if subpop:
            hd = self.take(subpop, axis=1)
        else:
            hd = self

        # determine output chunks - preserve axis0, change axis1, new axis2
        chunks = (hd.chunks[0], (1,)*len(hd.chunks[1]), (max_allele+1,))

        # mapper function
        def f(block):
            h = HaplotypeArray(block)
            return h.count_alleles(max_allele=max_allele)[:, None, :]

        # map blocks and reduce
        out = hd.map_blocks(f, chunks=chunks, new_axis=2).sum(axis=1)
        return AlleleCountsDaskArray(out)

    def count_alleles_subpops(self, subpops, max_allele=None):

        # if max_allele not specified, count all alleles
        if max_allele is None:
            max_allele = self.max().compute()[()]

        return {k: self.count_alleles(max_allele=max_allele, subpop=v)
                for k, v in subpops.items()}

    def map_alleles(self, mapping, **kwargs):

        def f(block, bmapping):
            h = HaplotypeArray(block)
            return h.map_alleles(bmapping)

        # obtain dask array
        mapping = da.from_array(mapping, chunks=(self.chunks[0], None))

        # map blocks
        out = da.map_blocks(f, self, mapping,
                            chunks=self.chunks)
        return view_subclass(out, HaplotypeDaskArray)

    def concatenate(self, others, axis=0):
        pass


# copy docstrings
copy_method_doc(HaplotypeDaskArray.to_genotypes,
                HaplotypeArray.to_genotypes)
copy_method_doc(HaplotypeDaskArray.count_alleles,
                HaplotypeArray.count_alleles)
copy_method_doc(HaplotypeDaskArray.count_alleles_subpops,
                HaplotypeArray.count_alleles_subpops)
copy_method_doc(HaplotypeDaskArray.map_alleles,
                HaplotypeArray.map_alleles)


class AlleleCountsDaskArray(DaskArrayWrapper, DisplayAs2D):
    pass
