# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import operator
from collections import namedtuple


import numpy as np


from allel.compat import string_types, copy_method_doc, integer_types
from allel.model.ndarray import GenotypeArray, HaplotypeArray, \
    AlleleCountsArray, recarray_to_html_str, recarray_display
from .backend_base import Backend, check_equal_length, get_column_names
from .backend_numpy import numpy_backend
from .backend_bcolz import bcolz_backend, bcolztmp_backend, \
    bcolz_gzip1_backend, bcolztmp_gzip1_backend
from .backend_h5py import h5tmp_backend, h5tmp_gzip1_backend, h5mem_backend, \
    h5mem_gzip1_backend


default_backend = bcolz_backend


def get_backend(backend=None):
    if backend is None:
        return default_backend
    elif isinstance(backend, string_types):
        # normalise backend
        backend = str(backend).lower()
        if backend in ['numpy', 'ndarray', 'np']:
            return numpy_backend
        elif backend in ['bcolz', 'carray']:
            return bcolz_backend
        elif backend in ['bcolztmp', 'carraytmp']:
            return bcolztmp_backend
        elif backend in ['bcolz_gzip1', 'bcolz_zlib1', 'carray_gzip1',
                         'carray_zlib1']:
            return bcolz_gzip1_backend
        elif backend in ['bcolztmp_gzip1', 'bcolztmp_zlib1',
                         'carraytmp_gzip1', 'carraytmp_zlib1']:
            return bcolztmp_gzip1_backend
        elif backend in ['hdf5', 'h5py', 'h5dmem', 'h5mem']:
            return h5mem_backend
        elif backend in ['h5dtmp', 'h5tmp']:
            return h5tmp_backend
        elif backend in ['h5tmp_gzip1', 'h5dtmp_gzip1', 'h5tmp_zlib1',
                         'h5dtmp_gzip1']:
            return h5tmp_gzip1_backend
        elif backend in ['h5mem_gzip1', 'h5dmem_gzip1', 'h5mem_zlib1',
                         'h5mem_gzip1']:
            return h5mem_gzip1_backend
        else:
            raise ValueError('unknown backend: %s' % backend)
    elif isinstance(backend, Backend):
        # custom backend
        return backend
    else:
        raise ValueError('expected None, string or Backend, found: %r'
                         % backend)


def is_array_like(a):
    return hasattr(a, 'shape') and hasattr(a, 'dtype')


def check_array_like(a, ndim=None):
    if isinstance(a, tuple):
        for x in a:
            check_array_like(x)
    else:
        if not is_array_like(a):
            raise ValueError(
                'expected array-like with shape and dtype, found %r' % a
            )
        if ndim is not None and len(a.shape) != ndim:
            raise ValueError(
                'expected array-like with %s dimensions, found %s' %
                (ndim, len(a.shape))
            )


class ChunkedArray(object):

    def __init__(self, data):
        check_array_like(data)
        self.data = data

    def __getitem__(self, *args):
        return self.data.__getitem__(*args)

    def __setitem__(self, key, value):
        return self.data.__setitem__(key, value)

    def __getattr__(self, item):
        return getattr(self.data, item)

    def __array__(self):
        return np.asarray(self.data[:])

    def __repr__(self):
        return '%s(%s, %s, %s.%s)' % \
               (type(self).__name__, str(self.shape), str(self.dtype),
                type(self.data).__module__, type(self.data).__name__)

    def __str__(self):
        return str(self.data)

    def __len__(self):
        return len(self.data)

    @property
    def ndim(self):
        return len(self.shape)

    def store(self, arr, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))
        backend.store(self, arr, **kwargs)

    def copy(self, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))
        out = backend.copy(self, **kwargs)
        return type(self)(out)

    def max(self, axis=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))
        return backend.amax(self, axis=axis, **kwargs)

    def min(self, axis=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))
        return backend.amin(self, axis=axis, **kwargs)

    def sum(self, axis=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))
        return backend.sum(self, axis=axis, **kwargs)

    def op_scalar(self, op, other, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))
        out = backend.op_scalar(self, op, other, **kwargs)
        return ChunkedArray(out)

    def __eq__(self, other, **kwargs):
        return self.op_scalar(operator.eq, other, **kwargs)

    def __ne__(self, other, **kwargs):
        return self.op_scalar(operator.ne, other, **kwargs)

    def __lt__(self, other, **kwargs):
        return self.op_scalar(operator.lt, other, **kwargs)

    def __gt__(self, other, **kwargs):
        return self.op_scalar(operator.gt, other, **kwargs)

    def __le__(self, other, **kwargs):
        return self.op_scalar(operator.le, other, **kwargs)

    def __ge__(self, other, **kwargs):
        return self.op_scalar(operator.ge, other, **kwargs)

    def __add__(self, other, **kwargs):
        return self.op_scalar(operator.add, other, **kwargs)

    def __floordiv__(self, other, **kwargs):
        return self.op_scalar(operator.floordiv, other, **kwargs)

    def __mod__(self, other, **kwargs):
        return self.op_scalar(operator.mod, other, **kwargs)

    def __mul__(self, other, **kwargs):
        return self.op_scalar(operator.mul, other, **kwargs)

    def __pow__(self, other, **kwargs):
        return self.op_scalar(operator.pow, other, **kwargs)

    def __sub__(self, other, **kwargs):
        return self.op_scalar(operator.sub, other, **kwargs)

    def __truediv__(self, other, **kwargs):
        return self.op_scalar(operator.truediv, other, **kwargs)

    def compress(self, condition, axis=0, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))
        out = backend.compress(self, condition, axis=axis, **kwargs)
        return type(self)(out)

    def take(self, indices, axis=0, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))
        out = backend.take(self, indices, axis=axis, **kwargs)
        return type(self)(out)

    def subset(self, sel0, sel1, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))
        out = backend.subset(self, sel0, sel1, **kwargs)
        return type(self)(out)

    def hstack(self, *others, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))
        tup = (self,) + others
        out = backend.hstack(tup, **kwargs)
        return type(self)(out)

    def vstack(self, *others, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))
        tup = (self,) + others
        out = backend.vstack(tup, **kwargs)
        return type(self)(out)


class GenotypeChunkedArray(ChunkedArray):
    """TODO

    """

    def __init__(self, data):
        self.check_input_data(data)
        super(GenotypeChunkedArray, self).__init__(data)
        self._mask = None

    @staticmethod
    def check_input_data(data):
        check_array_like(data, 3)
        if data.dtype.kind not in 'ui':
            raise TypeError('integer dtype required')

    def __getitem__(self, *args):
        out = self.data.__getitem__(*args)
        if is_array_like(out) \
                and len(self.shape) == len(out.shape) \
                and self.shape[2] == out.shape[2]:
            # dimensionality and ploidy preserved
            out = GenotypeArray(out)
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
        """Total number of genotype calls (n_variants * n_samples)."""
        return self.shape[0] * self.shape[1]

    @property
    def n_allele_calls(self):
        """Total number of allele calls (n_variants * n_samples * ploidy)."""
        return self.shape[0] * self.shape[1] * self.shape[2]

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask):

        # check input
        if not is_array_like(mask):
            mask = np.asarray(mask)
        check_array_like(mask, 2)
        if mask.shape != self.shape[:2]:
            raise ValueError('mask has incorrect shape')

        # store
        self._mask = mask

    def fill_masked(self, value=-1, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.fill_masked(value=value)

        out = backend.map_blocks(self, mapper, **kwargs)
        return GenotypeChunkedArray(out)

    def subset(self, sel0, sel1, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))
        out = backend.subset(self, sel0, sel1, **kwargs)
        g = GenotypeChunkedArray(out)
        if self.mask is not None:
            mask = backend.subset(self.mask, sel0, sel1, **kwargs)
            g.mask = mask
        return g

    def is_called(self, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_called()

        out = backend.map_blocks(self, mapper, **kwargs)
        return ChunkedArray(out)

    def is_missing(self, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_missing()

        out = backend.map_blocks(self, mapper, **kwargs)
        return ChunkedArray(out)

    def is_hom(self, allele=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_hom(allele=allele)

        out = backend.map_blocks(self, mapper, **kwargs)
        return ChunkedArray(out)

    def is_hom_ref(self, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_hom_ref()

        out = backend.map_blocks(self, mapper, **kwargs)
        return ChunkedArray(out)

    def is_hom_alt(self, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_hom_alt()

        out = backend.map_blocks(self, mapper, **kwargs)
        return ChunkedArray(out)

    def is_het(self, allele=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_het(allele=allele)

        out = backend.map_blocks(self, mapper, **kwargs)
        return ChunkedArray(out)

    def is_call(self, call, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_call(call)

        out = backend.map_blocks(self, mapper, **kwargs)
        return ChunkedArray(out)

    def count_called(self, axis=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_called()

        out = backend.sum(self, axis=axis, mapper=mapper, **kwargs)
        return out

    def count_missing(self, axis=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_missing()

        out = backend.sum(self, axis=axis, mapper=mapper, **kwargs)
        return out

    def count_hom(self, allele=None, axis=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_hom(allele=allele)

        out = backend.sum(self, axis=axis, mapper=mapper, **kwargs)
        return out

    def count_hom_ref(self, axis=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_hom_ref()

        out = backend.sum(self, axis=axis, mapper=mapper, **kwargs)
        return out

    def count_hom_alt(self, axis=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_hom_alt()

        out = backend.sum(self, axis=axis, mapper=mapper, **kwargs)
        return out

    def count_het(self, axis=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_het()

        out = backend.sum(self, axis=axis, mapper=mapper, **kwargs)
        return out

    def count_call(self, call, axis=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_call(call)

        out = backend.sum(self, axis=axis, mapper=mapper, **kwargs)
        return out

    def to_haplotypes(self, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.to_haplotypes()

        out = backend.map_blocks(self, mapper, **kwargs)
        return HaplotypeChunkedArray(out)

    def to_n_ref(self, fill=0, dtype='i1', **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.to_n_ref(fill=fill, dtype=dtype)

        out = backend.map_blocks(self, mapper, dtype=dtype, **kwargs)
        return ChunkedArray(out)

    def to_n_alt(self, fill=0, dtype='i1', **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.to_n_alt(fill=fill, dtype=dtype)

        out = backend.map_blocks(self, mapper, dtype=dtype, **kwargs)
        return ChunkedArray(out)

    def to_allele_counts(self, alleles=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        # determine alleles to count
        if alleles is None:
            m = self.max()
            alleles = list(range(m+1))

        def mapper(block):
            return block.to_allele_counts(alleles)

        out = backend.map_blocks(self, mapper, **kwargs)
        return ChunkedArray(out)

    def to_packed(self, boundscheck=True, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        if self.ploidy != 2:
            raise ValueError('can only pack diploid calls')

        if boundscheck:
            amx = self.max()
            if amx > 14:
                raise ValueError('max allele for packing is 14, found %s'
                                 % amx)
            amn = self.min()
            if amn < -1:
                raise ValueError('min allele for packing is -1, found %s'
                                 % amn)

        def mapper(block):
            return block.to_packed(boundscheck=False)

        out = backend.map_blocks(self, mapper, **kwargs)
        return ChunkedArray(out)

    @staticmethod
    def from_packed(packed, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        # check input
        check_array_like(packed)

        def mapper(block):
            return GenotypeArray.from_packed(block)

        out = backend.map_blocks(packed, mapper, **kwargs)
        return GenotypeChunkedArray(out)

    def count_alleles(self, max_allele=None, subpop=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        # if max_allele not specified, count all alleles
        if max_allele is None:
            max_allele = self.max()

        def mapper(block):
            return block.count_alleles(max_allele=max_allele, subpop=subpop)

        out = backend.map_blocks(self, mapper, **kwargs)
        return AlleleCountsChunkedArray(out)

    def count_alleles_subpops(self, subpops, max_allele=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        if max_allele is None:
            max_allele = self.max()

        def mapper(block):
            return block.count_alleles_subpops(subpops, max_allele=max_allele)

        out = backend.dict_map_blocks(self, mapper, **kwargs)
        for k, v in out.items():
            out[k] = AlleleCountsChunkedArray(v)
        return out

    def to_gt(self, phased=False, max_allele=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        if max_allele is None:
            max_allele = self.max()

        def mapper(block):
            return block.to_gt(phased=phased, max_allele=max_allele)

        out = backend.map_blocks(self, mapper, **kwargs)
        return ChunkedArray(out)

    # noinspection PyTypeChecker
    def map_alleles(self, mapping, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        # check inputs
        check_equal_length(self, mapping)

        # define mapping function
        def mapper(block, bmapping):
            return block.map_alleles(bmapping, copy=False)

        # execute map
        domain = (self, mapping)
        kwargs.setdefault('dtype', self.dtype)  # TODO needed?
        out = backend.map_blocks(domain, mapper, **kwargs)
        return GenotypeChunkedArray(out)


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
copy_method_doc(GenotypeChunkedArray.to_haplotypes,
                GenotypeArray.to_haplotypes)
copy_method_doc(GenotypeChunkedArray.to_n_ref, GenotypeArray.to_n_ref)
copy_method_doc(GenotypeChunkedArray.to_n_alt, GenotypeArray.to_n_alt)
copy_method_doc(GenotypeChunkedArray.to_allele_counts,
                GenotypeArray.to_allele_counts)
copy_method_doc(GenotypeChunkedArray.to_packed, GenotypeArray.to_packed)
GenotypeChunkedArray.from_packed.__doc__ = GenotypeArray.from_packed.__doc__
copy_method_doc(GenotypeChunkedArray.count_alleles,
                GenotypeArray.count_alleles)
copy_method_doc(GenotypeChunkedArray.count_alleles_subpops,
                GenotypeArray.count_alleles_subpops)
copy_method_doc(GenotypeChunkedArray.to_gt, GenotypeArray.to_gt)
copy_method_doc(GenotypeChunkedArray.map_alleles, GenotypeArray.map_alleles)
copy_method_doc(GenotypeChunkedArray.hstack, GenotypeArray.hstack)
copy_method_doc(GenotypeChunkedArray.vstack, GenotypeArray.vstack)


class HaplotypeChunkedArray(ChunkedArray):
    """TODO

    """

    def __init__(self, data):
        self.check_input_data(data)
        super(HaplotypeChunkedArray, self).__init__(data)

    @staticmethod
    def check_input_data(data):
        check_array_like(data, 2)
        if data.dtype.kind not in 'ui':
            raise TypeError('integer dtype required')

    def __getitem__(self, *args):
        out = self.data.__getitem__(*args)
        if is_array_like(out) and len(self.shape) == len(out.shape):
            out = HaplotypeArray(out)
        return out

    def _repr_html_(self):
        return self[:6].to_html_str(caption=repr(self))

    @property
    def n_variants(self):
        """Number of variants (length of first array dimension)."""
        return self.shape[0]

    @property
    def n_haplotypes(self):
        """Number of haplotypes (length of second array dimension)."""
        return self.shape[1]

    def to_genotypes(self, ploidy=2, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        # check ploidy is compatible
        if (self.n_haplotypes % ploidy) > 0:
            raise ValueError('incompatible ploidy')

        # build output
        def mapper(block):
            return block.to_genotypes(ploidy)

        out = backend.map_blocks(self, mapper, **kwargs)
        return GenotypeChunkedArray(out)

    def is_called(self, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))
        return backend.op_scalar(self, operator.ge, 0, **kwargs)

    def is_missing(self, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))
        return backend.op_scalar(self, operator.lt, 0, **kwargs)

    def is_ref(self, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))
        return backend.op_scalar(self, operator.eq, 0, **kwargs)

    def is_alt(self, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))
        return backend.op_scalar(self, operator.gt, 0, **kwargs)

    def is_call(self, allele, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))
        return backend.op_scalar(self, operator.eq, allele, **kwargs)

    def count_called(self, axis=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_called()

        out = backend.sum(self, axis=axis, mapper=mapper, **kwargs)
        return out

    def count_missing(self, axis=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_missing()

        out = backend.sum(self, axis=axis, mapper=mapper, **kwargs)
        return out

    def count_ref(self, axis=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_ref()

        out = backend.sum(self, axis=axis, mapper=mapper, **kwargs)
        return out

    def count_alt(self, axis=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_alt()

        out = backend.sum(self, axis=axis, mapper=mapper, **kwargs)
        return out

    def count_call(self, allele, axis=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_call(allele)

        out = backend.sum(self, axis=axis, mapper=mapper, **kwargs)
        return out

    def count_alleles(self, max_allele=None, subpop=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        # if max_allele not specified, count all alleles
        if max_allele is None:
            max_allele = self.max()

        def mapper(block):
            return block.count_alleles(max_allele=max_allele, subpop=subpop)

        out = backend.map_blocks(self, mapper, **kwargs)
        return AlleleCountsChunkedArray(out)

    def count_alleles_subpops(self, subpops, max_allele=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        if max_allele is None:
            max_allele = self.max()

        def mapper(block):
            return block.count_alleles_subpops(subpops, max_allele=max_allele)

        out = backend.dict_map_blocks(self, mapper, **kwargs)
        for k, v in out.items():
            out[k] = AlleleCountsChunkedArray(v)
        return out

    def map_alleles(self, mapping, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        # check inputs
        check_equal_length(self, mapping)

        # define mapping function
        def mapper(block, bmapping):
            return block.map_alleles(bmapping, copy=False)

        # execute map
        domain = (self, mapping)
        kwargs.setdefault('dtype', self.dtype)  # TODO needed?
        out = backend.map_blocks(domain, mapper, **kwargs)
        return HaplotypeChunkedArray(out)


# copy docstrings
copy_method_doc(HaplotypeChunkedArray.to_genotypes,
                HaplotypeArray.to_genotypes)
copy_method_doc(HaplotypeChunkedArray.count_alleles,
                HaplotypeArray.count_alleles)
copy_method_doc(HaplotypeChunkedArray.count_alleles_subpops,
                HaplotypeArray.count_alleles_subpops)
copy_method_doc(HaplotypeChunkedArray.map_alleles, HaplotypeArray.map_alleles)


class AlleleCountsChunkedArray(ChunkedArray):

    def __init__(self, data):
        self.check_input_data(data)
        super(AlleleCountsChunkedArray, self).__init__(data)

    @staticmethod
    def check_input_data(data):
        check_array_like(data, 2)
        if data.dtype.kind not in 'ui':
            raise TypeError('integer dtype required')

    def __getitem__(self, *args):
        out = self.data.__getitem__(*args)
        if is_array_like(out) and len(self.shape) == len(out.shape) and \
                out.shape[1] == self.shape[1]:
            out = AlleleCountsArray(out)
        return out

    def _repr_html_(self):
        return self[:6].to_html_str(caption=repr(self))

    @property
    def n_variants(self):
        """Number of variants (length of first array dimension)."""
        return self.shape[0]

    @property
    def n_alleles(self):
        """Number of alleles (length of second array dimension)."""
        return self.shape[1]

    def to_frequencies(self, fill=np.nan, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.to_frequencies(fill=fill)

        out = backend.map_blocks(self, mapper, **kwargs)
        return ChunkedArray(out)

    def allelism(self, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.allelism()

        out = backend.map_blocks(self, mapper, **kwargs)
        return ChunkedArray(out)

    def max_allele(self, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.max_allele()

        out = backend.map_blocks(self, mapper, **kwargs)
        return ChunkedArray(out)

    def is_variant(self, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_variant()

        out = backend.map_blocks(self, mapper, **kwargs)
        return ChunkedArray(out)

    def is_non_variant(self, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_non_variant()

        out = backend.map_blocks(self, mapper, **kwargs)
        return ChunkedArray(out)

    def is_segregating(self, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_segregating()

        out = backend.map_blocks(self, mapper, **kwargs)
        return ChunkedArray(out)

    def is_non_segregating(self, allele=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_non_segregating(allele=allele)

        out = backend.map_blocks(self, mapper, **kwargs)
        return ChunkedArray(out)

    def is_singleton(self, allele=1, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_singleton(allele=allele)

        out = backend.map_blocks(self, mapper, **kwargs)
        return ChunkedArray(out)

    def is_doubleton(self, allele=1, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_doubleton(allele=allele)

        out = backend.map_blocks(self, mapper, **kwargs)
        return ChunkedArray(out)

    def count_variant(self, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_variant()

        out = backend.sum(self, mapper=mapper, **kwargs)
        return out

    def count_non_variant(self, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_non_variant()

        out = backend.sum(self, mapper=mapper, **kwargs)
        return out

    def count_segregating(self, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_segregating()

        out = backend.sum(self, mapper=mapper, **kwargs)
        return out

    def count_non_segregating(self, allele=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_non_segregating(allele=allele)

        out = backend.sum(self, mapper=mapper, **kwargs)
        return out

    def count_singleton(self, allele=1, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_singleton(allele=allele)

        out = backend.sum(self, mapper=mapper, **kwargs)
        return out

    def count_doubleton(self, allele=1, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_doubleton(allele=allele)

        out = backend.sum(self, mapper=mapper, **kwargs)
        return out

    def map_alleles(self, mapping, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        # check inputs
        check_equal_length(self, mapping)

        # define mapping function
        def mapper(block, bmapping):
            return block.map_alleles(bmapping)

        # execute map
        domain = (self, mapping)
        kwargs.setdefault('dtype', self.dtype)  # TODO needed?
        out = backend.map_blocks(domain, mapper, **kwargs)
        return AlleleCountsChunkedArray(out)


copy_method_doc(AlleleCountsChunkedArray.allelism, AlleleCountsArray.allelism)
copy_method_doc(AlleleCountsChunkedArray.max_allele,
                AlleleCountsArray.max_allele)
copy_method_doc(AlleleCountsChunkedArray.map_alleles,
                AlleleCountsArray.map_alleles)


def check_table_like(data, names=None):
    if names is None:
        name = get_column_names(data)
    if len(names) < 1:
        raise ValueError('at least one column name is required')
    # should raise KeyError if name not present
    cols = tuple(data[n] for n in names)
    check_array_like(cols)
    check_equal_length(*cols)
    return names


class ChunkedTable(object):

    def __init__(self, data, names=None):
        self.data = data
        self.names = check_table_like(data, names)
        self.rowcls = namedtuple('row', self.names)

    def __getitem__(self, item):

        if isinstance(item, string_types):
            # return column
            return ChunkedArray(self.data[item])

        elif isinstance(item, integer_types):
            # return row as tuple
            return self.rowcls(*(self.data[n][item] for n in self.names))

        elif isinstance(item, slice):
            # load into numpy structured array
            if item.start is None:
                start = 0
            else:
                start = item.start
            if item.stop is None:
                stop = self.shape[0]
            else:
                stop = item.stop
            if item.step is None:
                step = 1
            else:
                step = item.step
            outshape = (stop - start) // step
            out = np.empty(outshape, dtype=self.dtype)
            for n in self.names:
                out[n] = self.data[n][item]
            return out.view(np.recarray)

        elif isinstance(item, (list, tuple)) and \
                all([isinstance(i, string_types) for i in item]):
            # assume names of columns, return table
            return ChunkedTable(self.data, names=item)

        else:
            raise NotImplementedError('unsupported item: %r' % item)

    def __getattr__(self, item):
        if item in self.names:
            return ChunkedArray(self.data[item])
        else:
            raise AttributeError(item)

    def __repr__(self):
        return '%s(%s, %s.%s)' % \
               (type(self).__name__, self.shape[0],
                type(self.data).__module__, type(self.data).__name__)

    def _repr_html_(self):
        caption = repr(self)
        ra = self[:6]
        return recarray_to_html_str(ra, limit=5, caption=caption)

    def display(self, limit, **kwargs):
        kwargs.setdefault('caption', repr(self))
        ra = self[:limit+1]
        return recarray_display(ra, limit=limit, **kwargs)

    @property
    def shape(self):
        return self.data[self.names[0]].shape[:1]

    def __len__(self):
        return self.shape[0]

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def dtype(self):
        l = []
        for n in self.names:
            c = self.data[n]
            # Need to account for multidimensional columns
            t = (n, c.dtype) if len(c.shape) == 1 else \
                (n, c.dtype, c.shape[1:])
            l.append(t)
        return np.dtype(l)

    def compress(self, condition, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))
        out = backend.compress_table(self, condition, **kwargs)
        return type(self)(out)

    def take(self, indices, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))
        out = backend.take_table(self, indices, **kwargs)
        return type(self)(out)

    def vstack(self, *others, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))
        tup = (self,) + others
        out = backend.vstack_table(tup, **kwargs)
        return type(self)(out)

    # TODO eval
    # TODO query
    # TODO addcol (and __setitem__?)
    # TODO delcol (and __delitem__?)
    # TODO store
    # TODO copy


# TODO write table classes
## ChunkedTable
## VariantChunkedTable
## FeatureChunkedTable
