# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import operator
from allel.compat import range, reduce, integer_types


import numpy as np
import bcolz


from allel.model import GenotypeArray, HaplotypeArray
from allel.constants import DIM_PLOIDY
from allel.util import asarray_ndim


__all__ = ['GenotypeCArray', 'HaplotypeCArray']


def _block_append(f, data, out):
    bs = data.chunklen
    for i in range(0, data.shape[0], bs):
        block = data[i:i+bs]
        out.append(f(block))


def _block_sum(data, axis=None):
    bs = data.chunklen

    if axis is None:
        out = 0
        for i in range(0, data.shape[0], bs):
            block = data[i:i+bs]
            out += np.sum(block)
        return out

    elif axis == 0:
        out = np.zeros((data.shape[1],), dtype=int)
        for i in range(0, data.shape[0], bs):
            block = data[i:i+bs]
            out += np.sum(block, axis=0)
        return out

    elif axis == 1:
        out = np.zeros((data.shape[0],), dtype=int)
        for i in range(0, data.shape[0], bs):
            block = data[i:i+bs]
            out[i:i+bs] += np.sum(block, axis=1)
        return out


def _block_max(data, axis=None):
    bs = data.chunklen
    out = None

    if axis is None:
        for i in range(0, data.shape[0], bs):
            block = data[i:i+bs]
            m = np.max(block)
            if out is None:
                out = m
            else:
                out = m if m > out else out
        return out

    elif axis == 0:
        for i in range(0, data.shape[0], bs):
            block = data[i:i+bs]
            m = np.max(block, axis=0)
            if m is None:
                out = m
            else:
                out = np.where(m > out, m, out)
        return out

    elif axis == 1:
        out = np.zeros((data.shape[0],), dtype=int)
        for i in range(0, data.shape[0], bs):
            block = data[i:i+bs]
            out[i:i+bs] = np.max(block, axis=1)
        return out


def _block_min(data, axis=None):
    bs = data.chunklen
    out = None

    if axis is None:
        for i in range(0, data.shape[0], bs):
            block = data[i:i+bs]
            m = np.min(block)
            if out is None:
                out = m
            else:
                out = m if m < out else out
        return out

    elif axis == 0:
        for i in range(0, data.shape[0], bs):
            block = data[i:i+bs]
            m = np.min(block, axis=0)
            if m is None:
                out = m
            else:
                out = np.where(m < out, m, out)
        return out

    elif axis == 1:
        out = np.zeros((data.shape[0],), dtype=int)
        for i in range(0, data.shape[0], bs):
            block = data[i:i+bs]
            out[i:i+bs] = np.min(block, axis=1)
        return out


def _block_compress(condition, data, axis):

    # check inputs
    condition = asarray_ndim(condition, 1)
    if axis not in {0, 1}:
        raise NotImplementedError('only axis 0 (variants) or 1 (samples) '
                                  'supported')

    if axis == 0:
        if condition.size != data.shape[0]:
            raise ValueError('length of condition must match length of '
                             'first dimension; expected %s, found %s' %
                             (data.shape[0], condition.size))

        # setup output
        n = np.count_nonzero(condition) \
            * reduce(operator.mul, data.shape[1:], 1)
        out = bcolz.zeros((0,) + data.shape[1:],
                          dtype=data.dtype,
                          expectedlen=n)

        # build output
        bs = data.chunklen
        for i in range(0, data.shape[0], bs):
            block = data[i:i+bs]
            vcond = condition[i:i+bs]
            out.append(np.compress(vcond, block, axis=0))

        return out

    elif axis == 1:
        if condition.size != data.shape[1]:
            raise ValueError('length of condition must match length of '
                             'second dimension; expected %s, found %s' %
                             (data.shape[1], condition.size))

        # setup output
        n = data.shape[0] \
            * np.count_nonzero(condition) \
            * reduce(operator.mul, data.shape[2:], 1)
        out = bcolz.zeros((0, np.count_nonzero(condition)) + data.shape[2:],
                          dtype=data.dtype,
                          expectedlen=n)

        # build output
        bs = data.chunklen
        for i in range(0, data.shape[0], bs):
            block = data[i:i+bs]
            out.append(np.compress(condition, block, axis=1))

        return out


def _block_take(data, indices, axis):

    # check inputs
    indices = asarray_ndim(indices, 1)
    if axis not in {0, 1}:
        raise NotImplementedError('only axis 0 (variants) or 1 (samples) '
                                  'supported')

    if axis == 0:
        condition = np.zeros((data.shape[0],), dtype=bool)
        condition[indices] = True
        return _block_compress(condition, data, axis=0)

    elif axis == 1:
        condition = np.zeros((data.shape[1],), dtype=bool)
        condition[indices] = True
        return _block_compress(condition, data, axis=1)


def _block_subset(cls, data, variants, samples):

    # check inputs
    variants = asarray_ndim(variants, 1, allow_none=True)
    samples = asarray_ndim(samples, 1, allow_none=True)
    if variants is None and samples is None:
        raise ValueError('variants and/or samples required')

    # if either variants or samples is None, use take/compress
    if samples is None:
        if variants.size < data.shape[0]:
            return _block_take(data, variants, axis=0)
        else:
            return _block_compress(variants, data, axis=0)
    elif variants is None:
        if samples.size < data.shape[1]:
            return _block_take(data, samples, axis=1)
        else:
            return _block_compress(samples, data, axis=1)

    # ensure boolean array for variants
    if variants.size < data.shape[0]:
        tmp = np.zeros((data.shape[0],), dtype=bool)
        tmp[variants] = True
        variants = tmp

    # ensure indices for samples
    if samples.size == data.shape[1]:
        samples = np.nonzero(samples)[0]

    # setup output
    n = np.count_nonzero(variants) \
        * samples.size \
        * reduce(operator.mul, data.shape[2:], 1)
    out = bcolz.zeros((0, samples.size) + data.shape[2:],
                      dtype=data.dtype,
                      expectedlen=n)

    # build output
    bs = data.chunklen
    for i in range(0, data.shape[0], bs):
        block = data[i:i+bs]
        vcond = variants[i:i+bs]
        x = cls(block, copy=False)
        out.append(x.subset(variants=vcond, samples=samples))

    return out


class GenotypeCArray(object):

    @staticmethod
    def _check_input_data(obj):

        # check dtype
        if obj.dtype.kind not in 'ui':
            raise TypeError('integer dtype required')

        # check dimensionality
        if obj.ndim != 3:
            raise TypeError('array with 3 dimensions required')

        # check length of ploidy dimension
        if obj.shape[DIM_PLOIDY] == 1:
            raise ValueError('use HaplotypeCArray for haploid calls')

    def __init__(self, data, copy=True, **kwargs):
        if copy or not isinstance(data, bcolz.carray):
            data = bcolz.carray(data, **kwargs)
        # check late to avoid creating an intermediate numpy array
        self._check_input_data(data)
        self.data = data

    def __getitem__(self, *args):
        out = self.data.__getitem__(*args)
        if hasattr(out, 'ndim') and out.ndim == 3:
            out = GenotypeArray(out, copy=False)
        return out

    def __array__(self):
        return self.data[:]

    @property
    def n_variants(self):
        """Number of variants (length of first array dimension)."""
        return self.data.shape[0]

    @property
    def n_samples(self):
        """Number of samples (length of second array dimension)."""
        return self.data.shape[1]

    @property
    def ploidy(self):
        """Sample ploidy (length of third array dimension)."""
        return self.data.shape[2]

    def __repr__(self):
        s = repr(self.data)
        s = 'GenotypeCArray' + s[6:]
        return s

    def compress(self, condition, axis):
        data = _block_compress(condition, self.data, axis)
        return GenotypeCArray(data, copy=False)

        # # check inputs
        # condition = asarray_ndim(condition, 1)
        # if axis not in {0, 1}:
        #     raise NotImplementedError('only axis 0 (variants) or 1 (samples) '
        #                               'supported')
        #
        # if axis == 0:
        #     if condition.size != self.n_variants:
        #         raise ValueError('length of condition must match length of '
        #                          'first dimension; expected %s, found %s' %
        #                          (self.n_variants, condition.size))
        #
        #     # setup output
        #     n = np.count_nonzero(condition) * self.n_samples * self.ploidy
        #     out = bcolz.zeros((0, self.n_samples, self.ploidy),
        #                       dtype=self.data.dtype,
        #                       expectedlen=n)
        #
        #     # build output
        #     bs = self.data.chunklen
        #     for i in range(0, self.n_variants, bs):
        #         block = self.data[i:i+bs]
        #         vcond = condition[i:i+bs]
        #         out.append(np.compress(vcond, block, axis=0))
        #
        #     return out
        #
        # elif axis == 1:
        #     if condition.size != self.n_samples:
        #         raise ValueError('length of condition must match length of '
        #                          'second dimension; expected %s, found %s' %
        #                          (self.n_samples, condition.size))
        #
        #     # setup output
        #     n = self.n_variants * np.count_nonzero(condition) * self.ploidy
        #     out = bcolz.zeros((0, np.count_nonzero(condition), self.ploidy),
        #                       dtype=self.data.dtype,
        #                       expectedlen=n)
        #
        #     # build output
        #     bs = self.data.chunklen
        #     for i in range(0, self.n_variants, bs):
        #         block = self.data[i:i+bs]
        #         out.append(np.compress(condition, block, axis=1))
        #
        #     return out

    def take(self, indices, axis):
        data = _block_take(self.data, indices, axis)
        return GenotypeCArray(data, copy=False)

        # # check inputs
        # indices = asarray_ndim(indices, 1)
        # if axis not in {0, 1}:
        #     raise NotImplementedError('only axis 0 (variants) or 1 (samples) '
        #                               'supported')
        #
        # if axis == 0:
        #     condition = np.zeros((self.n_variants,), dtype=bool)
        #     condition[indices] = True
        #     return self.compress(condition, axis=0)
        #
        # elif axis == 1:
        #     condition = np.zeros((self.n_samples,), dtype=bool)
        #     condition[indices] = True
        #     return self.compress(condition, axis=1)

    def subset(self, variants, samples):
        data = _block_subset(GenotypeArray, self.data, variants, samples)
        return GenotypeCArray(data, copy=False)

        # # check inputs
        # variants = asarray_ndim(variants, 1, allow_none=True)
        # samples = asarray_ndim(samples, 1, allow_none=True)
        # if variants is None and samples is None:
        #     raise ValueError('variants and/or samples required')
        #
        # # if either variants or samples is None, use take/compress
        # if samples is None:
        #     if variants.size < self.n_variants:
        #         return self.take(variants, axis=0)
        #     else:
        #         return self.compress(variants, axis=0)
        # elif variants is None:
        #     if samples.size < self.n_samples:
        #         return self.take(samples, axis=1)
        #     else:
        #         return self.compress(samples, axis=1)
        #
        # # ensure boolean array for variants
        # if variants.size < self.n_variants:
        #     tmp = np.zeros((self.n_variants,), dtype=bool)
        #     tmp[variants] = True
        #     variants = tmp
        #
        # # ensure indices for samples
        # if samples.size == self.n_variants:
        #     samples = np.nonzero(samples)[0]
        #
        # # setup output
        # n = np.count_nonzero(variants) * samples.size * self.ploidy
        # out = bcolz.zeros((0, samples.size, self.ploidy),
        #                   dtype=self.data.dtype,
        #                   expectedlen=n)
        #
        # # build output
        # bs = self.data.chunklen
        # for i in range(0, self.n_variants, bs):
        #     block = self.data[i:i+bs]
        #     vcond = variants[i:i+bs]
        #     g = GenotypeArray(block, copy=False)
        #     out.append(g.subset(variants=vcond, samples=samples))
        #
        # return out

    def is_called(self):

        # setup output
        n = self.n_variants * self.n_samples
        out = bcolz.zeros((0, self.n_samples), dtype='u1', expectedlen=n)

        # build output
        f = lambda data: GenotypeArray(data, copy=False).is_called()
        _block_append(f, self.data, out)

        return out

    def is_missing(self):

        # setup output
        n = self.n_variants * self.n_samples
        out = bcolz.zeros((0, self.n_samples), dtype='u1', expectedlen=n)

        # build output
        f = lambda data: GenotypeArray(data, copy=False).is_missing()
        _block_append(f, self.data, out)

        return out

    def is_hom(self, allele=None):

        # setup output
        n = self.n_variants * self.n_samples
        out = bcolz.zeros((0, self.n_samples), dtype='u1', expectedlen=n)

        # build output
        f = lambda data: GenotypeArray(data, copy=False).is_hom(allele=allele)
        _block_append(f, self.data, out)

        return out

    def is_hom_ref(self):

        # setup output
        n = self.n_variants * self.n_samples
        out = bcolz.zeros((0, self.n_samples), dtype='u1', expectedlen=n)

        # build output
        f = lambda data: GenotypeArray(data, copy=False).is_hom_ref()
        _block_append(f, self.data, out)

        return out

    def is_hom_alt(self):

        # setup output
        n = self.n_variants * self.n_samples
        out = bcolz.zeros((0, self.n_samples), dtype='u1', expectedlen=n)

        # build output
        f = lambda data: GenotypeArray(data, copy=False).is_hom_alt()
        _block_append(f, self.data, out)

        return out

    def is_het(self):

        # setup output
        n = self.n_variants * self.n_samples
        out = bcolz.zeros((0, self.n_samples), dtype='u1', expectedlen=n)

        # build output
        f = lambda data: GenotypeArray(data, copy=False).is_het()
        _block_append(f, self.data, out)

        return out

    def is_call(self, call):

        # setup output
        n = self.n_variants * self.n_samples
        out = bcolz.zeros((0, self.n_samples), dtype='u1', expectedlen=n)

        # build output
        f = lambda data: GenotypeArray(data, copy=False).is_call(call)
        _block_append(f, self.data, out)

        return out

    def count_called(self, axis=None):
        b = self.is_called()
        return _block_sum(b, axis=axis)

    def count_missing(self, axis=None):
        b = self.is_missing()
        return _block_sum(b, axis=axis)

    def count_hom(self, allele=None, axis=None):
        b = self.is_hom(allele=allele)
        return _block_sum(b, axis=axis)

    def count_hom_ref(self, axis=None):
        b = self.is_hom_ref()
        return _block_sum(b, axis=axis)

    def count_hom_alt(self, axis=None):
        b = self.is_hom_alt()
        return _block_sum(b, axis=axis)

    def count_het(self, axis=None):
        b = self.is_het()
        return _block_sum(b, axis=axis)

    def count_call(self, call, axis=None):
        b = self.is_call(call=call)
        return _block_sum(b, axis=axis)

    def view_haplotypes(self):
        # Unfortunately this cannot be implemented as a lightweight view,
        # so we have to copy.

        # setup output
        out = bcolz.zeros((0, self.n_samples * self.ploidy),
                          dtype=self.data.dtype,
                          chunklen=self.data.chunklen)

        # build output
        f = lambda block: block.reshape((block.shape[0], -1))
        _block_append(f, self.data, out)

        h = HaplotypeCArray(out, copy=False)
        return h

    def to_n_alt(self, fill=0):

        # setup output
        n = self.n_variants * self.n_samples
        out = bcolz.zeros((0, self.n_samples), dtype='i1', expectedlen=n)

        # build output
        f = lambda data: GenotypeArray(data, copy=False).to_n_alt(fill)
        _block_append(f, self.data, out)

        return out

    def max(self, axis=None):
        if axis not in {None, 0, 1}:
            raise NotImplementedError('only axis None, 0 (variants) or 1 '
                                      '(samples) supported')
        return _block_max(self.data, axis=axis)

    def min(self, axis=None):
        if axis not in {None, 0, 1}:
            raise NotImplementedError('only axis None, 0 (variants) or 1 '
                                      '(samples) supported')
        return _block_min(self.data, axis=axis)

    def to_allele_counts(self, alleles=None):

        # determine alleles to count
        if alleles is None:
            m = self.max()
            alleles = list(range(m+1))

        # set up output
        n = self.n_variants * self.n_samples * len(alleles)
        out = bcolz.zeros((0, self.n_samples, len(alleles)),
                          dtype='u1',
                          expectedlen=n)

        # build output
        def f(block):
            g = GenotypeArray(block, copy=False)
            return g.to_allele_counts(alleles)

        _block_append(f, self.data, out)

        return out

    def to_packed(self, boundscheck=True):

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

        # set up output
        n = self.n_variants * self.n_samples
        out = bcolz.zeros((0, self.n_samples), dtype='i1', expectedlen=n)

        # build output
        def f(block):
            g = GenotypeArray(block, copy=False)
            return g.to_packed(boundscheck=False)

        _block_append(f, self.data, out)

        return out

    @staticmethod
    def from_packed(packed):

        # check input
        if not isinstance(packed, (np.ndarray, bcolz.carray)):
            packed = np.asarray(packed)

        # set up output
        n = packed.shape[0] * packed.shape[1] * 2
        out = bcolz.zeros((0, packed.shape[1], 2), dtype='i1', expectedlen=n)

        # build output
        def f(block):
            return GenotypeArray.from_packed(block)
        _block_append(f, packed, out)

        return GenotypeCArray(out, copy=False)

    def allelism(self):
        out = bcolz.zeros((0,), dtype=int)

        def f(block):
            return GenotypeArray(block, copy=False).allelism()
        _block_append(f, self.data, out)
        return out

    def allele_number(self):
        out = bcolz.zeros((0,), dtype=int)

        def f(block):
            return GenotypeArray(block, copy=False).allele_number()
        _block_append(f, self.data, out)
        return out

    def allele_count(self, allele=1):
        out = bcolz.zeros((0,), dtype=int)

        def f(block):
            return GenotypeArray(block, copy=False).allele_count(allele=allele)
        _block_append(f, self.data, out)
        return out

    def allele_frequency(self, allele=1, fill=np.nan):
        out = bcolz.zeros((0,), dtype=float)

        def f(block):
            g = GenotypeArray(block, copy=False)
            af = g.allele_frequency(allele=allele, fill=fill)
            return af
        _block_append(f, self.data, out)

        return out

    def allele_counts(self, alleles=None):

        # if alleles not specified, count all alleles
        if alleles is None:
            m = self.max()
            alleles = list(range(m+1))

        # setup output
        out = bcolz.zeros((0, len(alleles)), dtype=int)

        def f(block):
            g = GenotypeArray(block, copy=False)
            return g.allele_counts(alleles=alleles)
        _block_append(f, self.data, out)

        return out

    def allele_frequencies(self, alleles=None, fill=np.nan):

        # if alleles not specified, count all alleles
        if alleles is None:
            m = self.max()
            alleles = list(range(m+1))

        # setup output
        out = bcolz.zeros((0, len(alleles)), dtype=float)

        def f(block):
            g = GenotypeArray(block, copy=False)
            af = g.allele_frequencies(alleles=alleles, fill=fill)
            return af
        _block_append(f, self.data, out)

        return out

    def is_variant(self):
        out = bcolz.zeros((0,), dtype=bool)

        def f(block):
            g = GenotypeArray(block, copy=False)
            return g.is_variant()
        _block_append(f, self.data, out)

        return out

    def is_non_variant(self):
        out = bcolz.zeros((0,), dtype=bool)

        def f(block):
            g = GenotypeArray(block, copy=False)
            return g.is_non_variant()
        _block_append(f, self.data, out)

        return out

    def is_segregating(self):
        out = bcolz.zeros((0,), dtype=bool)

        def f(block):
            g = GenotypeArray(block, copy=False)
            return g.is_segregating()
        _block_append(f, self.data, out)

        return out

    def is_non_segregating(self, allele=None):
        out = bcolz.zeros((0,), dtype=bool)

        def f(block):
            g = GenotypeArray(block, copy=False)
            return g.is_non_segregating(allele=allele)
        _block_append(f, self.data, out)

        return out

    def is_singleton(self, allele=1):
        out = bcolz.zeros((0,), dtype=bool)

        def f(block):
            g = GenotypeArray(block, copy=False)
            return g.is_singleton(allele=allele)
        _block_append(f, self.data, out)

        return out

    def is_doubleton(self, allele=1):
        out = bcolz.zeros((0,), dtype=bool)

        def f(block):
            g = GenotypeArray(block, copy=False)
            return g.is_doubleton(allele=allele)
        _block_append(f, self.data, out)

        return out

    def count_variant(self):
        return _block_sum(self.is_variant())

    def count_non_variant(self):
        return _block_sum(self.is_non_variant())

    def count_segregating(self):
        return _block_sum(self.is_segregating())

    def count_non_segregating(self, allele=None):
        return _block_sum(self.is_non_segregating(allele=allele))

    def count_singleton(self, allele=1):
        return _block_sum(self.is_singleton(allele=allele))

    def count_doubleton(self, allele=1):
        return _block_sum(self.is_doubleton(allele=allele))


class HaplotypeCArray(object):

    @staticmethod
    def _check_input_data(obj):

        # check dtype
        if obj.dtype.kind not in 'ui':
            raise TypeError('integer dtype required')

        # check dimensionality
        if obj.ndim != 2:
            raise TypeError('array with 2 dimensions required')

    def __init__(self, data, copy=True, **kwargs):
        if copy or not isinstance(data, bcolz.carray):
            data = bcolz.carray(data, **kwargs)
        # check late to avoid creating an intermediate numpy array
        self._check_input_data(data)
        self.data = data

    def __getitem__(self, *args):
        out = self.data.__getitem__(*args)
        if hasattr(out, 'ndim') and out.ndim == 2:
            out = HaplotypeArray(out, copy=False)
        return out

    def __array__(self):
        return self.data[:]

    @property
    def n_variants(self):
        """Number of variants (length of first array dimension)."""
        return self.data.shape[0]

    @property
    def n_haplotypes(self):
        """Number of haplotypes (length of second array dimension)."""
        return self.data.shape[1]

    def __repr__(self):
        s = repr(self.data)
        s = 'HaplotypeCArray' + s[6:]
        return s

    def compress(self, condition, axis):
        return _block_compress(condition, self.data, axis)

    def take(self, indices, axis):
        return _block_take(self.data, indices, axis)

    def subset(self, variants, samples):
        return _block_subset(HaplotypeArray, self.data, variants, samples)

    def view_genotypes(self, ploidy):
        # Unfortunately this cannot be implemented as a lightweight view,
        # so we have to copy.

        # check ploidy is compatible
        if (self.n_haplotypes % ploidy) > 0:
            raise ValueError('incompatible ploidy')

        # setup output
        n_samples = self.n_haplotypes / ploidy
        out = bcolz.zeros((0, n_samples, ploidy),
                          dtype=self.data.dtype,
                          chunklen=self.data.chunklen)

        # build output
        f = lambda block: block.reshape((block.shape[0], -1, ploidy))
        _block_append(f, self.data, out)

        g = GenotypeCArray(out, copy=False)
        return g

    def is_called(self):
        return self >= 0

    def is_missing(self):
        return self < 0

    def is_ref(self):
        return self == 0

    def is_alt(self, allele=None):
        if allele is None:
            return self > 0
        else:
            return self == allele

    def is_call(self, allele):
        return self == allele

    def _op(self, op, other):
        if not isinstance(other, integer_types):
            raise NotImplementedError('only supported for scalars')

        # setup output
        n = self.n_variants * self.n_haplotypes
        out = bcolz.zeros((0, self.n_haplotypes), dtype='u1', expectedlen=n)

        # build output
        f = lambda data: op(data, other)
        _block_append(f, self.data, out)

        return out

    def __eq__(self, other):
        return self._op(operator.eq, other)

    def __ne__(self, other):
        return self._op(operator.ne, other)

    def __lt__(self, other):
        return self._op(operator.lt, other)

    def __gt__(self, other):
        return self._op(operator.gt, other)

    def __le__(self, other):
        return self._op(operator.le, other)

    def __ge__(self, other):
        return self._op(operator.ge, other)

    def allelism(self):
        out = bcolz.zeros((0,), dtype=int)

        def f(data):
            return HaplotypeArray(data, copy=False).allelism()
        _block_append(f, self.data, out)
        return out

    def allele_number(self):
        out = bcolz.zeros((0,), dtype=int)

        def f(data):
            return HaplotypeArray(data, copy=False).allele_number()
        _block_append(f, self.data, out)
        return out

    def allele_count(self, allele=1):
        out = bcolz.zeros((0,), dtype=int)

        def f(data):
            return HaplotypeArray(data, copy=False).allele_count(allele=allele)
        _block_append(f, self.data, out)
        return out

    def allele_frequency(self, allele=1, fill=np.nan):
        out = bcolz.zeros((0,), dtype=float)

        def f(data):
            g = HaplotypeArray(data, copy=False)
            af = g.allele_frequency(allele=allele, fill=fill)
            return af
        _block_append(f, self.data, out)

        return out

    def allele_counts(self, alleles=None):
        out = bcolz.zeros((0,), dtype=int)

        def f(data):
            g = HaplotypeArray(data, copy=False)
            return g.allele_counts(alleles=alleles)
        _block_append(f, self.data, out)

        return out

    def allele_frequencies(self, alleles=None, fill=np.nan):
        out = bcolz.zeros((0,), dtype=float)

        def f(data):
            g = HaplotypeArray(data, copy=False)
            af = g.allele_frequencies(alleles=alleles, fill=fill)
            return af
        _block_append(f, self.data, out)

        return out

    def is_variant(self):
        out = bcolz.zeros((0,), dtype=bool)

        def f(data):
            g = HaplotypeArray(data, copy=False)
            return g.is_variant()
        _block_append(f, self.data, out)

        return out

    def is_non_variant(self):
        out = bcolz.zeros((0,), dtype=bool)

        def f(data):
            g = HaplotypeArray(data, copy=False)
            return g.is_non_variant()
        _block_append(f, self.data, out)

        return out

    def is_segregating(self):
        out = bcolz.zeros((0,), dtype=bool)

        def f(data):
            g = HaplotypeArray(data, copy=False)
            return g.is_segregating()
        _block_append(f, self.data, out)

        return out

    def is_non_segregating(self, allele=None):
        out = bcolz.zeros((0,), dtype=bool)

        def f(data):
            g = HaplotypeArray(data, copy=False)
            return g.is_non_segregating(allele=allele)
        _block_append(f, self.data, out)

        return out

    def is_singleton(self, allele=1):
        out = bcolz.zeros((0,), dtype=bool)

        def f(data):
            g = HaplotypeArray(data, copy=False)
            return g.is_singleton(allele=allele)
        _block_append(f, self.data, out)

        return out

    def is_doubleton(self, allele=1):
        out = bcolz.zeros((0,), dtype=bool)

        def f(data):
            g = HaplotypeArray(data, copy=False)
            return g.is_doubleton(allele=allele)
        _block_append(f, self.data, out)

        return out

    def count_variant(self):
        return _block_sum(self.is_variant())

    def count_non_variant(self):
        return _block_sum(self.is_non_variant())

    def count_segregating(self):
        return _block_sum(self.is_segregating())

    def count_non_segregating(self, allele=None):
        return _block_sum(self.is_non_segregating(allele=allele))

    def count_singleton(self, allele=1):
        return _block_sum(self.is_singleton(allele=allele))

    def count_doubleton(self, allele=1):
        return _block_sum(self.is_doubleton(allele=allele))
