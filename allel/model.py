# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np
import numexpr as ne


from allel.constants import *


class GenotypeArray(np.ndarray):
    """TODO

    """

    @staticmethod
    def _check_input_data(obj):
        obj = np.asarray(obj)

        # check dtype
        if obj.dtype.kind not in 'ui':
            raise TypeError('integer dtype required')

        # check dimensionality
        if obj.ndim != 3:
            raise TypeError('array with 3 dimensions required')

        return obj

    def __new__(cls, data):
        obj = cls._check_input_data(data)
        obj = obj.view(cls)
        return obj

    def __array_finalize__(self, obj):

        # called after constructor
        if obj is None:
            return

        # called after slice (new-from-template)
        if isinstance(obj, GenotypeArray):
            return

        # called after view
        GenotypeArray._check_input_data(obj)

    def __getslice__(self, *args, **kwargs):
        s = np.ndarray.__getslice__(self, *args, **kwargs)
        if hasattr(s, 'ndim'):
            if s.ndim == 3:
                return s
            elif s.ndim > 0:
                return np.asarray(s)
        return s

    def __getitem__(self, *args, **kwargs):
        s = np.ndarray.__getitem__(self, *args, **kwargs)
        if hasattr(s, 'ndim'):
            if s.ndim == 3:
                return s
            elif s.ndim > 0:
                return np.asarray(s)
        return s

    @property
    def n_variants(self):
        """TODO"""
        return self.shape[0]

    @property
    def n_samples(self):
        """TODO"""
        return self.shape[1]

    @property
    def ploidy(self):
        """TODO"""
        return self.shape[2]

    def __repr__(self):
        s = super(GenotypeArray, self).__repr__()
        return s[:-1] + ', n_variants=%s, n_samples=%s, ploidy=%s)' % \
                        (self.n_variants, self.n_samples, self.ploidy)

    # noinspection PyUnusedLocal
    def is_called(self):
        """TODO

        """

        # special case diploid
        if self.ploidy == DIPLOID:
            allele1 = self[..., 0]  # noqa
            allele2 = self[..., 1]  # noqa
            ex = '(allele1 >= 0) & (allele2 >= 0)'
            out = ne.evaluate(ex)

        # general ploidy case
        else:
            out = np.all(self >= 0, axis=DIM_PLOIDY)

        return out

    # noinspection PyUnusedLocal
    def is_missing(self):
        """TODO

        """

        # special case diploid
        if self.ploidy == DIPLOID:
            allele1 = self[..., 0]  # noqa
            allele2 = self[..., 1]  # noqa
            # call is missing if either allele is missing
            ex = '(allele1 < 0) | (allele2 < 0)'
            out = ne.evaluate(ex)

        # general ploidy case
        else:
            # call is missing if any allele is missing
            out = np.any(self < 0, axis=DIM_PLOIDY)

        return out

    # noinspection PyUnusedLocal
    def is_hom(self, allele=None):
        """TODO

        """

        # special case diploid
        if self.ploidy == DIPLOID:
            allele1 = self[..., 0]  # noqa
            allele2 = self[..., 1]  # noqa
            if allele is None:
                ex = '(allele1 >= 0) & (allele1  == allele2)'
            else:
                ex = '(allele1 == {0}) & (allele2 == {0})'.format(allele)
            out = ne.evaluate(ex)

        # general ploidy case
        else:
            if allele is None:
                allele1 = self[..., 0, None]  # noqa
                other_alleles = self[..., 1:]  # noqa
                ex = '(allele1 >= 0) & (allele1 == other_alleles)'
                out = np.all(ne.evaluate(ex), axis=DIM_PLOIDY)
            else:
                out = np.all(self == allele, axis=DIM_PLOIDY)

        return out

    def is_hom_ref(self):
        """TODO

        """

        return self.is_hom(allele=0)

    # noinspection PyUnusedLocal
    def is_hom_alt(self):
        """TODO

        """

        # special case diploid
        if self.ploidy == DIPLOID:
            allele1 = self[..., 0]  # noqa
            allele2 = self[..., 1]  # noqa
            ex = '(allele1 > 0) & (allele1  == allele2)'
            out = ne.evaluate(ex)

        # general ploidy case
        else:
            allele1 = self[..., 0, None]  # noqa
            other_alleles = self[..., 1:]  # noqa
            ex = '(allele1 > 0) & (allele1 == other_alleles)'
            out = np.all(ne.evaluate(ex), axis=DIM_PLOIDY)

        return out

    # noinspection PyUnusedLocal
    def is_het(self):
        """TODO

        """

        # special case diploid
        if self.ploidy == DIPLOID:
            allele1 = self[..., 0]  # noqa
            allele2 = self[..., 1]  # noqa
            ex = '(allele1 >= 0) & (allele2  >= 0) & (allele1 != allele2)'
            out = ne.evaluate(ex)

        # general ploidy case
        else:
            allele1 = self[..., 0, None]  # noqa
            other_alleles = self[..., 1:]  # noqa
            out = np.all(self >= 0, axis=DIM_PLOIDY) \
                & np.any(allele1 != other_alleles, axis=DIM_PLOIDY)

        return out

    # noinspection PyUnusedLocal
    def is_call(self, call):
        """TODO

        """

        # special case diploid
        if self.ploidy == DIPLOID:
            if not len(call) == DIPLOID:
                raise ValueError('invalid call: %r', call)
            allele1 = self[..., 0]  # noqa
            allele2 = self[..., 1]  # noqa
            ex = '(allele1 == {0}) & (allele2  == {1})'.format(*call)
            out = ne.evaluate(ex)

        # general ploidy case
        else:
            if not len(call) == self.ploidy:
                raise ValueError('invalid call: %r', call)
            call = np.asarray(call)[None, None, :]
            out = np.all(self == call, axis=DIM_PLOIDY)

        return out

    def count_called(self, axis=None):
        b = self.is_called()
        return np.sum(b, axis=axis)

    def count_missing(self, axis=None):
        b = self.is_missing()
        return np.sum(b, axis=axis)

    def count_hom(self, allele=None, axis=None):
        b = self.is_hom(allele=allele)
        return np.sum(b, axis=axis)

    def count_hom_ref(self, axis=None):
        b = self.is_hom_ref()
        return np.sum(b, axis=axis)

    def count_hom_alt(self, axis=None):
        b = self.is_hom_alt()
        return np.sum(b, axis=axis)

    def count_het(self, axis=None):
        b = self.is_het()
        return np.sum(b, axis=axis)

    def count_call(self, call, axis=None):
        b = self.is_call(call=call)
        return np.sum(b, axis=axis)

    def to_haplotypes(self):
        """TODO

        """

        # reshape, preserving size of variants dimension
        newshape = (self.shape[DIM_VARIANTS], -1)
        data = np.reshape(self, newshape)
        h = HaplotypeArray(data)
        return h

    @staticmethod
    def from_haplotypes(h, ploidy):
        """TODO

        """

        h = HaplotypeArray(h)
        return h.to_genotypes(ploidy=ploidy)

    def to_n_alt(self, fill=0):
        """TODO


        """

        # count number of alternate alleles
        out = np.empty(self.shape[:-1], dtype='i1')
        np.sum(self > 0, axis=DIM_PLOIDY, out=out)

        # fill missing calls
        if fill != 0:
            m = self.is_missing()
            out[m] = fill

        return out

    def to_allele_counts(self, alleles=None):
        """TODO

        """

        # determine alleles to count
        if alleles is None:
            m = self.max()
            alleles = list(range(m+1))

        # set up output array
        outshape = self.shape[:2] + (len(alleles),)
        out = np.zeros(outshape, dtype='u1')

        for i, allele in enumerate(alleles):
            # count alleles along ploidy dimension
            np.sum(self == allele, axis=DIM_PLOIDY, out=out[..., i])

        return out

    def to_packed(self, boundscheck=True):
        """TODO

        """

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

        from allel.opt.gt import pack_diploid

        # ensure int8 dtype
        if self.dtype == np.int8:
            data = self
        else:
            data = self.astype(dtype=np.int8)

        # pack data
        packed = pack_diploid(data)

        return packed

    @staticmethod
    def from_packed(packed):
        """TODO

        """

        # check arguments
        packed = np.asarray(packed)
        if packed.ndim != 2:
            raise ValueError('packed array must have 2 dimensions')
        if packed.dtype != np.uint8:
            packed = packed.astype(np.uint8)

        from allel.opt.gt import unpack_diploid
        data = unpack_diploid(packed)
        return GenotypeArray(data)

    def to_sparse(self, format='csr', **kwargs):
        """TODO

        """

        h = self.to_haplotypes()
        m = h.to_sparse(format=format, **kwargs)
        return m

    @staticmethod
    def from_sparse(m, ploidy, order=None, out=None):
        """TODO

        """

        h = HaplotypeArray.from_sparse(m, order=order, out=out)
        g = h.to_genotypes(ploidy=ploidy)
        return g

    def allelism(self):
        """TODO

        """

        return self.to_haplotypes().allelism()

    def allele_number(self):
        """TODO

        """

        return self.to_haplotypes().allele_number()

    def allele_count(self, allele=1):
        """TODO

        """

        return self.to_haplotypes().allele_count(allele=allele)

    def allele_frequency(self, allele=1, fill=0):
        """TODO

        """

        return self.to_haplotypes().allele_frequency(allele=allele, fill=fill)

    def allele_counts(self, alleles=None):
        """TODO

        """

        return self.to_haplotypes().allele_counts(alleles=alleles)

    def allele_frequencies(self, alleles=None, fill=0):
        """TODO

        """

        return self.to_haplotypes().allele_frequencies(alleles=alleles,
                                                       fill=fill)

    def is_variant(self):
        """TODO

        """

        return self.to_haplotypes().is_variant()

    def is_non_variant(self):
        """TODO

        """

        return self.to_haplotypes().is_non_variant()

    def is_segregating(self):
        """TODO

        """

        return self.to_haplotypes().is_segregating()

    def is_non_segregating(self, allele=None):
        """TODO

        """

        return self.to_haplotypes().is_non_segregating(allele=allele)

    def is_singleton(self, allele=1):
        """TODO

        """

        return self.to_haplotypes().is_singleton(allele=allele)

    def is_doubleton(self, allele=1):
        """TODO

        """

        return self.to_haplotypes().is_doubleton(allele=allele)

    def count_variant(self):
        return np.sum(self.is_variant())

    def count_non_variant(self):
        return np.sum(self.is_non_variant())

    def count_segregating(self):
        return np.sum(self.is_segregating())

    def count_non_segregating(self, allele=None):
        return np.sum(self.is_non_segregating(allele=allele))

    def count_singleton(self, allele=None):
        return np.sum(self.is_singleton(allele=allele))

    def count_doubleton(self, allele=None):
        return np.sum(self.is_doubleton(allele=allele))


class HaplotypeArray(np.ndarray):
    """TODO

    """

    @staticmethod
    def _check_input_data(obj):
        obj = np.asarray(obj)

        # check dtype
        if obj.dtype.kind not in 'ui':
            raise TypeError('integer dtype required')

        # check dimensionality
        if obj.ndim != 2:
            raise TypeError('array with 2 dimensions required')

        return obj

    def __new__(cls, data):
        obj = cls._check_input_data(data)
        obj = obj.view(cls)
        return obj

    def __array_finalize__(self, obj):

        # called after constructor
        if obj is None:
            return

        # called after slice (new-from-template)
        if isinstance(obj, HaplotypeArray):
            return

        # called after view
        HaplotypeArray._check_input_data(obj)

    def __getslice__(self, *args, **kwargs):
        s = np.ndarray.__getslice__(self, *args, **kwargs)
        if hasattr(s, 'ndim'):
            if s.ndim == 2:
                return s
            elif s.ndim > 0:
                return np.asarray(s)
        return s

    def __getitem__(self, *args, **kwargs):
        s = np.ndarray.__getitem__(self, *args, **kwargs)
        if hasattr(s, 'ndim'):
            if s.ndim == 2:
                return s
            elif s.ndim > 0:
                return np.asarray(s)
        return s

    @property
    def n_variants(self):
        """TODO"""
        return self.shape[0]

    @property
    def n_haplotypes(self):
        """TODO"""
        return self.shape[1]

    def __repr__(self):
        s = super(HaplotypeArray, self).__repr__()
        return s[:-1] + ', n_variants=%s, n_haplotypes=%s)' % \
                        (self.n_variants, self.n_haplotypes)

    def to_genotypes(self, ploidy):
        """TODO

        """

        # check ploidy is compatible
        if (self.n_haplotypes % ploidy) > 0:
            raise ValueError('incompatible ploidy')

        # reshape
        newshape = (self.shape[0], -1, ploidy)
        data = self.reshape(newshape)

        # wrap
        g = GenotypeArray(data)

        return g

    @staticmethod
    def from_genotypes(g):
        """TODO

        """

        g = GenotypeArray(g)
        h = g.to_haplotypes()
        return h

    def to_sparse(self, format='csr', **kwargs):
        """TODO

        """

        import scipy.sparse

        # check arguments
        f = {
            'bsr': scipy.sparse.bsr_matrix,
            'coo': scipy.sparse.coo_matrix,
            'csc': scipy.sparse.csc_matrix,
            'csr': scipy.sparse.csr_matrix,
            'dia': scipy.sparse.dia_matrix,
            'dok': scipy.sparse.dok_matrix,
            'lil': scipy.sparse.lil_matrix
        }
        if format not in f:
            raise ValueError('invalid format: %r' % format)

        # create sparse matrix
        m = f[format](self, **kwargs)

        return m

    @staticmethod
    def from_sparse(m, order=None, out=None):
        """TODO

        """

        import scipy.sparse

        # check arguments
        if not scipy.sparse.isspmatrix(m):
            raise ValueError('not a sparse matrix: %r' % m)

        # convert to dense array
        data = m.toarray(order=order, out=out)

        # wrap
        h = HaplotypeArray(data)

        return h

    def allelism(self):
        """TODO

        """

        # calculate allele counts
        ac = self.allele_counts()

        # count alleles present
        n = np.sum(ac > 0, axis=1)

        return n

    def allele_number(self):
        """TODO

        """

        # count non-missing calls over samples
        an = np.sum(self >= 0, axis=1)

        return an

    def allele_count(self, allele=1):
        """TODO

        """

        # count non-missing calls over samples
        return np.sum(self == allele, axis=1)

    def allele_frequency(self, allele=1, fill=0):
        """TODO

        """

        # intermediate variables
        an = self.allele_number()
        ac = self.allele_count(allele=allele)

        # calculate allele frequency, accounting for variants with no calls
        err = np.seterr(invalid='ignore')
        af = np.where(an > 0, ac / an, fill)
        np.seterr(**err)

        return af, ac, an

    def allele_counts(self, alleles=None):
        """TODO

        """

        # if alleles not specified, count all alleles
        if alleles is None:
            m = self.max()
            alleles = list(range(m+1))

        # set up output array
        ac = np.zeros((self.n_variants, len(alleles)), dtype='i4')

        # count alleles
        for i, allele in enumerate(alleles):
            np.sum(self == allele, axis=1, out=ac[:, i])

        return ac

    def allele_frequencies(self, alleles=None, fill=0):
        """TODO

        """

        # intermediate variables
        an = self.allele_number()[:, None]
        ac = self.allele_counts(alleles=alleles)

        # calculate allele frequency, accounting for variants with no calls
        err = np.seterr(invalid='ignore')
        af = np.where(an > 0, ac / an, fill)
        np.seterr(**err)

        return af, ac, an[:, 0]

    def is_variant(self):
        """TODO

        """

        # find variants with at least 1 non-reference allele
        out = np.sum(self > 0, axis=1) >= 1

        return out

    def is_non_variant(self):
        """TODO

        """

        # find variants with no non-reference alleles
        out = np.all(self <= 0, axis=1)

        return out

    def is_segregating(self):
        """TODO

        """

        # find segregating variants
        out = self.allelism() > 1

        return out

    def is_non_segregating(self, allele=None):
        """TODO

        """

        if allele is None:

            # find fixed variants
            out = self.allelism() <= 1

        else:

            # find fixed variants with respect to a specific allele
            ex = '(self < 0) | (self == {})'.format(allele)
            b = ne.evaluate(ex)
            out = np.all(b, axis=1)

        return out

    def is_singleton(self, allele=1):
        """TODO

        """

        # count allele
        ac = self.allele_count(allele=allele)

        # find singletons
        out = ac == 1

        return out

    def is_doubleton(self, allele=1):
        """TODO

        """

        # count allele
        ac = self.allele_count(allele=allele)

        # find doubletons
        out = ac == 2

        return out

    def count_variant(self):
        return np.sum(self.is_variant())

    def count_non_variant(self):
        return np.sum(self.is_non_variant())

    def count_segregating(self):
        return np.sum(self.is_segregating())

    def count_non_segregating(self, allele=None):
        return np.sum(self.is_non_segregating(allele=allele))

    def count_singleton(self, allele=None):
        return np.sum(self.is_singleton(allele=allele))

    def count_doubleton(self, allele=None):
        return np.sum(self.is_doubleton(allele=allele))


class PosArray(np.ndarray):
    """TODO

    """

    @staticmethod
    def _check_input_data(obj):
        obj = np.asarray(obj)

        # check dtype
        if obj.dtype.kind not in 'ui':
            raise TypeError('integer dtype required')

        # check dimensionality
        print(repr(obj))
        print(obj.ndim)
        if obj.ndim != 1:
            raise TypeError('array with 1 dimension required')

        # check sorted ascending
        if np.any(np.diff(obj) < 0):
            raise ValueError('array is not sorted')

        return obj

    def __new__(cls, data):
        obj = cls._check_input_data(data)
        obj = obj.view(cls)
        return obj

    def __array_finalize__(self, obj):

        # called after constructor
        if obj is None:
            return

        # called after slice (new-from-template)
        if isinstance(obj, PosArray):
            return

        # called after view
        PosArray._check_input_data(obj)

    def __getslice__(self, *args, **kwargs):
        s = np.ndarray.__getslice__(self, *args, **kwargs)
        if hasattr(s, 'ndim'):
            if s.ndim == 1:
                return s
            elif s.ndim > 0:
                return np.asarray(s)
        return s

    def __getitem__(self, *args, **kwargs):
        s = np.ndarray.__getitem__(self, *args, **kwargs)
        if hasattr(s, 'ndim'):
            if s.ndim == 1:
                return s
            elif s.ndim > 0:
                return np.asarray(s)
        return s

    @property
    def n_variants(self):
        """TODO"""
        return self.shape[0]

    def __repr__(self):
        s = super(PosArray, self).__repr__()
        return s[:-1] + ', n_variants=%s)' % self.n_variants

    def locate_position(self, p):
        """TODO

        """

        # find position
        index = np.searchsorted(self, p)
        if index < self.size and self[index] == p:
            return index
        else:
            return None

    def locate_positions(self, other, assume_unique=False):
        """TODO

        """

        # check inputs
        other = PosArray(other)

        # find intersection
        cond1 = np.in1d(self, other, assume_unique=assume_unique)
        cond2 = np.in1d(other, self, assume_unique=assume_unique)

        return cond1, cond2

    def intersect(self, other, assume_unique=False):
        """TODO

        """

        # check inputs
        other = PosArray(other)

        # find intersection
        cond = np.in1d(self, other, assume_unique=assume_unique)

        return np.compress(cond, self)

    def locate_interval(self, start=0, stop=None):
        """TODO

        """

        # locate start and stop indices
        start_index = np.searchsorted(self, start)
        stop_index = np.searchsorted(self, stop, side='right') \
            if stop is not None else None

        loc = slice(start_index, stop_index)
        return loc

    def locate_intervals(self, starts, stops):
        """TODO

        """

        # check inputs
        starts = np.asarray(starts)
        stops = np.asarray(stops)
        # TODO raise ValueError
        assert starts.ndim == stops.ndim == 1
        assert starts.shape[0] == stops.shape[0]

        # find indices of start and stop positions in pos
        start_indices = np.searchsorted(self, starts)
        stop_indices = np.searchsorted(self, stops, side='right')

        # find intervals overlapping at least one position
        cond2 = start_indices < stop_indices

        # find positions within at least one interval
        cond1 = np.zeros_like(self, dtype=np.bool)
        for i, j in zip(start_indices[cond2], stop_indices[cond2]):
            cond1[i:j] = True

        return cond1, cond2

    # TODO windowed counts