# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import collections


# third-party imports
import numpy as np


# internal imports
from allel.util import check_dtype_kind, check_shape, check_dtype, asarray_ndim, check_ndim,\
    ignore_invalid, check_dim0_aligned
from allel.compat import PY2


class ArrayBase(object):
    """Abstract base class that wraps a NumPy array."""

    @classmethod
    def _check_values(cls, data):
        pass

    def __init__(self, data, copy=False, **kwargs):
        values = np.array(data, copy=copy, **kwargs)
        self._check_values(values)
        self._values = values

    @property
    def values(self):
        """The underlying array of values."""
        return self._values

    def __getattr__(self, item):
        return getattr(self.values, item)

    def __getitem__(self, item):
        return self.values[item]

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)

    def __array__(self, *args):
        a = self.values
        if args:
            a = a.astype(args[0])
        return a

    def __str__(self):
        return str(self.values)

    def __repr__(self):
        r = '%s(%s, dtype=%s)\n' % (type(self).__name__, self.shape, self.dtype)
        r += str(self)
        return r

    def __eq__(self, other):
        return self.values == other

    def __ne__(self, other):
        return self.values != other

    def __lt__(self, other):
        return self.values < other

    def __gt__(self, other):
        return self.values > other

    def __le__(self, other):
        return self.values <= other

    def __ge__(self, other):
        return self.values >= other

    def __abs__(self):
        return abs(self.values)

    def __add__(self, other):
        return self.values + other

    def __and__(self, other):
        return self.values & other

    def __div__(self, other):
        return self.values.__div__(other)

    def __floordiv__(self, other):
        return self.values // other

    def __inv__(self):
        return ~self.values

    def __invert__(self):
        return ~self.values

    def __lshift__(self, other):
        return self.values << other

    def __mod__(self, other):
        return self.values % other

    def __mul__(self, other):
        return self.values * other

    def __neg__(self):
        return -self.values

    def __or__(self, other):
        return self.values | other

    def __pos__(self):
        return +self.values

    def __pow__(self, other):
        return self.values ** other

    def __rshift__(self, other):
        return self.values >> other

    def __sub__(self, other):
        return self.values - other

    def __truediv__(self, other):
        return self.values.__truediv__(other)

    def __xor__(self, other):
        return self.values ^ other

    def hstack(self, *others):
        """Stack arrays in sequence horizontally (column-wise)."""
        tup = (self,) + others
        return np.hstack(tup)

    def vstack(self, *others):
        """Stack arrays in sequence vertically (row-wise)."""
        tup = (self,) + others
        return np.vstack(tup)

    def dstack(self, *others):
        """Stack arrays depth-wise."""
        tup = (self,) + others
        return np.dstack(tup)

    def concatenate(self, *others, **kwargs):
        """Concatenate arrays."""
        tup = (self,) + others
        return np.concatenate(tup, **kwargs)

    def copy(self, *args, **kwargs):
        data = self.values.copy(*args, **kwargs)
        return type(self)(data)


class GenotypeBase(ArrayBase):
    """Abstract class for wrapping genotype calls."""

    @classmethod
    def _check_values(cls, data):
        check_dtype_kind(data, 'u', 'i')

    def __init__(self, data, copy=False, **kwargs):
        super(GenotypeBase, self).__init__(data, copy=copy, **kwargs)
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
        """A boolean mask associated with this genotype array, indicating
        genotype calls that should be filtered (i.e., excluded) from
        genotype and allele counting operations.

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 1], [1, 1]],
        ...                          [[0, 2], [-1, -1]]], dtype='i1')
        >>> g
        GenotypeArray((3, 2, 2), dtype=int8)
        0/0 0/1
        0/1 1/1
        0/2 ./.
        >>> g.count_called()
        5
        >>> g.count_alleles()
        AlleleCountsArray((3, 3), dtype=int32)
        3 1 0
        1 3 0
        1 0 1
        >>> mask = [[True, False], [False, True], [False, False]]
        >>> g.mask = mask
        >>> g
        GenotypeArray((3, 2, 2), dtype=int8)
        ./. 0/1
        0/1 ./.
        0/2 ./.
        >>> g.count_called()
        3
        >>> g.count_alleles()
        AlleleCountsArray((3, 3), dtype=int32)
        1 1 0
        1 1 0
        1 0 1

        Notes
        -----
        This is a lightweight genotype call mask and **not** a mask in the
        sense of a numpy masked array. This means that the mask will only be
        taken into account by the genotype and allele counting methods of this
        class, and is ignored by any of the generic methods on the ndarray
        class or by any numpy ufuncs.

        Note also that the mask may not survive any slicing, indexing or
        other subsetting procedures (e.g., call to :func:`numpy.compress` or
        :func:`numpy.take`). I.e., the mask will have to be similarly indexed
        then reapplied. The only exceptions are simple slicing operations
        that preserve the dimensionality and ploidy of the array, and the
        subset() method, both of which **will** preserve the mask if present.

        """
        return self._mask

    @mask.setter
    def mask(self, mask):
        if mask is not None:
            mask = np.asarray(mask)
            check_shape(mask, self.shape[:-1])
            check_dtype(mask, np.dtype(bool))
        self._mask = mask

    @property
    def is_phased(self):
        """TODO"""
        return self._is_phased

    @is_phased.setter
    def is_phased(self, is_phased):
        if is_phased is not None:
            is_phased = np.asarray(is_phased)
            check_shape(is_phased, self.shape[:-1])
            check_dtype(is_phased, np.dtype(bool))
        self._is_phased = is_phased

    def to_str(self, *args, **kwargs):
        raise NotImplementedError

    def to_html(self, *args, **kwargs):
        raise NotImplementedError

    def __str__(self):
        return self.to_str()

    def _repr_html_(self):
        return self.to_html()

    def fill_masked(self, value=-1, copy=True):
        """Fill masked genotype calls with a given value.

        Parameters
        ----------
        value : int, optional
            The fill value.
        copy : bool, optional
            If False, modify the array in place.

        Returns
        -------
        g : GenotypeArray

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 1], [1, 1]],
        ...                          [[0, 2], [-1, -1]]], dtype='i1')
        >>> mask = [[True, False], [False, True], [False, False]]
        >>> g.mask = mask
        >>> g.fill_masked().values
        array([[[-1, -1],
                [ 0,  1]],
               [[ 0,  1],
                [-1, -1]],
               [[ 0,  2],
                [-1, -1]]], dtype=int8)

        """

        if self.mask is None:
            raise ValueError('no mask is set')

        # apply the mask
        data = np.array(self.values, copy=copy)
        data[self.mask] = value

        if copy:
            out = type(self)(data)  # wrap
            out.is_phased = self.is_phased
            # don't set mask because it has been filled in
        else:
            out = self
            out.mask = None  # reset mask

        return out

    def is_called(self):
        """Find non-missing genotype calls.

        Returns
        -------
        out : ndarray, bool, shape (n_variants, n_samples)
            Array where elements are True if the genotype call matches the
            condition.

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 1], [1, 1]],
        ...                          [[0, 2], [-1, -1]]])
        >>> g.is_called()
        array([[ True,  True],
               [ True,  True],
               [ True, False]], dtype=bool)

        """

        out = np.all(self >= 0, axis=-1)

        # handle mask
        if self.mask is not None:
            out &= ~self.mask

        return out

    def is_missing(self):
        """Find missing genotype calls.

        Returns
        -------
        out : ndarray, bool, shape (n_variants, n_samples)
            Array where elements are True if the genotype call matches the
            condition.

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 1], [1, 1]],
        ...                          [[0, 2], [-1, -1]]])
        >>> g.is_missing()
        array([[False, False],
               [False, False],
               [False,  True]], dtype=bool)

        """

        out = np.any(self < 0, axis=-1)

        # handle mask
        if self.mask is not None:
            out |= self.mask

        return out

    def is_hom(self, allele=None):
        """Find genotype calls that are homozygous.

        Parameters
        ----------
        allele : int, optional
            Allele index.

        Returns
        -------
        out : ndarray, bool, shape (n_variants, n_samples)
            Array where elements are True if the genotype call matches the
            condition.

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 1], [1, 1]],
        ...                          [[2, 2], [-1, -1]]])
        >>> g.is_hom()
        array([[ True, False],
               [False,  True],
               [ True, False]], dtype=bool)
        >>> g.is_hom(allele=1)
        array([[False, False],
               [False,  True],
               [False, False]], dtype=bool)

        """

        if allele is None:
            allele1 = self[..., 0, np.newaxis]
            other_alleles = self[..., 1:]
            tmp = (allele1 >= 0) & (allele1 == other_alleles)
            out = np.all(tmp, axis=-1)
        else:
            out = np.all(self == allele, axis=-1)

        # handle mask
        if self.mask is not None:
            out &= ~self.mask

        return out

    def is_hom_ref(self):
        """Find genotype calls that are homozygous for the reference allele.

        Returns
        -------
        out : ndarray, bool, shape (n_variants, n_samples)
            Array where elements are True if the genotype call matches the
            condition.

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 1], [1, 1]],
        ...                          [[0, 2], [-1, -1]]])
        >>> g.is_hom_ref()
        array([[ True, False],
               [False, False],
               [False, False]], dtype=bool)

        """

        return self.is_hom(allele=0)

    def is_hom_alt(self):
        """Find genotype calls that are homozygous for any alternate (i.e.,
        non-reference) allele.

        Returns
        -------
        out : ndarray, bool, shape (n_variants, n_samples)
            Array where elements are True if the genotype call matches the
            condition.

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 1], [1, 1]],
        ...                          [[2, 2], [-1, -1]]])
        >>> g.is_hom_alt()
        array([[False, False],
               [False,  True],
               [ True, False]], dtype=bool)

        """

        allele1 = self[..., 0, np.newaxis]
        other_alleles = self[..., 1:]
        tmp = (allele1 > 0) & (allele1 == other_alleles)
        out = np.all(tmp, axis=-1)

        # handle mask
        if self.mask is not None:
            out &= ~self.mask

        return out

    def is_het(self, allele=None):
        """Find genotype calls that are heterozygous.

        Returns
        -------
        out : ndarray, bool, shape (n_variants, n_samples)
            Array where elements are True if the genotype call matches the
            condition.
        allele : int, optional
            Heterozygous allele.

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 1], [1, 1]],
        ...                          [[0, 2], [-1, -1]]])
        >>> g.is_het()
        array([[False,  True],
               [ True, False],
               [ True, False]], dtype=bool)
        >>> g.is_het(2)
        array([[False, False],
               [False, False],
               [ True, False]], dtype=bool)

        """

        allele1 = self[..., 0, np.newaxis]
        other_alleles = self[..., 1:]
        out = np.all(self >= 0, axis=-1) & np.any(allele1 != other_alleles, axis=-1)
        if allele is not None:
            out &= np.any(self == allele, axis=-1)

        # handle mask
        if self.mask is not None:
            out &= ~self.mask

        return out

    def is_call(self, call):
        """Locate genotypes with a given call.

        Parameters
        ----------
        call : array_like, int, shape (ploidy,)
            The genotype call to find.

        Returns
        -------
        out : ndarray, bool, shape (n_variants, n_samples)
            Array where elements are True if the genotype is `call`.

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 1], [1, 1]],
        ...                          [[0, 2], [-1, -1]]])
        >>> g.is_call((0, 2))
        array([[False, False],
               [False, False],
               [ True, False]], dtype=bool)

        """

        # guard conditions
        if not len(call) == self.shape[2]:
            raise ValueError('invalid call: %r', call)

        if self.ndim == 2:
            call = np.asarray(call)[np.newaxis, :]
        else:
            call = np.asarray(call)[np.newaxis, np.newaxis, :]
        out = np.all(self == call, axis=-1)

        # handle mask
        if self.mask is not None:
            out &= ~self.mask

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

    def count_het(self, allele=None, axis=None):
        b = self.is_het(allele=allele)
        return np.sum(b, axis=axis)

    def count_call(self, call, axis=None):
        b = self.is_call(call=call)
        return np.sum(b, axis=axis)

    def to_n_ref(self, fill=0, dtype='i1'):
        """Transform each genotype call into the number of
        reference alleles.

        Parameters
        ----------
        fill : int, optional
            Use this value to represent missing calls.
        dtype : dtype, optional
            Output dtype.

        Returns
        -------
        out : ndarray, int8, shape (n_variants, n_samples)
            Array of ref alleles per genotype call.

        Notes
        -----
        By default this function returns 0 for missing genotype calls
        **and** for homozygous non-reference genotype calls. Use the
        `fill` argument to change how missing calls are represented.

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 2], [1, 1]],
        ...                          [[2, 2], [-1, -1]]])
        >>> g.to_n_ref()
        array([[2, 1],
               [1, 0],
               [0, 0]], dtype=int8)
        >>> g.to_n_ref(fill=-1)
        array([[ 2,  1],
               [ 1,  0],
               [ 0, -1]], dtype=int8)

        """

        # count number of alternate alleles
        out = np.empty(self.shape[:-1], dtype=dtype)
        np.sum(self == 0, axis=-1, out=out)

        # fill missing calls
        if fill != 0:
            m = self.is_missing()
            out[m] = fill

        # handle mask
        if self.mask is not None:
            out[self.mask] = fill

        return out

    def to_n_alt(self, fill=0, dtype='i1'):
        """Transform each genotype call into the number of
        non-reference alleles.

        Parameters
        ----------
        fill : int, optional
            Use this value to represent missing calls.
        dtype : dtype, optional
            Output dtype.

        Returns
        -------
        out : ndarray, int, shape (n_variants, n_samples)
            Array of non-ref alleles per genotype call.

        Notes
        -----
        This function simply counts the number of non-reference
        alleles, it makes no distinction between different alternate
        alleles.

        By default this function returns 0 for missing genotype calls
        **and** for homozygous reference genotype calls. Use the
        `fill` argument to change how missing calls are represented.

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 2], [1, 1]],
        ...                          [[2, 2], [-1, -1]]])
        >>> g.to_n_alt()
        array([[0, 1],
               [1, 2],
               [2, 0]], dtype=int8)
        >>> g.to_n_alt(fill=-1)
        array([[ 0,  1],
               [ 1,  2],
               [ 2, -1]], dtype=int8)

        """

        # count number of alternate alleles
        out = np.empty(self.shape[:-1], dtype=dtype)
        np.sum(self > 0, axis=-1, out=out)

        # fill missing calls
        if fill != 0:
            m = self.is_missing()
            out[m] = fill

        # handle mask
        if self.mask is not None:
            out[self.mask] = fill

        return out

    def to_allele_counts(self, max_allele=None, dtype='u1'):
        """Transform genotype calls into allele counts per call.

        Parameters
        ----------
        max_allele : int, optional
            Highest allele index. Provide this value to speed up computation.
        dtype : dtype, optional
            Output dtype.

        Returns
        -------
        out : ndarray, uint8, shape (n_variants, n_samples, len(alleles))
            Array of allele counts per call.

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 2], [1, 1]],
        ...                          [[2, 2], [-1, -1]]])
        >>> g.to_allele_counts()
        array([[[2, 0, 0],
                [1, 1, 0]],
               [[1, 0, 1],
                [0, 2, 0]],
               [[0, 0, 2],
                [0, 0, 0]]], dtype=uint8)

        """

        # determine alleles to count
        if max_allele is None:
            max_allele = self.max()
        alleles = list(range(max_allele + 1))

        # set up output array
        outshape = self.shape[:-1] + (len(alleles),)
        out = np.zeros(outshape, dtype=dtype)

        for allele in alleles:
            # count alleles along ploidy dimension
            allele_match = self == allele
            if self.mask is not None:
                allele_match &= ~self.mask[..., np.newaxis]
            np.sum(allele_match, axis=-1, out=out[..., allele])

        return out

    def to_gt(self):
        """Convert genotype calls to VCF-style string representation.

        Returns
        -------
        gt : ndarray, string, shape (n_variants, n_samples)

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 2], [1, 1]],
        ...                          [[1, 2], [2, 1]],
        ...                          [[2, 2], [-1, -1]]])
        >>> g.to_gt()
        chararray([[b'0/0', b'0/1'],
               [b'0/2', b'1/1'],
               [b'1/2', b'2/1'],
               [b'2/2', b'./.']],
              dtype='|S3')
        >>> g.is_phased = np.ones(g.shape[:-1], dtype=bool)
        >>> g.to_gt()
        chararray([[b'0|0', b'0|1'],
               [b'0|2', b'1|1'],
               [b'1|2', b'2|1'],
               [b'2|2', b'.|.']],
              dtype='|S3')

        """

        # how many characters needed per allele call?
        max_allele = np.max(self)
        if max_allele <= 0:
            max_allele = 1
        nchar = int(np.floor(np.log10(max_allele))) + 1

        # convert to string
        a = self.astype((np.string_, nchar)).view(np.chararray)

        # recode missing alleles
        a[self < 0] = b'.'
        if self.mask is not None:
            a[self.mask[..., np.newaxis]] = b'.'

        # determine allele call separator
        if self.is_phased is None:
            sep = b'/'
        else:
            sep = np.empty(self.shape[:-1], dtype='S1').view(np.chararray)
            sep[self.is_phased] = b'|'
            sep[~self.is_phased] = b'/'

        # join via separator, coping with any ploidy
        gt = a[..., 0]
        for i in range(1, self.ploidy):
            gt = gt + sep + a[..., i]

        return gt

    def map_alleles(self, mapping, copy=True):
        pass  # TODO

    def copy(self, *args, **kwargs):
        data = self.values.copy(*args, **kwargs)
        out = type(self)(data)
        if self.mask is not None:
            out.mask = self.mask.copy()
        if self.is_phased is not None:
            out.is_phased = self.is_phased.copy()
        return out


def subset(data, sel0, sel1):

    # check inputs
    data = np.asarray(data)
    if data.ndim < 2:
        raise ValueError('data must have 2 or more dimensions')
    sel0 = asarray_ndim(sel0, 1, allow_none=True)
    sel1 = asarray_ndim(sel1, 1, allow_none=True)

    # ensure indices
    if sel0 is not None and sel0.dtype.kind == 'b':
        sel0, = np.nonzero(sel0)
    if sel1 is not None and sel1.dtype.kind == 'b':
        sel1, = np.nonzero(sel1)

    # ensure leading dimension indices can be broadcast correctly
    if sel0 is not None and sel1 is not None:
        sel0 = sel0[:, np.newaxis]

    # deal with None arguments
    if sel0 is None:
        sel0 = slice(None)
    if sel1 is None:
        sel1 = slice(None)

    return data[sel0, sel1]


class GenotypeArray(GenotypeBase):
    """Array of discrete genotype calls.

    Parameters
    ----------
    data : array_like, int, shape (n_variants, n_samples, ploidy)
        Genotype data.
    copy : bool, optional
        If True, make a copy of `data`.
    **kwargs : keyword arguments
        All keyword arguments are passed through to :func:`numpy.array`.

    Notes
    -----
    This class represents data on discrete genotype calls as a
    3-dimensional numpy array of integers. By convention the first
    dimension corresponds to the variants genotyped, the second
    dimension corresponds to the samples genotyped, and the third
    dimension corresponds to the ploidy of the samples.

    Each integer within the array corresponds to an **allele index**,
    where 0 is the reference allele, 1 is the first alternate allele,
    2 is the second alternate allele, ... and -1 (or any other
    negative integer) is a missing allele call. A single byte integer
    dtype (int8) can represent up to 127 distinct alleles, which is
    usually sufficient.  The actual alleles (i.e., the alternate
    nucleotide sequences) and the physical positions of the variants
    within the genome of an organism are stored in separate arrays,
    discussed elsewhere.

    Arrays of this class can store either **phased or unphased**
    genotype calls. If the genotypes are phased (i.e., haplotypes have
    been resolved) then individual haplotypes can be extracted by
    converting to a :class:`HaplotypeArray` then indexing the second
    dimension. If the genotype calls are unphased then the ordering of
    alleles along the third (ploidy) dimension is arbitrary. N.B.,
    this means that an unphased diploid heterozygous call could be
    stored as (0, 1) or equivalently as (1, 0).

    A genotype array can store genotype calls with any ploidy > 1. For
    haploid calls, use a :class:`HaplotypeArray`. Note that genotype
    arrays are not capable of storing calls for samples with differing
    or variable ploidy.

    With genotype data on large numbers of variants and/or samples,
    storing the genotype calls in memory as an uncompressed numpy
    array if integers may be impractical. For working with large
    arrays of genotype data, see the
    :class:`allel.model.chunked.GenotypeChunkedArray` class, which provides an
    alternative implementation of this interface using chunked compressed
    arrays.

    Examples
    --------

    Instantiate a genotype array::

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 1], [1, 1]],
        ...                          [[0, 2], [-1, -1]]], dtype='i1')
        >>> g.dtype
        dtype('int8')
        >>> g.ndim
        3
        >>> g.shape
        (3, 2, 2)
        >>> g.n_variants
        3
        >>> g.n_samples
        2
        >>> g.ploidy
        2
        >>> g
        GenotypeArray((3, 2, 2), dtype=int8)
        0/0 0/1
        0/1 1/1
        0/2 ./.

    Genotype calls for a single variant at all samples can be obtained
    by indexing the first dimension, e.g.::

        >>> g[1]
        GenotypeVector((2, 2), dtype=int8)
        0/1 1/1

    Genotype calls for a single sample at all variants can be obtained
    by indexing the second dimension, e.g.::

        >>> g[:, 1]
        GenotypeVector((3, 2), dtype=int8)
        0/1 1/1 ./.

    A genotype call for a single sample at a single variant can be
    obtained by indexing the first and second dimensions, e.g.::

        >>> g[1, 0]
        array([0, 1], dtype=int8)

    A genotype array can store polyploid calls, e.g.::

        >>> g = allel.GenotypeArray([[[0, 0, 0], [0, 0, 1]],
        ...                          [[0, 1, 1], [1, 1, 1]],
        ...                          [[0, 1, 2], [-1, -1, -1]]],
        ...                         dtype='i1')
        >>> g.ploidy
        3
        >>> g
        GenotypeArray((3, 2, 3), dtype=int8)
        0/0/0 0/0/1
        0/1/1 1/1/1
        0/1/2 ././.

    """

    @classmethod
    def _check_values(cls, data):
        super(GenotypeArray, cls)._check_values(data)
        check_ndim(data, 3)

    def __init__(self, data, copy=False, **kwargs):
        super(GenotypeArray, self).__init__(data, copy=copy, **kwargs)

    @property
    def n_variants(self):
        """Number of variants (length of first array dimension)."""
        return self.shape[0]

    @property
    def n_samples(self):
        """Number of samples (length of second array dimension)."""
        return self.shape[1]

    def __getitem__(self, item):
        s = self.values.__getitem__(item)

        # decide whether to wrap the result as GenotypeArray
        wrap_array = (
            hasattr(s, 'ndim') and s.ndim == 3 and  # dimensionality preserved
            s.shape[-1] == self.shape[-1] and  # ploidy preserved
            not _adds_newaxis(item)
        )
        if wrap_array:
            s = type(self)(s)
            if self.mask is not None:
                m = self.mask[item]
                s.mask = m
            if self.is_phased is not None:
                p = self.is_phased[item]
                s.is_phased = p
            return s

        # decide whether to wrap the result as GenotypeVector
        wrap_vector = (
            # row selection
            isinstance(item, int) or (
                # row selection
                isinstance(item, tuple) and
                len(item) == 2 and
                isinstance(item[0], int) and
                isinstance(item[1], (slice, list, np.ndarray, type(Ellipsis)))
            ) or (
                # column selection
                len(item) == 2 and
                isinstance(item[0], (slice, list, np.ndarray)) and
                isinstance(item[1], int)
            )
        )
        if wrap_vector:
            s = GenotypeVector(s)
            if self.mask is not None:
                m = self.mask[item]
                s.mask = m
            if self.is_phased is not None:
                p = self.is_phased[item]
                s.is_phased = p

        return s

    def _display_items(self, row_threshold, col_threshold, row_edgeitems, col_edgeitems):
        if row_threshold is None:
            row_threshold = self.shape[0]
        if col_threshold is None:
            col_threshold = self.shape[1]
        # ensure sensible edgeitems
        row_edgeitems = min(row_edgeitems, row_threshold // 2)
        col_edgeitems = min(col_edgeitems, col_threshold // 2)

        # determine indices of items to show
        if self.shape[0] > row_threshold:
            row_indices = list(range(row_edgeitems))
            row_indices += list(range(self.shape[0] - row_edgeitems, self.shape[0], 1))
        else:
            row_indices = list(range(self.shape[0]))
        if self.shape[1] > col_threshold:
            col_indices = list(range(col_edgeitems))
            col_indices += list(range(self.shape[1] - col_edgeitems, self.shape[1], 1))
        else:
            col_indices = list(range(self.shape[1]))

        # convert to gt
        tmp = self[np.array(row_indices)[:, np.newaxis], col_indices]
        if tmp.mask is not None:
            tmp.fill_masked(copy=False)
        gt = tmp.to_gt()
        n = gt.dtype.itemsize
        if PY2:
            items = [[x.rjust(n) for x in row] for row in gt]
        else:
            items = [[str(x, 'ascii').rjust(n) for x in row] for row in gt]

        # insert ellipsis
        if self.shape[1] > col_threshold:
            col_indices = (
                col_indices[:col_edgeitems] + ['...'] + col_indices[-col_edgeitems:]
            )
            items = [(row[:col_edgeitems] + [' ... '] + row[-col_edgeitems:])
                     for row in items]
        if self.shape[0] > row_threshold:
            row_indices = (
                row_indices[:row_edgeitems] + ['...'] + row_indices[-row_edgeitems:]
            )
            items = items[:row_edgeitems] + [['...']] + items[-row_edgeitems:]

        return row_indices, col_indices, items

    def to_str(self, row_threshold=6, col_threshold=10, row_edgeitems=3,
               col_edgeitems=5):
        _, _, items = self._display_items(row_threshold, col_threshold, row_edgeitems,
                                          col_edgeitems)
        s = ''
        for row in items:
            s += ' '.join(row) + '\n'
        return s

    def to_html(self, row_threshold=6, col_threshold=10, row_edgeitems=3, col_edgeitems=5,
                caption=None):
        row_indices, col_indices, items = self._display_items(
            row_threshold, col_threshold, row_edgeitems, col_edgeitems
        )
        # N.B., table captions don't render in jupyter notebooks on GitHub,
        # so put caption outside table element
        if caption is None:
            caption = '%s(%s, dtype=%s)\n' % (type(self).__name__, self.shape, self.dtype)
        # sanitize caption
        caption = caption.replace('<', '&lt;')
        caption = caption.strip().replace('\n', '<br/>')
        html = caption
        html += '<table>'
        html += '<tr><th></th>'
        html += ''.join(['<th style="text-align: center">%s</th>' % i
                         for i in col_indices])
        html += '</tr>'
        for row_index, row in zip(row_indices, items):
            if row_index == '...':
                html += '<tr><th style="text-align: center">...</th>' \
                        '<td style="text-align: center" colspan=%s>...</td></tr>' % \
                        (len(col_indices) + 1)
            else:
                html += '<tr><th style="text-align: center">%s</th>' % row_index
                html += ''.join(['<td style="text-align: center">%s</td>' % item
                                 for item in row])
                html += '</tr>'
        html += '</table>'
        return html

    def display(self, row_threshold=6, col_threshold=10, row_edgeitems=3,
                col_edgeitems=5, caption=None):
        html = self.to_html(row_threshold, col_threshold, row_edgeitems, col_edgeitems,
                            caption)
        from IPython.display import display_html
        display_html(html, raw=True)

    def displayall(self, caption=None):
        self.display(row_threshold=None, col_threshold=None, caption=caption)

    def to_haplotypes(self, copy=False):
        """Reshape a genotype array to view it as haplotypes by
        dropping the ploidy dimension.

        Parameters
        ----------
        copy : bool, optional
            If True, copy data.

        Returns
        -------
        h : HaplotypeArray, shape (n_variants, n_samples * ploidy)
            Haplotype array.

        Notes
        -----
        If genotype calls are unphased, the haplotypes returned by
        this function will bear no resemblance to the true haplotypes.

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 1], [1, 1]],
        ...                          [[0, 2], [-1, -1]]])
        >>> g.to_haplotypes()
        HaplotypeArray((3, 4), dtype=int64)
        0 0 0 1
        0 1 1 1
        0 2 . .

        """

        # reshape, preserving size of variants dimension
        newshape = (self.shape[0], -1)
        data = np.reshape(self, newshape)
        h = HaplotypeArray(data, copy=copy)
        return h

    def to_packed(self, boundscheck=True):
        """Pack diploid genotypes into a single byte for each genotype,
        using the left-most 4 bits for the first allele and the right-most 4
        bits for the second allele. Allows single byte encoding of diploid
        genotypes for variants with up to 15 alleles.

        Parameters
        ----------
        boundscheck : bool, optional
            If False, do not check that minimum and maximum alleles are
            compatible with bit-packing.

        Returns
        -------
        packed : ndarray, uint8, shape (n_variants, n_samples)
            Bit-packed genotype array.

        Notes
        -----
        If a mask has been set, it is ignored by this function.

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 2], [1, 1]],
        ...                          [[2, 2], [-1, -1]]], dtype='i1')
        >>> g.to_packed()
        array([[  0,   1],
               [  2,  17],
               [ 34, 239]], dtype=uint8)

        """

        # TODO use check_ploidy
        if self.shape[2] != 2:
            raise ValueError('can only pack diploid calls')

        if boundscheck:
            amx = self.max()
            if amx > 14:
                raise ValueError('max allele for packing is 14, found %s' % amx)
            amn = self.min()
            if amn < -1:
                raise ValueError('min allele for packing is -1, found %s' % amn)

        from allel.opt.model import genotype_pack_diploid

        # ensure int8 dtype
        if self.dtype.type == np.int8:
            data = self.values
        else:
            data = self.astype(dtype=np.int8)

        # pack data
        packed = genotype_pack_diploid(data)

        return packed

    @staticmethod
    def from_packed(packed):
        """Unpack diploid genotypes that have been bit-packed into single
        bytes.

        Parameters
        ----------
        packed : ndarray, uint8, shape (n_variants, n_samples)
            Bit-packed diploid genotype array.

        Returns
        -------
        g : GenotypeArray, shape (n_variants, n_samples, 2)
            Genotype array.

        Examples
        --------

        >>> import allel
        >>> import numpy as np
        >>> packed = np.array([[0, 1],
        ...                    [2, 17],
        ...                    [34, 239]], dtype='u1')
        >>> allel.GenotypeArray.from_packed(packed)
        GenotypeArray((3, 2, 2), dtype=int8)
        0/0 0/1
        0/2 1/1
        2/2 ./.

        """

        # check arguments
        packed = np.asarray(packed)
        check_ndim(packed, 2)
        check_dtype(packed, np.dtype('u1'))

        from allel.opt.model import genotype_unpack_diploid
        data = genotype_unpack_diploid(packed)
        return GenotypeArray(data)

    # noinspection PyShadowingBuiltins
    def to_sparse(self, format='csr', **kwargs):
        """Convert into a sparse matrix.

        Parameters
        ----------
        format : {'coo', 'csc', 'csr', 'dia', 'dok', 'lil'}
            Sparse matrix format.
        kwargs : keyword arguments
            Passed through to sparse matrix constructor.

        Returns
        -------
        m : scipy.sparse.spmatrix
            Sparse matrix

        Notes
        -----

        If a mask has been set, it is ignored by this function.

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 0]],
        ...                          [[0, 1], [0, 1]],
        ...                          [[1, 1], [0, 0]],
        ...                          [[0, 0], [-1, -1]]], dtype='i1')
        >>> m = g.to_sparse(format='csr')
        >>> m
        <4x4 sparse matrix of type '<class 'numpy.int8'>'
            with 6 stored elements in Compressed Sparse Row format>
        >>> m.data
        array([ 1,  1,  1,  1, -1, -1], dtype=int8)
        >>> m.indices
        array([1, 3, 0, 1, 2, 3], dtype=int32)
        >>> m.indptr
        array([0, 0, 2, 4, 6], dtype=int32)

        """

        h = self.to_haplotypes()
        m = h.to_sparse(format=format, **kwargs)
        return m

    @staticmethod
    def from_sparse(m, ploidy, order=None, out=None):
        """Construct a genotype array from a sparse matrix.

        Parameters
        ----------
        m : scipy.sparse.spmatrix
            Sparse matrix
        ploidy : int
            The sample ploidy.
        order : {'C', 'F'}, optional
            Whether to store data in C (row-major) or Fortran (column-major)
            order in memory.
        out : ndarray, shape (n_variants, n_samples), optional
            Use this array as the output buffer.

        Returns
        -------
        g : GenotypeArray, shape (n_variants, n_samples, ploidy)
            Genotype array.

        Examples
        --------

        >>> import allel
        >>> import numpy as np
        >>> import scipy.sparse
        >>> data = np.array([ 1,  1,  1,  1, -1, -1], dtype=np.int8)
        >>> indices = np.array([1, 3, 0, 1, 2, 3], dtype=np.int32)
        >>> indptr = np.array([0, 0, 2, 4, 6], dtype=np.int32)
        >>> m = scipy.sparse.csr_matrix((data, indices, indptr))
        >>> g = allel.GenotypeArray.from_sparse(m, ploidy=2)
        >>> g
        GenotypeArray((4, 2, 2), dtype=int8)
        0/0 0/0
        0/1 0/1
        1/1 0/0
        0/0 ./.

        """

        h = HaplotypeArray.from_sparse(m, order=order, out=out)
        g = h.to_genotypes(ploidy=ploidy)
        return g

    def haploidify_samples(self):
        """Construct a pseudo-haplotype for each sample by randomly
        selecting an allele from each genotype call.

        Returns
        -------
        h : HaplotypeArray

        Notes
        -----
        If a mask has been set, it is ignored by this function.

        Examples
        --------

        >>> import allel
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 2], [1, 1]],
        ...                          [[1, 2], [2, 1]],
        ...                          [[2, 2], [-1, -1]]])
        >>> g.haploidify_samples()
        HaplotypeArray((4, 2), dtype=int64)
        0 1
        0 1
        1 1
        2 .
        >>> g = allel.GenotypeArray([[[0, 0, 0], [0, 0, 1]],
        ...                          [[0, 1, 1], [1, 1, 1]],
        ...                          [[0, 1, 2], [-1, -1, -1]]])
        >>> g.haploidify_samples()
        HaplotypeArray((3, 2), dtype=int64)
        0 0
        1 1
        2 .

        """

        # N.B., this implementation is obscure and uses more memory than
        # necessary, TODO review

        # define the range of possible indices, e.g., diploid => (0, 1)
        index_range = np.arange(0, self.ploidy, dtype='u1')

        # create a random index for each genotype call
        indices = np.random.choice(index_range, size=self.n_calls, replace=True)

        # reshape genotype data so it's suitable for passing to np.choose
        # by merging the variants and samples dimensions
        choices = self.reshape(-1, self.ploidy).T

        # now use random indices to haploidify
        data = np.choose(indices, choices)

        # reshape the haploidified data to restore the variants and samples
        # dimensions
        data = data.reshape((self.n_variants, self.n_samples))

        # view as haplotype array
        h = HaplotypeArray(data, copy=False)

        return h

    def count_alleles(self, max_allele=None, subpop=None):
        """Count the number of calls of each allele per variant.

        Parameters
        ----------
        max_allele : int, optional
            The highest allele index to count. Alleles above this will be
            ignored.
        subpop : sequence of ints, optional
            Indices of samples to include in count.

        Returns
        -------
        ac : AlleleCountsArray

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 2], [1, 1]],
        ...                          [[2, 2], [-1, -1]]])
        >>> g.count_alleles()
        AlleleCountsArray((3, 3), dtype=int32)
        3 1 0
        1 2 1
        0 0 2
        >>> g.count_alleles(max_allele=1)
        AlleleCountsArray((3, 2), dtype=int32)
        3 1
        1 2
        0 0

        """

        # check inputs
        subpop = asarray_ndim(subpop, 1, allow_none=True, dtype=np.int64)
        if subpop is not None:
            if np.any(subpop >= self.shape[1]):
                raise ValueError('index out of bounds')
            if np.any(subpop < 0):
                raise ValueError('negative indices not supported')

        # determine alleles to count
        if max_allele is None:
            max_allele = self.max()

        if self.dtype == np.dtype('i1'):
            # use optimisations
            from allel.opt.model import genotype_int8_count_alleles, \
                genotype_int8_count_alleles_masked, \
                genotype_int8_count_alleles_subpop, \
                genotype_int8_count_alleles_subpop_masked

            if subpop is None and self.mask is None:
                ac = genotype_int8_count_alleles(self.values, max_allele)
            elif subpop is None:
                ac = genotype_int8_count_alleles_masked(
                    self.values, self.mask.view(dtype='u1'), max_allele
                )
            elif self.mask is None:
                ac = genotype_int8_count_alleles_subpop(self.values, max_allele, subpop)
            else:
                ac = genotype_int8_count_alleles_subpop_masked(
                    self.values, self.mask.view(dtype='u1'), max_allele, subpop
                )

        else:
            # set up output array
            ac = np.zeros((self.shape[0], max_allele + 1), dtype='i4')

            # extract subpop
            g = self
            if subpop is not None:
                g = g[:, subpop]

            # count alleles
            alleles = list(range(max_allele + 1))
            for allele in alleles:
                allele_match = g == allele
                if g.mask is not None:
                    allele_match &= ~g.mask[:, :, None]
                np.sum(allele_match, axis=(1, 2), out=ac[:, allele])

        return AlleleCountsArray(ac, copy=False)

    def count_alleles_subpops(self, subpops, max_allele=None):
        """Count alleles for multiple subpopulations simultaneously.

        Parameters
        ----------
        subpops : dict (string -> sequence of ints)
            Mapping of subpopulation names to sample indices.
        max_allele : int, optional
            The highest allele index to count. Alleles above this will be
            ignored.

        Returns
        -------
        out : dict (string -> AlleleCountsArray)
            A mapping of subpopulation names to allele counts arrays.

        """

        if max_allele is None:
            max_allele = self.max()

        out = {name: self.count_alleles(max_allele=max_allele, subpop=subpop)
               for name, subpop in subpops.items()}

        return out

    def compress(self, condition, axis=0):
        out = self.values.compress(condition, axis=axis)
        if axis in {0, 1}:
            out = type(self)(out)
            if self.mask is not None:
                out.mask = self.mask.compress(condition, axis=axis)
            if self.is_phased is not None:
                out.is_phased = self.is_phased.compress(condition, axis=axis)
        return out

    def take(self, indices, axis=0):
        out = self.values.take(indices, axis=axis)
        if axis in {0, 1}:
            out = type(self)(out)
            if self.mask is not None:
                out.mask = self.mask.take(indices, axis=axis)
            if self.is_phased is not None:
                out.is_phased = self.is_phased.take(indices, axis=axis)
        return out

    def subset(self, sel0=None, sel1=None):
        """Make a sub-selection of variants and samples.

        Parameters
        ----------
        sel0 : array_like
            Boolean array or list of indices selecting variants.
        sel1 : array_like
            Boolean array or list of indices selecting samples.

        Returns
        -------
        out : GenotypeArray

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1], [1, 1]],
        ...                          [[0, 1], [1, 1], [1, 2]],
        ...                          [[0, 2], [-1, -1], [-1, -1]]])
        >>> g.subset([0, 1], [0, 2])
        GenotypeArray((2, 2, 2), dtype=int64)
        0/0 1/1
        0/1 1/2

        See Also
        --------
        GenotypeArray.take, GenotypeArray.compress

        """

        out = type(self)(subset(self.values, sel0, sel1))
        if self.mask is not None:
            out.mask = subset(self.mask, sel0, sel1)
        if self.is_phased is not None:
            out.is_phased = subset(self.is_phased, sel0, sel1)
        return out

    def hstack(self, *others):
        """Stack arrays in sequence horizontally (column-wise)."""
        out = super(GenotypeArray, self).hstack(*others)
        out = type(self)(out)
        return out

    def vstack(self, *others):
        """Stack arrays in sequence vertically (row-wise)."""
        out = super(GenotypeArray, self).vstack(*others)
        out = type(self)(out)
        return out

    def concatenate(self, *others, **kwargs):
        """Concatenate arrays."""
        out = super(GenotypeArray, self).concatenate(*others, **kwargs)
        axis = kwargs.get('axis', 0)
        if axis in {0, 1}:
            out = type(self)(out)
        return out

    def map_alleles(self, mapping, copy=True):
        """Transform alleles via a mapping.

        Parameters
        ----------
        mapping : ndarray, int8, shape (n_variants, max_allele)
            An array defining the allele mapping for each variant.
        copy : bool, optional
            If True, return a new array; if False, apply mapping in place
            (only applies for arrays with dtype int8; all other dtypes
            require a copy).

        Returns
        -------
        gm : GenotypeArray

        Notes
        -----
        If a mask has been set, it is ignored by this function.

        Examples
        --------

        >>> import allel
        >>> import numpy as np
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 2], [1, 1]],
        ...                          [[1, 2], [2, 1]],
        ...                          [[2, 2], [-1, -1]]], dtype='i1')
        >>> mapping = np.array([[1, 2, 0],
        ...                     [2, 0, 1],
        ...                     [2, 1, 0],
        ...                     [0, 2, 1]], dtype='i1')
        >>> g.map_alleles(mapping)
        GenotypeArray((4, 2, 2), dtype=int8)
        1/1 1/2
        2/1 0/0
        1/0 0/1
        1/1 ./.

        Notes
        -----
        For arrays with dtype int8 an optimised implementation is used which is
        faster and uses far less memory. It is recommended to convert arrays to
        dtype int8 where possible before calling this method.

        See Also
        --------
        create_allele_mapping

        """

        h = self.to_haplotypes()
        hm = h.map_alleles(mapping, copy=copy)
        gm = hm.to_genotypes(ploidy=self.shape[2])
        return gm


def _adds_newaxis(item):
    if item is None:
        return True
    elif item is np.newaxis:
        return True
    elif isinstance(item, tuple):
        return any((i is None or i is np.newaxis) for i in item)
    return False


class GenotypeVector(GenotypeBase):

    @classmethod
    def _check_values(cls, data):
        super(GenotypeVector, cls)._check_values(data)
        check_ndim(data, 2)

    def __init__(self, data, copy=False, **kwargs):
        super(GenotypeVector, self).__init__(data, copy=copy, **kwargs)

    def __getitem__(self, item):
        s = self.values.__getitem__(item)
        # decide whether to wrap the result
        wrap = (
            hasattr(s, 'ndim') and s.ndim == self.ndim and  # dimensionality preserved
            s.shape[-1] == self.shape[-1] and  # ploidy preserved
            not _adds_newaxis(item)
        )
        if wrap:
            s = type(self)(s)
            if self.mask is not None:
                m = self.mask[item]
                s.mask = m
            if self.is_phased is not None:
                p = self.is_phased[item]
                s.is_phased = p
            return s
        return s

    def _display_items(self, threshold, edgeitems):
        if threshold is None:
            threshold = self.shape[0]

        # ensure sensible edgeitems
        edgeitems = min(edgeitems, threshold // 2)

        # determine indices of items to show
        if self.shape[0] > threshold:
            indices = list(range(edgeitems))
            indices += list(range(self.shape[0] - edgeitems, self.shape[0], 1))
        else:
            indices = list(range(self.shape[0]))

        # convert to gt
        tmp = self[indices]
        if tmp.mask is not None:
            tmp.fill_masked(copy=False)
        gt = tmp.to_gt()
        if PY2:
            items = list(gt)
        else:
            items = [str(x, 'ascii') for x in gt]

        # insert ellipsis
        if self.shape[0] > threshold:
            indices = indices[:edgeitems] + [' ... '] + indices[-edgeitems:]
            items = items[:edgeitems] + [' ... '] + items[-edgeitems:]

        return indices, items

    def to_str(self, threshold=10, edgeitems=5):
        _, items = self._display_items(threshold, edgeitems)
        s = ' '.join(items)
        return s

    def to_html(self, threshold=10, edgeitems=5, caption=None):
        indices, items = self._display_items(threshold, edgeitems)
        # N.B., table captions don't render in jupyter notebooks on GitHub,
        # so put caption outside table element
        if caption is None:
            caption = '%s(%s, dtype=%s)\n' % (type(self).__name__, self.shape, self.dtype)
        # sanitize caption
        caption = caption.replace('<', '&lt;')
        caption = caption.strip().replace('\n', '<br/>')
        html = caption
        html += '<table>'
        html += '<tr>'
        html += ''.join(['<th style="text-align: center">%s</th>' % i
                         for i in indices])
        html += '</tr>'
        html += '<tr>'
        html += ''.join(['<td style="text-align: center">%s</td>' % item
                         for item in items])
        html += '</tr>'
        html += '</table>'
        return html

    def display(self, threshold=10, edgeitems=5, caption=None):
        html = self.to_html(threshold, edgeitems, caption)
        from IPython.display import display_html
        display_html(html, raw=True)

    def displayall(self, caption=None):
        self.display(threshold=None, caption=caption)

    def compress(self, condition, axis=0):
        out = self.values.compress(condition, axis=axis)
        if axis == 0:
            out = type(self)(out)
            if self.mask is not None:
                out.mask = self.mask.compress(condition, axis=axis)
            if self.is_phased is not None:
                out.is_phased = self.is_phased.compress(condition, axis=axis)
        return out

    def take(self, indices, axis=0):
        out = self.values.take(indices, axis=axis)
        if axis == 0:
            out = type(self)(out)
            if self.mask is not None:
                out.mask = self.mask.take(indices, axis=axis)
            if self.is_phased is not None:
                out.is_phased = self.is_phased.take(indices, axis=axis)
        return out

    def vstack(self, *others):
        """Stack arrays in sequence vertically (row-wise)."""
        out = super(GenotypeVector, self).vstack(*others)
        out = type(self)(out)
        return out

    def concatenate(self, *others, **kwargs):
        """Concatenate arrays."""
        out = super(GenotypeVector, self).concatenate(*others, **kwargs)
        axis = kwargs.get('axis', 0)
        if axis == 0:
            out = type(self)(out)
        return out

    def to_haplotypes(self, copy=False):
        return HaplotypeArray(self, copy=copy)


class HaplotypeArray(ArrayBase):
    """Array of haplotypes.

    Parameters
    ----------
    data : array_like, int, shape (n_variants, n_haplotypes)
        Haplotype data.
    **kwargs : keyword arguments
        All keyword arguments are passed through to :func:`numpy.array`.

    Notes
    -----
    This class represents haplotype data as a 2-dimensional numpy
    array of integers. By convention the first dimension corresponds
    to the variants genotyped, the second dimension corresponds to the
    haplotypes.

    Each integer within the array corresponds to an **allele index**,
    where 0 is the reference allele, 1 is the first alternate allele,
    2 is the second alternate allele, ... and -1 (or any other
    negative integer) is a missing allele call.

    If adjacent haplotypes originate from the same sample, then a
    haplotype array can also be viewed as a genotype array. However,
    this is not a requirement.

    With data on large numbers of variants and/or haplotypes,
    storing the data in memory as an uncompressed numpy
    array if integers may be impractical. For working with large
    arrays of haplotype data, see the
    :class:`allel.model.chunked.HaplotypeChunkedArray` class, which provides an
    alternative implementation of this interface using chunked compressed
    arrays.

    Examples
    --------

    Instantiate a haplotype array::

        >>> import allel
        >>> h = allel.HaplotypeArray([[0, 0, 0, 1],
        ...                           [0, 1, 1, 1],
        ...                           [0, 2, -1, -1]], dtype='i1')
        >>> h.dtype
        dtype('int8')
        >>> h.ndim
        2
        >>> h.shape
        (3, 4)
        >>> h.n_variants
        3
        >>> h.n_haplotypes
        4
        >>> h
        HaplotypeArray((3, 4), dtype=int8)
        0 0 0 1
        0 1 1 1
        0 2 . .

    Allele calls for a single variant at all haplotypes can be obtained
    by indexing the first dimension, e.g.::

        >>> h[1]
        array([0, 1, 1, 1], dtype=int8)

    A single haplotype can be obtained by indexing the second
    dimension, e.g.::

        >>> h[:, 1]
        array([0, 1, 2], dtype=int8)

    An allele call for a single haplotype at a single variant can be
    obtained by indexing the first and second dimensions, e.g.::

        >>> h[1, 0]
        0

    View haplotypes as diploid genotypes::

        >>> h.to_genotypes(ploidy=2)
        GenotypeArray((3, 2, 2), dtype=int8)
        0/0 0/1
        0/1 1/1
        0/2 ./.

    """

    @classmethod
    def _check_values(cls, data):
        check_dtype_kind(data, 'u', 'i')
        check_ndim(data, 2)

    def __init__(self, data, copy=False, **kwargs):
        super(HaplotypeArray, self).__init__(data, copy=copy, **kwargs)

    @property
    def n_variants(self):
        """Number of variants."""
        return self.shape[0]

    @property
    def n_haplotypes(self):
        """Number of haplotypes."""
        return self.shape[1]

    def _display_items(self, row_threshold, col_threshold, row_edgeitems, col_edgeitems):
        if row_threshold is None:
            row_threshold = self.shape[0]
        if col_threshold is None:
            col_threshold = self.shape[1]

        # ensure sensible edgeitems
        row_edgeitems = min(row_edgeitems, row_threshold // 2)
        col_edgeitems = min(col_edgeitems, col_threshold // 2)

        # determine indices of items to show
        if self.shape[0] > row_threshold:
            row_indices = list(range(row_edgeitems))
            row_indices += list(range(self.shape[0] - row_edgeitems, self.shape[0], 1))
        else:
            row_indices = list(range(self.shape[0]))
        if self.shape[1] > col_threshold:
            col_indices = list(range(col_edgeitems))
            col_indices += list(range(self.shape[1] - col_edgeitems, self.shape[1], 1))
        else:
            col_indices = list(range(self.shape[1]))

        # convert to stringy thingy
        tmp = self[np.array(row_indices)[:, np.newaxis], col_indices]
        max_allele = np.max(tmp)
        if max_allele <= 0:
            max_allele = 1
        n = int(np.floor(np.log10(max_allele))) + 1
        t = tmp.astype((np.string_, n)).view(np.chararray)
        # recode missing alleles
        t[tmp < 0] = b'.'
        if PY2:
            items = [[x.rjust(n) for x in row] for row in t]
        else:
            items = [[str(x, 'ascii').rjust(n) for x in row] for row in t]

        # insert ellipsis
        if self.shape[1] > col_threshold:
            col_indices = (
                col_indices[:col_edgeitems] + ['...'] + col_indices[-col_edgeitems:]
            )
            items = [(row[:col_edgeitems] + [' ... '] + row[-col_edgeitems:])
                     for row in items]
        if self.shape[0] > row_threshold:
            row_indices = (
                row_indices[:row_edgeitems] + ['...'] + row_indices[-row_edgeitems:]
            )
            items = items[:row_edgeitems] + [['...']] + items[-row_edgeitems:]

        return row_indices, col_indices, items

    def to_str(self, row_threshold=6, col_threshold=20, row_edgeitems=3,
               col_edgeitems=10):
        _, _, items = self._display_items(row_threshold, col_threshold, row_edgeitems,
                                          col_edgeitems)
        s = ''
        for row in items:
            s += ' '.join(row) + '\n'
        return s

    def to_html(self, row_threshold=6, col_threshold=20, row_edgeitems=3, col_edgeitems=10,
                caption=None):
        # TODO refactor with GenotypeArray.to_html
        row_indices, col_indices, items = self._display_items(
            row_threshold, col_threshold, row_edgeitems, col_edgeitems
        )
        # N.B., table captions don't render in jupyter notebooks on GitHub,
        # so put caption outside table element
        if caption is None:
            caption = '%s(%s, dtype=%s)\n' % (type(self).__name__, self.shape, self.dtype)
        # sanitize caption
        caption = caption.replace('<', '&lt;')
        caption = caption.strip().replace('\n', '<br/>')
        html = caption
        html += '<table>'
        html += '<tr><th></th>'
        html += ''.join(['<th style="text-align: center">%s</th>' % i
                         for i in col_indices])
        html += '</tr>'
        for row_index, row in zip(row_indices, items):
            if row_index == '...':
                html += '<tr><th style="text-align: center">...</th>' \
                        '<td style="text-align: center" colspan="%s">...</td></tr>' % \
                        len(col_indices)
            else:
                html += '<tr><th style="text-align: center">%s</th>' % \
                        row_index
                html += ''.join(['<td style="text-align: center">%s</td>' % item
                                 for item in row])
                html += '</tr>'
        html += '</table>'
        return html

    def __str__(self):
        return self.to_str()

    def _repr_html_(self):
        return self.to_html()

    def display(self, row_threshold=6, col_threshold=10, row_edgeitems=3,
                col_edgeitems=5, caption=None):
        html = self.to_html(row_threshold, col_threshold, row_edgeitems,
                            col_edgeitems, caption)
        from IPython.display import display_html
        display_html(html, raw=True)

    def displayall(self, caption=None):
        self.display(row_threshold=None, col_threshold=None, caption=caption)

    def __getitem__(self, item):
        s = self.values.__getitem__(item)

        # decide whether to wrap the result as GenotypeArray
        wrap_array = (
            hasattr(s, 'ndim') and s.ndim == 2 and  # dimensionality preserved
            not _adds_newaxis(item)
        )
        if wrap_array:
            s = type(self)(s)
            return s

        return s

    def subset(self, sel0=None, sel1=None):
        """Make a sub-selection of variants and haplotypes.

        Parameters
        ----------
        sel0 : array_like
            Boolean array or list of indices selecting variants.
        sel1 : array_like
            Boolean array or list of indices selecting haplotypes.

        Returns
        -------
        out : HaplotypeArray

        See Also
        --------
        HaplotypeArray.take, HaplotypeArray.compress

        """

        return type(self)(subset(self, sel0, sel1))

    def compress(self, condition, axis=0):
        out = self.values.compress(condition, axis=axis)
        if axis in {0, 1}:
            out = type(self)(out)
        return out

    def take(self, indices, axis=0):
        out = self.values.take(indices, axis=axis)
        if axis in {0, 1}:
            out = type(self)(out)
        return out

    def hstack(self, *others):
        """Stack arrays in sequence horizontally (column-wise)."""
        out = super(HaplotypeArray, self).hstack(*others)
        out = type(self)(out)
        return out

    def vstack(self, *others):
        """Stack arrays in sequence vertically (row-wise)."""
        out = super(HaplotypeArray, self).vstack(*others)
        out = type(self)(out)
        return out

    def concatenate(self, *others, **kwargs):
        """Concatenate arrays."""
        out = super(HaplotypeArray, self).concatenate(*others, **kwargs)
        axis = kwargs.get('axis', 0)
        if axis in {0, 1}:
            out = type(self)(out)
        return out

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

    def count_called(self, axis=None):
        b = self.is_called()
        return np.sum(b, axis=axis)

    def count_missing(self, axis=None):
        b = self.is_missing()
        return np.sum(b, axis=axis)

    def count_ref(self, axis=None):
        b = self.is_ref()
        return np.sum(b, axis=axis)

    def count_alt(self, axis=None):
        b = self.is_alt()
        return np.sum(b, axis=axis)

    def count_call(self, allele, axis=None):
        b = self.is_call(allele=allele)
        return np.sum(b, axis=axis)

    def to_genotypes(self, ploidy, copy=False):
        """Reshape a haplotype array to view it as genotypes by restoring the
        ploidy dimension.

        Parameters
        ----------
        ploidy : int
            The sample ploidy.
        copy : bool, optional
            If True, make a copy of data.

        Returns
        -------
        g : ndarray, int, shape (n_variants, n_samples, ploidy)
            Genotype array (sharing same underlying buffer).
        copy : bool, optional
            If True, copy the data.

        Examples
        --------

        >>> import allel
        >>> h = allel.HaplotypeArray([[0, 0, 0, 1],
        ...                           [0, 1, 1, 1],
        ...                           [0, 2, -1, -1]], dtype='i1')
        >>> h.to_genotypes(ploidy=2)
        GenotypeArray((3, 2, 2), dtype=int8)
        0/0 0/1
        0/1 1/1
        0/2 ./.

        """

        # check ploidy is compatible
        if (self.shape[1] % ploidy) > 0:
            raise ValueError('incompatible ploidy')

        # reshape
        newshape = (self.shape[0], -1, ploidy)
        data = self.reshape(newshape)

        # wrap
        g = GenotypeArray(data, copy=copy)

        return g

    # noinspection PyShadowingBuiltins
    def to_sparse(self, format='csr', **kwargs):
        """Convert into a sparse matrix.

        Parameters
        ----------
        format : {'coo', 'csc', 'csr', 'dia', 'dok', 'lil'}
            Sparse matrix format.
        kwargs : keyword arguments
            Passed through to sparse matrix constructor.

        Returns
        -------
        m : scipy.sparse.spmatrix
            Sparse matrix

        Examples
        --------

        >>> import allel
        >>> h = allel.HaplotypeArray([[0, 0, 0, 0],
        ...                           [0, 1, 0, 1],
        ...                           [1, 1, 0, 0],
        ...                           [0, 0, -1, -1]], dtype='i1')
        >>> m = h.to_sparse(format='csr')
        >>> m
        <4x4 sparse matrix of type '<class 'numpy.int8'>'
            with 6 stored elements in Compressed Sparse Row format>
        >>> m.data
        array([ 1,  1,  1,  1, -1, -1], dtype=int8)
        >>> m.indices
        array([1, 3, 0, 1, 2, 3], dtype=int32)
        >>> m.indptr
        array([0, 0, 2, 4, 6], dtype=int32)

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
        """Construct a haplotype array from a sparse matrix.

        Parameters
        ----------
        m : scipy.sparse.spmatrix
            Sparse matrix
        order : {'C', 'F'}, optional
            Whether to store data in C (row-major) or Fortran (column-major)
            order in memory.
        out : ndarray, shape (n_variants, n_samples), optional
            Use this array as the output buffer.

        Returns
        -------
        h : HaplotypeArray, shape (n_variants, n_haplotypes)
            Haplotype array.

        Examples
        --------

        >>> import allel
        >>> import numpy as np
        >>> import scipy.sparse
        >>> data = np.array([ 1,  1,  1,  1, -1, -1], dtype=np.int8)
        >>> indices = np.array([1, 3, 0, 1, 2, 3], dtype=np.int32)
        >>> indptr = np.array([0, 0, 2, 4, 6], dtype=np.int32)
        >>> m = scipy.sparse.csr_matrix((data, indices, indptr))
        >>> h = allel.HaplotypeArray.from_sparse(m)
        >>> h
        HaplotypeArray((4, 4), dtype=int8)
        0 0 0 0
        0 1 0 1
        1 1 0 0
        0 0 . .

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

    def count_alleles(self, max_allele=None, subpop=None):
        """Count the number of calls of each allele per variant.

        Parameters
        ----------
        max_allele : int, optional
            The highest allele index to count. Alleles greater than this
            index will be ignored.
        subpop : array_like, int, optional
            Indices of haplotypes to include.

        Returns
        -------
        ac : AlleleCountsArray, int, shape (n_variants, n_alleles)

        Examples
        --------

        >>> import allel
        >>> h = allel.HaplotypeArray([[0, 0, 0, 1],
        ...                           [0, 1, 1, 1],
        ...                           [0, 2, -1, -1]], dtype='i1')
        >>> ac = h.count_alleles()
        >>> ac
        AlleleCountsArray((3, 3), dtype=int32)
        3 1 0
        1 3 0
        1 0 1

        """

        # check inputs
        subpop = asarray_ndim(subpop, 1, allow_none=True, dtype=np.int64)
        if subpop is not None:
            if np.any(subpop >= self.shape[1]):
                raise ValueError('index out of bounds')
            if np.any(subpop < 0):
                raise ValueError('negative indices not supported')

        # determine alleles to count
        if max_allele is None:
            max_allele = self.max()

        if self.dtype == np.dtype('i1'):
            # use optimisations
            from allel.opt.model import haplotype_int8_count_alleles, \
                haplotype_int8_count_alleles_subpop
            if subpop is None:
                ac = haplotype_int8_count_alleles(self.values, max_allele)

            else:
                ac = haplotype_int8_count_alleles_subpop(self.values, max_allele, subpop)

        else:
            # set up output array
            ac = np.zeros((self.shape[0], max_allele + 1), dtype='i4')

            # extract subpop
            if subpop is not None:
                h = self[:, subpop]
            else:
                h = self

            # count alleles
            alleles = list(range(max_allele + 1))
            for allele in alleles:
                np.sum(h == allele, axis=1, out=ac[:, allele])

        return AlleleCountsArray(ac, copy=False)

    def count_alleles_subpops(self, subpops, max_allele=None):
        """Count alleles for multiple subpopulations simultaneously.

        Parameters
        ----------
        subpops : dict (string -> sequence of ints)
            Mapping of subpopulation names to sample indices.
        max_allele : int, optional
            The highest allele index to count. Alleles above this will be
            ignored.

        Returns
        -------
        out : dict (string -> AlleleCountsArray)
            A mapping of subpopulation names to allele counts arrays.

        """

        if max_allele is None:
            max_allele = self.max()

        out = {name: self.count_alleles(max_allele=max_allele, subpop=subpop)
               for name, subpop in subpops.items()}

        return out

    def map_alleles(self, mapping, copy=True):
        """Transform alleles via a mapping.

        Parameters
        ----------
        mapping : ndarray, int8, shape (n_variants, max_allele)
            An array defining the allele mapping for each variant.
        copy : bool, optional
            If True, return a new array; if False, apply mapping in place
            (only applies for arrays with dtype int8; all other dtypes
            require a copy).

        Returns
        -------
        hm : HaplotypeArray

        Examples
        --------

        >>> import allel
        >>> import numpy as np
        >>> h = allel.HaplotypeArray([[0, 0, 0, 1],
        ...                           [0, 1, 1, 1],
        ...                           [0, 2, -1, -1]], dtype='i1')
        >>> mapping = np.array([[1, 2, 0],
        ...                     [2, 0, 1],
        ...                     [2, 1, 0]], dtype='i1')
        >>> h.map_alleles(mapping)
        HaplotypeArray((3, 4), dtype=int8)
        1 1 1 2
        2 0 0 0
        2 0 . .

        Notes
        -----

        For arrays with dtype int8 an optimised implementation is used which is
        faster and uses far less memory. It is recommended to convert arrays to
        dtype int8 where possible before calling this method.

        See Also
        --------

        create_allele_mapping

        """

        # check inputs
        mapping = asarray_ndim(mapping, 2)
        check_dim0_aligned(self, mapping)

        if self.dtype == np.dtype('i1'):
            # use optimisation
            mapping = np.asarray(mapping, dtype='i1')
            from allel.opt.model import haplotype_int8_map_alleles
            data = haplotype_int8_map_alleles(self.values, mapping, copy=copy)

        else:
            # use numpy indexing
            i = np.arange(self.shape[0]).reshape((-1, 1))
            data = mapping[i, self]
            data[self < 0] = -1

        return HaplotypeArray(data, copy=False)

    def prefix_argsort(self):
        """Return indices that would sort the haplotypes by prefix."""
        return np.lexsort(self[::-1])

    def distinct(self):
        """Return sets of indices for each distinct haplotype."""

        # setup collection
        d = collections.defaultdict(set)

        # iterate over haplotypes
        for i in range(self.shape[1]):

            # hash the haplotype
            k = hash(self[:, i].tobytes())

            # collect
            d[k].add(i)

        # extract sets, sorted by most common
        return sorted(d.values(), key=len, reverse=True)

    def distinct_counts(self):
        """Return counts for each distinct haplotype."""

        # hash the haplotypes
        k = [hash(self[:, i].tobytes()) for i in range(self.shape[1])]

        # count and sort
        # noinspection PyArgumentList
        counts = sorted(collections.Counter(k).values(), reverse=True)

        return np.asarray(counts)

    def distinct_frequencies(self):
        """Return frequencies for each distinct haplotype."""

        c = self.distinct_counts()
        n = self.shape[1]
        return c / n


class AlleleCountsArray(ArrayBase):
    """Array of allele counts.

    Parameters
    ----------
    data : array_like, int, shape (n_variants, n_alleles)
        Allele counts data.
    copy : bool, optional
        If True, make a copy of `data`.
    **kwargs : keyword arguments
        All keyword arguments are passed through to :func:`numpy.array`.

    Notes
    -----
    This class represents allele counts as a 2-dimensional numpy
    array of integers. By convention the first dimension corresponds
    to the variants genotyped, the second dimension corresponds to the
    alleles counted.

    Examples
    --------

    Obtain allele counts from a genotype array:

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 1], [1, 1]],
        ...                          [[0, 2], [-1, -1]]], dtype='i1')
        >>> ac = g.count_alleles()
        >>> ac
        AlleleCountsArray((3, 3), dtype=int32)
        3 1 0
        1 3 0
        1 0 1
        >>> ac.dtype
        dtype('int32')
        >>> ac.shape
        (3, 3)
        >>> ac.n_variants
        3
        >>> ac.n_alleles
        3

    Allele counts for a single variant can be obtained by indexing the first
    dimension, e.g.::

        >>> ac[1]
        array([1, 3, 0], dtype=int32)

    Allele counts for a specific allele can be obtained by indexing the
    second dimension, e.g., reference allele counts:

        >>> ac[:, 0]
        array([3, 1, 1], dtype=int32)

    Calculate the total number of alleles called for each variant:

        >>> import numpy as np
        >>> n = np.sum(ac, axis=1)
        >>> n
        array([4, 4, 2])

    """

    def __init__(self, data, copy=False, **kwargs):
        super(AlleleCountsArray, self).__init__(data, copy=copy, **kwargs)

    @classmethod
    def _check_values(cls, data):
        check_dtype_kind(data, 'u', 'i')
        check_ndim(data, 2)

    def __add__(self, other):
        ret = super(AlleleCountsArray, self).__add__(other)
        if hasattr(ret, 'shape') and ret.shape == self.shape:
            ret = AlleleCountsArray(ret)
        return ret

    def __sub__(self, other):
        ret = super(AlleleCountsArray, self).__sub__(other)
        if hasattr(ret, 'shape') and ret.shape == self.shape:
            ret = AlleleCountsArray(ret)
        return ret

    def __getitem__(self, item):
        s = self.values.__getitem__(item)

        # decide whether to wrap the result
        wrap_array = (
            hasattr(s, 'ndim') and s.ndim == 2 and  # dimensionality preserved
            self.shape[1] == s.shape[1] and  # number of alleles preserved
            not _adds_newaxis(item)
        )
        if wrap_array:
            s = type(self)(s)
            return s

        return s

    def __str__(self):
        return self.to_str()

    def _repr_html_(self):
        return self.to_html()

    @property
    def n_variants(self):
        """Number of variants (length of first array dimension)."""
        return self.shape[0]

    @property
    def n_alleles(self):
        """Number of alleles (length of second array dimension)."""
        return self.shape[1]

    def _display_items(self, threshold, edgeitems):
        if threshold is None:
            threshold = self.shape[0]

        # ensure sensible edgeitems
        edgeitems = min(edgeitems, threshold // 2)

        # determine indices of items to show
        if self.shape[0] > threshold:
            indices = list(range(edgeitems))
            indices += list(range(self.shape[0] - edgeitems, self.shape[0], 1))
        else:
            indices = list(range(self.shape[0]))

        # convert to stringy thingy
        tmp = self[indices]
        max_value = np.max(tmp)
        if max_value <= 0:
            max_value = 1
        n = int(np.floor(np.log10(max_value))) + 1
        t = tmp.astype((np.string_, n)).view(np.chararray)
        if PY2:
            items = [[x.rjust(n) for x in row] for row in t]
        else:
            items = [[str(x, 'ascii').rjust(n) for x in row] for row in t]

        # insert ellipsis
        if self.shape[0] > threshold:
            indices = (
                indices[:edgeitems] + ['...'] + indices[-edgeitems:]
            )
            items = items[:edgeitems] + [['...']] + items[-edgeitems:]

        return indices, items

    def to_str(self, threshold=6, edgeitems=3):
        _, items = self._display_items(threshold, edgeitems)
        s = ''
        for row in items:
            s += ' '.join(row) + '\n'
        return s

    def to_html(self, threshold=6, edgeitems=3, caption=None):
        indices, items = self._display_items(threshold, edgeitems)
        # N.B., table captions don't render in jupyter notebooks on GitHub,
        # so put caption outside table element
        if caption is None:
            caption = '%s(%s, dtype=%s)\n' % (type(self).__name__, self.shape, self.dtype)
        # sanitize caption
        caption = caption.replace('<', '&lt;')
        caption = caption.strip().replace('\n', '<br/>')
        html = caption
        html += '<table>'
        html += '<tr><th></th>'
        html += ''.join(['<th style="text-align: center">%s</th>' % i
                         for i in range(self.shape[1])])
        html += '</tr>'
        for row_index, row in zip(indices, items):
            if row_index == '...':
                html += '<tr><th style="text-align: center">...</th>' \
                        '<td style="text-align: center" colspan="%s">...</td></tr>' % \
                        self.shape[1]
            else:
                html += '<tr><th style="text-align: center">%s</th>' % row_index
                html += ''.join(['<td style="text-align: center">%s</td>' % item
                                 for item in row])
                html += '</tr>'
        html += '</table>'
        return html

    def display(self, threshold=6, edgeitems=3, caption=None):
        html = self.to_html(threshold, edgeitems, caption)
        from IPython.display import display_html
        display_html(html, raw=True)

    def displayall(self, caption=None):
        self.display(threshold=None, caption=caption)

    def to_frequencies(self, fill=np.nan):
        """Compute allele frequencies.

        Parameters
        ----------
        fill : float, optional
            Value to use when number of allele calls is 0.

        Returns
        -------
        af : ndarray, float, shape (n_variants, n_alleles)

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 2], [1, 1]],
        ...                          [[2, 2], [-1, -1]]])
        >>> ac = g.count_alleles()
        >>> ac.to_frequencies()
        array([[ 0.75,  0.25,  0.  ],
               [ 0.25,  0.5 ,  0.25],
               [ 0.  ,  0.  ,  1.  ]])

        """

        n = np.sum(self, axis=1)[:, None]
        with ignore_invalid():
            af = np.where(n > 0, self / n, fill)

        return af

    def allelism(self):
        """Determine the number of distinct alleles observed for each variant.

        Returns
        -------
        n : ndarray, int, shape (n_variants,)
            Allelism array.

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 2], [1, 1]],
        ...                          [[2, 2], [-1, -1]]])
        >>> ac = g.count_alleles()
        >>> ac.allelism()
        array([2, 3, 1])

        """

        return np.sum(self > 0, axis=1)

    def max_allele(self):
        """Return the highest allele index for each variant.

        Returns
        -------
        n : ndarray, int, shape (n_variants,)
            Allele index array.

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 2], [1, 1]],
        ...                          [[2, 2], [-1, -1]]])
        >>> ac = g.count_alleles()
        >>> ac.max_allele()
        array([1, 2, 2], dtype=int8)

        """

        out = np.empty(self.shape[0], dtype='i1')
        out.fill(-1)
        for i in range(self.shape[1]):
            d = self[:, i] > 0
            out[d] = i
        return out

    def is_variant(self):
        """Find variants with at least one non-reference allele call.

        Returns
        -------
        out : ndarray, bool, shape (n_variants,)
            Boolean array where elements are True if variant matches the
            condition.

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 0]],
        ...                          [[0, 0], [0, 1]],
        ...                          [[0, 2], [1, 1]],
        ...                          [[2, 2], [-1, -1]]])
        >>> ac = g.count_alleles()
        >>> ac.is_variant()
        array([False,  True,  True,  True], dtype=bool)

        """

        return np.any(self[:, 1:] > 0, axis=1)

    def is_non_variant(self):
        """Find variants with no non-reference allele calls.

        Returns
        -------
        out : ndarray, bool, shape (n_variants,)
            Boolean array where elements are True if variant matches the
            condition.

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 0]],
        ...                          [[0, 0], [0, 1]],
        ...                          [[0, 2], [1, 1]],
        ...                          [[2, 2], [-1, -1]]])
        >>> ac = g.count_alleles()
        >>> ac.is_non_variant()
        array([ True, False, False, False], dtype=bool)

        """

        return np.all(self[:, 1:] == 0, axis=1)

    def is_segregating(self):
        """Find segregating variants (where more than one allele is observed).

        Returns
        -------
        out : ndarray, bool, shape (n_variants,)
            Boolean array where elements are True if variant matches the
            condition.

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 0]],
        ...                          [[0, 0], [0, 1]],
        ...                          [[0, 2], [1, 1]],
        ...                          [[2, 2], [-1, -1]]])
        >>> ac = g.count_alleles()
        >>> ac.is_segregating()
        array([False,  True,  True, False], dtype=bool)

        """

        return self.allelism() > 1

    def is_non_segregating(self, allele=None):
        """Find non-segregating variants (where at most one allele is
        observed).

        Parameters
        ----------
        allele : int, optional
            Allele index.

        Returns
        -------
        out : ndarray, bool, shape (n_variants,)
            Boolean array where elements are True if variant matches the
            condition.

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 0]],
        ...                          [[0, 0], [0, 1]],
        ...                          [[0, 2], [1, 1]],
        ...                          [[2, 2], [-1, -1]]])
        >>> ac = g.count_alleles()
        >>> ac.is_non_segregating()
        array([ True, False, False,  True], dtype=bool)
        >>> ac.is_non_segregating(allele=2)
        array([False, False, False,  True], dtype=bool)

        """

        if allele is None:
            return self.allelism() <= 1
        else:
            return (self.allelism() == 1) & (self[:, allele] > 0)

    def is_singleton(self, allele):
        """Find variants with a single call for the given allele.

        Parameters
        ----------
        allele : int, optional
            Allele index.

        Returns
        -------
        out : ndarray, bool, shape (n_variants,)
            Boolean array where elements are True if variant matches the
            condition.

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 0]],
        ...                          [[0, 0], [0, 1]],
        ...                          [[1, 1], [1, 2]],
        ...                          [[2, 2], [-1, -1]]])
        >>> ac = g.count_alleles()
        >>> ac.is_singleton(allele=1)
        array([False,  True, False, False], dtype=bool)
        >>> ac.is_singleton(allele=2)
        array([False, False,  True, False], dtype=bool)

        """

        return self[:, allele] == 1

    def is_doubleton(self, allele):
        """Find variants with exactly two calls for the given allele.

        Parameters
        ----------
        allele : int, optional
            Allele index.

        Returns
        -------
        out : ndarray, bool, shape (n_variants,)
            Boolean array where elements are True if variant matches the
            condition.

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 0]],
        ...                          [[0, 0], [1, 1]],
        ...                          [[1, 1], [1, 2]],
        ...                          [[2, 2], [-1, -1]]])
        >>> ac = g.count_alleles()
        >>> ac.is_doubleton(allele=1)
        array([False,  True, False, False], dtype=bool)
        >>> ac.is_doubleton(allele=2)
        array([False, False, False,  True], dtype=bool)

        """

        return self[:, allele] == 2

    def is_biallelic(self):
        """Find biallelic variants.

        Returns
        -------
        out : ndarray, bool, shape (n_variants,)
            Boolean array where elements are True if variant matches the
            condition.

        """
        return self.allelism() == 2

    def is_biallelic_01(self, min_mac=None):
        """Find variants biallelic for the reference (0) and first alternate
        (1) allele.

        Parameters
        ----------
        min_mac : int, optional
            Minimum minor allele count.

        Returns
        -------
        out : ndarray, bool, shape (n_variants,)
            Boolean array where elements are True if variant matches the
            condition.

        """
        loc = self.is_biallelic() & (self.max_allele() == 1)
        if min_mac is not None:
            loc = loc & (self[:, :2].min(axis=1) >= min_mac)
        return loc

    def count_variant(self):
        return np.sum(self.is_variant())

    def count_non_variant(self):
        return np.sum(self.is_non_variant())

    def count_segregating(self):
        return np.sum(self.is_segregating())

    def count_non_segregating(self, allele=None):
        return np.sum(self.is_non_segregating(allele=allele))

    def count_singleton(self, allele=1):
        return np.sum(self.is_singleton(allele=allele))

    def count_doubleton(self, allele=1):
        return np.sum(self.is_doubleton(allele=allele))

    def map_alleles(self, mapping):
        """Transform alleles via a mapping.

        Parameters
        ----------
        mapping : ndarray, int8, shape (n_variants, max_allele)
            An array defining the allele mapping for each variant.

        Returns
        -------
        ac : AlleleCountsArray

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 0]],
        ...                          [[0, 0], [0, 1]],
        ...                          [[0, 2], [1, 1]],
        ...                          [[2, 2], [-1, -1]]])
        >>> ac = g.count_alleles()
        >>> ac
        AlleleCountsArray((4, 3), dtype=int32)
        4 0 0
        3 1 0
        1 2 1
        0 0 2
        >>> mapping = [[1, 0, 2],
        ...            [1, 0, 2],
        ...            [2, 1, 0],
        ...            [1, 2, 0]]
        >>> ac.map_alleles(mapping)
        AlleleCountsArray((4, 3), dtype=int64)
        0 4 0
        1 3 0
        1 2 1
        2 0 0

        See Also
        --------
        create_allele_mapping

        """

        mapping = asarray_ndim(mapping, 2)
        check_dim0_aligned(self, mapping)

        # setup output array
        out = np.empty_like(mapping)

        # apply transformation
        i = np.arange(self.shape[0]).reshape((-1, 1))
        out[i, mapping] = self

        return type(self)(out)

    def compress(self, condition, axis=0):
        out = self.values.compress(condition, axis=axis)
        if axis == 0:
            out = type(self)(out)
        return out

    def take(self, indices, axis=0):
        out = self.values.take(indices, axis=axis)
        if axis == 0:
            out = type(self)(out)
        return out

    def vstack(self, *others):
        """Stack arrays in sequence vertically (row-wise)."""
        out = super(AlleleCountsArray, self).vstack(*others)
        out = type(self)(out)
        return out

    def concatenate(self, *others, **kwargs):
        """Concatenate arrays."""
        out = super(AlleleCountsArray, self).concatenate(*others, **kwargs)
        axis = kwargs.get('axis', 0)
        if axis == 0:
            out = type(self)(out)
        return out
