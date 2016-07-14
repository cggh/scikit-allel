# -*- coding: utf-8 -*-
"""This module defines NumPy array classes for variant call data.

Please note, functions and command line utilities for converting variant call
data from the VCF file format into NumPy arrays and HDF5 files are available
from the `vcfnp <https://github.com/alimanfoo/vcfnp>`_ package.

"""
from __future__ import absolute_import, print_function, division


import logging
import itertools
import bisect
import collections


import numpy as np


from allel.compat import PY2
from allel.constants import DIM_PLOIDY, DIPLOID
from allel.util import ignore_invalid, asarray_ndim, check_dim0_aligned, \
    ensure_dim1_aligned
from allel.io import write_vcf, iter_gff3


__all__ = ['GenotypeArray', 'HaplotypeArray', 'AlleleCountsArray',
           'SortedIndex', 'UniqueIndex', 'SortedMultiIndex', 'VariantTable',
           'FeatureTable']


logger = logging.getLogger(__name__)
debug = logger.debug


def subset(data, sel0, sel1):

    # check inputs
    data = np.asarray(data)
    if data.ndim < 2:
        raise ValueError('data must have 2 or more dimensions')
    sel0 = asarray_ndim(sel0, 1, allow_none=True)
    sel1 = asarray_ndim(sel1, 1, allow_none=True)

    # ensure indices
    if sel0 is not None and sel0.dtype.kind == 'b':
        sel0 = np.nonzero(sel0)[0]
    if sel1 is not None and sel1.dtype.kind == 'b':
        sel1 = np.nonzero(sel1)[0]

    # ensure leading dimension indices can be broadcast correctly
    if sel0 is not None and sel1 is not None:
        sel0 = sel0[:, None]

    # deal with None arguments
    if sel0 is None:
        sel0 = slice(None)
    if sel1 is None:
        sel1 = slice(None)

    return data[sel0, sel1]


class ArrayAug(np.ndarray):

    def __repr__(self):
        s = '%s(%s, dtype=%s)\n' % (type(self).__name__, self.shape,
                                    self.dtype)
        s += str(self)
        return s

    def hstack(self, *others):
        """Stack arrays in sequence horizontally (column wise)."""
        tup = (self,) + others
        a = np.hstack(tup)
        return type(self)(a, copy=False)

    def vstack(self, *others):
        """Stack arrays in sequence vertically (row wise)."""
        tup = (self,) + others
        a = np.vstack(tup)
        return type(self)(a, copy=False)

    def reshape(self, *args, **kwargs):
        # return as vanilla array
        a = super(ArrayAug, self).reshape(*args, **kwargs)
        return np.asarray(a)

    def flatten(self, *args, **kwargs):
        # return as vanilla array
        a = super(ArrayAug, self).flatten(*args, **kwargs)
        return np.asarray(a)

    def ravel(self, *args, **kwargs):
        # return as vanilla array
        a = super(ArrayAug, self).ravel(*args, **kwargs)
        return np.asarray(a)

    def transpose(self, *args, **kwargs):
        # return as vanilla array
        a = super(ArrayAug, self).transpose(*args, **kwargs)
        return np.asarray(a)

    @property
    def T(self):
        # return as vanilla array
        a = super(ArrayAug, self).T
        return np.asarray(a)


class RecArrayAug(np.recarray):

    def __repr__(self):
        s = '%s(%s, dtype=%s)\n' % (type(self).__name__, self.shape,
                                    self.dtype)
        s += str(self)
        return s

    def _repr_html_(self):
        return recarray_to_html_str(self)

    def display(self, limit=5, **kwargs):
        return recarray_display(self, limit=limit, **kwargs)

    def displayall(self, **kwargs):
        return self.display(limit=None, **kwargs)

    @classmethod
    def from_hdf5_group(cls, *args, **kwargs):
        a = recarray_from_hdf5_group(*args, **kwargs)
        return cls(a, copy=False)

    def to_hdf5_group(self, parent, name, **kwargs):
        return recarray_to_hdf5_group(self, parent, name, **kwargs)

    def eval(self, expression, vm='python'):
        """Evaluate an expression against the table columns.

        Parameters
        ----------
        expression : string
            Expression to evaluate.
        vm : {'numexpr', 'python'}
            Virtual machine to use.

        Returns
        -------
        result : ndarray

        """

        if vm == 'numexpr':
            import numexpr as ne
            return ne.evaluate(expression, local_dict=self)
        else:
            if PY2:
                # locals must be a mapping
                m = {k: self[k] for k in self.dtype.names}
            else:
                m = self
            return eval(expression, dict(), m)

    def query(self, expression, vm='python'):
        """Evaluate expression and then use it to extract rows from the table.

        Parameters
        ----------
        expression : string
            Expression to evaluate.
        vm : {'numexpr', 'python'}
            Virtual machine to use.

        Returns
        -------
        result : structured array

        """

        condition = self.eval(expression, vm=vm)
        return self.compress(condition)


class IntegerArray(ArrayAug):

    def astype(self, dtype, *args, **kwargs):
        x = super(IntegerArray, self).astype(dtype, *args, **kwargs)
        if x.dtype.kind not in 'iu':
            x = np.asarray(x)
        return x


class GenotypeArray(IntegerArray):
    """Array of discrete genotype calls.

    Parameters
    ----------
    data : array_like, int, shape (n_variants, n_samples, ploidy)
        Genotype data.
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

    Genotype calls for a single variant at all samples can be obtained
    by indexing the first dimension, e.g.::

        >>> g[1]
        array([[0, 1],
               [1, 1]], dtype=int8)

    Genotype calls for a single sample at all variants can be obtained
    by indexing the second dimension, e.g.::

        >>> g[:, 1]
        array([[ 0,  1],
               [ 1,  1],
               [-1, -1]], dtype=int8)

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

    """

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
            raise ValueError('use HaplotypeArray for haploid calls')

    def __new__(cls, data, **kwargs):
        kwargs.setdefault('copy', False)
        obj = np.array(data, **kwargs)
        cls._check_input_data(obj)
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

    # noinspection PyUnusedLocal
    def __array_wrap__(self, out_arr, context=None):
        # don't wrap results of any ufuncs
        return np.asarray(out_arr)

    def __getslice__(self, *args, **kwargs):
        s = np.ndarray.__getslice__(self, *args, **kwargs)
        if hasattr(s, 'ndim'):
            if s.ndim == 3 and self.shape[2] == s.shape[2]:
                # dimensionality and ploidy preserved
                if hasattr(self, 'mask') and self.mask is not None:
                    # attempt to slice mask
                    m = self.mask.__getslice__(*args)
                    s.mask = m
                return s
            elif s.ndim > 0:
                return np.asarray(s)
        return s

    def __getitem__(self, *args, **kwargs):
        s = np.ndarray.__getitem__(self, *args, **kwargs)
        if hasattr(s, 'ndim'):
            if s.ndim == 3 and self.shape[2] == s.shape[2]:
                # dimensionality and ploidy preserved
                if hasattr(self, 'mask') and self.mask is not None:
                    # attempt to slice mask
                    m = self.mask.__getitem__(*args)
                    s.mask = m
                return s
            elif s.ndim > 0:
                return np.asarray(s)
        return s

    def to_html_str(self, limit=5, caption=None, cols=None):
        import petl as etl
        n, m, _ = self.shape

        # choose how many variants to display
        limit = min(n, limit)

        # choose which columns to display
        if cols is None:
            if m <= 10:
                # display all
                cidx = list(range(m))
            else:
                # display subset
                cidx = [0, 1, 2, 3, 4, m-5, m-4, m-3, m-2, m-1]
        else:
            cidx = cols

        # prepare data for display
        gt = self[:limit+1][:, cidx].to_gt()
        if not PY2:
            gt = [[str(v, 'ascii') for v in row] for row in gt]

        # prepare table
        tbl = (
            etl
            .wrap(gt)
            .pushheader(cidx)
            .addrownumbers(start=0)
            .rename('row', '')
        )

        if cols is None and m > 10:
            # insert a spacer column
            tbl = tbl.addcolumn('...', ['...'] * limit, index=6)

        # construct caption
        if caption is None:
            caption = 'GenotypeArray(%s, dtype=%s)' % (self.shape, self.dtype)
        caption = caption.replace('<', '&lt;')
        caption = caption.replace('\n', '<br/>')

        # build HTML
        # noinspection PyProtectedMember
        html = etl.util.vis._display_html(tbl,
                                          caption=caption,
                                          limit=limit,
                                          td_styles={'': 'font-weight: bold'},
                                          index_header=False)
        return html

    def _repr_html_(self):
        return self.to_html_str()

    @property
    def n_variants(self):
        """Number of variants (length of first array dimension)."""
        return self.shape[0]

    @property
    def n_samples(self):
        """Number of samples (length of second array dimension)."""
        return self.shape[1]

    @property
    def ploidy(self):
        """Sample ploidy (length of third array dimension)."""
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
        """A boolean mask associated with this genotype array, indicating
        genotype calls that should be filtered (i.e., excluded) from
        genotype and allele counting operations.

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 1], [1, 1]],
        ...                          [[0, 2], [-1, -1]]], dtype='i1')
        >>> g.count_called()
        5
        >>> g.count_alleles()
        AlleleCountsArray((3, 3), dtype=int32)
        [[3 1 0]
         [1 3 0]
         [1 0 1]]
        >>> mask = [[True, False], [False, True], [False, False]]
        >>> g.mask = mask
        >>> g.count_called()
        3
        >>> g.count_alleles()
        AlleleCountsArray((3, 3), dtype=int32)
        [[1 1 0]
         [1 1 0]
         [1 0 1]]

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
        if hasattr(self, '_mask'):
            return self._mask
        else:
            return None

    @mask.setter
    def mask(self, mask):

        # check input
        if mask is not None:
            mask = asarray_ndim(mask, 2)
            if mask.shape != self.shape[:2]:
                raise ValueError('mask has incorrect shape')

        # store
        self._mask = mask

    def fill_masked(self, value=-1, mask=None, copy=True):
        """Fill masked genotype calls with a given value.

        Parameters
        ----------
        value : int, optional
            The fill value.
        mask : array_like, bool, shape (n_variants, n_samples), optional
            A boolean array where True elements indicate genotype calls to be
            filled. If not provided, value of the `mask` property will be used.
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
        >>> g.fill_masked()
        GenotypeArray((3, 2, 2), dtype=int8)
        [[[-1 -1]
          [ 0  1]]
         [[ 0  1]
          [-1 -1]]
         [[ 0  2]
          [-1 -1]]]

        """

        # determine mask
        if mask is None and self.mask is None:
            raise ValueError('no mask found')
        mask = mask if mask is not None else self.mask
        mask = asarray_ndim(mask, 2)
        if mask.shape != self.shape[:2]:
            raise ValueError('mask has incorrect shape')

        # decide whether to copy
        if copy:
            a = self.copy()
        else:
            a = self

        # apply the mask
        a[mask, ...] = value

        return a.view(GenotypeArray)

    def subset(self, sel0=None, sel1=None):
        """Make a sub-selection of variants and samples.

        Parameters
        ----------
        sel0 : array_like
            Boolean array or list of indices selecting variants.
        sel0 : array_like
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
        [[[0 0]
          [1 1]]
         [[0 1]
          [1 2]]]

        See Also
        --------
        numpy.take, numpy.compress

        """

        data = subset(self, sel0, sel1)
        g = GenotypeArray(data, copy=False)
        if hasattr(self, 'mask') and self.mask is not None:
            m = subset(self.mask, sel0, sel1)
            g.mask = m
        return g

    # noinspection PyUnusedLocal
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
        import numexpr as ne

        # special case diploid
        if self.shape[2] == DIPLOID:
            allele1 = self[..., 0]  # noqa
            allele2 = self[..., 1]  # noqa
            expr = '(allele1 >= 0) & (allele2 >= 0)'
            out = ne.evaluate(expr)

        # general ploidy case
        else:
            out = np.all(self >= 0, axis=2)

        # handle mask
        if hasattr(self, 'mask') and self.mask is not None:
            out &= ~self.mask

        return out

    # noinspection PyUnusedLocal
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
        import numexpr as ne

        # special case diploid
        if self.shape[2] == DIPLOID:
            allele1 = self[..., 0]  # noqa
            allele2 = self[..., 1]  # noqa
            # call is missing if either allele is missing
            ex = '(allele1 < 0) | (allele2 < 0)'
            out = ne.evaluate(ex)

        # general ploidy case
        else:
            # call is missing if any allele is missing
            out = np.any(self < 0, axis=2)

        # handle mask
        if hasattr(self, 'mask') and self.mask is not None:
            out |= self.mask

        return out

    # noinspection PyUnusedLocal
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
        import numexpr as ne

        # special case diploid
        if self.shape[2] == DIPLOID:
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
                out = np.all(ne.evaluate(ex), axis=2)
            else:
                out = np.all(self == allele, axis=2)

        # handle mask
        if hasattr(self, 'mask') and self.mask is not None:
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

    # noinspection PyUnusedLocal
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
        import numexpr as ne

        # special case diploid
        if self.shape[2] == DIPLOID:
            allele1 = self[..., 0]  # noqa
            allele2 = self[..., 1]  # noqa
            ex = '(allele1 > 0) & (allele1  == allele2)'
            out = ne.evaluate(ex)

        # general ploidy case
        else:
            allele1 = self[..., 0, None]  # noqa
            other_alleles = self[..., 1:]  # noqa
            ex = '(allele1 > 0) & (allele1 == other_alleles)'
            out = np.all(ne.evaluate(ex), axis=2)

        # handle mask
        if hasattr(self, 'mask') and self.mask is not None:
            out &= ~self.mask

        return out

    # noinspection PyUnusedLocal
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
        import numexpr as ne

        # special case diploid
        if self.shape[2] == DIPLOID:
            allele1 = self[..., 0]  # noqa
            allele2 = self[..., 1]  # noqa
            ex = '(allele1 >= 0) & (allele2  >= 0) & (allele1 != allele2)'
            if allele is not None:
                ex += ' & ((allele1 == {0}) | (allele2 == {0}))' \
                    .format(allele)
            out = ne.evaluate(ex)

        # general ploidy case
        else:
            allele1 = self[..., 0, None]  # noqa
            other_alleles = self[..., 1:]  # noqa
            out = np.all(self >= 0, axis=2) \
                & np.any(allele1 != other_alleles, axis=2)
            if allele is not None:
                out &= np.any(self == allele, axis=2)

        # handle mask
        if hasattr(self, 'mask') and self.mask is not None:
            out &= ~self.mask

        return out

    # noinspection PyUnusedLocal
    def is_call(self, call):
        """Find genotypes with a given call.

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
        import numexpr as ne

        # special case diploid
        if self.shape[2] == DIPLOID:
            if not len(call) == DIPLOID:
                raise ValueError('invalid call: %r', call)
            allele1 = self[..., 0]  # noqa
            allele2 = self[..., 1]  # noqa
            ex = '(allele1 == {0}) & (allele2  == {1})'.format(*call)
            out = ne.evaluate(ex)

        # general ploidy case
        else:
            if not len(call) == self.shape[2]:
                raise ValueError('invalid call: %r', call)
            call = np.asarray(call)[None, None, :]
            out = np.all(self == call, axis=2)

        # handle mask
        if hasattr(self, 'mask') and self.mask is not None:
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
        [[3 1 0]
         [1 2 1]
         [0 0 2]]
        >>> g.count_alleles(max_allele=1)
        AlleleCountsArray((3, 2), dtype=int32)
        [[3 1]
         [1 2]
         [0 0]]

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

        if self.dtype.type == np.int8:
            # use optimisations
            from allel.opt.model import genotype_int8_count_alleles, \
                genotype_int8_count_alleles_masked, \
                genotype_int8_count_alleles_subpop, \
                genotype_int8_count_alleles_subpop_masked

            if subpop is None:
                if hasattr(self, 'mask') and self.mask is not None:
                    ac = genotype_int8_count_alleles_masked(
                        self, self.mask.view(dtype='u1'), max_allele
                    )
                else:
                    ac = genotype_int8_count_alleles(self, max_allele)

            else:
                if hasattr(self, 'mask') and self.mask is not None:
                    ac = genotype_int8_count_alleles_subpop_masked(
                        self, self.mask.view(dtype='u1'), max_allele, subpop
                    )
                else:
                    ac = genotype_int8_count_alleles_subpop(
                        self, max_allele, subpop
                    )

        else:
            # set up output array
            ac = np.zeros((self.shape[0], max_allele + 1), dtype='i4')

            # extract subpop
            if subpop is not None:
                g = self[:, subpop]
            else:
                g = self

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

    def to_haplotypes(self, copy=False):
        """Reshape a genotype array to view it as haplotypes by
        dropping the ploidy dimension.

        Returns
        -------
        h : HaplotypeArray, shape (n_variants, n_samples * ploidy)
            Haplotype array.
        copy : bool, optional
            If True, make a copy of the data.

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
        [[ 0  0  0  1]
         [ 0  1  1  1]
         [ 0  2 -1 -1]]

        """

        # reshape, preserving size of variants dimension
        newshape = (self.shape[0], -1)
        data = np.reshape(self, newshape)
        h = HaplotypeArray(data, copy=copy)
        return h

    def to_n_ref(self, fill=0, dtype='i1'):
        """Transform each genotype call into the number of
        reference alleles.

        Parameters
        ----------
        fill : int, optional
            Use this value to represent missing calls.

        Returns
        -------
        out : ndarray, int, shape (n_variants, n_samples)
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
        out = np.empty((self.shape[0], self.shape[1]), dtype=dtype)
        np.sum(self == 0, axis=2, out=out)

        # fill missing calls
        if fill != 0:
            m = self.is_missing()
            out[m] = fill

        # handle mask
        if hasattr(self, 'mask') and self.mask is not None:
            out[self.mask] = fill

        return out

    def to_n_alt(self, fill=0, dtype='i1'):
        """Transform each genotype call into the number of
        non-reference alleles.

        Parameters
        ----------
        fill : int, optional
            Use this value to represent missing calls.

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
        out = np.empty((self.shape[0], self.shape[1]), dtype=dtype)
        np.sum(self > 0, axis=2, out=out)

        # fill missing calls
        if fill != 0:
            m = self.is_missing()
            out[m] = fill

        # handle mask
        if hasattr(self, 'mask') and self.mask is not None:
            out[self.mask] = fill

        return out

    def to_allele_counts(self, alleles=None):
        """Transform genotype calls into allele counts per call.

        Parameters
        ----------
        alleles : sequence of ints, optional
            If not None, count only the given alleles. (By default, count all
            alleles.)

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
        >>> g.to_allele_counts(alleles=(0, 1))
        array([[[2, 0],
                [1, 1]],
               [[1, 0],
                [0, 2]],
               [[0, 0],
                [0, 0]]], dtype=uint8)

        """

        # determine alleles to count
        if alleles is None:
            m = self.max()
            alleles = list(range(m+1))

        # set up output array
        outshape = (self.shape[0], self.shape[1], len(alleles))
        out = np.zeros(outshape, dtype='u1')

        for i, allele in enumerate(alleles):
            # count alleles along ploidy dimension
            allele_match = self == allele
            if hasattr(self, 'mask') and self.mask is not None:
                allele_match &= ~self.mask[:, :, None]
            np.sum(allele_match, axis=2, out=out[..., i])

        return out

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

        if self.shape[2] != 2:
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

        from allel.opt.model import genotype_pack_diploid

        # ensure int8 dtype
        if self.dtype.type == np.int8:
            data = self
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
        [[[ 0  0]
          [ 0  1]]
         [[ 0  2]
          [ 1  1]]
         [[ 2  2]
          [-1 -1]]]

        """

        # check arguments
        packed = np.asarray(packed)
        if packed.ndim != 2:
            raise ValueError('packed array must have 2 dimensions')
        if packed.dtype != np.uint8:
            packed = packed.astype(np.uint8)

        from allel.opt.model import genotype_unpack_diploid
        data = genotype_unpack_diploid(packed)
        return GenotypeArray(data)

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
        [[[ 0  0]
          [ 0  0]]
         [[ 0  1]
          [ 0  1]]
         [[ 1  1]
          [ 0  0]]
         [[ 0  0]
          [-1 -1]]]

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
        [[ 0  1]
         [ 0  1]
         [ 1  1]
         [ 2 -1]]
        >>> g = allel.GenotypeArray([[[0, 0, 0], [0, 0, 1]],
        ...                          [[0, 1, 1], [1, 1, 1]],
        ...                          [[0, 1, 2], [-1, -1, -1]]])
        >>> g.haploidify_samples()
        HaplotypeArray((3, 2), dtype=int64)
        [[ 0  0]
         [ 1  1]
         [ 2 -1]]

        """

        # N.B., this implementation is obscure and uses more memory that
        # necessary, TODO review

        # define the range of possible indices, e.g., diploid => (0, 1)
        index_range = np.arange(0, self.shape[2], dtype='u1')

        # create a random index for each genotype call
        indices = np.random.choice(index_range,
                                   size=(self.shape[0] * self.shape[1]),
                                   replace=True)

        # reshape genotype data so it's suitable for passing to np.choose
        # by merging the variants and samples dimensions
        choices = self.reshape(-1, self.shape[2]).T

        # now use random indices to haploidify
        data = np.choose(indices, choices)

        # reshape the haploidified data to restore the variants and samples
        # dimensions
        data = data.reshape((self.shape[0], self.shape[1]))

        # view as haplotype array
        h = HaplotypeArray(data, copy=False)

        return h

    # noinspection PyUnusedLocal
    def to_gt(self, phased=False, max_allele=None):
        """Convert genotype calls to VCF-style string representation.

        Parameters
        ----------
        phased : bool, optional
            Determines separator.
        max_allele : int, optional
            Manually specify max allele index.

        Returns
        -------
        gt : ndarray, string, shape (n_variants, n_samples)

        Notes
        -----
        If a mask has been set, it is ignored by this function.

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
        >>> g.to_gt(phased=True)
        chararray([[b'0|0', b'0|1'],
               [b'0|2', b'1|1'],
               [b'1|2', b'2|1'],
               [b'2|2', b'.|.']],
              dtype='|S3')

        """

        # determine separator
        if phased:
            sep = b'|'  # noqa
        else:
            sep = b'/'  # noqa

        # how many characters needed?
        if max_allele is None:
            max_allele = np.max(self)
        if max_allele <= 0:
            max_allele = 1
        nchar = int(np.floor(np.log10(max_allele))) + 1

        # convert to string
        a = self.astype((np.string_, nchar)).view(np.chararray)

        # recode missing alleles
        a[a.startswith(b'-')] = b'.'

        # join via separator
        expr = "a[..., 0]"
        for i in range(1, self.shape[2]):
            expr += " + sep + a[..., %s]" % i
        gt = eval(expr)

        return gt

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
        [[[ 1  1]
          [ 1  2]]
         [[ 2  1]
          [ 0  0]]
         [[ 1  0]
          [ 0  1]]
         [[ 1  1]
          [-1 -1]]]

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


class HaplotypeArray(IntegerArray):
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
        [[[ 0  0]
          [ 0  1]]
         [[ 0  1]
          [ 1  1]]
         [[ 0  2]
          [-1 -1]]]

    """

    @staticmethod
    def _check_input_data(obj):

        # check dtype
        if obj.dtype.kind not in 'ui':
            raise TypeError('integer dtype required')

        # check dimensionality
        if obj.ndim != 2:
            raise TypeError('array with 2 dimensions required')

    def __new__(cls, data, **kwargs):
        kwargs.setdefault('copy', False)
        obj = np.array(data, **kwargs)
        cls._check_input_data(obj)
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

    # noinspection PyUnusedLocal
    def __array_wrap__(self, out_arr, context=None):
        # don't wrap results of any ufuncs
        return np.asarray(out_arr)

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

    def to_html_str(self, limit=5, caption=None, cols=None):
        import petl as etl
        n, m = self.shape

        # choose how many variants to display
        limit = min(n, limit)

        # choose which columns to display
        if cols is None:
            if m <= 10:
                # display all
                cidx = list(range(m))
            else:
                # display subset
                cidx = [0, 1, 2, 3, 4, m-5, m-4, m-3, m-2, m-1]
        else:
            cidx = cols

        # prepare data for display
        h = self[:limit+1][:, cidx]

        # prepare table
        tbl = (
            etl
            .wrap(h)
            .pushheader(cidx)
            .addrownumbers(start=0)
            .rename('row', '')
        )

        if cols is None and m > 10:
            # insert a spacer column
            tbl = tbl.addcolumn('...', ['...'] * limit, index=6)

        # construct caption
        if caption is None:
            caption = 'HaplotypeArray(%s, dtype=%s)' % (self.shape, self.dtype)
        caption = caption.replace('<', '&lt;')
        caption = caption.replace('\n', '<br/>')

        # build HTML
        # noinspection PyProtectedMember
        html = etl.util.vis._display_html(tbl,
                                          caption=caption,
                                          limit=limit,
                                          td_styles={'': 'font-weight: bold'},
                                          index_header=False)
        return html

    def _repr_html_(self):
        return self.to_html_str()

    @property
    def n_variants(self):
        """Number of variants (length of first dimension)."""
        return self.shape[0]

    @property
    def n_haplotypes(self):
        """Number of haplotypes (length of second dimension)."""
        return self.shape[1]

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
        numpy.take, numpy.compress

        """

        return HaplotypeArray(subset(self, sel0, sel1), copy=False)

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
        [[[ 0  0]
          [ 0  1]]
         [[ 0  1]
          [ 1  1]]
         [[ 0  2]
          [-1 -1]]]

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
        [[ 0  0  0  0]
         [ 0  1  0  1]
         [ 1  1  0  0]
         [ 0  0 -1 -1]]

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
        [[3 1 0]
         [1 3 0]
         [1 0 1]]

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

        if self.dtype.type == np.int8:
            # use optimisations
            from allel.opt.model import haplotype_int8_count_alleles, \
                haplotype_int8_count_alleles_subpop
            if subpop is None:
                ac = haplotype_int8_count_alleles(self, max_allele)

            else:
                ac = haplotype_int8_count_alleles_subpop(self, max_allele,
                                                         subpop)

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
        [[ 1  1  1  2]
         [ 2  0  0  0]
         [ 2  0 -1 -1]]

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

        if self.dtype.type == np.int8:
            # use optimisation
            mapping = np.asarray(mapping, dtype='i1')
            from allel.opt.model import haplotype_int8_map_alleles
            data = haplotype_int8_map_alleles(self, mapping, copy=copy)

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
        counts = sorted(collections.Counter(k).values(), reverse=True)

        return np.asarray(counts)

    def distinct_frequencies(self):
        """Return frequencies for each distinct haplotype."""

        c = self.distinct_counts()
        n = self.shape[1]
        return c / n


class AlleleCountsArray(IntegerArray):
    """Array of allele counts.

    Parameters
    ----------
    data : array_like, int, shape (n_variants, n_alleles)
        Allele counts data.
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
        [[3 1 0]
         [1 3 0]
         [1 0 1]]
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

    @staticmethod
    def _check_input_data(obj):

        # check dtype
        if obj.dtype.kind not in 'ui':
            raise TypeError('integer dtype required')

        # check dimensionality
        if obj.ndim != 2:
            raise TypeError('array with 2 dimensions required')

    def __new__(cls, data, **kwargs):
        kwargs.setdefault('copy', False)
        obj = np.array(data, **kwargs)
        cls._check_input_data(obj)
        obj = obj.view(cls)
        return obj

    def __array_finalize__(self, obj):

        # called after constructor
        if obj is None:
            return

        # called after slice (new-from-template)
        if isinstance(obj, AlleleCountsArray):
            return

        # called after view
        AlleleCountsArray._check_input_data(obj)

    # noinspection PyUnusedLocal
    def __array_wrap__(self, out_arr, context=None):
        # don't wrap results of any ufuncs
        return np.asarray(out_arr)

    def __getslice__(self, *args, **kwargs):
        s = np.ndarray.__getslice__(self, *args, **kwargs)
        if hasattr(s, 'ndim') and s.ndim > 0:
            if s.ndim == 2 and s.shape[1] == self.shape[1]:
                # wrap only if number of alleles is preserved
                return AlleleCountsArray(s, copy=False)
            return np.asarray(s)
        return s

    def __getitem__(self, *args, **kwargs):
        s = np.ndarray.__getitem__(self, *args, **kwargs)
        if hasattr(s, 'ndim') and s.ndim > 0:
            if s.ndim == 2 and s.shape[1] == self.shape[1]:
                # wrap only if number of alleles is preserved
                return AlleleCountsArray(s, copy=False)
            return np.asarray(s)
        return s

    def __add__(self, other):
        ret = super(AlleleCountsArray, self).__add__(other)
        if hasattr(other, 'shape') and other.shape == self.shape:
            ret = AlleleCountsArray(ret)
        return ret

    def __sub__(self, other):
        ret = super(AlleleCountsArray, self).__sub__(other)
        if hasattr(other, 'shape') and other.shape == self.shape:
            ret = AlleleCountsArray(ret)
        return ret

    def to_html_str(self, limit=5, caption=None):
        import petl as etl
        ac = self[:limit+1]
        tbl = (
            etl
            .wrap(ac)
            .pushheader(list(range(ac.shape[1])))
            .addrownumbers(start=0)
            .rename('row', '')
        )

        if caption is None:
            caption = 'AlleleCountsArray(%s, dtype=%s)' \
                      % (self.shape, self.dtype)
        caption = caption.replace('<', '&lt;')
        caption = caption.replace('\n', '<br/>')

        # noinspection PyProtectedMember
        html = etl.util.vis._display_html(tbl,
                                          caption=caption,
                                          limit=limit,
                                          td_styles={'': 'font-weight: bold'},
                                          index_header=False)
        return html

    def _repr_html_(self):
        return self.to_html_str()

    @property
    def n_variants(self):
        """Number of variants (length of first array dimension)."""
        return self.shape[0]

    @property
    def n_alleles(self):
        """Number of alleles (length of second array dimension)."""
        return self.shape[1]

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
        loc = (self.is_biallelic() &
               (self.max_allele() == 1))
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
        [[4 0 0]
         [3 1 0]
         [1 2 1]
         [0 0 2]]
        >>> mapping = [[1, 0, 2],
        ...            [1, 0, 2],
        ...            [2, 1, 0],
        ...            [1, 2, 0]]
        >>> ac.map_alleles(mapping)
        AlleleCountsArray((4, 3), dtype=int64)
        [[0 4 0]
         [1 3 0]
         [1 2 1]
         [2 0 0]]

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

        return AlleleCountsArray(out)


class SortedIndex(ArrayAug):
    """Index of sorted values, e.g., positions from a single chromosome or
    contig.

    Parameters
    ----------
    data : array_like
        Values in ascending order.
    **kwargs : keyword arguments
        All keyword arguments are passed through to :func:`numpy.array`.

    Notes
    -----
    Values must be given in ascending order, although duplicate values
    may be present (i.e., values must be monotonically increasing).

    Examples
    --------

    >>> import allel
    >>> idx = allel.SortedIndex([2, 5, 14, 15, 42, 42, 77], dtype='i4')
    >>> idx.dtype
    dtype('int32')
    >>> idx.ndim
    1
    >>> idx.shape
    (7,)
    >>> idx.is_unique
    False

    """

    @staticmethod
    def _check_input_data(obj):

        # check dimensionality
        if obj.ndim != 1:
            raise TypeError('array with 1 dimension required')

        # check sorted ascending
        if np.any(obj[:-1] > obj[1:]):
            raise ValueError('array is not monotonically increasing')

    def __new__(cls, data, **kwargs):
        kwargs.setdefault('copy', False)
        obj = np.array(data, **kwargs)
        cls._check_input_data(obj)
        obj = obj.view(cls)
        return obj

    def __array_finalize__(self, obj):

        # called after constructor
        if obj is None:
            return

        # called after slice (new-from-template)
        if isinstance(obj, SortedIndex):
            return

        # called after view
        SortedIndex._check_input_data(obj)

    # noinspection PyUnusedLocal
    def __array_wrap__(self, out_arr, context=None):
        # don't wrap results of any ufuncs
        return np.asarray(out_arr)

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
    def is_unique(self):
        """True if no duplicate entries."""
        if not hasattr(self, '_is_unique'):
            self._is_unique = ~np.any(self[:-1] == self[1:])
        return self._is_unique

    def locate_key(self, key):
        """Get index location for the requested key.

        Parameters
        ----------
        key : int
            Value to locate.

        Returns
        -------
        loc : int or slice
            Location of `key` (will be slice if there are duplicate entries).

        Examples
        --------

        >>> import allel
        >>> idx = allel.SortedIndex([3, 6, 6, 11])
        >>> idx.locate_key(3)
        0
        >>> idx.locate_key(11)
        3
        >>> idx.locate_key(6)
        slice(1, 3, None)
        >>> try:
        ...     idx.locate_key(2)
        ... except KeyError as e:
        ...     print(e)
        ...
        2

        """

        left = np.searchsorted(self, key, side='left')
        right = bisect.bisect_right(self, key)
        diff = right - left
        if diff == 0:
            raise KeyError(key)
        elif diff == 1:
            return left
        else:
            return slice(left, right)

    def locate_intersection(self, other):
        """Locate the intersection with another array.

        Parameters
        ----------
        other : array_like, int
            Array of values to intersect.

        Returns
        -------
        loc : ndarray, bool
            Boolean array with location of intersection.
        loc_other : ndarray, bool
            Boolean array with location in `other` of intersection.

        Examples
        --------

        >>> import allel
        >>> idx1 = allel.SortedIndex([3, 6, 11, 20, 35])
        >>> idx2 = allel.SortedIndex([4, 6, 20, 39])
        >>> loc1, loc2 = idx1.locate_intersection(idx2)
        >>> loc1
        array([False,  True, False,  True, False], dtype=bool)
        >>> loc2
        array([False,  True,  True, False], dtype=bool)
        >>> idx1[loc1]
        SortedIndex((2,), dtype=int64)
        [ 6 20]
        >>> idx2[loc2]
        SortedIndex((2,), dtype=int64)
        [ 6 20]

        """

        # check inputs
        other = SortedIndex(other, copy=False)

        # find intersection
        assume_unique = self.is_unique and other.is_unique
        loc = np.in1d(self, other, assume_unique=assume_unique)
        loc_other = np.in1d(other, self, assume_unique=assume_unique)

        return loc, loc_other

    def locate_keys(self, keys, strict=True):
        """Get index locations for the requested keys.

        Parameters
        ----------
        keys : array_like, int
            Array of keys to locate.
        strict : bool, optional
            If True, raise KeyError if any keys are not found in the index.

        Returns
        -------
        loc : ndarray, bool
            Boolean array with location of values.

        Examples
        --------

        >>> import allel
        >>> idx1 = allel.SortedIndex([3, 6, 11, 20, 35])
        >>> idx2 = allel.SortedIndex([4, 6, 20, 39])
        >>> loc = idx1.locate_keys(idx2, strict=False)
        >>> loc
        array([False,  True, False,  True, False], dtype=bool)
        >>> idx1[loc]
        SortedIndex((2,), dtype=int64)
        [ 6 20]

        """

        # check inputs
        keys = SortedIndex(keys, copy=False)

        # find intersection
        loc, found = self.locate_intersection(keys)

        if strict and np.any(~found):
            raise KeyError(keys[~found])

        return loc

    def intersect(self, other):
        """Intersect with `other` sorted index.

        Parameters
        ----------
        other : array_like, int
            Array of values to intersect with.

        Returns
        -------
        out : SortedIndex
            Values in common.

        Examples
        --------

        >>> import allel
        >>> idx1 = allel.SortedIndex([3, 6, 11, 20, 35])
        >>> idx2 = allel.SortedIndex([4, 6, 20, 39])
        >>> idx1.intersect(idx2)
        SortedIndex((2,), dtype=int64)
        [ 6 20]

        """

        loc = self.locate_keys(other, strict=False)
        return np.compress(loc, self)

    def locate_range(self, start=None, stop=None):
        """Locate slice of index containing all entries within `start` and
        `stop` values **inclusive**.

        Parameters
        ----------
        start : int, optional
            Start value.
        stop : int, optional
            Stop value.

        Returns
        -------
        loc : slice
            Slice object.

        Examples
        --------

        >>> import allel
        >>> idx = allel.SortedIndex([3, 6, 11, 20, 35])
        >>> loc = idx.locate_range(4, 32)
        >>> loc
        slice(1, 4, None)
        >>> idx[loc]
        SortedIndex((3,), dtype=int64)
        [ 6 11 20]

        """

        # locate start and stop indices
        if start is None:
            start_index = 0
        else:
            start_index = bisect.bisect_left(self, start)
        if stop is None:
            stop_index = len(self)
        else:
            stop_index = bisect.bisect_right(self, stop)

        if stop_index - start_index == 0:
            raise KeyError(start, stop)

        loc = slice(start_index, stop_index)
        return loc

    def intersect_range(self, start=None, stop=None):
        """Intersect with range defined by `start` and `stop` values
        **inclusive**.

        Parameters
        ----------
        start : int, optional
            Start value.
        stop : int, optional
            Stop value.

        Returns
        -------
        idx : SortedIndex

        Examples
        --------

        >>> import allel
        >>> idx = allel.SortedIndex([3, 6, 11, 20, 35])
        >>> idx.intersect_range(4, 32)
        SortedIndex((3,), dtype=int64)
        [ 6 11 20]

        """

        try:
            loc = self.locate_range(start=start, stop=stop)
        except KeyError:
            return self[0:0]
        else:
            return self[loc]

    def locate_intersection_ranges(self, starts, stops):
        """Locate the intersection with a set of ranges.

        Parameters
        ----------
        starts : array_like, int
            Range start values.
        stops : array_like, int
            Range stop values.

        Returns
        -------
        loc : ndarray, bool
            Boolean array with location of entries found.
        loc_ranges : ndarray, bool
            Boolean array with location of ranges containing one or more
            entries.

        Examples
        --------

        >>> import allel
        >>> import numpy as np
        >>> idx = allel.SortedIndex([3, 6, 11, 20, 35])
        >>> ranges = np.array([[0, 2], [6, 17], [12, 15], [31, 35],
        ...                    [100, 120]])
        >>> starts = ranges[:, 0]
        >>> stops = ranges[:, 1]
        >>> loc, loc_ranges = idx.locate_intersection_ranges(starts, stops)
        >>> loc
        array([False,  True,  True, False,  True], dtype=bool)
        >>> loc_ranges
        array([False,  True, False,  True, False], dtype=bool)
        >>> idx[loc]
        SortedIndex((3,), dtype=int64)
        [ 6 11 35]
        >>> ranges[loc_ranges]
        array([[ 6, 17],
               [31, 35]])

        """

        # check inputs
        starts = asarray_ndim(starts, 1)
        stops = asarray_ndim(stops, 1)
        check_dim0_aligned(starts, stops)

        # find indices of start and stop values in idx
        start_indices = np.searchsorted(self, starts)
        stop_indices = np.searchsorted(self, stops, side='right')

        # find intervals overlapping at least one value
        loc_ranges = start_indices < stop_indices

        # find values within at least one interval
        loc = np.zeros(self.shape, dtype=np.bool)
        for i, j in zip(start_indices[loc_ranges], stop_indices[loc_ranges]):
            loc[i:j] = True

        return loc, loc_ranges

    def locate_ranges(self, starts, stops, strict=True):
        """Locate items within the given ranges.

        Parameters
        ----------
        starts : array_like, int
            Range start values.
        stops : array_like, int
            Range stop values.
        strict : bool, optional
            If True, raise KeyError if any ranges contain no entries.

        Returns
        -------
        loc : ndarray, bool
            Boolean array with location of entries found.

        Examples
        --------

        >>> import allel
        >>> import numpy as np
        >>> idx = allel.SortedIndex([3, 6, 11, 20, 35])
        >>> ranges = np.array([[0, 2], [6, 17], [12, 15], [31, 35],
        ...                    [100, 120]])
        >>> starts = ranges[:, 0]
        >>> stops = ranges[:, 1]
        >>> loc = idx.locate_ranges(starts, stops, strict=False)
        >>> loc
        array([False,  True,  True, False,  True], dtype=bool)
        >>> idx[loc]
        SortedIndex((3,), dtype=int64)
        [ 6 11 35]

        """

        loc, found = self.locate_intersection_ranges(starts, stops)

        if strict and np.any(~found):
            raise KeyError(starts[~found], stops[~found])

        return loc

    def intersect_ranges(self, starts, stops):
        """Intersect with a set of ranges.

        Parameters
        ----------
        starts : array_like, int
            Range start values.
        stops : array_like, int
            Range stop values.

        Returns
        -------
        idx : SortedIndex

        Examples
        --------

        >>> import allel
        >>> import numpy as np
        >>> idx = allel.SortedIndex([3, 6, 11, 20, 35])
        >>> ranges = np.array([[0, 2], [6, 17], [12, 15], [31, 35],
        ...                    [100, 120]])
        >>> starts = ranges[:, 0]
        >>> stops = ranges[:, 1]
        >>> idx.intersect_ranges(starts, stops)
        SortedIndex((3,), dtype=int64)
        [ 6 11 35]

        """

        loc = self.locate_ranges(starts, stops, strict=False)
        return np.compress(loc, self)


class UniqueIndex(ArrayAug):
    """Array of unique values (e.g., variant or sample identifiers).

    Parameters
    ----------
    data : array_like
        Values.
    **kwargs : keyword arguments
        All keyword arguments are passed through to :func:`numpy.array`.

    Notes
    -----
    This class represents an arbitrary set of unique values, e.g., sample or
    variant identifiers.

    There is no need for values to be sorted. However, all values must be
    unique within the array, and must be hashable objects.

    Examples
    --------

    >>> import allel
    >>> idx = allel.UniqueIndex(['A', 'C', 'B', 'F'])
    >>> idx.dtype
    dtype('<U1')
    >>> idx.ndim
    1
    >>> idx.shape
    (4,)

    """

    @staticmethod
    def _check_input_data(obj):

        # check dimensionality
        if obj.ndim != 1:
            raise TypeError('array with 1 dimension required')

        # check unique
        # noinspection PyTupleAssignmentBalance
        _, counts = np.unique(obj, return_counts=True)
        if np.any(counts > 1):
            raise ValueError('values are not unique')

    def __new__(cls, data, **kwargs):
        kwargs.setdefault('copy', False)
        obj = np.array(data, **kwargs)
        cls._check_input_data(obj)
        obj = obj.view(cls)
        lookup = {v: i for i, v in enumerate(obj)}
        obj.lookup = lookup
        return obj

    def __array_finalize__(self, obj):

        # called after constructor
        if obj is None:
            return

        # called after slice (new-from-template)
        if isinstance(obj, UniqueIndex):
            return

        # called after view
        UniqueIndex._check_input_data(obj)

    # noinspection PyUnusedLocal
    def __array_wrap__(self, out_arr, context=None):
        # don't wrap results of any ufuncs
        return np.asarray(out_arr)

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

    def locate_key(self, key):
        """Get index location for the requested key.

        Parameters
        ----------
        key : object
            Key to locate.

        Returns
        -------
        loc : int
            Location of `key`.

        Examples
        --------

        >>> import allel
        >>> idx = allel.UniqueIndex(['A', 'C', 'B', 'F'])
        >>> idx.locate_key('A')
        0
        >>> idx.locate_key('B')
        2
        >>> try:
        ...     idx.locate_key('X')
        ... except KeyError as e:
        ...     print(e)
        ...
        'X'

        """

        return self.lookup[key]

    def locate_intersection(self, other):
        """Locate the intersection with another array.

        Parameters
        ----------
        other : array_like
            Array to intersect.

        Returns
        -------
        loc : ndarray, bool
            Boolean array with location of intersection.
        loc_other : ndarray, bool
            Boolean array with location in `other` of intersection.

        Examples
        --------

        >>> import allel
        >>> idx1 = allel.UniqueIndex(['A', 'C', 'B', 'F'])
        >>> idx2 = allel.UniqueIndex(['X', 'F', 'G', 'C', 'Z'])
        >>> loc1, loc2 = idx1.locate_intersection(idx2)
        >>> loc1
        array([False,  True, False,  True], dtype=bool)
        >>> loc2
        array([False,  True, False,  True, False], dtype=bool)
        >>> idx1[loc1]
        UniqueIndex((2,), dtype=<U1)
        ['C' 'F']
        >>> idx2[loc2]
        UniqueIndex((2,), dtype=<U1)
        ['F' 'C']

        """

        # check inputs
        other = UniqueIndex(other)

        # find intersection
        assume_unique = True
        loc = np.in1d(self, other, assume_unique=assume_unique)
        loc_other = np.in1d(other, self, assume_unique=assume_unique)

        return loc, loc_other

    def locate_keys(self, keys, strict=True):
        """Get index locations for the requested keys.

        Parameters
        ----------
        keys : array_like
            Array of keys to locate.
        strict : bool, optional
            If True, raise KeyError if any keys are not found in the index.

        Returns
        -------
        loc : ndarray, bool
            Boolean array with location of keys.

        Examples
        --------

        >>> import allel
        >>> idx = allel.UniqueIndex(['A', 'C', 'B', 'F'])
        >>> idx.locate_keys(['F', 'C'])
        array([False,  True, False,  True], dtype=bool)
        >>> idx.locate_keys(['X', 'F', 'G', 'C', 'Z'], strict=False)
        array([False,  True, False,  True], dtype=bool)

        """

        # check inputs
        keys = UniqueIndex(keys)

        # find intersection
        loc, found = self.locate_intersection(keys)

        if strict and np.any(~found):
            raise KeyError(keys[~found])

        return loc

    def intersect(self, other):
        """Intersect with `other`.

        Parameters
        ----------
        other : array_like
            Array to intersect.

        Returns
        -------
        out : UniqueIndex

        Examples
        --------

        >>> import allel
        >>> idx1 = allel.UniqueIndex(['A', 'C', 'B', 'F'])
        >>> idx2 = allel.UniqueIndex(['X', 'F', 'G', 'C', 'Z'])
        >>> idx1.intersect(idx2)
        UniqueIndex((2,), dtype=<U1)
        ['C' 'F']
        >>> idx2.intersect(idx1)
        UniqueIndex((2,), dtype=<U1)
        ['F' 'C']

        """

        loc = self.locate_keys(other, strict=False)
        return np.compress(loc, self)


class SortedMultiIndex(object):
    """Two-level index of sorted values, e.g., variant positions from two or
    more chromosomes/contigs.

    Parameters
    ----------
    l1 : array_like
        First level values in ascending order.
    l2 : array_like
        Second level values, in ascending order within each sub-level.
    copy : bool, optional
        If True, inputs will be copied into new arrays.

    Examples
    --------

    >>> import allel
    >>> chrom = ['chr1', 'chr1', 'chr2', 'chr2', 'chr2', 'chr3']
    >>> pos = [1, 4, 2, 5, 5, 3]
    >>> idx = allel.SortedMultiIndex(chrom, pos)
    >>> len(idx)
    6

    """

    def __init__(self, l1, l2, copy=False):
        l1 = SortedIndex(l1, copy=copy)
        l2 = np.array(l2, copy=copy)
        l2 = asarray_ndim(l2, 1)
        check_dim0_aligned(l1, l2)
        self.l1 = l1
        self.l2 = l2

    def __repr__(self):
        s = ('SortedMultiIndex(%s)\n' % len(self))
        return s

    def __str__(self):
        s = ('SortedMultiIndex(%s)\n' % len(self))
        return s

    def locate_key(self, k1, k2=None):
        """
        Get index location for the requested key.

        Parameters
        ----------
        k1 : object
            Level 1 key.
        k2 : object, optional
            Level 2 key.

        Returns
        -------
        loc : int or slice
            Location of requested key (will be slice if there are duplicate
            entries).

        Examples
        --------

        >>> import allel
        >>> chrom = ['chr1', 'chr1', 'chr2', 'chr2', 'chr2', 'chr3']
        >>> pos = [1, 4, 2, 5, 5, 3]
        >>> idx = allel.SortedMultiIndex(chrom, pos)
        >>> idx.locate_key('chr1')
        slice(0, 2, None)
        >>> idx.locate_key('chr1', 4)
        1
        >>> idx.locate_key('chr2', 5)
        slice(3, 5, None)
        >>> try:
        ...     idx.locate_key('chr3', 4)
        ... except KeyError as e:
        ...     print(e)
        ...
        ('chr3', 4)

        """

        loc1 = self.l1.locate_key(k1)
        if k2 is None:
            return loc1
        if isinstance(loc1, slice):
            offset = loc1.start
            try:
                loc2 = SortedIndex(self.l2[loc1], copy=False).locate_key(k2)
            except KeyError:
                # reraise with more information
                raise KeyError(k1, k2)
            else:
                if isinstance(loc2, slice):
                    loc = slice(offset + loc2.start, offset + loc2.stop)
                else:
                    # assume singleton
                    loc = offset + loc2
        else:
            # singleton match in l1
            v = self.l2[loc1]
            if v == k2:
                loc = loc1
            else:
                raise KeyError(k1, k2)
        return loc

    def locate_range(self, k1, start=None, stop=None):
        """Locate slice of index containing all entries within the range
        `key`:`start`-`stop` **inclusive**.

        Parameters
        ----------
        key : object
            Level 1 key value.
        start : object, optional
            Level 2 start value.
        stop : object, optional
            Level 2 stop value.

        Returns
        -------
        loc : slice
            Slice object.

        Examples
        --------

        >>> import allel
        >>> chrom = ['chr1', 'chr1', 'chr2', 'chr2', 'chr2', 'chr3']
        >>> pos = [1, 4, 2, 5, 5, 3]
        >>> idx = allel.SortedMultiIndex(chrom, pos)
        >>> idx.locate_range('chr1')
        slice(0, 2, None)
        >>> idx.locate_range('chr1', 1, 4)
        slice(0, 2, None)
        >>> idx.locate_range('chr2', 3, 7)
        slice(3, 5, None)
        >>> try:
        ...     idx.locate_range('chr3', 4, 9)
        ... except KeyError as e:
        ...     print(e)
        ('chr3', 4, 9)

        """

        loc1 = self.l1.locate_key(k1)
        if start is None and stop is None:
            loc = loc1
        elif isinstance(loc1, slice):
            offset = loc1.start
            idx = SortedIndex(self.l2[loc1], copy=False)
            try:
                loc2 = idx.locate_range(start, stop)
            except KeyError:
                raise KeyError(k1, start, stop)
            else:
                loc = slice(offset + loc2.start, offset + loc2.stop)
        else:
            # singleton match in l1
            v = self.l2[loc1]
            if start <= v <= stop:
                loc = loc1
            else:
                raise KeyError(k1, start, stop)
        # ensure slice is always returned
        if not isinstance(loc, slice):
            loc = slice(loc, loc + 1)
        return loc

    def __len__(self):
        return len(self.l1)


class VariantTable(RecArrayAug):
    """Table (catalogue) of variants.

    Parameters
    ----------
    data : array_like, structured, shape (n_variants,)
        Variant records.
    index : string or pair of strings, optional
        Names of columns to use for positional index, e.g., 'POS' if table
        contains a 'POS' column and records from a single chromosome/contig,
        or ('CHROM', 'POS') if table contains records from multiple
        chromosomes/contigs.
    **kwargs : keyword arguments, optional
        Further keyword arguments are passed through to
        :func:`numpy.rec.array`.

    Examples
    --------
    Instantiate a table from existing data::

        >>> import allel
        >>> records = [[b'chr1', 2, 35, 4.5, (1, 2)],
        ...            [b'chr1', 7, 12, 6.7, (3, 4)],
        ...            [b'chr2', 3, 78, 1.2, (5, 6)],
        ...            [b'chr2', 9, 22, 4.4, (7, 8)],
        ...            [b'chr3', 6, 99, 2.8, (9, 10)]]
        >>> dtype = [('CHROM', 'S4'),
        ...          ('POS', 'u4'),
        ...          ('DP', int),
        ...          ('QD', float),
        ...          ('AC', (int, 2))]
        >>> vt = allel.VariantTable(records, dtype=dtype,
        ...                         index=('CHROM', 'POS'))
        >>> vt.names
        ('CHROM', 'POS', 'DP', 'QD', 'AC')
        >>> vt.n_variants
        5

    Access a column::

        >>> vt['DP']
        array([35, 12, 78, 22, 99])

    Access multiple columns::

        >>> vt[['DP', 'QD']]  # doctest: +ELLIPSIS
        VariantTable((5,), dtype=(numpy.record, [('DP', '<i8'), ('QD', '<f8...
        [(35, 4.5) (12, 6.7) (78, 1.2) (22, 4.4) (99, 2.8)]

    Access a row::

        >>> vt[2]
        (b'chr2', 3, 78, 1.2, array([5, 6]))

    Access multiple rows::

        >>> vt[2:4]  # doctest: +ELLIPSIS
        VariantTable((2,), dtype=(numpy.record, [('CHROM', 'S4'), ('POS', '...
        [(b'chr2', 3, 78, 1.2, array([5, 6])) (b'chr2', 9, 22, 4.4, array([...

    Evaluate expressions against the table::

        >>> vt.eval('DP > 30')
        array([ True, False,  True, False,  True], dtype=bool)
        >>> vt.eval('(DP > 30) & (QD > 4)')
        array([ True, False, False, False, False], dtype=bool)
        >>> vt.eval('DP * 2')
        array([ 70,  24, 156,  44, 198])

    Query the table::

        >>> vt.query('DP > 30')  # doctest: +ELLIPSIS
        VariantTable((3,), dtype=(numpy.record, [('CHROM', 'S4'), ('POS', '...
        [(b'chr1', 2, 35, 4.5, array([1, 2])) (b'chr2', 3, 78, 1.2, array([...
         (b'chr3', 6, 99, 2.8, array([ 9, 10]))]
        >>> vt.query('(DP > 30) & (QD > 4)')  # doctest: +ELLIPSIS
        VariantTable((1,), dtype=(numpy.record, [('CHROM', 'S4'), ('POS', '...
        [(b'chr1', 2, 35, 4.5, array([1, 2]))]

    Use the index to query variants::

        >>> vt.query_region(b'chr2', 1, 10)  # doctest: +ELLIPSIS
        VariantTable((2,), dtype=(numpy.record, [('CHROM', 'S4'), ('POS', '...
        [(b'chr2', 3, 78, 1.2, array([5, 6])) (b'chr2', 9, 22, 4.4, array([...

    """

    def __new__(cls, data, index=None, **kwargs):
        kwargs.setdefault('copy', False)
        obj = np.rec.array(data, **kwargs)
        obj = obj.view(cls)
        # initialise index
        # noinspection PyArgumentList
        cls.set_index(obj, index)
        return obj

    def __array_finalize__(self, obj):

        # called after constructor
        if obj is None:
            return

        # called after slice (new-from-template)
        if isinstance(obj, VariantTable):
            return

        # called after view - nothing to do
        # VariantTable._check_input_data(obj)

    # noinspection PyUnusedLocal
    def __array_wrap__(self, out_arr, context=None):
        # don't wrap results of any ufuncs
        return np.asarray(out_arr)

    def __getslice__(self, *args, **kwargs):
        s = np.ndarray.__getslice__(self, *args, **kwargs)
        if hasattr(s, 'ndim') and s.ndim > 0:
            if s.dtype.names is not None:
                return VariantTable(s, copy=False)
            else:
                return np.asarray(s)
        return s

    def __getitem__(self, *args, **kwargs):
        s = np.ndarray.__getitem__(self, *args, **kwargs)
        if hasattr(s, 'ndim') and s.ndim > 0:
            if s.dtype.names is not None:
                return VariantTable(s, copy=False)
            else:
                return np.asarray(s)
        return s

    @property
    def n_variants(self):
        """Number of variants (length of first dimension)."""
        return self.shape[0]

    @property
    def names(self):
        """Column names."""
        return self.dtype.names

    def set_index(self, index):
        """Set or reset the index.

        Parameters
        ----------
        index : string or pair of strings, optional
            Names of columns to use for positional index, e.g., 'POS' if table
            contains a 'POS' column and records from a single
            chromosome/contig, or ('CHROM', 'POS') if table contains records
            from multiple chromosomes/contigs.

        """
        if index is None:
            pass
        elif isinstance(index, str):
            index = SortedIndex(self[index], copy=False)
        elif isinstance(index, (tuple, list)) and len(index) == 2:
            index = SortedMultiIndex(self[index[0]], self[index[1]],
                                     copy=False)
        else:
            raise ValueError('invalid index argument, expected string or '
                             'pair of strings, found %s' % repr(index))
        self.index = index

    def query_position(self, chrom=None, position=None):
        """Query the table, returning row or rows matching the given genomic
        position.

        Parameters
        ----------
        chrom : string, optional
            Chromosome/contig.
        position : int, optional
            Position (1-based).

        Returns
        -------
        result : row or VariantTable

        """

        if self.index is None:
            raise ValueError('no index has been set')
        if isinstance(self.index, SortedIndex):
            # ignore chrom
            loc = self.index.locate_key(position)
        else:
            loc = self.index.locate_key(chrom, position)
        return self[loc]

    def query_region(self, chrom=None, start=None, stop=None):
        """Query the table, returning row or rows within the given genomic
        region.

        Parameters
        ----------
        chrom : string, optional
            Chromosome/contig.
        start : int, optional
            Region start position (1-based).
        stop : int, optional
            Region stop position (1-based).

        Returns
        -------
        result : VariantTable

        """
        if self.index is None:
            raise ValueError('no index has been set')
        if isinstance(self.index, SortedIndex):
            # ignore chrom
            loc = self.index.locate_range(start, stop)
        else:
            loc = self.index.locate_range(chrom, start, stop)
        return self[loc]

    def to_vcf(self, path, rename=None, number=None, description=None,
               fill=None, write_header=True):
        r"""Write to a variant call format (VCF) file.

        Parameters
        ----------
        path : string
            File path.
        rename : dict, optional
            Rename these columns in the VCF.
        number : dict, optional
            Override the number specified in INFO headers.
        description : dict, optional
            Descriptions for the INFO and FILTER headers.
        fill : dict, optional
            Fill values used for missing data in the table.

        Examples
        --------
        Setup a variant table to write out::

            >>> import allel
            >>> chrom = [b'chr1', b'chr1', b'chr2', b'chr2', b'chr3']
            >>> pos = [2, 6, 3, 8, 1]
            >>> id = ['a', 'b', 'c', 'd', 'e']
            >>> ref = [b'A', b'C', b'T', b'G', b'N']
            >>> alt = [(b'T', b'.'),
            ...        (b'G', b'.'),
            ...        (b'A', b'C'),
            ...        (b'C', b'A'),
            ...        (b'X', b'.')]
            >>> qual = [1.2, 2.3, 3.4, 4.5, 5.6]
            >>> filter_qd = [True, True, True, False, False]
            >>> filter_dp = [True, False, True, False, False]
            >>> dp = [12, 23, 34, 45, 56]
            >>> qd = [12.3, 23.4, 34.5, 45.6, 56.7]
            >>> flg = [True, False, True, False, True]
            >>> ac = [(1, -1), (3, -1), (5, 6), (7, 8), (9, -1)]
            >>> xx = [(1.2, 2.3), (3.4, 4.5), (5.6, 6.7), (7.8, 8.9),
            ...       (9.0, 9.9)]
            >>> columns = [chrom, pos, id, ref, alt, qual, filter_dp,
            ...            filter_qd, dp, qd, flg, ac, xx]
            >>> records = list(zip(*columns))
            >>> dtype = [('chrom', 'S4'),
            ...          ('pos', 'u4'),
            ...          ('ID', 'S1'),
            ...          ('ref', 'S1'),
            ...          ('alt', ('S1', 2)),
            ...          ('qual', 'f4'),
            ...          ('filter_dp', bool),
            ...          ('filter_qd', bool),
            ...          ('dp', int),
            ...          ('qd', float),
            ...          ('flg', bool),
            ...          ('ac', (int, 2)),
            ...          ('xx', (float, 2))]
            >>> vt = allel.VariantTable(records, dtype=dtype)

        Now write out to VCF and inspect the result::

            >>> rename = {'dp': 'DP', 'qd': 'QD', 'filter_qd': 'QD'}
            >>> fill = {'ALT': b'.', 'ac': -1}
            >>> number = {'ac': 'A'}
            >>> description = {'ac': 'Allele counts', 'filter_dp': 'Low depth'}
            >>> vt.to_vcf('example.vcf', rename=rename, fill=fill,
            ...           number=number, description=description)
            >>> print(open('example.vcf').read())  # doctest: +ELLIPSIS
            ##fileformat=VCFv4.1
            ##fileDate=...
            ##source=...
            ##INFO=<ID=DP,Number=1,Type=Integer,Description="">
            ##INFO=<ID=QD,Number=1,Type=Float,Description="">
            ##INFO=<ID=ac,Number=A,Type=Integer,Description="Allele counts">
            ##INFO=<ID=flg,Number=0,Type=Flag,Description="">
            ##INFO=<ID=xx,Number=2,Type=Float,Description="">
            ##FILTER=<ID=QD,Description="">
            ##FILTER=<ID=dp,Description="Low depth">
            #CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO
            chr1	2	a	A	T	1.2	QD;dp	DP=12;QD=12.3;ac=1;flg;xx=...
            chr1	6	b	C	G	2.3	QD	DP=23;QD=23.4;ac=3;xx=3.4,4.5
            chr2	3	c	T	A,C	3.4	QD;dp	DP=34;QD=34.5;ac=5,6;flg;x...
            chr2	8	d	G	C,A	4.5	PASS	DP=45;QD=45.6;ac=7,8;xx=7...
            chr3	1	e	N	X	5.6	PASS	DP=56;QD=56.7;ac=9;flg;xx=...

        """

        write_vcf(path, variants=self, rename=rename, number=number,
                  description=description, fill=fill,
                  write_header=write_header)


def sample_to_haplotype_selection(indices, ploidy):
    return [(i * ploidy) + n for i in indices for n in range(ploidy)]


# TODO factor out common table code


class FeatureTable(RecArrayAug):
    """Table of genomic features (e.g., genes, exons, etc.).

    Parameters
    ----------
    data : array_like, structured, shape (n_variants,)
        Variant records.
    index : pair or triplet of strings, optional
        Names of columns to use for positional index, e.g., ('start',
        'stop') if table contains 'start' and 'stop' columns and records
        from a single chromosome/contig, or ('seqid', 'start', 'end') if table
        contains records from multiple chromosomes/contigs.
    **kwargs : keyword arguments, optional
        Further keyword arguments are passed through to
        :func:`numpy.rec.array`.

    """

    def __new__(cls, data, index=None, **kwargs):
        kwargs.setdefault('copy', False)
        obj = np.rec.array(data, **kwargs)
        obj = obj.view(cls)
        # TODO initialise interval index
        return obj

    def __array_finalize__(self, obj):

        # called after constructor
        if obj is None:
            return

        # called after slice (new-from-template)
        if isinstance(obj, FeatureTable):
            return

        # called after view - nothing to do
        # VariantTable._check_input_data(obj)

    # noinspection PyUnusedLocal
    def __array_wrap__(self, out_arr, context=None):
        # don't wrap results of any ufuncs
        return np.asarray(out_arr)

    def __getslice__(self, *args, **kwargs):
        s = np.ndarray.__getslice__(self, *args, **kwargs)
        if hasattr(s, 'ndim') and s.ndim > 0:
            if s.dtype.names is not None:
                return FeatureTable(s, copy=False)
            else:
                return np.asarray(s)
        return s

    def __getitem__(self, *args, **kwargs):
        s = np.ndarray.__getitem__(self, *args, **kwargs)
        if hasattr(s, 'ndim') and s.ndim > 0:
            if s.dtype.names is not None:
                return FeatureTable(s, copy=False)
            else:
                return np.asarray(s)
        return s

    @property
    def n_features(self):
        """Number of features (length of first dimension)."""
        return self.shape[0]

    @property
    def names(self):
        """Column names."""
        return self.dtype.names

    def query_region(self, chrom=None, start=None, stop=None):
        """TODO

        """
        # TODO use interval index
        pass

    def to_mask(self, size, start_name='start', stop_name='end'):
        """Construct a mask array where elements are True if the fall within
        features in the table.

        Parameters
        ----------

        size : int
            Size of chromosome/contig.
        start_name : string, optional
            Name of column with start coordinates.
        stop_name : string, optional
            Name of column with stop coordinates.

        Returns
        -------

        mask : ndarray, bool

        """
        m = np.zeros(size, dtype=bool)
        for start, stop in self[[start_name, stop_name]]:
            m[start-1:stop] = True
        return m

    @staticmethod
    def from_gff3(path, attributes=None, region=None,
                  score_fill=-1, phase_fill=-1, attributes_fill=b'.',
                  dtype=None):
        """Read a feature table from a GFF3 format file.

        Parameters
        ----------
        path : string
            File path.
        attributes : list of strings, optional
            List of columns to extract from the "attributes" field.
        region : string, optional
            Genome region to extract. If given, file must be position
            sorted, bgzipped and tabix indexed. Tabix must also be installed
            and on the system path.
        score_fill : object, optional
            Value to use where score field has a missing value.
        phase_fill : object, optional
            Value to use where phase field has a missing value.
        attributes_fill : object or list of objects, optional
            Value(s) to use where attribute field(s) have a missing value.
        dtype : numpy dtype, optional
            Manually specify a dtype.

        Returns
        -------
        ft : FeatureTable

        """

        # setup iterator
        recs = iter_gff3(path, attributes=attributes, region=region,
                         score_fill=score_fill, phase_fill=phase_fill,
                         attributes_fill=attributes_fill)

        # determine dtype from sample of initial records
        if dtype is None:
            names = 'seqid', 'source', 'type', 'start', 'end', 'score', \
                    'strand', 'phase'
            if attributes is not None:
                names += tuple(attributes)
            recs_sample = list(itertools.islice(recs, 1000))
            a = np.rec.array(recs_sample, names=names)
            dtype = a.dtype
            recs = itertools.chain(recs_sample, recs)

        a = np.fromiter(recs, dtype=dtype)
        ft = FeatureTable(a, copy=False)
        return ft


def create_allele_mapping(ref, alt, alleles, dtype='i1'):
    """Create an array mapping variant alleles into a different allele index
    system.

    Parameters
    ----------
    ref : array_like, S1, shape (n_variants,)
        Reference alleles.
    alt : array_like, S1, shape (n_variants, n_alt_alleles)
        Alternate alleles.
    alleles : array_like, S1, shape (n_variants, n_alleles)
        Alleles defining the new allele indexing.

    Returns
    -------
    mapping : ndarray, int8, shape (n_variants, n_alt_alleles + 1)

    Examples
    --------
    Example with biallelic variants::

        >>> import allel
        >>> from allel.model.ndarray import create_allele_mapping
        >>> ref = [b'A', b'C', b'T', b'G']
        >>> alt = [b'T', b'G', b'C', b'A']
        >>> alleles = [[b'A', b'T'],  # no transformation
        ...            [b'G', b'C'],  # swap
        ...            [b'T', b'A'],  # 1 missing
        ...            [b'A', b'C']]  # 1 missing
        >>> mapping = create_allele_mapping(ref, alt, alleles)
        >>> mapping
        array([[ 0,  1],
               [ 1,  0],
               [ 0, -1],
               [-1,  0]], dtype=int8)

    Example with multiallelic variants::

        >>> ref = [b'A', b'C', b'T']
        >>> alt = [[b'T', b'G'],
        ...        [b'A', b'T'],
        ...        [b'G', b'.']]
        >>> alleles = [[b'A', b'T'],
        ...            [b'C', b'T'],
        ...            [b'G', b'A']]
        >>> mapping = create_allele_mapping(ref, alt, alleles)
        >>> mapping
        array([[ 0,  1, -1],
               [ 0, -1,  1],
               [-1,  0, -1]], dtype=int8)

    See Also
    --------
    GenotypeArray.map_alleles, HaplotypeArray.map_alleles,
    AlleleCountsArray.map_alleles

    """

    ref = asarray_ndim(ref, 1)
    alt = asarray_ndim(alt, 1, 2)
    alleles = asarray_ndim(alleles, 1, 2)
    check_dim0_aligned(ref, alt, alleles)

    # reshape for convenience
    ref = ref[:, None]
    if alt.ndim == 1:
        alt = alt[:, None]
    if alleles.ndim == 1:
        alleles = alleles[:, None]
    source_alleles = np.append(ref, alt, axis=1)

    # setup output array
    out = np.empty(source_alleles.shape, dtype=dtype)
    out.fill(-1)

    # find matches
    for ai in range(source_alleles.shape[1]):
        match = source_alleles[:, ai, None] == alleles
        match_i, match_j = match.nonzero()
        out[match_i, ai] = match_j

    return out


def locate_fixed_differences(ac1, ac2):
    """Locate variants with no shared alleles between two populations.

    Parameters
    ----------
    ac1 : array_like, int, shape (n_variants, n_alleles)
        Allele counts array from the first population.
    ac2 : array_like, int, shape (n_variants, n_alleles)
        Allele counts array from the second population.

    Returns
    -------
    loc : ndarray, bool, shape (n_variants,)

    See Also
    --------
    allel.stats.diversity.windowed_df

    Examples
    --------

    >>> import allel
    >>> from allel.model.ndarray import locate_fixed_differences
    >>> g = allel.GenotypeArray([[[0, 0], [0, 0], [1, 1], [1, 1]],
    ...                          [[0, 1], [0, 1], [0, 1], [0, 1]],
    ...                          [[0, 1], [0, 1], [1, 1], [1, 1]],
    ...                          [[0, 0], [0, 0], [1, 1], [2, 2]],
    ...                          [[0, 0], [-1, -1], [1, 1], [-1, -1]]])
    >>> ac1 = g.count_alleles(subpop=[0, 1])
    >>> ac2 = g.count_alleles(subpop=[2, 3])
    >>> loc_df = locate_fixed_differences(ac1, ac2)
    >>> loc_df
    array([ True, False, False,  True,  True], dtype=bool)

    """

    # check inputs
    ac1 = asarray_ndim(ac1, 2)
    ac2 = asarray_ndim(ac2, 2)
    check_dim0_aligned(ac1, ac2)
    ac1, ac2 = ensure_dim1_aligned(ac1, ac2)

    # stack allele counts for convenience
    pac = np.dstack([ac1, ac2])

    # count numbers of alleles called in each population
    pan = np.sum(pac, axis=1)

    # count the numbers of populations with each allele
    npa = np.sum(pac > 0, axis=2)

    # locate variants with allele calls in both populations
    non_missing = np.all(pan > 0, axis=1)

    # locate variants where all alleles are only found in a single population
    no_shared_alleles = np.all(npa <= 1, axis=1)

    return non_missing & no_shared_alleles


def locate_private_alleles(*acs):
    """Locate alleles that are found only in a single population.

    Parameters
    ----------
    *acs : array_like, int, shape (n_variants, n_alleles)
        Allele counts arrays from each population.

    Returns
    -------
    loc : ndarray, bool, shape (n_variants, n_alleles)
        Boolean array where elements are True if allele is private to a
        single population.

    Examples
    --------

    >>> import allel
    >>> from allel.model.ndarray import locate_private_alleles
    >>> g = allel.GenotypeArray([[[0, 0], [0, 0], [1, 1], [1, 1]],
    ...                          [[0, 1], [0, 1], [0, 1], [0, 1]],
    ...                          [[0, 1], [0, 1], [1, 1], [1, 1]],
    ...                          [[0, 0], [0, 0], [1, 1], [2, 2]],
    ...                          [[0, 0], [-1, -1], [1, 1], [-1, -1]]])
    >>> ac1 = g.count_alleles(subpop=[0, 1])
    >>> ac2 = g.count_alleles(subpop=[2])
    >>> ac3 = g.count_alleles(subpop=[3])
    >>> loc_private_alleles = locate_private_alleles(ac1, ac2, ac3)
    >>> loc_private_alleles
    array([[ True, False, False],
           [False, False, False],
           [ True, False, False],
           [ True,  True,  True],
           [ True,  True, False]], dtype=bool)
    >>> loc_private_variants = np.any(loc_private_alleles, axis=1)
    >>> loc_private_variants
    array([ True, False,  True,  True,  True], dtype=bool)

    """

    # check inputs
    acs = [asarray_ndim(ac, 2) for ac in acs]
    debug([ac.shape for ac in acs])
    check_dim0_aligned(*acs)
    debug([ac.shape for ac in acs])
    acs = ensure_dim1_aligned(*acs)
    debug([ac.shape for ac in acs])

    # stack allele counts for convenience
    pac = np.dstack(acs)
    debug(pac.shape)

    # count the numbers of populations with each allele
    npa = np.sum(pac > 0, axis=2)

    # locate alleles found only in a single population
    loc_pa = npa == 1

    return loc_pa


def array_to_hdf5(a, parent, name, **kwargs):
    """Write a Numpy array to an HDF5 dataset.

    Parameters
    ----------
    a : ndarray
        Data to write.
    parent : string or h5py group
        Parent HDF5 file or group. If a string, will be treated as HDF5 file
        name.
    name : string
        Name or path of dataset to write data into.
    kwargs : keyword arguments
        Passed through to h5py require_dataset() function.

    Returns
    -------
    h5d : h5py dataset

    """

    import h5py

    h5f = None

    if isinstance(parent, str):
        h5f = h5py.File(parent, mode='a')
        parent = h5f

    try:

        kwargs.setdefault('chunks', True)  # auto-chunking
        kwargs.setdefault('dtype', a.dtype)
        kwargs.setdefault('compression', 'gzip')
        h5d = parent.require_dataset(name, shape=a.shape, **kwargs)
        h5d[...] = a
        return h5d

    finally:
        if h5f is not None:
            h5f.close()


def recarray_to_html_str(ra, limit=5, caption=None):
    # use implementation from petl
    import petl as etl
    tbl = etl.fromarray(ra)
    if caption is None:
        caption = '%s(%s, dtype=%s)' \
                  % (type(ra).__name__, ra.shape, ra.dtype)
    caption = caption.replace('<', '&lt;')
    caption = caption.replace('\n', '<br/>')
    # noinspection PyProtectedMember
    html = etl.util.vis._display_html(tbl,
                                      caption=caption,
                                      limit=limit,
                                      index_header=False)
    return html


def recarray_display(ra, limit=5, caption=None, **kwargs):
    # use implementation from petl
    import petl as etl
    tbl = etl.fromarray(ra)
    kwargs.setdefault('index_header', False)
    if caption is None:
        caption = '%s(%s, dtype=%s)' \
                  % (type(ra).__name__, ra.shape, ra.dtype)
    caption = caption.replace('<', '&lt;')
    caption = caption.replace('\n', '<br/>')
    return tbl.display(limit=limit, caption=caption, **kwargs)


def recarray_from_hdf5_group(*args, **kwargs):
    """Load a recarray from columns stored as separate datasets with an
    HDF5 group.

    Either provide an h5py group as a single positional argument,
    or provide two positional arguments giving the HDF5 file path and the
    group node path within the file.

    The following optional parameters may be given.

    Parameters
    ----------
    start : int, optional
        Index to start loading from.
    stop : int, optional
        Index to finish loading at.
    condition : array_like, bool, optional
        A 1-dimensional boolean array of the same length as the columns of the
        table to load, indicating a selection of rows to load.

    """

    import h5py

    h5f = None

    if len(args) == 1:
        group = args[0]

    elif len(args) == 2:
        file_path, node_path = args
        h5f = h5py.File(file_path, mode='r')
        try:
            group = h5f[node_path]
        except:
            h5f.close()
            raise

    else:
        raise ValueError('bad arguments; expected group or (file_path, '
                         'node_path), found %s' % repr(args))

    try:

        if not isinstance(group, h5py.Group):
            raise ValueError('expected group, found %r' % group)

        # determine dataset names to load
        available_dataset_names = [n for n in group.keys()
                                   if isinstance(group[n], h5py.Dataset)]
        names = kwargs.pop('names', available_dataset_names)
        names = [str(n) for n in names]  # needed for PY2
        for n in names:
            if n not in set(group.keys()):
                raise ValueError('name not found: %s' % n)
            if not isinstance(group[n], h5py.Dataset):
                raise ValueError('name does not refer to a dataset: %s, %r'
                                 % (n, group[n]))

        # check datasets are aligned
        datasets = [group[n] for n in names]
        length = datasets[0].shape[0]
        for d in datasets[1:]:
            if d.shape[0] != length:
                raise ValueError('datasets must be of equal length')

        # determine start and stop parameters for load
        start = kwargs.pop('start', 0)
        stop = kwargs.pop('stop', length)

        # check condition
        condition = kwargs.pop('condition', None)
        condition = asarray_ndim(condition, 1, allow_none=True)
        if condition is not None and condition.size != length:
            raise ValueError('length of condition does not match length '
                             'of datasets')

        # setup output data
        dtype = [(n, d.dtype, d.shape[1:]) for n, d in zip(names, datasets)]
        ra = np.empty(length, dtype=dtype)

        for n, d in zip(names, datasets):
            a = d[start:stop]
            if condition is not None:
                a = np.compress(condition[start:stop], a, axis=0)
            ra[n] = a

        return ra

    finally:
        if h5f is not None:
            h5f.close()


def recarray_to_hdf5_group(ra, parent, name, **kwargs):
    """Write each column in a recarray to a dataset in an HDF5 group.

    Parameters
    ----------
    parent : string or h5py group
        Parent HDF5 file or group. If a string, will be treated as HDF5 file
        name.
    name : string
        Name or path of group to write data into.
    kwargs : keyword arguments
        Passed through to h5py require_dataset() function.

    Returns
    -------
    h5g : h5py group

    """

    import h5py

    h5f = None

    if isinstance(parent, str):
        h5f = h5py.File(parent, mode='a')
        parent = h5f

    try:

        h5g = parent.require_group(name)
        for n in ra.dtype.names:
            array_to_hdf5(ra[n], h5g, n, **kwargs)

        return h5g

    finally:
        if h5f is not None:
            h5f.close()
