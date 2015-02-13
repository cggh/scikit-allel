# -*- coding: utf-8 -*-
"""
This module defines array classes for variant call data.

"""
from __future__ import absolute_import, print_function, division


import logging


import numpy as np
import numexpr as ne


from allel.util import ignore_invalid


logger = logging.getLogger(__name__)
debug = logger.debug


from allel.constants import DIM_SAMPLES, DIM_PLOIDY, DIPLOID


class GenotypeArray(np.ndarray):
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

    In many cases the number of distinct alleles for each variant is
    small, e.g., less than 10, or even 2 (all variants are
    biallelic). In these cases a genotype array is not the most
    compact way of storing genotype data in memory. This class defines
    functions for bit-packing diploid genotype calls into single
    bytes, and for transforming genotype arrays into sparse matrices,
    which can assist in cases where memory usage needs to be
    minimised. Note however that these more compact representations do
    not allow the same flexibility in terms of using numpy universal
    functions to access and manipulate data.

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
        ...                          [[0, 1, 2], [-1, -1, -1]]], dtype='i1')
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
        """Constructor."""
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

    def __repr__(self):
        s = super(GenotypeArray, self).__repr__()
        return s[:-1] + ', n_variants=%s, n_samples=%s, ploidy=%s)' % \
                        (self.n_variants, self.n_samples, self.ploidy)

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
        """Find genotype calls that are heterozygous.

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
        >>> g.is_het()
        array([[False,  True],
               [ True, False],
               [ True, False]], dtype=bool)

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

    def view_haplotypes(self):
        """Reshape a genotype array to view it as haplotypes by
        dropping the ploidy dimension.

        Returns
        -------

        h : HaplotypeArray, shape (n_variants, n_samples * ploidy)
            Haplotype array (sharing same underlying buffer).

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
        >>> g.view_haplotypes()
        HaplotypeArray([[ 0,  0,  0,  1],
               [ 0,  1,  1,  1],
               [ 0,  2, -1, -1]], n_variants=3, n_haplotypes=4)

        """

        # reshape, preserving size of variants dimension
        newshape = (self.n_variants, -1)
        data = np.reshape(self, newshape)
        h = HaplotypeArray(data, copy=False)
        return h

    def to_n_alt(self, fill=0):
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
        out = np.empty((self.n_variants, self.n_samples), dtype='i1')
        np.sum(self > 0, axis=DIM_PLOIDY, out=out)

        # fill missing calls
        if fill != 0:
            m = self.is_missing()
            out[m] = fill

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
        ...                         [[0, 2], [1, 1]],
        ...                         [[2, 2], [-1, -1]]])
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
        outshape = (self.n_variants, self.n_samples, len(alleles))
        out = np.zeros(outshape, dtype='u1')

        for i, allele in enumerate(alleles):
            # count alleles along ploidy dimension
            np.sum(self == allele, axis=DIM_PLOIDY, out=out[..., i])

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
        GenotypeArray([[[ 0,  0],
                [ 0,  1]],
               [[ 0,  2],
                [ 1,  1]],
               [[ 2,  2],
                [-1, -1]]], dtype=int8, n_variants=3, n_samples=2, ploidy=2)

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

        h = self.view_haplotypes()
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
        GenotypeArray([[[ 0,  0],
                [ 0,  0]],
               [[ 0,  1],
                [ 0,  1]],
               [[ 1,  1],
                [ 0,  0]],
               [[ 0,  0],
                [-1, -1]]], dtype=int8, n_variants=4, n_samples=2, ploidy=2)

        """

        h = HaplotypeArray.from_sparse(m, order=order, out=out)
        g = h.view_genotypes(ploidy=ploidy)
        return g

    def allelism(self):
        """Determine the number of distinct alleles for each variant.

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
        >>> g.allelism()
        array([2, 3, 1])

        """

        return self.view_haplotypes().allelism()

    def allele_number(self):
        """Count the number of non-missing allele calls per variant.

        Returns
        -------

        an : ndarray, int, shape (n_variants,)
            Allele number array.

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 2], [1, 1]],
        ...                          [[2, 2], [-1, -1]]])
        >>> g.allele_number()
        array([4, 4, 2])

        """

        return self.view_haplotypes().allele_number()

    def allele_count(self, allele=1):
        """Count the number of calls of the given allele per variant.

        Parameters
        ----------

        allele : int, optional
            Allele index.

        Returns
        -------

        ac : ndarray, int, shape (n_variants,)
            Allele count array.

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 2], [1, 1]],
        ...                          [[2, 2], [-1, -1]]])
        >>> g.allele_count(allele=1)
        array([1, 2, 0])
        >>> g.allele_count(allele=2)
        array([0, 1, 2])

        """

        return self.view_haplotypes().allele_count(allele=allele)

    def allele_frequency(self, allele=1, fill=np.nan):
        """Calculate the frequency of the given allele per variant.

        Parameters
        ----------

        allele : int, optional
            Allele index.
        fill : int, optional
            The value to use where all genotype calls are missing for a
            variant.

        Returns
        -------

        af : ndarray, float, shape (n_variants,)
            Allele frequency array.
        ac : ndarray, int, shape (n_variants,)
            Allele count array (numerator).
        an : ndarray, int, shape (n_variants,)
            Allele number array (denominator).

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 2], [1, 1]],
        ...                          [[2, 2], [-1, -1]]])
        >>> af, ac, an = g.allele_frequency(allele=1)
        >>> af
        array([ 0.25,  0.5 ,  0.  ])
        >>> af, ac, an = g.allele_frequency(allele=2)
        >>> af
        array([ 0.  ,  0.25,  1.  ])

        """

        return self.view_haplotypes().allele_frequency(allele=allele,
                                                       fill=fill)

    def allele_counts(self, alleles=None):
        """Count the number of calls of each allele per variant.

        Parameters
        ----------

        alleles : sequence of ints, optional
            The alleles to count. If None, all alleles will be counted.

        Returns
        -------

        ac : ndarray, int, shape (n_variants, len(alleles))
            Allele counts array.

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 2], [1, 1]],
        ...                          [[2, 2], [-1, -1]]])
        >>> g.allele_counts()
        array([[3, 1, 0],
               [1, 2, 1],
               [0, 0, 2]], dtype=int32)
        >>> g.allele_counts(alleles=(1, 2))
        array([[1, 0],
               [2, 1],
               [0, 2]], dtype=int32)

        """

        return self.view_haplotypes().allele_counts(alleles=alleles)

    def allele_frequencies(self, alleles=None, fill=np.nan):
        """Calculate the frequency of each allele per variant.

        Parameters
        ----------

        alleles : sequence of ints, optional
            The alleles to calculate frequency of. If None, all allele
            frequencies will be calculated.
        fill : int, optional
            The value to use where all genotype calls are missing for a
            variant.

        Returns
        -------

        af : ndarray, float, shape (n_variants, len(alleles))
            Allele frequencies array.
        ac : ndarray, int, shape (n_variants, len(alleles))
            Allele counts array (numerator).
        an : ndarray, int, shape (n_variants,)
            Allele number array (denominator).

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 2], [1, 1]],
        ...                          [[2, 2], [-1, -1]]])
        >>> af, ac, an = g.allele_frequencies()
        >>> af
        array([[ 0.75,  0.25,  0.  ],
               [ 0.25,  0.5 ,  0.25],
               [ 0.  ,  0.  ,  1.  ]])
        >>> af, ac, an = g.allele_frequencies(alleles=(1, 2))
        >>> af
        array([[ 0.25,  0.  ],
               [ 0.5 ,  0.25],
               [ 0.  ,  1.  ]])
        """

        return self.view_haplotypes().allele_frequencies(alleles=alleles,
                                                         fill=fill)

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
        >>> g.is_variant()
        array([False,  True,  True,  True], dtype=bool)

        """

        return self.view_haplotypes().is_variant()

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
        >>> g.is_non_variant()
        array([ True, False, False, False], dtype=bool)

        """

        return self.view_haplotypes().is_non_variant()

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
        >>> g.is_segregating()
        array([False,  True,  True, False], dtype=bool)

        """

        return self.view_haplotypes().is_segregating()

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
        >>> g.is_non_segregating()
        array([ True, False, False,  True], dtype=bool)
        >>> g.is_non_segregating(allele=2)
        array([False, False, False,  True], dtype=bool)

        """

        return self.view_haplotypes().is_non_segregating(allele=allele)

    def is_singleton(self, allele=1):
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
        >>> g.is_singleton(allele=1)
        array([False,  True, False, False], dtype=bool)
        >>> g.is_singleton(allele=2)
        array([False, False,  True, False], dtype=bool)

        """

        return self.view_haplotypes().is_singleton(allele=allele)

    def is_doubleton(self, allele=1):
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
        >>> g.is_doubleton(allele=1)
        array([False,  True, False, False], dtype=bool)
        >>> g.is_doubleton(allele=2)
        array([False, False, False,  True], dtype=bool)

        """

        return self.view_haplotypes().is_doubleton(allele=allele)

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

    def haploidify_samples(self):
        """Construct a pseudo-haplotype for each sample by randomly
        selecting an allele from each genotype call.

        Returns
        -------

        h : HaplotypeArray

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
        HaplotypeArray([[ 0,  1],
               [ 0,  1],
               [ 1,  1],
               [ 2, -1]], n_variants=4, n_haplotypes=2)
        >>> g = allel.GenotypeArray([[[0, 0, 0], [0, 0, 1]],
        ...                          [[0, 1, 1], [1, 1, 1]],
        ...                          [[0, 1, 2], [-1, -1, -1]]])
        >>> g.haploidify_samples()
        HaplotypeArray([[ 0,  0],
               [ 1,  1],
               [ 2, -1]], n_variants=3, n_haplotypes=2)

        """

        # N.B., this implementation is obscure and uses more memory that
        # necessary, TODO review

        # define the range of possible indices, e.g., diploid => (0, 1)
        index_range = np.arange(0, self.ploidy, dtype='u1')

        # create a random index for each genotype call
        indices = np.random.choice(index_range,
                                   size=(self.n_variants * self.n_samples),
                                   replace=True)

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

    def heterozygosity_observed(self, fill=np.nan):
        """Calculate the rate of observed heterozygosity for each variant.

        Parameters
        ----------

        fill : float, optional
            Use this value for variants where all calls are missing.

        Returns
        -------

        ho : ndarray, float, shape (n_variants,)
            Observed heterozygosity

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 0], [0, 0]],
        ...                          [[0, 0], [0, 1], [1, 1]],
        ...                          [[0, 0], [1, 1], [2, 2]],
        ...                          [[1, 1], [1, 2], [-1, -1]]])
        >>> g.heterozygosity_observed()
        array([ 0.        ,  0.33333333,  0.        ,  0.5       ])

        """

        # count hets
        n_het = self.count_het(axis=DIM_SAMPLES)
        n_called = self.count_called(axis=DIM_SAMPLES)

        # calculate rate of observed heterozygosity, accounting for variants
        # where all calls are missing
        with ignore_invalid():
            ho = np.where(n_called > 0, n_het / n_called, fill)

        return ho

    def heterozygosity_expected(self, fill=np.nan):
        """Calculate the expected rate of heterozygosity for each variant
        under Hardy-Weinberg equilibrium.

        Parameters
        ----------

        fill : float, optional
            Use this value for variants where all calls are missing.

        Returns
        -------

        he : ndarray, float, shape (n_variants,)
            Expected heterozygosity

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 0], [0, 0]],
        ...                          [[0, 0], [0, 1], [1, 1]],
        ...                          [[0, 0], [1, 1], [2, 2]],
        ...                          [[1, 1], [1, 2], [-1, -1]]])
        >>> g.heterozygosity_expected()
        array([ 0.        ,  0.5       ,  0.66666667,  0.375     ])

        """

        # calculate allele frequencies
        af, _, an = self.allele_frequencies(fill=0)

        # calculate expected heterozygosity
        out = 1 - np.sum(np.power(af, self.ploidy), axis=1)

        # fill values where allele frequencies could not be calculated
        out[an == 0] = fill

        return out

    def inbreeding_coefficient(self, fill=np.nan):
        """Calculate the inbreeding coefficient for each variant.

        Parameters
        ----------

        fill : float, optional
            Use this value for variants where the expected heterozygosity is
            zero.

        Returns
        -------

        f : ndarray, float, shape (n_variants,)
            Inbreeding coefficient.

        Notes
        -----

        The inbreeding coefficient is calculated as *1 - (Ho/He)* where *Ho* is
        the observed heterozygosity and *He* is the expected heterozygosity.

        Examples
        --------

        >>> import allel
        >>> import numpy as np
        >>> g = allel.GenotypeArray([[[0, 0], [0, 0], [0, 0]],
        ...                          [[0, 0], [0, 1], [1, 1]],
        ...                          [[0, 0], [1, 1], [2, 2]],
        ...                          [[1, 1], [1, 2], [-1, -1]]])
        >>> g.heterozygosity_observed()
        array([ 0.        ,  0.33333333,  0.        ,  0.5       ])
        >>> g.heterozygosity_expected()
        array([ 0.        ,  0.5       ,  0.66666667,  0.375     ])
        >>> g.inbreeding_coefficient()
        array([        nan,  0.33333333,  1.        , -0.33333333])

        """

        # calculate observed and expected heterozygosity
        ho = self.heterozygosity_observed()
        he = self.heterozygosity_expected()

        # calculate inbreeding coefficient, accounting for variants with no
        # expected heterozygosity
        with ignore_invalid():
            f = np.where(he > 0, 1 - (ho / he), fill)

        return f

    def mean_pairwise_difference(self, fill=np.nan):
        """Calculate the mean number of pairwise differences between
        haplotypes for each variant.

        Parameters
        ----------

        fill : float
            Use this value where there are no pairs to compare (e.g.,
            all allele calls are missing).

        Returns
        -------

        m : ndarray, float, shape (n_variants,)

        Notes
        -----

        The values returned by this function can be summed over a genome
        region and divided by the number of accessible bases to estimate
        nucleotide diversity.

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 0]],
        ...                          [[0, 0], [0, 1]],
        ...                          [[0, 0], [1, 1]],
        ...                          [[0, 1], [1, 1]],
        ...                          [[1, 1], [1, 1]],
        ...                          [[0, 0], [1, 2]],
        ...                          [[0, 1], [1, 2]],
        ...                          [[0, 1], [-1, -1]]])
        >>> g.mean_pairwise_difference()
        array([ 0.        ,  0.5       ,  0.66666667,  0.5       ,  0.        ,
                0.83333333,  0.83333333,  1.        ])

        """

        return self.view_haplotypes().mean_pairwise_difference(fill=fill)


class HaplotypeArray(np.ndarray):
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

        >>> h.view_genotypes(ploidy=2)
        GenotypeArray([[[ 0,  0],
                [ 0,  1]],
               [[ 0,  1],
                [ 1,  1]],
               [[ 0,  2],
                [-1, -1]]], dtype=int8, n_variants=3, n_samples=2, ploidy=2)

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
        """Constructor."""
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

    @property
    def n_variants(self):
        """Number of variants (length of first dimension)."""
        return self.shape[0]

    @property
    def n_haplotypes(self):
        """Number of haplotypes (length of second dimension)."""
        return self.shape[1]

    def __repr__(self):
        s = super(HaplotypeArray, self).__repr__()
        return s[:-1] + ', n_variants=%s, n_haplotypes=%s)' % \
                        (self.n_variants, self.n_haplotypes)

    def view_genotypes(self, ploidy):
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

        Examples
        --------

        >>> import allel
        >>> h = allel.HaplotypeArray([[0, 0, 0, 1],
        ...                           [0, 1, 1, 1],
        ...                           [0, 2, -1, -1]], dtype='i1')
        >>> h.view_genotypes(ploidy=2)
        GenotypeArray([[[ 0,  0],
                [ 0,  1]],
               [[ 0,  1],
                [ 1,  1]],
               [[ 0,  2],
                [-1, -1]]], dtype=int8, n_variants=3, n_samples=2, ploidy=2)

        """

        # check ploidy is compatible
        if (self.n_haplotypes % ploidy) > 0:
            raise ValueError('incompatible ploidy')

        # reshape
        newshape = (self.n_variants, -1, ploidy)
        data = self.reshape(newshape)

        # wrap
        g = GenotypeArray(data, copy=False)

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
        HaplotypeArray([[ 0,  0,  0,  0],
               [ 0,  1,  0,  1],
               [ 1,  1,  0,  0],
               [ 0,  0, -1, -1]], dtype=int8, n_variants=4, n_haplotypes=4)

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
        """Determine the number of distinct alleles for each variant.

        Returns
        -------

        n : ndarray, int, shape (n_variants,)
            Allelism array.

        """

        # calculate allele counts
        ac = self.allele_counts()

        # count alleles present
        n = np.sum(ac > 0, axis=1)

        return n

    def allele_number(self):
        """Count the number of non-missing allele calls per variant.

        Returns
        -------

        an : ndarray, int, shape (n_variants,)
            Allele number array.

        """

        # count non-missing calls over samples
        an = np.sum(self >= 0, axis=1)

        return an

    def allele_count(self, allele=1):
        """Count the number of calls of the given allele per variant.

        Parameters
        ----------

        allele : int, optional
            Allele index.

        Returns
        -------

        ac : ndarray, int, shape (n_variants,)
            Allele count array.

        """

        # count non-missing calls over samples
        return np.sum(self == allele, axis=1)

    def allele_frequency(self, allele=1, fill=np.nan):
        """Calculate the frequency of the given allele per variant.

        Parameters
        ----------

        allele : int, optional
            Allele index.
        fill : int, optional
            The value to use where all genotype calls are missing for a
            variant.

        Returns
        -------

        af : ndarray, float, shape (n_variants,)
            Allele frequency array.
        ac : ndarray, int, shape (n_variants,)
            Allele count array (numerator).
        an : ndarray, int, shape (n_variants,)
            Allele number array (denominator).

        """

        # intermediate variables
        an = self.allele_number()
        ac = self.allele_count(allele=allele)

        # calculate allele frequency, accounting for variants with no calls
        with ignore_invalid():
            af = np.where(an > 0, ac / an, fill)

        return af, ac, an

    def allele_counts(self, alleles=None):
        """Count the number of calls of each allele per variant.

        Parameters
        ----------

        alleles : sequence of ints, optional
            The alleles to count. If None, all alleles will be counted.

        Returns
        -------

        ac : ndarray, int, shape (n_variants, len(alleles))
            Allele counts array.

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

    def allele_frequencies(self, alleles=None, fill=np.nan):
        """Calculate the frequency of each allele per variant.

        Parameters
        ----------

        alleles : sequence of ints, optional
            The alleles to calculate frequency of. If None, all allele
            frequencies will be calculated.
        fill : int, optional
            The value to use where all genotype calls are missing for a
            variant.

        Returns
        -------

        af : ndarray, float, shape (n_variants, len(alleles))
            Allele frequencies array.
        ac : ndarray, int, shape (n_variants, len(alleles))
            Allele counts array (numerator).
        an : ndarray, int, shape (n_variants,)
            Allele number array (denominator).

        """

        # intermediate variables
        an = self.allele_number()[:, None]
        ac = self.allele_counts(alleles=alleles)

        # calculate allele frequency, accounting for variants with no calls
        with ignore_invalid():
            af = np.where(an > 0, ac / an, fill)

        return af, ac, an[:, 0]

    def is_variant(self):
        """Find variants with at least one non-reference allele call.

        Returns
        -------

        out : ndarray, bool, shape (n_variants,)
            Boolean array where elements are True if variant matches the
            condition.

        """

        # find variants with at least 1 non-reference allele
        out = np.sum(self > 0, axis=1) >= 1

        return out

    def is_non_variant(self):
        """Find variants with no non-reference allele calls.

        Returns
        -------

        out : ndarray, bool, shape (n_variants,)
            Boolean array where elements are True if variant matches the
            condition.

        """

        # find variants with no non-reference alleles
        out = np.all(self <= 0, axis=1)

        return out

    def is_segregating(self):
        """Find segregating variants (where more than one allele is observed).

        Returns
        -------

        out : ndarray, bool, shape (n_variants,)
            Boolean array where elements are True if variant matches the
            condition.

        """

        # find segregating variants
        out = self.allelism() > 1

        return out

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

        """

        # count allele
        ac = self.allele_count(allele=allele)

        # find singletons
        out = ac == 1

        return out

    def is_doubleton(self, allele=1):
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

    def count_singleton(self, allele=1):
        return np.sum(self.is_singleton(allele=allele))

    def count_doubleton(self, allele=1):
        return np.sum(self.is_doubleton(allele=allele))

    def mean_pairwise_difference(self, fill=np.nan):
        """Calculate the mean number of pairwise differences between
        haplotypes for each variant.

        Parameters
        ----------

        fill : float
            Use this value where there are no pairs to compare (e.g.,
            all allele calls are missing).

        Returns
        -------

        m : ndarray, float, shape (n_variants,)

        Notes
        -----

        The values returned by this function can be summed over a genome
        region and divided by the number of accessible bases to estimate
        nucleotide diversity.

        Examples
        --------

        >>> import allel
        >>> h = allel.HaplotypeArray([[0, 0, 0, 0],
        ...                           [0, 0, 0, 1],
        ...                           [0, 0, 1, 1],
        ...                           [0, 1, 1, 1],
        ...                           [1, 1, 1, 1],
        ...                           [0, 0, 1, 2],
        ...                           [0, 1, 1, 2],
        ...                           [0, 1, -1, -1]])
        >>> h.mean_pairwise_difference()
        array([ 0.        ,  0.5       ,  0.66666667,  0.5       ,  0.        ,
                0.83333333,  0.83333333,  1.        ])

        """

        # This function calculates the mean number of pairwise differences
        # between haplotypes, generalising to any number of alleles.

        # number of non-missing alleles (i.e., haplotypes) for each variant
        an = self.allele_number()

        # allele counts for each variant
        ac = self.allele_counts()

        # total number of pairwise comparisons for each variant:
        # (an choose 2)
        n_pairs = an * (an - 1) / 2

        # number of pairwise comparisons where there is no difference:
        # sum of (ac choose 2) for each allele (i.e., number of ways to
        # choose the same allele twice)
        n_same = np.sum(ac * (ac - 1) / 2, axis=1)

        # number of pairwise differences
        n_diff = n_pairs - n_same

        # mean number of pairwise differences, accounting for cases where
        # there are no pairs
        with ignore_invalid():
            m = np.where(n_pairs > 0, n_diff / n_pairs, fill)

        return m


class PositionIndex(np.ndarray):
    """Array of positions from a single chromosome or contig.

    Parameters
    ----------

    data : array_like, int
        Positions (1-based) in ascending order.
    **kwargs : keyword arguments
        All keyword arguments are passed through to :func:`numpy.array`.

    Notes
    -----

    This class represents the genomic positions of a set of variants as a
    1-dimensional numpy array of integers.

    Each integer within the array is a 1-based coordinate position within a
    single chromosome or contig. Positions must be given in ascending order,
    although duplicate positions may be present (i.e., positions must be
    monotonically increasing).

    Examples
    --------

    >>> import allel
    >>> pos = allel.PositionIndex([2, 5, 14, 15, 42, 42, 77], dtype='i4')
    >>> pos.dtype
    dtype('int32')
    >>> pos.ndim
    1
    >>> pos.shape
    (7,)
    >>> pos.is_unique
    False

    """

    @staticmethod
    def _check_input_data(obj):

        # check dtype
        if obj.dtype.kind not in 'ui':
            raise TypeError('integer dtype required')

        # check dimensionality
        if obj.ndim != 1:
            raise TypeError('array with 1 dimension required')

        # check sorted ascending
        diff = np.diff(obj)
        if np.any(diff < 0):
            raise ValueError('array is not monotonically increasing')

    def __new__(cls, data, **kwargs):
        """Constructor."""
        obj = np.array(data, **kwargs)
        cls._check_input_data(obj)
        obj = obj.view(cls)
        return obj

    def __array_finalize__(self, obj):

        # called after constructor
        if obj is None:
            return

        # called after slice (new-from-template)
        if isinstance(obj, PositionIndex):
            return

        # called after view
        PositionIndex._check_input_data(obj)

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
            self._is_unique = ~np.any(np.diff(self) == 0)
        return self._is_unique

    def locate_key(self, key):
        """Get index location for the requested key.

        Parameters
        ----------

        key : int
            Position to locate.

        Returns
        -------

        loc : int or slice
            Location of `key` (will be slice if there are duplicate entries).

        Examples
        --------

        >>> import allel
        >>> pos = allel.PositionIndex([3, 6, 6, 11])
        >>> pos.locate_key(3)
        0
        >>> pos.locate_key(11)
        3
        >>> pos.locate_key(6)
        slice(1, 3, None)
        >>> try:
        ...     pos.locate_key(2)
        ... except KeyError as e:
        ...     print(e)
        ...
        2

        """

        left = np.searchsorted(self, key, side='left')
        right = np.searchsorted(self, key, side='right')
        diff = right - left
        if diff == 0:
            raise KeyError(key)
        elif diff == 1:
            return left
        else:
            return slice(left, right)

    def locate_intersection(self, other):
        """Locate the intersection with another position array.

        Parameters
        ----------

        other : array_like, int
            Array of positions to intersect.

        Returns
        -------

        loc : ndarray, bool
            Boolean array with location of intersection.
        loc_other : ndarray, bool
            Boolean array with location in `other` of intersection.

        Examples
        --------

        >>> import allel
        >>> pos1 = allel.PositionIndex([3, 6, 11, 20, 35])
        >>> pos2 = allel.PositionIndex([4, 6, 20, 39])
        >>> loc1, loc2 = pos1.locate_intersection(pos2)
        >>> loc1
        array([False,  True, False,  True, False], dtype=bool)
        >>> loc2
        array([False,  True,  True, False], dtype=bool)
        >>> pos1[loc1]
        PositionIndex([ 6, 20])
        >>> pos2[loc2]
        PositionIndex([ 6, 20])

        """

        # check inputs
        other = PositionIndex(other)

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
            Boolean array with location of positions.

        Examples
        --------

        >>> import allel
        >>> pos1 = allel.PositionIndex([3, 6, 11, 20, 35])
        >>> pos2 = allel.PositionIndex([4, 6, 20, 39])
        >>> loc = pos1.locate_keys(pos2, strict=False)
        >>> loc
        array([False,  True, False,  True, False], dtype=bool)
        >>> pos1[loc]
        PositionIndex([ 6, 20])

        """

        # check inputs
        keys = PositionIndex(keys)

        # find intersection
        loc, found = self.locate_intersection(keys)

        if strict and np.any(~found):
            raise KeyError(keys[~found])

        return loc

    def intersect(self, other):
        """Intersect with `other` positions.

        Parameters
        ----------

        other : array_like, int
            Array of positions to locate.

        Returns
        -------

        out : PositionIndex
            Positions in common.

        Examples
        --------

        >>> import allel
        >>> pos1 = allel.PositionIndex([3, 6, 11, 20, 35])
        >>> pos2 = allel.PositionIndex([4, 6, 20, 39])
        >>> pos1.intersect(pos2)
        PositionIndex([ 6, 20])

        """

        loc = self.locate_keys(other, strict=False)
        return np.compress(loc, self)

    def locate_range(self, start=None, stop=None):
        """Locate slice of index containing all entries within `start` and
        `stop` positions **inclusive**.

        Parameters
        ----------

        start : int, optional
            Start position.
        stop : int, optional
            Stop position.

        Returns
        -------

        loc : slice
            Slice object.

        Examples
        --------

        >>> import allel
        >>> pos = allel.PositionIndex([3, 6, 11, 20, 35])
        >>> loc = pos.locate_range(4, 32)
        >>> loc
        slice(1, 4, None)
        >>> pos[loc]
        PositionIndex([ 6, 11, 20])

        """

        # locate start and stop indices
        if start is None:
            start_index = 0
        else:
            start_index = np.searchsorted(self, start)
        if stop is None:
            stop_index = len(self)
        else:
            stop_index = np.searchsorted(self, stop, side='right')

        if stop_index - start_index == 0:
            raise KeyError(start, stop)

        loc = slice(start_index, stop_index)
        return loc

    def intersect_range(self, start=None, stop=None):
        """Intersect with range defined by `start` and `stop` positions
        **inclusive**.

        Parameters
        ----------

        start : int, optional
            Start position.
        stop : int, optional
            Stop position.

        Returns
        -------

        pos : PositionIndex

        Examples
        --------

        >>> import allel
        >>> pos = allel.PositionIndex([3, 6, 11, 20, 35])
        >>> pos.intersect_range(4, 32)
        PositionIndex([ 6, 11, 20])

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
            Range start positions.
        stops : array_like, int
            Range stop positions.

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
        >>> pos = allel.PositionIndex([3, 6, 11, 20, 35])
        >>> ranges = np.array([[0, 2], [6, 17], [12, 15], [31, 35],
        ...                    [100, 120]])
        >>> starts = ranges[:, 0]
        >>> stops = ranges[:, 1]
        >>> loc, loc_ranges = pos.locate_intersection_ranges(starts, stops)
        >>> loc
        array([False,  True,  True, False,  True], dtype=bool)
        >>> loc_ranges
        array([False,  True, False,  True, False], dtype=bool)
        >>> pos[loc]
        PositionIndex([ 6, 11, 35])
        >>> ranges[loc_ranges]
        array([[ 6, 17],
               [31, 35]])

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
        loc_ranges = start_indices < stop_indices

        # find positions within at least one interval
        loc = np.zeros(self.shape, dtype=np.bool)
        for i, j in zip(start_indices[loc_ranges], stop_indices[loc_ranges]):
            loc[i:j] = True

        return loc, loc_ranges

    def locate_ranges(self, starts, stops, strict=True):
        """Locate items within the given ranges.

        Parameters
        ----------

        starts : array_like, int
            Range start positions.
        stops : array_like, int
            Range stop positions.
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
        >>> pos = allel.PositionIndex([3, 6, 11, 20, 35])
        >>> ranges = np.array([[0, 2], [6, 17], [12, 15], [31, 35],
        ...                    [100, 120]])
        >>> starts = ranges[:, 0]
        >>> stops = ranges[:, 1]
        >>> loc = pos.locate_ranges(starts, stops, strict=False)
        >>> loc
        array([False,  True,  True, False,  True], dtype=bool)
        >>> pos[loc]
        PositionIndex([ 6, 11, 35])

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
            Range start positions.
        stops : array_like, int
            Range stop positions.

        Returns
        -------

        pos : PositionIndex

        Examples
        --------

        >>> import allel
        >>> import numpy as np
        >>> pos = allel.PositionIndex([3, 6, 11, 20, 35])
        >>> ranges = np.array([[0, 2], [6, 17], [12, 15], [31, 35],
        ...                    [100, 120]])
        >>> starts = ranges[:, 0]
        >>> stops = ranges[:, 1]
        >>> pos.intersect_ranges(starts, stops)
        PositionIndex([ 6, 11, 35])

        """

        loc = self.locate_ranges(starts, stops, strict=False)
        return np.compress(loc, self)


class LabelIndex(np.ndarray):
    """Array of labels (e.g., variant or sample identifiers).

    Parameters
    ----------

    data : array_like
        Labels.
    **kwargs : keyword arguments
        All keyword arguments are passed through to :func:`numpy.array`.

    Notes
    -----

    This class represents an arbitrary set of unique labels, e.g., sample or
    variant identifiers.

    There is no need for labels to be sorted. However, all labels must be
    unique within the array, and must be hashable objects.

    Examples
    --------

    >>> import allel
    >>> lbl = allel.LabelIndex(['A', 'C', 'B', 'F'])
    >>> lbl.dtype
    dtype('<U1')
    >>> lbl.ndim
    1
    >>> lbl.shape
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
        """Constructor."""
        obj = np.array(data, **kwargs)
        cls._check_input_data(obj)
        obj = obj.view(cls)
        return obj

    def __array_finalize__(self, obj):

        # called after constructor
        if obj is None:
            return

        # called after slice (new-from-template)
        if isinstance(obj, LabelIndex):
            return

        # called after view
        LabelIndex._check_input_data(obj)

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
        >>> lbl = allel.LabelIndex(['A', 'C', 'B', 'F'])
        >>> lbl.locate_key('A')
        0
        >>> lbl.locate_key('B')
        2
        >>> try:
        ...     lbl.locate_key('X')
        ... except KeyError as e:
        ...     print(e)
        ...
        'X'

        """

        # TODO review implementation for performance with larger arrays

        loc = np.nonzero(self == key)[0]
        if len(loc) == 0:
            raise KeyError(key)
        return loc[0]

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
        >>> lbl1 = allel.LabelIndex(['A', 'C', 'B', 'F'])
        >>> lbl2 = allel.LabelIndex(['X', 'F', 'G', 'C', 'Z'])
        >>> loc1, loc2 = lbl1.locate_intersection(lbl2)
        >>> loc1
        array([False,  True, False,  True], dtype=bool)
        >>> loc2
        array([False,  True, False,  True, False], dtype=bool)
        >>> lbl1[loc1]
        LabelIndex(['C', 'F'],
              dtype='<U1')
        >>> lbl2[loc2]
        LabelIndex(['F', 'C'],
              dtype='<U1')

        """

        # TODO review implementation for performance with larger arrays

        # check inputs
        other = LabelIndex(other)

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
        >>> lbl = allel.LabelIndex(['A', 'C', 'B', 'F'])
        >>> lbl.locate_keys(['F', 'C'])
        array([False,  True, False,  True], dtype=bool)
        >>> lbl.locate_keys(['X', 'F', 'G', 'C', 'Z'], strict=False)
        array([False,  True, False,  True], dtype=bool)

        """

        # check inputs
        keys = LabelIndex(keys)

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

        out : LabelIndex

        Examples
        --------

        >>> import allel
        >>> lbl1 = allel.LabelIndex(['A', 'C', 'B', 'F'])
        >>> lbl2 = allel.LabelIndex(['X', 'F', 'G', 'C', 'Z'])
        >>> lbl1.intersect(lbl2)
        LabelIndex(['C', 'F'],
              dtype='<U1')
        >>> lbl2.intersect(lbl1)
        LabelIndex(['F', 'C'],
              dtype='<U1')

        """

        loc = self.locate_keys(other, strict=False)
        return np.compress(loc, self)
