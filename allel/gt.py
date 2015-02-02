# -*- coding: utf-8 -*-
"""
This module provides functions for manipulating arrays of discrete genotype
calls.

Conventions
-----------

By convention, a *genotype array* is an array of discrete genotype
calls for a set of samples at a set of variants, represented as a
3-dimensional numpy array of integers. The first dimension corresponds
to the variants genotyped (i.e., the length of the first dimension is
equal to the number of variants), the second dimension corresponds to
the samples genotyped (i.e., the length of the second dimension equals
the number of samples), and the third dimension corresponds to the
ploidy of the samples.

For example, the array `g` below is a genotype array storing genotype
calls for 3 variants in 2 samples with ploidy 2 (i.e., diploid)::

    >>> import numpy as np
    >>> g = np.array([[[0, 0], [0, 1]],
    ...               [[0, 1], [1, 1]],
    ...               [[0, 2], [-1, -1]]], dtype='i1')
    >>> g.dtype
    dtype('int8')
    >>> g.ndim
    3
    >>> n_variants, n_samples, ploidy = g.shape
    >>> n_variants
    3
    >>> n_samples
    2
    >>> ploidy
    2

An array of genotype calls for a single variant at all samples can be
obtained by indexing the first dimension, e.g.::

    >>> g[1]
    array([[0, 1],
           [1, 1]], dtype=int8)

An array of genotype calls for a single sample at all variants can be
obtained by indexing the second dimension, e.g.::

    >>> g[:, 1]
    array([[ 0,  1],
           [ 1,  1],
           [-1, -1]], dtype=int8)

A genotype call for a specific sample at a specific variant can be obtained
by indexing the first and second dimensions, e.g.::

    >>> g[0, 0]
    array([0, 0], dtype=int8)
    >>> g[0, 1]
    array([0, 1], dtype=int8)
    >>> g[1, 1]
    array([1, 1], dtype=int8)
    >>> g[2, 1]
    array([-1, -1], dtype=int8)

Allelism
~~~~~~~~

Each integer within the array corresponds to an *allele index*, where
0 is the reference allele, 1 is the first alternate allele, 2 is the
second alternate allele, ... and -1 (or any other negative integer) is
a missing call. The actual alleles (i.e., the alternate nucleotide
sequences) and the physical positions of the variants within the
genome of an organism are stored in separate arrays, discussed
elsewhere.

A single byte integer dtype (int8) can represent up to 127 distinct
alleles, which is usually sufficient for most applications. A larger
integer dtype can be used if more alleles are required.

In many applications the number of distinct alleles for each variant
is small, e.g., less than 10, or even 2 (all variants are
biallelic). In these cases the canonical genotype array is not the
most compact way of storing genotype data in memory. This module
defines functions for bit-packing diploid genotype calls into single
bytes, and for transforming genotype arrays into sparse matrices,
which can assist in cases where memory usage needs to be minimised. Note
however that these more compact representations do not allow the same
flexibility in terms of using numpy universal functions to access and
manipulate data.

Phased and unphased calls
~~~~~~~~~~~~~~~~~~~~~~~~~

The canonical genotype array can store either phased or unphased
genotype calls. If the genotypes are phased (i.e., haplotypes have
been resolved) then individual haplotypes can be extracted by indexing
the second and third dimensions, e.g.::

    >>> # view the maternal haplotype for the second sample
    ... g[:, 1, 0]
    array([ 0,  1, -1], dtype=int8)

If the genotype calls are unphased then slices along the third
dimension have no meaning, and the ordering of alleles along the third
dimension is arbitrary. N.B, this means that an unphased diploid
heterozygous call could be stored as (0, 1) or equivalently as (1, 0).

Ploidy
~~~~~~

A genotype array can store genotype calls with any ploidy. For
example, the array `g_triploid` below has triploid calls for 2 samples
at 3 variants::

    >>> g_triploid = np.array([[[0, 0, 0], [0, 0, 1]],
    ...                        [[0, 1, 1], [1, 1, 1]],
    ...                        [[0, 1, 2], [-1, -1, -1]]], dtype='i1')

In the special case of haploid genotype calls, the third dimension can be
dropped, e.g., haploid calls for 2 samples at 3 variants::

    >>> g_haploid = np.array([[0, 1],
    ...                       [1, 1],
    ...                       [2, -1]], dtype='i1')

All the functions in this package assume that if a genotype array has only
two dimensions then it contains haploid calls.

Note that genotype arrays are not capable of storing calls for samples with
differing or variable ploidy.

Variant positions
~~~~~~~~~~~~~~~~~

Some functions in this module require a *positions array* to be
provided. A positions array is an array of integers corresponding to
the genomic positions of variants within a single chromosome or
contig. All values in a positions array must be given in increasing
order, i.e., it is assumed that variants are ordered according to
their genomic position.

**N.B., it is assumed that all positions are given using 1-based
coordinates.**

"""


from __future__ import absolute_import, print_function, division


import numpy as np
import numexpr as ne


from allel.compat import range
from allel.errors import ArgumentError


HAPLOID = 1
DIPLOID = 2
DIM_VARIANTS = 0
DIM_SAMPLES = 1
DIM_PLOIDY = 2
# packed representation of some common diploid genotypes
B00 = 0
B01 = 1
B10 = 16
B11 = 17
BMISSING = 239


def _check_genotype_array(g):

    # ensure we have a numpy array
    g = np.asarray(g)

    # check dimensionality
    if g.ndim == 2:
        # assume haploid
        ploidy = HAPLOID
    elif g.ndim == 3:
        ploidy = g.shape[2]
        if ploidy == HAPLOID:
            # drop empty ploidy dimension
            g = g[:, :, 0]
    else:
        raise ArgumentError('expected 2 or 3 dimensions, found %s' % g.ndim)

    return g, ploidy


def _check_haplotype_array(h):

    # ensure we have a numpy array
    h = np.asarray(h)

    # check dimensionality
    if h.ndim != 2:
        raise ArgumentError('expected 2 dimensions, found %s' % h.ndim)

    return h


def _check_boolean_array(b):

    # ensure we have a numpy boolean array
    b = np.asarray(b).view(dtype='b1')

    # check dimensionality
    if b.ndim not in {1, 2}:
        raise ArgumentError('expected 1 or 2 dimensions, found %s' % b.ndim)

    return b


def _check_pos_array(pos):

    # ensure we have a numpy array
    pos = np.asarray(pos)

    # check dimensionality
    if pos.ndim != 1:
        raise ArgumentError('expected one dimension, found %s' % pos.ndim)

    # check positions are sorted
    if np.any(np.diff(pos) < 0):
        raise ArgumentError('array is not sorted')

    return pos


def _check_axis(axis):
    if axis is None:
        return None
    elif axis == 'variants':
        return DIM_VARIANTS
    elif axis == 'samples':
        return DIM_SAMPLES
    elif axis == 'ploidy':
        return DIM_PLOIDY
    elif axis in {0, 1, 2}:
        return axis
    elif isinstance(axis, (list, tuple)):
        return tuple(_check_axis(a) for a in axis)
    else:
        raise ArgumentError('invalid axis: %r' % axis)


def _check_allele(allele):
    if allele is None:
        return None
    elif allele in {'ref', 'reference'}:
        return 0
    elif allele in {'alt', 'alternate'}:
        return 1
    elif isinstance(allele, int):
        return allele
    else:
        raise ArgumentError('invalid allele: %r' % allele)


def _check_alleles(alleles):
    if alleles is None:
        return None
    elif isinstance(alleles, (tuple, list)):
        return tuple(_check_allele(a) for a in alleles)
    else:
        raise ArgumentError('invalid alleles: %r' % alleles)


def _check_ploidy(ploidy):
    if ploidy is None:
        return None
    elif ploidy == 'haploid':
        return HAPLOID
    elif ploidy == 'diploid':
        return DIPLOID
    elif isinstance(ploidy, int) and ploidy > 0:
        return ploidy
    else:
        raise ArgumentError('invalid ploidy: %r' % ploidy)


def is_missing(g):
    """Find missing genotype calls.

    Parameters
    ----------

    g : array_like, int, shape (n_variants, n_samples, ploidy)
        Genotype array.

    Returns
    -------

    out : ndarray, bool, shape (n_variants, n_samples)
        Array where elements are True if the genotype call matches the
        condition.

    Examples
    --------

    >>> import allel
    >>> import numpy as np
    >>> g = np.array([[[0, 0], [0, 1]],
    ...               [[0, 1], [1, 1]],
    ...               [[0, 2], [-1, -1]]], dtype='i1')
    >>> allel.gt.is_missing(g)
    array([[False, False],
           [False, False],
           [False,  True]], dtype=bool)

    """

    # check input array
    g, ploidy = _check_genotype_array(g)

    # special case haploid
    if ploidy == HAPLOID:
        out = g < 0

    # special case diploid
    elif ploidy == DIPLOID:
        allele1 = g[..., 0]  # noqa
        allele2 = g[..., 1]  # noqa
        # call is missing if either allele is missing
        ex = '(allele1 < 0) | (allele2 < 0)'
        out = ne.evaluate(ex)

    # general ploidy case
    else:
        # call is missing if any allele is missing
        out = np.any(g < 0, axis=DIM_PLOIDY)

    return out


def is_called(g):
    """Find non-missing genotype calls.

    Parameters
    ----------

    g : array_like, int, shape (n_variants, n_samples, ploidy)
        Genotype array.

    Returns
    -------

    out : ndarray, bool, shape (n_variants, n_samples)
        Array where elements are True if the genotype call matches the
        condition.

    Examples
    --------

    >>> import allel
    >>> import numpy as np
    >>> g = np.array([[[0, 0], [0, 1]],
    ...               [[0, 1], [1, 1]],
    ...               [[0, 2], [-1, -1]]], dtype='i1')
    >>> allel.gt.is_called(g)
    array([[ True,  True],
           [ True,  True],
           [ True, False]], dtype=bool)

    """

    # check input array
    g, ploidy = _check_genotype_array(g)

    # special case haploid
    if ploidy == HAPLOID:
        out = g >= 0

    # special case diploid
    elif ploidy == DIPLOID:
        allele1 = g[..., 0]  # noqa
        allele2 = g[..., 1]  # noqa
        ex = '(allele1 >= 0) & (allele2 >= 0)'
        out = ne.evaluate(ex)

    # general ploidy case
    else:
        out = np.all(g >= 0, axis=DIM_PLOIDY)

    return out


def is_hom(g, allele=None):
    """Find genotype calls that are homozygous.

    Parameters
    ----------

    g : array_like, int, shape (n_variants, n_samples, ploidy)
        Genotype array.
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
    >>> import numpy as np
    >>> g = np.array([[[0, 0], [0, 1]],
    ...               [[0, 1], [1, 1]],
    ...               [[2, 2], [-1, -1]]], dtype='i1')
    >>> allel.gt.is_hom(g)
    array([[ True, False],
           [False,  True],
           [ True, False]], dtype=bool)
    >>> allel.gt.is_hom(g, allele=1)
    array([[False, False],
           [False,  True],
           [False, False]], dtype=bool)

    """

    # check inputs
    g, ploidy = _check_genotype_array(g)
    allele = _check_allele(allele)

    # special case haploid
    if ploidy == HAPLOID:
        if allele is None:
            # trivially true if non-missing
            out = g >= 0
        else:
            out = g == allele

    # special case diploid
    elif ploidy == DIPLOID:
        allele1 = g[..., 0]  # noqa
        allele2 = g[..., 1]  # noqa
        if allele is None:
            ex = '(allele1 >= 0) & (allele1  == allele2)'
        else:
            ex = '(allele1 == {0}) & (allele2 == {0})'.format(allele)
        out = ne.evaluate(ex)

    # general ploidy case
    else:
        if allele is None:
            allele1 = g[..., 0, None]  # noqa
            other_alleles = g[..., 1:]  # noqa
            ex = '(allele1 >= 0) & (allele1 == other_alleles)'
            out = np.all(ne.evaluate(ex), axis=DIM_PLOIDY)
        else:
            out = np.all(g == allele, axis=DIM_PLOIDY)

    return out


def is_hom_ref(g):
    """Find genotype calls that are homozygous for the reference allele.

    Parameters
    ----------

    g : array_like, int, shape (n_variants, n_samples, ploidy)
        Genotype array.

    Returns
    -------

    out : ndarray, bool, shape (n_variants, n_samples)
        Array where elements are True if the genotype call matches the
        condition.

    Examples
    --------

    >>> import allel
    >>> import numpy as np
    >>> g = np.array([[[0, 0], [0, 1]],
    ...               [[0, 1], [1, 1]],
    ...               [[2, 2], [-1, -1]]], dtype='i1')
    >>> allel.gt.is_hom_ref(g)
    array([[ True, False],
           [False, False],
           [False, False]], dtype=bool)

    """

    return is_hom(g, allele=0)


def is_hom_alt(g):
    """Find genotype calls that are homozygous for any alternate (i.e.,
    non-reference) allele.

    Parameters
    ----------

    g : array_like, int, shape (n_variants, n_samples, ploidy)
        Genotype array.

    Returns
    -------

    out : ndarray, bool, shape (n_variants, n_samples)
        Array where elements are True if the genotype call matches the
        condition.

    Examples
    --------

    >>> import allel
    >>> import numpy as np
    >>> g = np.array([[[0, 0], [0, 1]],
    ...               [[0, 1], [1, 1]],
    ...               [[2, 2], [-1, -1]]], dtype='i1')
    >>> allel.gt.is_hom_alt(g)
    array([[False, False],
           [False,  True],
           [ True, False]], dtype=bool)

    """

    # check input array
    g, ploidy = _check_genotype_array(g)

    # special case haploid
    if ploidy == HAPLOID:
        out = g > 0

    # special case diploid
    elif ploidy == DIPLOID:
        allele1 = g[..., 0]  # noqa
        allele2 = g[..., 1]  # noqa
        ex = '(allele1 > 0) & (allele1  == allele2)'
        out = ne.evaluate(ex)

    # general ploidy case
    else:
        allele1 = g[..., 0, None]  # noqa
        other_alleles = g[..., 1:]  # noqa
        ex = '(allele1 > 0) & (allele1 == other_alleles)'
        out = np.all(ne.evaluate(ex), axis=DIM_PLOIDY)

    return out


def is_het(g):
    """Find genotype calls that are heterozygous.

    Parameters
    ----------

    g : array_like, int, shape (n_variants, n_samples, ploidy)
        Genotype array.

    Returns
    -------

    out : ndarray, bool, shape (n_variants, n_samples)
        Array where elements are True if the genotype call matches the
        condition.

    Examples
    --------

    >>> import allel
    >>> import numpy as np
    >>> g = np.array([[[0, 0], [0, 1]],
    ...               [[0, 1], [1, 1]],
    ...               [[0, 2], [-1, -1]]], dtype='i1')
    >>> allel.gt.is_het(g)
    array([[False,  True],
           [ True, False],
           [ True, False]], dtype=bool)

    """

    # check input array
    g, ploidy = _check_genotype_array(g)

    # special case haploid
    if ploidy == HAPLOID:
        # trivially false
        out = np.zeros(g.shape[:2], dtype=np.bool)

    # special case diploid
    elif ploidy == DIPLOID:
        allele1 = g[..., 0]  # noqa
        allele2 = g[..., 1]  # noqa
        ex = '(allele1 >= 0) & (allele2  >= 0) & (allele1 != allele2)'
        out = ne.evaluate(ex)

    # general ploidy case
    else:
        allele1 = g[..., 0, None]  # noqa
        other_alleles = g[..., 1:]  # noqa
        out = np.all(g >= 0, axis=DIM_PLOIDY) \
            & np.any(allele1 != other_alleles, axis=DIM_PLOIDY)

    return out


def is_call(g, call):
    """Find genotypes with a given call.

    Parameters
    ----------

    g : array_like, int, shape (n_variants, n_samples, ploidy)
        Genotype array.
    call : int or array_like, int, shape (ploidy,)
        The genotype call to find.

    Returns
    -------

    out : ndarray, bool, shape (n_variants, n_samples)
        Array where elements are True if the genotype is `call`.

    Examples
    --------

    >>> import allel
    >>> import numpy as np
    >>> g = np.array([[[0, 0], [0, 1]],
    ...               [[0, 1], [1, 1]],
    ...               [[0, 2], [-1, -1]]], dtype='i1')
    >>> allel.gt.is_call(g, (0, 2))
    array([[False, False],
           [False, False],
           [ True, False]], dtype=bool)

    """

    # check input array
    g, ploidy = _check_genotype_array(g)

    # special case haploid
    if ploidy == HAPLOID:
        if not isinstance(call, int):
            raise ArgumentError('invalid call: %r' % call)
        out = g == call

    # special case diploid
    elif ploidy == DIPLOID:
        if not len(call) == DIPLOID:
            raise ArgumentError('invalid call: %r', call)
        allele1 = g[..., 0]  # noqa
        allele2 = g[..., 1]  # noqa
        ex = '(allele1 == {0}) & (allele2  == {1})'.format(*call)
        out = ne.evaluate(ex)

    # general ploidy case
    else:
        if not len(call) == ploidy:
            raise ArgumentError('invalid call: %r', call)
        call = np.asarray(call)[None, None, :]
        out = np.all(g == call, axis=DIM_PLOIDY)

    return out


################################
# Genotype array transformations
################################


def to_haplotypes(g):
    """Reshape a genotype array to view it as haplotypes by dropping the
    ploidy dimension.

    Parameters
    ----------

    g : array_like, int, shape (n_variants, n_samples, ploidy)
        Genotype array.

    Returns
    -------

    h : ndarray, int, shape (n_variants, n_samples * ploidy)
        Haplotype array.

    Notes
    -----

    If genotype calls are unphased, the haplotypes returned by this function
    will bear no resemblance to the true haplotypes.

    Examples
    --------

    >>> import allel
    >>> import numpy as np
    >>> g = np.array([[[0, 0], [0, 1]],
    ...               [[0, 1], [1, 1]],
    ...               [[0, 2], [-1, -1]]], dtype='i1')
    >>> allel.gt.to_haplotypes(g)
    array([[ 0,  0,  0,  1],
           [ 0,  1,  1,  1],
           [ 0,  2, -1, -1]], dtype=int8)

    """

    # check input array
    g, ploidy = _check_genotype_array(g)

    # special case haploid
    if ploidy == HAPLOID:
        h = g

    else:
        # reshape, preserving size of variants dimension
        newshape = (g.shape[DIM_VARIANTS], -1)
        h = np.reshape(g, newshape)

    return h


def from_haplotypes(h, ploidy):
    """Reshape a haplotype array to view it as genotypes by restoring the
    ploidy dimension.

    Parameters
    ----------

    h : array_like, int, shape (n_variants, n_samples * ploidy)
        Haplotype array.
    ploidy : int
        The sample ploidy.

    Returns
    -------

    g : ndarray, int, shape (n_variants, n_samples, ploidy)
        Genotype array.

    Examples
    --------

    >>> import allel
    >>> import numpy as np
    >>> h = np.array([[0, 0, 0, 1],
    ...               [0, 1, 1, 1],
    ...               [0, 2, -1, -1]], dtype='i1')
    >>> allel.gt.from_haplotypes(h, ploidy=2)
    array([[[ 0,  0],
            [ 0,  1]],
           [[ 0,  1],
            [ 1,  1]],
           [[ 0,  2],
            [-1, -1]]], dtype=int8)

    """

    # check inputs
    h = _check_haplotype_array(h)
    ploidy = _check_ploidy(ploidy)

    # special case haploid
    if ploidy == HAPLOID:
        # noop
        g = h

    else:
        # reshape
        newshape = (h.shape[0], -1, ploidy)
        g = h.reshape(newshape)

    return g


def to_n_alt(g, fill=0):
    """Transform each genotype call into the number of non-reference alleles.

    Parameters
    ----------

    g : array_like, int, shape (n_variants, n_samples, ploidy)
        Genotype array.
    fill : int, optional
        Use this value to represent missing calls.

    Returns
    -------

    out : ndarray, int, shape (n_variants, n_samples)
        Array of non-ref alleles per genotype call.

    Notes
    -----

    This function simply counts the number of non-reference alleles,
    it makes no distinction between different alternate alleles.

    By default this function returns 0 for missing genotype calls **and** for
    homozygous reference genotype calls. Use the `fill` argument to change
    how missing calls are represented.

    Examples
    --------

    >>> import allel
    >>> import numpy as np
    >>> g = np.array([[[0, 0], [0, 1]],
    ...               [[0, 2], [1, 1]],
    ...               [[2, 2], [-1, -1]]], dtype='i1')
    >>> allel.gt.to_n_alt(g)
    array([[0, 1],
           [1, 2],
           [2, 0]], dtype=int8)
    >>> allel.gt.to_n_alt(g, fill=-1)
    array([[ 0,  1],
           [ 1,  2],
           [ 2, -1]], dtype=int8)

    """

    # check input array
    g, ploidy = _check_genotype_array(g)

    # special case haploid
    if ploidy == HAPLOID:
        out = np.asarray(g > 0).view(dtype='i1')

    else:
        # count number of alternate alleles
        out = np.empty(g.shape[:-1], dtype='i1')
        np.sum(g > 0, axis=DIM_PLOIDY, out=out)

    if fill != 0:
        m = is_missing(g)
        out[m] = fill

    return out


def to_allele_counts(g, alleles=None):
    """Transform genotype calls into allele counts per call.

    Parameters
    ----------

    g : array_like, int, shape (n_variants, n_samples, ploidy)
        Genotype array.
    alleles : sequence of ints, optional
        If not None, count only the given alleles. (By default, count all
        possible alleles.)

    Returns
    -------

    out : ndarray, uint8, shape (n_variants, n_samples, len(alleles))
        Array of allele counts per call.

    Examples
    --------

    >>> import allel
    >>> import numpy as np
    >>> g = np.array([[[0, 0], [0, 1]],
    ...               [[0, 2], [1, 1]],
    ...               [[2, 2], [-1, -1]]], dtype='i1')
    >>> allel.gt.to_allele_counts(g)
    array([[[2, 0, 0],
            [1, 1, 0]],
           [[1, 0, 1],
            [0, 2, 0]],
           [[0, 0, 2],
            [0, 0, 0]]], dtype=uint8)
    >>> allel.gt.to_allele_counts(g, alleles=(0, 1))
    array([[[2, 0],
            [1, 1]],
           [[1, 0],
            [0, 2]],
           [[0, 0],
            [0, 0]]], dtype=uint8)

    """

    # check inputs
    g, ploidy = _check_genotype_array(g)
    if alleles is None:
        m = np.amax(g)
        alleles = list(range(m+1))

    # set up output array
    outshape = g.shape[:2] + (len(alleles),)
    out = np.zeros(outshape, dtype='u1')

    # special case haploid
    if ploidy == HAPLOID:
        for i, allele in enumerate(alleles):
            out[..., i] = g == allele

    else:
        for i, allele in enumerate(alleles):
            # count alleles along ploidy dimension
            np.sum(g == allele, axis=DIM_PLOIDY, out=out[..., i])

    return out


def to_packed(g, boundscheck=True):
    """Pack diploid genotypes into a single byte for each genotype,
    using the left-most 4 bits for the first allele and the right-most 4 bits
    for the second allele. Allows single byte encoding of diploid genotypes
    for variants with up to 15 alleles.

    Parameters
    ----------

    g : array_like, int, shape (n_variants, n_samples, 2)
        Genotype array.
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
    >>> import numpy as np
    >>> g = np.array([[[0, 0], [0, 1]],
    ...               [[0, 2], [1, 1]],
    ...               [[2, 2], [-1, -1]]], dtype='i1')
    >>> allel.gt.to_packed(g)
    array([[  0,   1],
           [  2,  17],
           [ 34, 239]], dtype=uint8)

    """

    # check inputs
    g, ploidy = _check_genotype_array(g)
    if ploidy != 2:
        raise ArgumentError('unexpected ploidy: %s' % ploidy)

    if boundscheck:
        amx = np.amax(g)
        if amx > 14:
            raise ArgumentError('max allele for packing is 14, found %s' % amx)
        amn = np.amin(g)
        if amn < -1:
            raise ArgumentError('min allele for packing is -1, found %s' % amn)

    from allel.opt.gt import pack_diploid
    packed = pack_diploid(g.view(dtype='i1'))

    return packed


def from_packed(packed):
    """Unpack diploid genotypes that have been bit-packed into single bytes.

    Parameters
    ----------

    packed : ndarray, uint8, shape (n_variants, n_samples)
        Bit-packed diploid genotype array.

    Returns
    -------

    g : array_like, int, shape (n_variants, n_samples, 2)
        Genotype array.

    Examples
    --------

    >>> import allel
    >>> import numpy as np
    >>> packed = np.array([[0, 1],
    ...                    [2, 17],
    ...                    [34, 239]], dtype='u1')
    >>> allel.gt.from_packed(packed)
    array([[[ 0,  0],
            [ 0,  1]],
           [[ 0,  2],
            [ 1,  1]],
           [[ 2,  2],
            [-1, -1]]], dtype=int8)

    """

    # check inputs
    packed = np.asarray(packed)
    if packed.ndim != 2:
        raise ArgumentError('expected 2 dimensions, found: %s' % packed.ndim)

    from allel.opt.gt import unpack_diploid
    g = unpack_diploid(packed.view(dtype='u1'))

    return g


# TODO to_csr
# TODO from_csr
# TODO to_csc
# TODO from_csc


###############################
# Allele frequency calculations
###############################


def max_allele(g, axis=None):
    """Return the highest allele index.

    Parameters
    ----------

    g : array_like, int, shape (n_variants, n_samples, ploidy)
        Genotype array.

    Returns
    -------

    out : int or ndarray, int
        The highest allele index.

    Examples
    --------

    >>> import allel
    >>> import numpy as np
    >>> g = np.array([[[0, 0], [0, 1]],
    ...               [[0, 2], [1, 1]],
    ...               [[2, 2], [-1, -1]]], dtype='i1')
    >>> allel.gt.max_allele(g)
    2
    >>> allel.gt.max_allele(g, axis=(0, 2))
    array([2, 1], dtype=int8)
    >>> allel.gt.max_allele(g, axis=(1, 2))
    array([1, 2, 2], dtype=int8)

    """

    # check inputs
    g, _ = _check_genotype_array(g)
    axis = _check_axis(axis)

    return np.amax(g, axis=axis)


def allelism(g):
    """Determine the number of distinct alleles for each variant.

    Parameters
    ----------

    g : array_like, int, shape (n_variants, n_samples, ploidy)
        Genotype array.

    Returns
    -------

    n : ndarray, int, shape (n_variants,)
        Allelism array.

    Examples
    --------

    >>> import allel
    >>> import numpy as np
    >>> g = np.array([[[0, 0], [0, 1]],
    ...               [[0, 2], [1, 1]],
    ...               [[2, 2], [-1, -1]]], dtype='i1')
    >>> allel.gt.allelism(g)
    array([2, 3, 1])

    """

    # calculate allele counts
    ac = allele_counts(g)

    # count alleles present
    n = np.sum(ac > 0, axis=1)

    return n


def allele_number(g):
    """Count the number of non-missing allele calls per variant.

    Parameters
    ----------

    g : array_like, int, shape (n_variants, n_samples, ploidy)
        Genotype array.

    Returns
    -------

    an : ndarray, int, shape (n_variants,)
        Allele number array.

    Examples
    --------

    >>> import allel
    >>> import numpy as np
    >>> g = np.array([[[0, 0], [0, 1]],
    ...               [[0, 2], [1, 1]],
    ...               [[2, 2], [-1, -1]]], dtype='i1')
    >>> allel.gt.allele_number(g)
    array([4, 4, 2])

    """

    # transform
    h = to_haplotypes(g)

    # count non-missing calls over samples
    an = np.sum(h >= 0, axis=1)

    return an


def allele_count(g, allele=1):
    """Count the number of calls of the given allele per variant.

    Parameters
    ----------

    g : array_like, int, shape (n_variants, n_samples, ploidy)
        Genotype array.
    allele : int, optional
        Allele index.

    Returns
    -------

    ac : ndarray, int, shape (n_variants,)
        Allele count array.

    Examples
    --------

    >>> import allel
    >>> import numpy as np
    >>> g = np.array([[[0, 0], [0, 1]],
    ...               [[0, 2], [1, 1]],
    ...               [[2, 2], [-1, -1]]], dtype='i1')
    >>> allel.gt.allele_count(g, allele=1)
    array([1, 2, 0])
    >>> allel.gt.allele_count(g, allele=2)
    array([0, 1, 2])

    """

    # check inputs
    g, ploidy = _check_genotype_array(g)
    allele = _check_allele(allele)

    # transform
    h = to_haplotypes(g)

    # count non-missing calls over samples
    return np.sum(h == allele, axis=1)


def allele_frequency(g, allele=1, fill=0):
    """Calculate the frequency of the given allele per variant.

    Parameters
    ----------

    g : array_like, int, shape (n_variants, n_samples, ploidy)
        Genotype array.
    allele : int, optional
        Allele index.
    fill : int, optional
        The value to use where all genotype calls are missing for a variant.

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
    >>> import numpy as np
    >>> g = np.array([[[0, 0], [0, 1]],
    ...               [[0, 2], [1, 1]],
    ...               [[2, 2], [-1, -1]]], dtype='i1')
    >>> af, ac, an = allel.gt.allele_frequency(g, allele=1)
    >>> af
    array([ 0.25,  0.5 ,  0.  ])
    >>> af, ac, an = allel.gt.allele_frequency(g, allele=2)
    >>> af
    array([ 0.  ,  0.25,  1.  ])

    """

    # check inputs
    g, ploidy = _check_genotype_array(g)
    allele = _check_allele(allele)

    # intermediate variables
    an = allele_number(g)
    ac = allele_count(g, allele=allele)

    # calculate allele frequency, accounting for variants with no allele calls
    err = np.seterr(invalid='ignore')
    af = np.where(an > 0, ac / an, fill)
    np.seterr(**err)

    return af, ac, an


def allele_counts(g, alleles=None):
    """Count the number of calls of each allele per variant.

    Parameters
    ----------

    g : array_like, int, shape (n_variants, n_samples, ploidy)
        Genotype array.
    alleles : sequence of ints, optional
        The alleles to count. If None, all alleles will be counted.

    Returns
    -------

    ac : ndarray, int, shape (n_variants, len(alleles))
        Allele counts array.

    Examples
    --------

    >>> import allel
    >>> import numpy as np
    >>> g = np.array([[[0, 0], [0, 1]],
    ...               [[0, 2], [1, 1]],
    ...               [[2, 2], [-1, -1]]], dtype='i1')
    >>> allel.gt.allele_counts(g)
    array([[3, 1, 0],
           [1, 2, 1],
           [0, 0, 2]], dtype=int32)
    >>> allel.gt.allele_counts(g, alleles=(1, 2))
    array([[1, 0],
           [2, 1],
           [0, 2]], dtype=int32)

    """

    # check inputs
    g, ploidy = _check_genotype_array(g)
    alleles = _check_alleles(alleles)

    # transform
    h = to_haplotypes(g)

    # determine number of variants
    n_variants = h.shape[0]

    # if alleles not specified, count all alleles
    if alleles is None:
        m = np.amax(h)
        alleles = list(range(m+1))

    # set up output array
    ac = np.zeros((n_variants, len(alleles)), dtype='i4')

    # count alleles
    for i, allele in enumerate(alleles):
        np.sum(h == allele, axis=1, out=ac[:, i])

    return ac


def allele_frequencies(g, alleles=None, fill=0):
    """Calculate the frequency of each allele per variant.

    Parameters
    ----------

    g : array_like, int, shape (n_variants, n_samples, ploidy)
        Genotype array.
    alleles : sequence of ints, optional
        The alleles to calculate frequency of. If None, all allele frequencies
        will be calculated.
    fill : int, optional
        The value to use where all genotype calls are missing for a variant.

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
    >>> import numpy as np
    >>> g = np.array([[[0, 0], [0, 1]],
    ...               [[0, 2], [1, 1]],
    ...               [[2, 2], [-1, -1]]], dtype='i1')
    >>> af, ac, an = allel.gt.allele_frequencies(g)
    >>> af
    array([[ 0.75,  0.25,  0.  ],
           [ 0.25,  0.5 ,  0.25],
           [ 0.  ,  0.  ,  1.  ]])
    >>> af, ac, an = allel.gt.allele_frequencies(g, alleles=(1, 2))
    >>> af
    array([[ 0.25,  0.  ],
           [ 0.5 ,  0.25],
           [ 0.  ,  1.  ]])

    """

    # check inputs
    g, ploidy = _check_genotype_array(g)
    alleles = _check_alleles(alleles)

    # intermediate variables
    an = allele_number(g)[:, None]
    ac = allele_counts(g, alleles=alleles)

    # calculate allele frequency, accounting for variants with no allele calls
    err = np.seterr(invalid='ignore')
    af = np.where(an > 0, ac / an, fill)
    np.seterr(**err)

    return af, ac, an[:, 0]


def is_variant(g):
    """Find variants with at least one non-reference allele call.

    Parameters
    ----------

    g : array_like, int, shape (n_variants, n_samples, ploidy)
        Genotype array.

    Returns
    -------

    out : ndarray, bool, shape (n_variants,)
        Boolean array where elements are True if variant matches the condition.

    Examples
    --------

    >>> import allel
    >>> import numpy as np
    >>> g = np.array([[[0, 0], [0, 0]],
    ...               [[0, 0], [0, 1]],
    ...               [[0, 2], [1, 1]],
    ...               [[2, 2], [-1, -1]]], dtype='i1')
    >>> allel.gt.is_variant(g)
    array([False,  True,  True,  True], dtype=bool)

    """

    # check inputs
    g, ploidy = _check_genotype_array(g)

    # transform
    h = to_haplotypes(g)

    # find variants with at least 1 non-reference allele
    out = np.sum(h > 0, axis=1) >= 1

    return out


def is_non_variant(g):
    """Find variants with no non-reference allele calls.

    Parameters
    ----------

    g : array_like, int, shape (n_variants, n_samples, ploidy)
        Genotype array.

    Returns
    -------

    out : ndarray, bool, shape (n_variants,)
        Boolean array where elements are True if variant matches the condition.

    Examples
    --------

    >>> import allel
    >>> import numpy as np
    >>> g = np.array([[[0, 0], [0, 0]],
    ...               [[0, 0], [0, 1]],
    ...               [[0, 2], [1, 1]],
    ...               [[2, 2], [-1, -1]]], dtype='i1')
    >>> allel.gt.is_non_variant(g)
    array([ True, False, False, False], dtype=bool)

    """

    # check inputs
    g, ploidy = _check_genotype_array(g)

    # transform
    h = to_haplotypes(g)

    # find variants with no non-reference alleles
    out = np.all(h <= 0, axis=1)

    return out


def is_segregating(g):
    """Find segregating variants (where more than one allele is observed).

    Parameters
    ----------

    g : array_like, int, shape (n_variants, n_samples, ploidy)
        Genotype array.

    Returns
    -------

    out : ndarray, bool, shape (n_variants,)
        Boolean array where elements are True if variant matches the condition.

    Examples
    --------

    >>> import allel
    >>> import numpy as np
    >>> g = np.array([[[0, 0], [0, 0]],
    ...               [[0, 0], [0, 1]],
    ...               [[1, 1], [1, 2]],
    ...               [[2, 2], [-1, -1]]], dtype='i1')
    >>> allel.gt.is_segregating(g)
    array([False,  True,  True, False], dtype=bool)

    """

    # check inputs
    g, ploidy = _check_genotype_array(g)

    # count distinct alleles
    n_alleles = allelism(g)

    # find segregating variants
    out = n_alleles > 1

    return out


def is_non_segregating(g, allele=None):
    """Find non-segregating variants (where at most one allele is observed).

    Parameters
    ----------

    g : array_like, int, shape (n_variants, n_samples, ploidy)
        Genotype array.
    allele : int, optional
        Allele index.

    Returns
    -------

    out : ndarray, bool, shape (n_variants,)
        Boolean array where elements are True if variant matches the condition.

    Examples
    --------

    >>> import allel
    >>> import numpy as np
    >>> g = np.array([[[0, 0], [0, 0]],
    ...               [[0, 0], [0, 1]],
    ...               [[1, 1], [1, 2]],
    ...               [[2, 2], [-1, -1]]], dtype='i1')
    >>> allel.gt.is_non_segregating(g)
    array([ True, False, False,  True], dtype=bool)

    """

    # check inputs
    g, ploidy = _check_genotype_array(g)
    allele = _check_allele(allele)

    if allele is None:

        # count distinct alleles
        n_alleles = allelism(g)

        # find fixed variants
        out = n_alleles <= 1

    else:

        # transform
        h = to_haplotypes(g)

        # find fixed variants with respect to a specific allele
        out = np.all((h < 0) | (h == allele), axis=1)

    return out


def is_singleton(g, allele=1):
    """Find variants with a single call for the given allele.

    Parameters
    ----------

    g : array_like, int, shape (n_variants, n_samples, ploidy)
        Genotype array.
    allele : int, optional
        Allele index.

    Returns
    -------

    out : ndarray, bool, shape (n_variants,)
        Boolean array where elements are True if variant matches the condition.

    Examples
    --------

    >>> import allel
    >>> import numpy as np
    >>> g = np.array([[[0, 0], [0, 0]],
    ...               [[0, 0], [0, 1]],
    ...               [[1, 1], [1, 2]],
    ...               [[2, 2], [-1, -1]]], dtype='i1')
    >>> allel.gt.is_singleton(g, allele=1)
    array([False,  True, False, False], dtype=bool)
    >>> allel.gt.is_singleton(g, allele=2)
    array([False, False,  True, False], dtype=bool)

    """

    # check inputs
    g, ploidy = _check_genotype_array(g)
    allele = _check_allele(allele)

    # count allele
    ac = allele_count(g, allele=allele)

    # find singletons
    out = ac == 1

    return out


def is_doubleton(g, allele=1):
    """Find variants with exactly two calls for the given allele.

    Parameters
    ----------

    g : array_like, int, shape (n_variants, n_samples, ploidy)
        Genotype array.
    allele : int, optional
        Allele index.

    Returns
    -------

    out : ndarray, bool, shape (n_variants,)
        Boolean array where elements are True if variant matches the condition.

    Examples
    --------

    >>> import allel
    >>> import numpy as np
    >>> g = np.array([[[0, 0], [0, 0]],
    ...               [[0, 0], [1, 1]],
    ...               [[1, 1], [1, 2]],
    ...               [[2, 2], [-1, -1]]], dtype='i1')
    >>> allel.gt.is_doubleton(g, allele=1)
    array([False,  True, False, False], dtype=bool)
    >>> allel.gt.is_doubleton(g, allele=2)
    array([False, False, False,  True], dtype=bool)

    """

    # check inputs
    g, ploidy = _check_genotype_array(g)
    allele = _check_allele(allele)

    # count allele
    ac = allele_count(g, allele=allele)

    # find doubletons
    out = ac == 2

    return out


####################
# Counting functions
####################


def count(b, axis=None):
    """Count nonzero (i.e., True) elements, optionally along the given axis.

    Parameters
    ----------

    b : array_like, bool, shape (n_variants, n_samples) or (n_variants,)
        Boolean array.
    axis : int, optional
        If not None, count along the given axis (0=variants, 1=samples).

    Returns
    -------

    count : int or ndarray, int
        Number of nonzero elements.

    Examples
    --------

    >>> import allel
    >>> import numpy as np
    >>> g = np.array([[[0, 0], [0, 0]],
    ...               [[0, 0], [1, 1]],
    ...               [[1, 1], [1, 2]],
    ...               [[2, 2], [-1, -1]]], dtype='i1')
    >>> b = allel.gt.is_called(g)
    >>> allel.gt.count(b)
    7
    >>> allel.gt.count(b, axis='variants')
    array([4, 3])
    >>> allel.gt.count(b, axis='samples')
    array([2, 2, 2, 1])
    >>> b = allel.gt.is_variant(g)
    >>> allel.gt.count(b)
    3

    """

    # check inputs
    b = _check_boolean_array(b)
    axis = _check_axis(axis)

    return np.sum(b, axis=axis)


def windowed_count(pos, b, window, start=None, stop=None):
    """Count nonzero (i.e., True) elements in non-overlapping windows over a
    single chromosome or contig.

    Parameters
    ----------

    pos : array_like, int, shape (n_variants,)
        Positions array.
    b : array_like, bool, shape (n_variants,) or (n_variants, n_samples)
        Boolean array.
    window : int
        Window size.
    start : int, optional
        Start position.
    stop : int, optional
        Stop position.

    Returns
    -------

    counts : ndarray, int, shape (n_bins,) or (n_bins, n_samples)
        Counts array.
    bin_edges : ndarray, int, shape (n_bins,)
        Bin edges used for counting.

    Examples
    --------

    >>> import allel
    >>> import numpy as np
    >>> g = np.array([[[0, 0], [0, 0]],
    ...               [[0, 1], [0, 1]],
    ...               [[1, 1], [1, 2]],
    ...               [[2, 2], [-1, -1]]], dtype='i1')
    >>> pos = np.array([2, 14, 15, 27])
    >>> b = allel.gt.is_variant(g)
    >>> counts, bin_edges = allel.gt.windowed_count(pos, b, window=10)
    >>> bin_edges
    array([ 2, 12, 22, 32])
    >>> counts
    array([0, 2, 1])
    >>> counts, bin_edges = allel.gt.windowed_count(pos, b, window=10,
    ...                                             start=1,
    ...                                             stop=27)
    >>> bin_edges
    array([ 1, 11, 21, 27])
    >>> counts
    array([0, 2, 1])
    >>> b = allel.gt.is_het(g)
    >>> counts, bin_edges = allel.gt.windowed_count(pos, b, window=10)
    >>> bin_edges
    array([ 2, 12, 22, 32])
    >>> counts
    array([[0, 0],
           [1, 2],
           [0, 0]])

    """

    # check inputs
    pos = _check_pos_array(pos)
    b = _check_boolean_array(b)
    if pos.shape[0] != b.shape[0]:
        raise ArgumentError('arrays do not have matching length for first '
                            'dimension: pos %s, b %s' % (pos.shape[0],
                                                         b.shape[0]))

    # determine bin edges
    bin_start = np.amin(pos) if start is None else start
    bin_stop = np.amax(pos) if stop is None else stop
    bin_edges = np.arange(bin_start, bin_stop, window)
    if stop is None and bin_edges[-1] < bin_stop:
        # add one more window to ensure stop is included
        bin_edges = np.append(bin_edges, bin_edges[-1] + window)
    elif stop is not None and bin_edges[-1] < bin_stop:
        # add one more window to ensure explicit stop is final edge
        bin_edges = np.append(bin_edges, stop)

    if b.ndim == 1:
        pos_incl = np.compress(b, pos)
        counts, _ = np.histogram(pos_incl, bins=bin_edges)

    else:
        # assume 2D
        n_bins = len(bin_edges) - 1
        n_samples = b.shape[1]
        counts = np.empty((n_bins, n_samples), dtype=int)
        for i in range(n_samples):
            pos_incl = np.compress(b[:, i], pos)
            h, _ = np.histogram(pos_incl, bins=bin_edges)
            counts[:, i] = h

    return counts, bin_edges


def windowed_density(pos, b, window, start=None, stop=None,
                     is_accessible=None, fill=0):
    """Calculate the per-base-pair density of nonzero (i.e., True) elements in
    non-overlapping windows over a single chromosome or contig.

    Parameters
    ----------

    pos : array_like, int, shape (n_variants,)
        Positions array.
    b : array_like, bool, shape (n_variants,) or (n_variants, n_samples)
        Boolean array.
    window : int
        Window size.
    start : int, optional
        Start position.
    stop : int, optional
        Stop position.
    is_accessible : array_like, bool, shape (len(contig),), optional
        Accessibility mask. If provided, the size of each window will be
        calculated as the number of accessible positions, rather than the
        window size.

    Returns
    -------

    densities : ndarray, int, shape (n_bins,) or (n_bins, n_samples)
        Densities array.
    counts : ndarray, int, shape (n_bins,) or (n_bins, n_samples)
        Counts array.
    bin_edges : ndarray, int, shape (n_bins,)
        Bin edges used for counting.

    Examples
    --------

    >>> import allel
    >>> import numpy as np
    >>> g = np.array([[[0, 0], [0, 0]],
    ...               [[0, 1], [0, 1]],
    ...               [[1, 1], [1, 2]],
    ...               [[2, 2], [-1, -1]]], dtype='i1')
    >>> pos = np.array([1, 14, 15, 27])
    >>> b = allel.gt.is_variant(g)
    >>> densities, counts, bin_edges = allel.gt.windowed_density(pos, b,
    ...                                                          window=10)
    >>> bin_edges
    array([ 1, 11, 21, 31])
    >>> counts
    array([0, 2, 1])
    >>> densities
    array([ 0.        ,  0.2       ,  0.09090909])
    >>> # Density calculations can take into account the number of accessible
    ... # positions within each window, e.g.:
    ... is_accessible = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...                           1, 1, 1, 1, 0, 0, 1, 1, 0, 0,
    ...                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=bool)
    >>> densities, counts, bin_edges = allel.gt.windowed_density(
    ...     pos, b, window=10, is_accessible=is_accessible, fill=np.nan
    ... )
    >>> bin_edges
    array([ 1, 11, 21, 31])
    >>> counts
    array([0, 2, 1])
    >>> densities
    array([        nan,  0.33333333,  0.09090909])

    """

    counts, bin_edges = windowed_count(pos, b, window, start=start, stop=stop)

    # determine window sizes (i.e., bin widths)
    if is_accessible is None:
        # assume all genome positions are accessible
        bin_widths = np.diff(bin_edges)
        # final bin includes right edge
        bin_widths[-1] += 1
    else:
        is_accessible = np.asarray(is_accessible)
        if is_accessible.ndim != 1:
            raise ArgumentError('expected 1 dimension, found %s' %
                                is_accessible.ndim)
        pos_accessible, = np.nonzero(is_accessible)
        # convert to 1-based coordinates
        pos_accessible += 1
        bin_widths, _ = np.histogram(pos_accessible, bins=bin_edges)

    if counts.ndim > 1:
        bin_widths = bin_widths[:, None]

    # calculate densities
    err = np.seterr(invalid='ignore')
    densities = np.where(bin_widths > 0, counts / bin_widths, fill)
    np.seterr(**err)

    return densities, counts, bin_edges


####################
# Plotting functions
####################


def plot_discrete_calldata():
    """TODO

    """
    pass


def plot_continuous_calldata():
    """TODO

    """
    pass


def plot_diploid_genotypes():
    """TODO

    """
    pass


def plot_genotype_counts_by_sample():
    """TODO

    """
    pass


def plot_genotype_counts_by_variant():
    """TODO

    """
    pass


def plot_continuous_calldata_by_sample():
    """TODO

    """
    pass


def plot_windowed_call_count():
    """TODO

    """
    pass


def plot_windowed_call_density():
    """TODO

    """
    pass
