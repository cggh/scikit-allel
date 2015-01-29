# -*- coding: utf-8 -*-
"""
This module provides functions for manipulating arrays of discrete genotype
calls.

Conventions
-----------

By convention, an array of discrete genotype calls for a set of
samples at a set of variants is canonically represented as a
3-dimensional numpy array of integers.  The first dimension
corresponds to the variants genotyped (i.e., the length of the first
dimension is equal to the number of variants), the second dimension
corresponds to the samples genotyped (i.e., the length of the second
dimension equals the number of samples), and the third dimension
corresponds to the ploidy of the samples.

For example, the array `g` below is a discrete genotype array storing
genotype calls for 3 variants in 2 samples with ploidy 2 (i.e.,
diploid)::

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

Each integer within the array corresponds to an allele call, where 0
is the reference allele, 1 is the first alternate allele, 2 is the
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

The canonical genotype array can store genotype calls with any ploidy. For
example, the array `g_triploid` below has triploid calls for 2 samples at
3 variants::

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

"""


from __future__ import absolute_import, print_function, division


import numpy as np
import numexpr as ne


from allel.errors import ArgumentError


HAPLOID = 1
DIPLOID = 2
DIM_VARIANTS = 0
DIM_SAMPLES = 1
DIM_PLOIDY = 2


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


def is_missing(g):
    """Find missing genotype calls.

    Parameters
    ----------

    g : array_like, int, shape (n_variants, n_samples, ploidy)
        Genotype array.

    Returns
    -------

    out : ndarray, bool, shape (n_variants, n_samples)
        Array where elements are True if the genotype call is missing.

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
        Array where elements are True if the genotype call is non-missing.

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
        If not None, find calls that are homozygous for the given allele.

    Returns
    -------

    out : ndarray, bool, shape (n_variants, n_samples)
        Array where elements are True if the genotype call is homozygous.

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

    # check input array
    g, ploidy = _check_genotype_array(g)

    # special case haploid
    if ploidy == HAPLOID:
        if allele is None:
            # trivially true
            out = np.ones(g.shape, dtype=np.bool)
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
        Array where elements are True if the genotype call is homozygous for
        the reference allele.

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
        Array where elements are True if the genotype call is homozygous for
        an alternate allele.

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
        Array where elements are True if the genotype call is heterozygous.

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
    call : int (haploid) or array_like, int, shape (ploidy,)
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


def _check_axis(axis):
    if axis is None:
        return None
    elif axis == 'variants':
        return DIM_VARIANTS
    elif axis == 'samples':
        return DIM_SAMPLES
    elif axis in {0, 1}:
        return axis
    else:
        raise ArgumentError('unexpected axis: %r' % axis)


def _count_true(a, axis):
    axis = _check_axis(axis)
    return np.sum(a, axis=axis)


def count_missing(g, axis=None):
    """Count missing genotype calls.

    Parameters
    ----------

    g : array_like, int, shape (n_variants, n_samples, ploidy)
        Genotype array.
    axis : int, optional
        If not None, sum over the given axis (0=variants, 1=samples).

    Returns
    -------

    count : int or ndarray
        Number of matching genotype calls.

    """

    return _count_true(is_missing(g), axis)


def count_called(g, axis=None):
    """Count non-missing genotype calls.

    Parameters
    ----------

    g : array_like, int, shape (n_variants, n_samples, ploidy)
        Genotype array.
    axis : int, optional
        If not None, sum over the given axis (0=variants, 1=samples).

    Returns
    -------

    count : int or ndarray
        Number of matching genotype calls.

    """

    return _count_true(is_called(g), axis)


def count_hom(g, allele=None, axis=None):
    """Count homozygous genotype calls.

    Parameters
    ----------

    g : array_like, int, shape (n_variants, n_samples, ploidy)
        Genotype array.
    allele : int, optional
        If not None, find calls that are homozygous for the given allele.
    axis : int, optional
        If not None, sum over the given axis (0=variants, 1=samples).

    Returns
    -------

    count : int or ndarray
        Number of matching genotype calls.

    """

    return _count_true(is_hom(g, allele=allele), axis)


def count_hom_ref(g, axis=None):
    """Count homozygous reference genotype calls.

    Parameters
    ----------

    g : array_like, int, shape (n_variants, n_samples, ploidy)
        Genotype array.
    axis : int, optional
        If not None, sum over the given axis (0=variants, 1=samples).

    Returns
    -------

    count : int or ndarray
        Number of matching genotype calls.

    """

    return _count_true(is_hom_ref(g), axis)


def count_hom_alt(g, axis=None):
    """Count homozygous alternate genotype calls.

    Parameters
    ----------

    g : array_like, int, shape (n_variants, n_samples, ploidy)
        Genotype array.
    axis : int, optional
        If not None, sum over the given axis (0=variants, 1=samples).

    Returns
    -------

    count : int or ndarray
        Number of matching genotype calls.

    """

    return _count_true(is_hom_alt(g), axis)


def count_het(g, axis=None):
    """Count heterozygous genotype calls.

    Parameters
    ----------

    g : array_like, int, shape (n_variants, n_samples, ploidy)
        Genotype array.
    axis : int, optional
        If not None, sum over the given axis (0=variants, 1=samples).

    Returns
    -------

    count : int or ndarray
        Number of matching genotype calls.

    """

    return _count_true(is_het(g), axis)


def count_call(g, call, axis=None):
    """Count genotype calls.

    Parameters
    ----------

    g : array_like, int, shape (n_variants, n_samples, ploidy)
        Genotype array.
    call : int (haploid) or array_like, int, shape (ploidy,)
        The genotype call to find.
    axis : int, optional
        If not None, sum over the given axis (0=variants, 1=samples).

    Returns
    -------

    count : int or ndarray
        Number of matching genotype calls.

    """

    return _count_true(is_call(g, call), axis)


################################
# Genotype array transformations
################################


def as_haplotypes(g):
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
    >>> allel.gt.as_haplotypes(g)
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


def as_n_alt(g):
    """Transform each genotype call into the number of non-reference alleles.

    Parameters
    ----------

    g : array_like, int, shape (n_variants, n_samples, ploidy)
        Genotype array.

    Returns
    -------

    out : ndarray, int, shape (n_variants, n_samples)
        Array of non-ref alleles per genotype call.

    Notes
    -----

    This function simply counts the number of non-reference alleles,
    it makes no distinction between different alternate alleles.

    This function returns 0 for missing genotype calls **and** for
    homozygous reference genotype calls, because in both cases the number of
    non-reference alleles is zero.

    Examples
    --------

    >>> import allel
    >>> import numpy as np
    >>> g = np.array([[[0, 0], [0, 1]],
    ...               [[0, 2], [1, 1]],
    ...               [[2, 2], [-1, -1]]], dtype='i1')
    >>> allel.gt.as_n_alt(g)
    array([[0, 1],
           [1, 2],
           [2, 0]], dtype=uint8)

    """

    # check input array
    g, ploidy = _check_genotype_array(g)

    # special case haploid
    if ploidy == HAPLOID:
        out = (g > 0).astype('u1')

    else:
        # count number of alternate alleles
        out = np.empty(g.shape[:-1], dtype='u1')
        np.sum(g > 0, axis=DIM_PLOIDY, out=out)

    return out


def as_012(g):
    """TODO

    """
    pass


def as_allele_counts(g):
    """TODO

    """
    pass


def pack_diploid(g):
    """TODO

    """
    pass


def unpack_diploid(g):
    """TODO

    """
    pass


###############################
# Allele frequency calculations
###############################


def max_allele(g, axis=None):
    """TODO

    """
    pass


def allelism(g):
    """TODO

    """
    pass


def allele_number(g):
    """TODO

    """
    pass


def allele_count(g, allele=1):
    """TODO

    """
    pass


def allele_frequency(g, allele=1):
    """TODO

    """
    pass


def allele_counts(g, alleles=None):
    """TODO

    """
    pass


def allele_frequencies(g, alleles=None):
    """TODO

    """
    pass


def is_variant(g):
    """TODO

    """
    pass


def is_non_variant(g):
    """TODO

    """
    pass


def is_segregating(g):
    """TODO

    """
    pass


def is_non_segregating(g):
    """TODO

    """
    pass


def is_singleton(g):
    """TODO

    """
    pass


def is_doubleton(g):
    """TODO

    """
    pass


def count_variant(g):
    """TODO

    """
    pass


def count_non_variant(g):
    """TODO

    """
    pass


def count_segregating(g):
    """TODO

    """
    pass


def count_non_segregating(g):
    """TODO

    """
    pass


def count_singleton(g):
    """TODO

    """
    pass


def count_doubleton(g):
    """TODO

    """
    pass


def windowed_genotype_counts():
    """TODO

    """
    pass


def windowed_genotype_density():
    """TODO

    """
    pass


def windowed_genotype_rate():
    """TODO

    """
    pass


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


def plot_windowed_genotype_counts():
    """TODO

    """
    pass


def plot_windowed_genotype_density():
    """TODO

    """
    pass


def plot_windowed_genotype_rate():
    """TODO

    """
    pass
