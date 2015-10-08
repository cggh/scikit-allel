# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np


from allel.util import asarray_ndim


def sfs(dac):
    """Compute the site frequency spectrum given derived allele counts at
    a set of biallelic variants.

    Parameters
    ----------
    dac : array_like, int, shape (n_variants,)
        Array of derived allele counts.

    Returns
    -------
    sfs : ndarray, int
        Array where the kth element is the number of variant sites with k
        derived alleles.

    """

    # check input
    dac = asarray_ndim(dac, 1)

    # compute site frequency spectrum
    s = np.bincount(dac)

    return s


def sfs_folded(ac):
    """Compute the folded site frequency spectrum given reference and
    alternate allele counts at a set of biallelic variants.

    Parameters
    ----------
    ac : array_like, int, shape (n_variants, 2)
        Allele counts array.

    Returns
    -------
    sfs_folded : ndarray, int
        Array where the kth element is the number of variant sites with a
        minor allele count of k.

    """

    # check input
    ac = asarray_ndim(ac, 2)
    assert ac.shape[1] == 2, 'only biallelic variants are supported'

    # compute minor allele counts
    mac = np.amin(ac, axis=1)

    # compute folded site frequency spectrum
    s = np.bincount(mac)

    return s


def sfs_scaled(dac):
    """Compute the site frequency spectrum scaled such that a constant value is
    expected across the spectrum for neutral variation and constant
    population size.

    Parameters
    ----------
    dac : array_like, int, shape (n_variants,)
        Array of derived allele counts.

    Returns
    -------
    sfs_scaled : ndarray, int
        An array where the value of the kth element is the number of variants
        with k derived alleles, multiplied by k.

    """

    # compute site frequency spectrum
    s = sfs(dac)

    # apply scaling
    k = np.arange(s.size)
    s *= k

    return s


def sfs_folded_scaled(ac, m=None):
    """Compute the folded site frequency spectrum scaled such that a constant
    value is expected across the spectrum for neutral variation and constant
    population size.

    Parameters
    ----------
    ac : array_like, int, shape (n_variants, 2)
        Allele counts array.
    m : int, optional
        The total number of chromosomes called at each variant site. Equal to
        the number of samples multiplied by the ploidy. If not provided,
        will be inferred to be the maximum value of the sum of reference and
        alternate allele counts present in `ac`.

    Returns
    -------
    sfs_folded_scaled : ndarray, int
        An array where the value of the kth element is the number of variants
        with minor allele count k, multiplied by the scaling factor
        (k * (m - k) / m).

    """

    # compute the site frequency spectrum
    s = sfs_folded(ac)

    # determine the total number of chromosomes called
    if m is None:
        m = np.amax(np.sum(ac, axis=1))

    # apply scaling
    k = np.arange(s.size)
    s *= k * (m - k) / m

    return s


def joint_sfs(dac1, dac2):
    """Compute the joint site frequency spectrum between two populations.

    Parameters
    ----------
    dac1 : array_like, int, shape (n_variants,)
        Derived allele counts for the first population.
    dac2 : array_like, int, shape (n_variants,)
        Derived allele counts for the second population.

    Returns
    -------
    joint_sfs : ndarray, int
        Array where the (i, j)th element is the number of variant sites with i
        derived alleles in the first population and j derived alleles in the
        second population.

    """

    # check inputs
    dac1 = asarray_ndim(dac1, 1)
    dac2 = asarray_ndim(dac2, 1)

    # compute site frequency spectrum
    n = np.max(dac1) + 1
    m = np.max(dac2) + 1
    s = np.bincount(dac1 * m + dac2)
    s.resize((n, m))
    return s


def joint_sfs_folded(ac1, ac2):
    """Compute the joint folded site frequency spectrum between two
    populations.

    Parameters
    ----------
    ac1 : array_like, int, shape (n_variants, 2)
        Allele counts for the first population.
    ac2 : array_like, int, shape (n_variants, 2)
        Allele counts for the second population.

    Returns
    -------
    joint_sfs_folded : ndarray, int
        Array where the (i, j)th element is the number of variant sites with a
        minor allele count of i in the first population and j in the second
        population.

    """

    # check inputs
    ac1 = asarray_ndim(ac1, 2)
    ac2 = asarray_ndim(ac2, 2)
    assert ac1.shape[1] == ac2.shape[1] == 2, \
        'only biallelic variants are supported'

    # compute minor allele counts
    mac1 = np.amin(ac1, axis=1)
    mac2 = np.amin(ac2, axis=1)

    # compute site frequency spectrum
    m = np.max(mac1) + 1
    n = np.max(mac2) + 1
    s = np.bincount(mac1 * n + mac2)
    s.resize((m, n))
    return s


def joint_sfs_scaled(dac1, dac2):
    """Compute the joint site frequency spectrum between two populations,
    scaled such that a constant value is expected across the spectrum for
    neutral variation, constant population size and unrelated populations.

    Parameters
    ----------
    dac1 : array_like, int, shape (n_variants,)
        Derived allele counts for the first population.
    dac2 : array_like, int, shape (n_variants,)
        Derived allele counts for the second population.

    Returns
    -------
    joint_sfs_scaled : ndarray, int
        Array where the (i, j)th element is the scaled frequency of variant
        sites with i derived alleles in the first population and j derived
        alleles in the second population.

    """

    # compute site frequency spectrum
    s = joint_sfs(dac1, dac2)

    # apply scaling
    k = np.arange(s.shape[0])[:, None]
    s *= k
    k = np.arange(s.shape[1])[None, :]
    s *= k

    return s


def joint_sfs_folded_scaled(ac1, ac2, m=None, n=None):
    """Compute the joint folded site frequency spectrum between two
    populations, scaled such that a constant value is expected across the
    spectrum for neutral variation, constant population size and unrelated
    populations.

    Parameters
    ----------
    ac1 : array_like, int, shape (n_variants, 2)
        Allele counts for the first population.
    ac2 : array_like, int, shape (n_variants, 2)
        Allele counts for the second population.

    Returns
    -------
    joint_sfs_folded_scaled : ndarray, int
        Array where the (i, j)th element is the scaled frequency of variant
        sites with a minor allele count of i in the first population and j
        in the second population.

    """

    # compute site frequency spectrum
    s = joint_sfs_folded(ac1, ac2)

    # determine the total number of chromosomes called
    if m is None:
        m = np.amax(np.sum(ac1, axis=1))
    if n is None:
        n = np.amax(np.sum(ac2, axis=1))

    # apply scaling
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            s[i, j] *= i * j * (m-i) * (n-j)

    return s
