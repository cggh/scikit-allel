# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np


from allel.model.ndarray import GenotypeArray
from allel.util import ignore_invalid, asarray_ndim


def heterozygosity_observed(g, fill=np.nan):
    """Calculate the rate of observed heterozygosity for each variant.

    Parameters
    ----------

    g : array_like, int, shape (n_variants, n_samples, ploidy)
        Genotype array.
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
    >>> allel.stats.heterozygosity_observed(g)
    array([ 0.        ,  0.33333333,  0.        ,  0.5       ])

    """

    # check inputs
    if not hasattr(g, 'count_het') or not hasattr(g, 'count_called'):
        g = GenotypeArray(g, copy=False)

    # count hets
    n_het = np.asarray(g.count_het(axis=1))
    n_called = np.asarray(g.count_called(axis=1))

    # calculate rate of observed heterozygosity, accounting for variants
    # where all calls are missing
    with ignore_invalid():
        ho = np.where(n_called > 0, n_het / n_called, fill)

    return ho


def heterozygosity_expected(af, ploidy, fill=np.nan):
    """Calculate the expected rate of heterozygosity for each variant
    under Hardy-Weinberg equilibrium.

    Parameters
    ----------

    af : array_like, float, shape (n_variants, n_alleles)
        Allele frequencies array.
    ploidy : int
        Sample ploidy.
    fill : float, optional
        Use this value for variants where allele frequencies do not sum to 1.

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
    >>> af = g.count_alleles().to_frequencies()
    >>> allel.stats.heterozygosity_expected(af, ploidy=2)
    array([ 0.        ,  0.5       ,  0.66666667,  0.375     ])

    """

    # check inputs
    af = asarray_ndim(af, 2)

    # calculate expected heterozygosity
    out = 1 - np.sum(np.power(af, ploidy), axis=1)

    # fill values where allele frequencies could not be calculated
    af_sum = np.sum(af, axis=1)
    with ignore_invalid():
        out[(af_sum < 1) | np.isnan(af_sum)] = fill

    return out


def inbreeding_coefficient(g, fill=np.nan):
    """Calculate the inbreeding coefficient for each variant.

    Parameters
    ----------

    g : array_like, int, shape (n_variants, n_samples, ploidy)
        Genotype array.
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
    >>> g = allel.GenotypeArray([[[0, 0], [0, 0], [0, 0]],
    ...                          [[0, 0], [0, 1], [1, 1]],
    ...                          [[0, 0], [1, 1], [2, 2]],
    ...                          [[1, 1], [1, 2], [-1, -1]]])
    >>> allel.stats.inbreeding_coefficient(g)
    array([        nan,  0.33333333,  1.        , -0.33333333])

    """

    # check inputs
    if not hasattr(g, 'count_het') or not hasattr(g, 'count_called'):
        g = GenotypeArray(g, copy=False)

    # calculate observed and expected heterozygosity
    ho = heterozygosity_observed(g)
    af = g.count_alleles().to_frequencies()
    he = heterozygosity_expected(af, ploidy=g.shape[-1], fill=0)

    # calculate inbreeding coefficient, accounting for variants with no
    # expected heterozygosity
    with ignore_invalid():
        f = np.where(he > 0, 1 - (ho / he), fill)

    return f
