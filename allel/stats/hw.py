# -*- coding: utf-8 -*-
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
    >>> allel.heterozygosity_observed(g)
    array([0.        , 0.33333333, 0.        , 0.5       ])

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


def heterozygosity_blue(g, fill=np.nan, corrected=True, ploidy=None, kinship=None):
    """Calculate the expected rate of heterozygosity for each variant while
    optionally correcting for kinship among samples following Harris and
    DeGiorgio (2017).

    Parameters
    ----------
    g : array_like, int, shape (n_variants, n_samples, max_ploidy)
        Genotype array.
    fill : float, optional
        Use this value for variants with invalid inputs.
    corrected : bool, optional
        If True, values are corrected based on total number of alleles
        or by weighted mean kiship.
    ploidy : array_like, int, shape (n_variants, n_samples), optional
        Specify variant-wise ploidy of each sample.
    kinship : array_like, float, shape (n_samples, n_samples), optional
        A symmetric matrix of kinship among samples.

    Returns
    -------
    he : ndarray, float, shape (n_variants,)
        Expected heterozygosity

    Notes
    -----
    If ploidy is specified then genotype calls in which the number
    of allele calls differ from their specified ploidy will be treated
    as missing genotypes.
    If kinship is not specified then heterozygosity expected will be
    calculated which is equivalent to heterozygosity blue when
    samples are non-inbred and unrelated to one another.

    Examples
    --------
    >>> import numpy as np
    >>> import allel
    >>> # tetraploid genotypes
    >>> g = allel.GenotypeArray([
    ...     [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    ...     [[0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 1, 1]],
    ...     [[0, 1, 1, 1], [0, 0, 2, 2], [2, 2, 2, 3]],
    ...     [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]],
    ...     [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]],
    ...     [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
    ...     [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, -1, -1]],
    ... ])
    >>> # kinship matrix indicating non-related samples
    >>> k = [[0.25, 0.00, 0.00],
    ...      [0.00, 0.25, 0.00],
    ...      [0.00, 0.00, 0.25]]
    >>> allel.heterozygosity_blue(g, kinship=k)
    array([0.        , 0.53030303, 0.75757576, 0.81818182, 0.72727273, 1.        , 0.57142857])

    """
    # check inputs
    if not hasattr(g, 'to_allele_counts') or not hasattr(g, 'ploidy'):
        g = GenotypeArray(g, copy=False)

    ploidy = g.ploidy if ploidy is None else np.array(ploidy, copy=False)

    # allele count must match ploidy
    gac = g.to_allele_counts()
    gac[gac.values.sum(axis=-1) != ploidy] = 0

    if kinship is None:
        # heterozygosity expected
        ac = gac.count_alleles()
        af = ac.to_frequencies()

        # uncorrected expected het
        out = 1 - np.sum(np.power(af, 2), axis=1)

        # correction based on total number of alleles
        if corrected:
            n_alleles = ac.sum(-1)
            out *= n_alleles / (n_alleles - 1)

    else:
        # heterozygosity blue
        out = np.zeros(len(g))
        gaf = gac.to_frequencies()
        kinship = np.array(kinship, copy=False)

        # inverse kinship matrix must be calculate only for non-missing samples
        # group genotypes based on missing samples to minimise re-computation
        called_mtx, called_idx = np.unique(gac.is_called(), axis=0, return_inverse=True)
        for idx, called in enumerate(called_mtx):

            # variants with this set of called samples
            variants = called_idx == idx

            if np.all(~called):
                # no called genotypes
                out[variants] = np.nan
                continue

            # kinship sub-matrix
            k_sub = kinship[np.ix_(called, called)]

            try:
                # inverse of sub-matrix
                k_inv = np.linalg.inv(k_sub)

            except np.linalg.LinAlgError:
                # cannot be calculated
                out[variants] = np.nan
                continue

            # calculate weights from inverse kinship matrix
            weights = k_inv.sum(axis=-1) / k_inv.sum()

            # blue of allele freqs
            af = np.nansum(weights.reshape(-1, 1) * gaf[np.ix_(variants, called)], axis=-2)

            # uncorrected het_blue
            het_blue = 1 - np.sum(np.power(af, 2), axis=-1)

            # correction with weighted mean kinship
            if corrected:
                wmk = np.sum(np.outer(weights, weights) * k_sub)
                het_blue *= (1 / (1 - wmk))

            # update output
            out[variants] = het_blue

    if fill is not np.nan:
        out[np.isnan(out)] = fill

    return out


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
    >>> allel.heterozygosity_expected(af, ploidy=2)
    array([0.        , 0.5       , 0.66666667, 0.375     ])

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
    >>> allel.inbreeding_coefficient(g)
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
