# -*- coding: utf-8 -*-
import numpy as np


from allel.model.ndarray import GenotypeArray
from allel.util import ignore_invalid, asarray_ndim


def heterozygosity_individual(g, fill=np.nan, corrected=True, ploidy=None):
    """Calculate the gammetic heterozygosity of each individual
    at each variant.

    Parameters
    ----------
    g : array_like, int, shape (n_variants, n_samples, ploidy)
        Genotype array.
    fill : float, optional
        Use this value for variants with invalid inputs.
    corrected : bool, optional
        If True, values are corrected for ploidy level.
    ploidy : array_like, int, (n_variants, n_samples), optional
        Specify ploidy of each genotype call.
    
    Returns
    -------
    hi : ndarray, float, shape (n_variants, n_samples)
        Observed individual heterozygosity
    
    Notes
    -----
    Individual heterozygosity is calculated assuming polysomic inheritance
    for polyploid genotype arrays following Hardy (2016).

    If the ploidy argument is used then the ploidy specified for each 
    genotype call must be equal to the number of alleles called for that
    genotype or else the genotype will be treated as missing.

    Examples
    --------

    >>> import allel
    >>> g = allel.GenotypeArray([[[0, 0, -1, -1], [0, 0, 0, 0]],
    ...                          [[0, 1, -1, -1], [0, 0, 0, 1]],
    ...                          [[1, 1, -1, -1], [0, 1, 1, 2]],
    ...                          [[0, -1, -1, -1], [0, 1, 2, 3]],
    ...                          [[-1, -1, -1, -1], [0, 1, -1, -1]]])
    >>> ploidy = [[2, 4],
    ...           [2, 4],
    ...           [2, 4],
    ...           [2, 4],
    ...           [2, 4]]
    >>> allel.heterozygosity_individual(g, ploidy=ploidy)
    array([[0.        , 0.        ],
           [1.        , 0.5       ],
           [0.        , 0.83333333],
           [       nan, 1.        ],
           [       nan,        nan]])

    """
    # check inputs
    if not hasattr(g, 'ploidy') or not hasattr(g, 'to_allele_counts'):
        g = GenotypeArray(g, copy=False)

    # use array ploidy if none provided
    ploidy = g.ploidy if ploidy is None else np.array(ploidy)
    
    # convert to genotype allele counts, remove those not matching ploidy
    gac = g.to_allele_counts()
    is_called = gac.values.sum(axis=-1) == ploidy
    gac[~is_called] = 0

    # genotype allele frequencies with partial genotypes removed
    gaf = gac.to_frequencies()

    # correction for ploidy level
    with ignore_invalid():
        correction = ploidy / (ploidy - 1) if corrected else 1

    # matrix of observed heterozygosity per sample
    hi = (1 - np.sum(np.power(gaf, 2), axis=-1)) * correction

    if fill is not np.nan:
        hi[np.isnan(hi)] = fill

    return hi


def heterozygosity_observed(g, fill=np.nan, corrected=True, ploidy=None):
    """Calculate the rate of observed heterozygosity for each variant.

    Parameters
    ----------

    g : array_like, int, shape (n_variants, n_samples, ploidy)
        Genotype array.
    fill : float, optional
        Use this value for variants where all calls are missing.
    corrected : bool, optional
        If True, values are corrected for ploidy level.
    ploidy : array_like, int, (n_variants, n_samples), optional
        Specify ploidy of each genotype call.

    Returns
    -------

    ho : ndarray, float, shape (n_variants,)
        Observed heterozygosity

    Notes
    -----

    Observed heterozygosity is calculated assuming polysomic inheritance
    for polyploid genotype arrays following Hardy (2016) (also see Meirmans
    and Liu 2018).

    By default, observed heterozygosity is corrected for ploidy by
    *Ho = Hu * (ploidy / (ploidy - 1))* where *Ho* is the corrected observed
    heterozygosity and *Hu* is the uncorrected observed heterozygosity
    described by Meirmans and Liu (2018).

    If the ploidy argument is used then the ploidy specified for each 
    genotype call must be equal to the number of called alleles for that
    genotype or else the genotype will be treated as missing.

    Examples
    --------

    Diploid genotype array

    >>> import allel
    >>> g = allel.GenotypeArray([[[0, 0], [0, 0], [0, 0]],
    ...                          [[0, 0], [0, 1], [1, 1]],
    ...                          [[0, 0], [1, 1], [2, 2]],
    ...                          [[1, 1], [1, 2], [-1, -1]]])
    >>> allel.heterozygosity_observed(g)
    array([0.        , 0.33333333, 0.        , 0.5       ])

    Tetraploid genotype array

    >>> import allel
    >>> g = allel.GenotypeArray([[[0, 0, 0, 0], [0, 0, 0, 0]],
    ...                          [[0, 0, 0, 0], [0, 0, 0, 1]],
    ...                          [[0, 0, 1, 1], [0, 0, 1, 2]],
    ...                          [[0, 0, 1, 1], [0, 0, -1, -1]],
    ...                          [[0, 1, 2, 3], [-1, -1, -1, -1]]])
    >>> allel.heterozygosity_observed(g)
    array([0.        , 0.25      , 0.75      , 0.66666667, 1.        ])

    """
    hi = heterozygosity_individual(
        g, 
        fill=np.nan, 
        corrected=corrected, 
        ploidy=ploidy
    )
    sum_het = np.nansum(hi, axis=-1)
    
    n_called = np.nansum(~np.isnan(hi), axis=-1)

    with ignore_invalid():
        ho = np.where(n_called > 0, sum_het / n_called, fill)

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
