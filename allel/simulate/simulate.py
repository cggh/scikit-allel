from allel.model.dask import GenotypeDaskArray, GenotypeAlleleCountsDaskArray, AlleleCountsDaskArray

import numpy as np
import dask.array as da


def simulate_genotypes(n_variants, n_samples, p=(0.95, 0.05), ploidy=2):
    """generate genotypes from a random distribution

    Parameters
    ----------
    n_variants : int
        number of variants to generate
    n_samples : int
        number of samples to generate
    p : tuple, float
        probability of each allele, must sum to 1.
        This is used to implicitly specify the number of alleles
    ploidy : int
        ploidy of individuals

    Returns
    -------
    GenotypeDaskArray: int8, shape (nvariants, nsamples, ploidy)

    Notes
    -----
    For speed and efficiency all variants are drawn from the same distribution.
    For a more "realistic" simulate this simple function may want to be extended.

    """
    a = np.arange(0, len(p), dtype="int8")

    g = da.random.choice(
        a, size=(n_variants, n_samples, ploidy), p=p)

    return GenotypeDaskArray(g)


def simulate_allele_counts_array(n_variants, n_samples, p=(0.95, 0.05), ploidy=2):
    """generate an allele counts array from a random distribution

    Parameters
    ----------
    n_variants : int
        number of variants to generate
    n_samples : int
        number of samples to generate
    p : tuple, float
        probability of each allele, must sum to 1.
        This is used to implicitly specify the number of alleles
    ploidy : int
        ploidy of individuals

    Returns
    -------
    GenotypeAlleleCountsDaskArray: int64, shape (nvariants, nsamples, n_alleles)

    Notes
    -----
    For speed and efficiency all variants are drawn from the same distribution.
    For a more "realistic" simulate this simple function may want to be extended.

    """

    aca = da.random.multinomial(
        ploidy,
        p,
        size=(n_variants, n_samples))

    return GenotypeAlleleCountsDaskArray(aca)


def simulate_allele_counts(n_variants, n_samples, p=(0.95, 0.05), ploidy=2):
    """generate allele counts from a random distribution

    Parameters
    ----------
    n_variants : int
        number of variants to generate
    n_samples : int
        number of samples to generate
    p : tuple, float
        probability of each allele, must sum to 1.
        This is used to implicitly specify the number of alleles
    ploidy : int
        ploidy of individuals

    Returns
    -------
    GenotypeDaskArray: int8, shape (nvariants, n_alleles)

    Notes
    -----
    For speed and efficiency all variants are drawn from the same distribution.
    For a more "realistic" simulate this simple function may want to be extended.

    """

    ac = da.random.multinomial(
        ploidy * n_samples,
        p,
        size=(n_variants,))

    return AlleleCountsDaskArray(ac)
