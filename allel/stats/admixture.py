# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


from allel.model.ndarray import AlleleCountsArray
from allel.util import asarray_ndim, check_dim0_aligned
from allel.stats.window import moving_statistic
from allel.stats.misc import jackknife


import numpy as np


def h_hat(ac):
    """Unbiased estimator for h, where 2*h is the heterozygosity
    of the population.

    Parameters
    ----------
    ac : array_like, int, shape (n_variants, 2)
        Allele counts array for a single population.

    Returns
    -------
    h_hat : ndarray, float, shape (n_variants,)

    Notes
    -----
    Used in Patterson (2012) for calculation of various statistics.

    """

    # check inputs
    ac = asarray_ndim(ac, 2)
    assert ac.shape[1] == 2, 'only biallelic variants supported'

    # compute allele number
    an = ac.sum(axis=1)

    # compute estimator
    x = (ac[:, 0] * ac[:, 1]) / (an * (an - 1))

    return x


def patterson_f2(aca, acb):
    """Unbiased estimator for F2(A, B), the branch length between populations
    A and B.

    Parameters
    ----------
    aca : array_like, int, shape (n_variants, 2)
        Allele counts for population A.
    acb : array_like, int, shape (n_variants, 2)
        Allele counts for population B.

    Returns
    -------
    f2 : ndarray, float, shape (n_variants,)

    Notes
    -----
    See Patterson (2012), Appendix A.

    """

    # check inputs
    aca = AlleleCountsArray(aca, copy=False)
    assert aca.shape[1] == 2, 'only biallelic variants supported'
    acb = AlleleCountsArray(acb, copy=False)
    assert acb.shape[1] == 2, 'only biallelic variants supported'
    check_dim0_aligned(aca, acb)

    # compute allele numbers
    sa = aca.sum(axis=1)
    sb = acb.sum(axis=1)

    # compute heterozygosities
    ha = h_hat(aca)
    hb = h_hat(acb)

    # compute sample frequencies for the alternate allele
    a = aca.to_frequencies()[:, 1]
    b = acb.to_frequencies()[:, 1]

    # compute estimator
    x = ((a - b) ** 2) - (ha / sa) - (hb / sb)

    return x


# noinspection PyPep8Naming
def patterson_f3(acc, aca, acb):
    """Unbiased estimator for F3(C; A, B), the three-population test for
    admixture in population C.

    Parameters
    ----------
    acc : array_like, int, shape (n_variants, 2)
        Allele counts for the test population (C).
    aca : array_like, int, shape (n_variants, 2)
        Allele counts for the first source population (A).
    acb : array_like, int, shape (n_variants, 2)
        Allele counts for the second source population (B).

    Returns
    -------
    T : ndarray, float, shape (n_variants,)
        Un-normalized f3 estimates per variant.
    B : ndarray, float, shape (n_variants,)
        Estimates for heterozygosity in population C.

    Notes
    -----
    See Patterson (2012), main text and Appendix A.

    For un-normalized f3 statistics, ignore the `B` return value.

    To compute the f3* statistic, which is normalized by heterozygosity in
    population C to remove numerical dependence on the allele frequency
    spectrum, compute ``np.sum(T) / np.sum(B)``.

    """

    # check inputs
    aca = AlleleCountsArray(aca, copy=False)
    assert aca.shape[1] == 2, 'only biallelic variants supported'
    acb = AlleleCountsArray(acb, copy=False)
    assert acb.shape[1] == 2, 'only biallelic variants supported'
    acc = AlleleCountsArray(acc, copy=False)
    assert acc.shape[1] == 2, 'only biallelic variants supported'
    check_dim0_aligned(aca, acb, acc)

    # compute allele number and heterozygosity in test population
    sc = acc.sum(axis=1)
    hc = h_hat(acc)

    # compute sample frequencies for the alternate allele
    a = aca.to_frequencies()[:, 1]
    b = acb.to_frequencies()[:, 1]
    c = acc.to_frequencies()[:, 1]

    # compute estimator
    T = ((c - a) * (c - b)) - (hc / sc)
    B = 2 * hc

    return T, B


def patterson_d(aca, acb, acc, acd):
    """Unbiased estimator for D(A, B; C, D), the normalised four-population
    test for admixture between (A or B) and (C or D), also known as the
    "ABBA BABA" test.

    Parameters
    ----------
    aca : array_like, int, shape (n_variants, 2),
        Allele counts for population A.
    acb : array_like, int, shape (n_variants, 2)
        Allele counts for population B.
    acc : array_like, int, shape (n_variants, 2)
        Allele counts for population C.
    acd : array_like, int, shape (n_variants, 2)
        Allele counts for population D.

    Returns
    -------
    num : ndarray, float, shape (n_variants,)
        Numerator (un-normalised f4 estimates).
    den : ndarray, float, shape (n_variants,)
        Denominator.

    Notes
    -----
    See Patterson (2012), main text and Appendix A.

    For un-normalized f4 statistics, ignore the `den` return value.

    """

    # check inputs
    aca = AlleleCountsArray(aca, copy=False)
    assert aca.shape[1] == 2, 'only biallelic variants supported'
    acb = AlleleCountsArray(acb, copy=False)
    assert acb.shape[1] == 2, 'only biallelic variants supported'
    acc = AlleleCountsArray(acc, copy=False)
    assert acc.shape[1] == 2, 'only biallelic variants supported'
    acd = AlleleCountsArray(acd, copy=False)
    assert acd.shape[1] == 2, 'only biallelic variants supported'
    check_dim0_aligned(aca, acb, acc, acd)

    # compute sample frequencies for the alternate allele
    a = aca.to_frequencies()[:, 1]
    b = acb.to_frequencies()[:, 1]
    c = acc.to_frequencies()[:, 1]
    d = acd.to_frequencies()[:, 1]

    # compute estimator
    num = (a - b) * (c - d)
    den = (a + b - (2 * a * b)) * (c + d - (2 * c * d))

    return num, den


# noinspection PyPep8Naming
def moving_patterson_f3(acc, aca, acb, size, start=0, stop=None, step=None,
                        normed=True):
    """Estimate F3(C; A, B) in moving windows.

    Parameters
    ----------
    acc : array_like, int, shape (n_variants, 2)
        Allele counts for the test population (C).
    aca : array_like, int, shape (n_variants, 2)
        Allele counts for the first source population (A).
    acb : array_like, int, shape (n_variants, 2)
        Allele counts for the second source population (B).
    size : int
        The window size (number of variants).
    start : int, optional
        The index at which to start.
    stop : int, optional
        The index at which to stop.
    step : int, optional
        The number of variants between start positions of windows. If not
        given, defaults to the window size, i.e., non-overlapping windows.
    normed : bool, optional
        If False, use un-normalised f3 values.

    Returns
    -------
    f3 : ndarray, float, shape (n_windows,)
        Estimated value of the statistic in each window.

    """

    # calculate per-variant values
    T, B = patterson_f3(acc, aca, acb)

    # calculate value of statistic within each block
    if normed:
        T_bsum = moving_statistic(T, statistic=np.nansum, size=size,
                                  start=start, stop=stop, step=step)
        B_bsum = moving_statistic(B, statistic=np.nansum, size=size,
                                  start=start, stop=stop, step=step)
        f3 = T_bsum / B_bsum

    else:
        f3 = moving_statistic(T, statistic=np.nanmean, size=size,
                              start=start, stop=stop, step=step)

    return f3


def moving_patterson_d(aca, acb, acc, acd, size, start=0, stop=None,
                       step=None):
    """Estimate D(A, B; C, D) in moving windows.

    Parameters
    ----------
    aca : array_like, int, shape (n_variants, 2),
        Allele counts for population A.
    acb : array_like, int, shape (n_variants, 2)
        Allele counts for population B.
    acc : array_like, int, shape (n_variants, 2)
        Allele counts for population C.
    acd : array_like, int, shape (n_variants, 2)
        Allele counts for population D.
    size : int
        The window size (number of variants).
    start : int, optional
        The index at which to start.
    stop : int, optional
        The index at which to stop.
    step : int, optional
        The number of variants between start positions of windows. If not
        given, defaults to the window size, i.e., non-overlapping windows.

    Returns
    -------
    d : ndarray, float, shape (n_windows,)
        Estimated value of the statistic in each window.

    """

    # calculate per-variant values
    num, den = patterson_d(aca, acb, acc, acd)

    # N.B., nans can occur if any of the populations have completely missing
    # genotype calls at a variant (i.e., allele number is zero). Here we
    # assume that is rare enough to be negligible.

    # compute the numerator and denominator within each window
    num_sum = moving_statistic(num, statistic=np.nansum, size=size,
                               start=start, stop=stop, step=step)
    den_sum = moving_statistic(den, statistic=np.nansum, size=size,
                               start=start, stop=stop, step=step)

    # calculate the statistic values in each block
    d = num_sum / den_sum

    return d


# noinspection PyPep8Naming
def average_patterson_f3(acc, aca, acb, blen, normed=True):
    """Estimate F3(C; A, B) and standard error using the block-jackknife.

    Parameters
    ----------
    acc : array_like, int, shape (n_variants, 2)
        Allele counts for the test population (C).
    aca : array_like, int, shape (n_variants, 2)
        Allele counts for the first source population (A).
    acb : array_like, int, shape (n_variants, 2)
        Allele counts for the second source population (B).
    blen : int
        Block size (number of variants).
    normed : bool, optional
        If False, use un-normalised f3 values.

    Returns
    -------
    f3 : float
        Estimated value of the statistic using all data.
    se : float
        Estimated standard error.
    z : float
        Z-score (number of standard errors from zero).
    vb : ndarray, float, shape (n_blocks,)
        Value of the statistic in each block.
    vj : ndarray, float, shape (n_blocks,)
        Values of the statistic from block-jackknife resampling.

    Notes
    -----
    See Patterson (2012), main text and Appendix A.

    See Also
    --------
    allel.stats.admixture.patterson_f3

    """

    # calculate per-variant values
    T, B = patterson_f3(acc, aca, acb)

    # N.B., nans can occur if any of the populations have completely missing
    # genotype calls at a variant (i.e., allele number is zero). Here we
    # assume that is rare enough to be negligible.

    # calculate overall value of statistic
    if normed:
        f3 = np.nansum(T) / np.nansum(B)
    else:
        f3 = np.nanmean(T)

    # calculate value of statistic within each block
    if normed:
        T_bsum = moving_statistic(T, statistic=np.nansum, size=blen)
        B_bsum = moving_statistic(B, statistic=np.nansum, size=blen)
        vb = T_bsum / B_bsum
        _, se, vj = jackknife((T_bsum, B_bsum),
                              statistic=lambda t, b: np.sum(t) / np.sum(b))

    else:
        vb = moving_statistic(T, statistic=np.nanmean, size=blen)
        _, se, vj = jackknife(vb, statistic=np.mean)

    # compute Z score
    z = f3 / se

    return f3, se, z, vb, vj


def average_patterson_d(aca, acb, acc, acd, blen):
    """Estimate D(A, B; C, D) and standard error using the block-jackknife.

    Parameters
    ----------
    aca : array_like, int, shape (n_variants, 2),
        Allele counts for population A.
    acb : array_like, int, shape (n_variants, 2)
        Allele counts for population B.
    acc : array_like, int, shape (n_variants, 2)
        Allele counts for population C.
    acd : array_like, int, shape (n_variants, 2)
        Allele counts for population D.
    blen : int
        Block size (number of variants).

    Returns
    -------
    d : float
        Estimated value of the statistic using all data.
    se : float
        Estimated standard error.
    z : float
        Z-score (number of standard errors from zero).
    vb : ndarray, float, shape (n_blocks,)
        Value of the statistic in each block.
    vj : ndarray, float, shape (n_blocks,)
        Values of the statistic from block-jackknife resampling.

    Notes
    -----
    See Patterson (2012), main text and Appendix A.

    See Also
    --------
    allel.stats.admixture.patterson_d

    """

    # calculate per-variant values
    num, den = patterson_d(aca, acb, acc, acd)

    # N.B., nans can occur if any of the populations have completely missing
    # genotype calls at a variant (i.e., allele number is zero). Here we
    # assume that is rare enough to be negligible.

    # calculate overall estimate
    d_avg = np.nansum(num) / np.nansum(den)

    # compute the numerator and denominator within each block
    num_bsum = moving_statistic(num, statistic=np.nansum, size=blen)
    den_bsum = moving_statistic(den, statistic=np.nansum, size=blen)

    # calculate the statistic values in each block
    vb = num_bsum / den_bsum

    # estimate standard error
    _, se, vj = jackknife((num_bsum, den_bsum),
                          statistic=lambda n, d: np.sum(n) / np.sum(d))

    # compute Z score
    z = d_avg / se

    return d_avg, se, z, vb, vj


# backwards compatibility
blockwise_patterson_f3 = average_patterson_f3
blockwise_patterson_d = average_patterson_d
