# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import logging


import numpy as np


from allel.model.ndarray import SortedIndex, AlleleCountsArray
from allel.model.util import locate_fixed_differences
from allel.util import asarray_ndim, ignore_invalid, check_dim0_aligned, \
    ensure_dim1_aligned
from allel.stats.window import windowed_statistic, per_base, moving_statistic


logger = logging.getLogger(__name__)
debug = logger.debug


def mean_pairwise_difference(ac, an=None, fill=np.nan):
    """Calculate for each variant the mean number of pairwise differences
    between chromosomes sampled from within a single population.

    Parameters
    ----------

    ac : array_like, int, shape (n_variants, n_alleles)
        Allele counts array.
    an : array_like, int, shape (n_variants,), optional
        Allele numbers. If not provided, will be calculated from `ac`.
    fill : float
        Use this value where there are no pairs to compare (e.g.,
        all allele calls are missing).

    Returns
    -------

    mpd : ndarray, float, shape (n_variants,)

    Notes
    -----

    The values returned by this function can be summed over a genome
    region and divided by the number of accessible bases to estimate
    nucleotide diversity, a.k.a. *pi*.

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
    >>> ac = h.count_alleles()
    >>> allel.stats.mean_pairwise_difference(ac)
    array([ 0.        ,  0.5       ,  0.66666667,  0.5       ,  0.        ,
            0.83333333,  0.83333333,  1.        ])

    See Also
    --------

    sequence_diversity, windowed_diversity

    """

    # This function calculates the mean number of pairwise differences
    # between haplotypes within a single population, generalising to any number
    # of alleles.

    # check inputs
    ac = asarray_ndim(ac, 2)

    # total number of haplotypes
    if an is None:
        an = np.sum(ac, axis=1)
    else:
        an = asarray_ndim(an, 1)
        check_dim0_aligned(ac, an)

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
        mpd = np.where(n_pairs > 0, n_diff / n_pairs, fill)

    return mpd


def mean_pairwise_difference_between(ac1, ac2, an1=None, an2=None,
                                     fill=np.nan):
    """Calculate for each variant the mean number of pairwise differences
    between chromosomes sampled from two different populations.

    Parameters
    ----------

    ac1 : array_like, int, shape (n_variants, n_alleles)
        Allele counts array from the first population.
    ac2 : array_like, int, shape (n_variants, n_alleles)
        Allele counts array from the second population.
    an1 : array_like, int, shape (n_variants,), optional
        Allele numbers for the first population. If not provided, will be
        calculated from `ac1`.
    an2 : array_like, int, shape (n_variants,), optional
        Allele numbers for the second population. If not provided, will be
        calculated from `ac2`.
    fill : float
        Use this value where there are no pairs to compare (e.g.,
        all allele calls are missing).

    Returns
    -------

    mpd : ndarray, float, shape (n_variants,)

    Notes
    -----

    The values returned by this function can be summed over a genome
    region and divided by the number of accessible bases to estimate
    nucleotide divergence between two populations, a.k.a. *Dxy*.

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
    >>> ac1 = h.count_alleles(subpop=[0, 1])
    >>> ac2 = h.count_alleles(subpop=[2, 3])
    >>> allel.stats.mean_pairwise_difference_between(ac1, ac2)
    array([ 0.  ,  0.5 ,  1.  ,  0.5 ,  0.  ,  1.  ,  0.75,   nan])

    See Also
    --------

    sequence_divergence, windowed_divergence

    """

    # This function calculates the mean number of pairwise differences
    # between haplotypes from two different populations, generalising to any
    # number of alleles.

    # check inputs
    ac1 = asarray_ndim(ac1, 2)
    ac2 = asarray_ndim(ac2, 2)
    check_dim0_aligned(ac1, ac2)
    ac1, ac2 = ensure_dim1_aligned(ac1, ac2)

    # total number of haplotypes sampled from each population
    if an1 is None:
        an1 = np.sum(ac1, axis=1)
    else:
        an1 = asarray_ndim(an1, 1)
        check_dim0_aligned(ac1, an1)
    if an2 is None:
        an2 = np.sum(ac2, axis=1)
    else:
        an2 = asarray_ndim(an2, 1)
        check_dim0_aligned(ac2, an2)

    # total number of pairwise comparisons for each variant
    n_pairs = an1 * an2

    # number of pairwise comparisons where there is no difference:
    # sum of (ac1 * ac2) for each allele (i.e., number of ways to
    # choose the same allele twice)
    n_same = np.sum(ac1 * ac2, axis=1)

    # number of pairwise differences
    n_diff = n_pairs - n_same

    # mean number of pairwise differences, accounting for cases where
    # there are no pairs
    with ignore_invalid():
        mpd = np.where(n_pairs > 0, n_diff / n_pairs, fill)

    return mpd


def sequence_diversity(pos, ac, start=None, stop=None,
                       is_accessible=None):
    """Estimate nucleotide diversity within a given region.

    Parameters
    ----------

    pos : array_like, int, shape (n_items,)
        Variant positions, using 1-based coordinates, in ascending order.
    ac : array_like, int, shape (n_variants, n_alleles)
        Allele counts array.
    start : int, optional
        The position at which to start (1-based).
    stop : int, optional
        The position at which to stop (1-based).
    is_accessible : array_like, bool, shape (len(contig),), optional
        Boolean array indicating accessibility status for all positions in the
        chromosome/contig.

    Returns
    -------

    pi : ndarray, float, shape (n_windows,)
        Nucleotide diversity.

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
    ...                          [[0, 1], [-1, -1]],
    ...                          [[-1, -1], [-1, -1]]])
    >>> ac = g.count_alleles()
    >>> pos = [2, 4, 7, 14, 15, 18, 19, 25, 27]
    >>> pi = allel.stats.sequence_diversity(pos, ac, start=1, stop=31)
    >>> pi
    0.13978494623655915

    """

    # check inputs
    if not isinstance(pos, SortedIndex):
        pos = SortedIndex(pos, copy=False)
    ac = asarray_ndim(ac, 2)
    is_accessible = asarray_ndim(is_accessible, 1, allow_none=True)

    # deal with subregion
    if start is not None or stop is not None:
        loc = pos.locate_range(start, stop)
        pos = pos[loc]
        ac = ac[loc]
    if start is None:
        start = pos[0]
    if stop is None:
        stop = pos[-1]

    # calculate mean pairwise difference
    mpd = mean_pairwise_difference(ac, fill=0)

    # sum differences over variants
    mpd_sum = np.sum(mpd)

    # calculate value per base
    if is_accessible is None:
        n_bases = stop - start + 1
    else:
        n_bases = np.count_nonzero(is_accessible[start-1:stop])

    pi = mpd_sum / n_bases
    return pi


def sequence_divergence(pos, ac1, ac2, an1=None, an2=None, start=None,
                        stop=None, is_accessible=None):
    """Estimate nucleotide divergence between two populations within a
    given region.

    Parameters
    ----------

    pos : array_like, int, shape (n_items,)
        Variant positions, using 1-based coordinates, in ascending order.
    ac1 : array_like, int, shape (n_variants, n_alleles)
        Allele counts array for the first population.
    ac2 : array_like, int, shape (n_variants, n_alleles)
        Allele counts array for the second population.
    an1 : array_like, int, shape (n_variants,), optional
        Allele numbers for the first population. If not provided, will be
        calculated from `ac1`.
    an2 : array_like, int, shape (n_variants,), optional
        Allele numbers for the second population. If not provided, will be
        calculated from `ac2`.
    start : int, optional
        The position at which to start (1-based).
    stop : int, optional
        The position at which to stop (1-based).
    is_accessible : array_like, bool, shape (len(contig),), optional
        Boolean array indicating accessibility status for all positions in the
        chromosome/contig.

    Returns
    -------

    Dxy : ndarray, float, shape (n_windows,)
        Nucleotide divergence.

    Examples
    --------

    Simplest case, two haplotypes in each population::

        >>> import allel
        >>> h = allel.HaplotypeArray([[0, 0, 0, 0],
        ...                           [0, 0, 0, 1],
        ...                           [0, 0, 1, 1],
        ...                           [0, 1, 1, 1],
        ...                           [1, 1, 1, 1],
        ...                           [0, 0, 1, 2],
        ...                           [0, 1, 1, 2],
        ...                           [0, 1, -1, -1],
        ...                           [-1, -1, -1, -1]])
        >>> ac1 = h.count_alleles(subpop=[0, 1])
        >>> ac2 = h.count_alleles(subpop=[2, 3])
        >>> pos = [2, 4, 7, 14, 15, 18, 19, 25, 27]
        >>> dxy = sequence_divergence(pos, ac1, ac2, start=1, stop=31)
        >>> dxy
        0.12096774193548387

    """

    # check inputs
    if not isinstance(pos, SortedIndex):
        pos = SortedIndex(pos, copy=False)
    ac1 = asarray_ndim(ac1, 2)
    ac2 = asarray_ndim(ac2, 2)
    if an1 is not None:
        an1 = asarray_ndim(an1, 1)
    if an2 is not None:
        an2 = asarray_ndim(an2, 1)
    is_accessible = asarray_ndim(is_accessible, 1, allow_none=True)

    # handle start/stop
    if start is not None or stop is not None:
        loc = pos.locate_range(start, stop)
        pos = pos[loc]
        ac1 = ac1[loc]
        ac2 = ac2[loc]
        if an1 is not None:
            an1 = an1[loc]
        if an2 is not None:
            an2 = an2[loc]
    if start is None:
        start = pos[0]
    if stop is None:
        stop = pos[-1]

    # calculate mean pairwise difference between the two populations
    mpd = mean_pairwise_difference_between(ac1, ac2, an1=an1, an2=an2, fill=0)

    # sum differences over variants
    mpd_sum = np.sum(mpd)

    # calculate value per base, N.B., expect pos is 1-based
    if is_accessible is None:
        n_bases = stop - start + 1
    else:
        n_bases = np.count_nonzero(is_accessible[start-1:stop])

    dxy = mpd_sum / n_bases

    return dxy


def windowed_diversity(pos, ac, size=None, start=None, stop=None, step=None,
                       windows=None, is_accessible=None, fill=np.nan):
    """Estimate nucleotide diversity in windows over a single
    chromosome/contig.

    Parameters
    ----------

    pos : array_like, int, shape (n_items,)
        Variant positions, using 1-based coordinates, in ascending order.
    ac : array_like, int, shape (n_variants, n_alleles)
        Allele counts array.
    size : int, optional
        The window size (number of bases).
    start : int, optional
        The position at which to start (1-based).
    stop : int, optional
        The position at which to stop (1-based).
    step : int, optional
        The distance between start positions of windows. If not given,
        defaults to the window size, i.e., non-overlapping windows.
    windows : array_like, int, shape (n_windows, 2), optional
        Manually specify the windows to use as a sequence of (window_start,
        window_stop) positions, using 1-based coordinates. Overrides the
        size/start/stop/step parameters.
    is_accessible : array_like, bool, shape (len(contig),), optional
        Boolean array indicating accessibility status for all positions in the
        chromosome/contig.
    fill : object, optional
        The value to use where a window is completely inaccessible.

    Returns
    -------

    pi : ndarray, float, shape (n_windows,)
        Nucleotide diversity in each window.
    windows : ndarray, int, shape (n_windows, 2)
        The windows used, as an array of (window_start, window_stop) positions,
        using 1-based coordinates.
    n_bases : ndarray, int, shape (n_windows,)
        Number of (accessible) bases in each window.
    counts : ndarray, int, shape (n_windows,)
        Number of variants in each window.

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
    ...                          [[0, 1], [-1, -1]],
    ...                          [[-1, -1], [-1, -1]]])
    >>> ac = g.count_alleles()
    >>> pos = [2, 4, 7, 14, 15, 18, 19, 25, 27]
    >>> pi, windows, n_bases, counts = allel.stats.windowed_diversity(
    ...     pos, ac, size=10, start=1, stop=31
    ... )
    >>> pi
    array([ 0.11666667,  0.21666667,  0.09090909])
    >>> windows
    array([[ 1, 10],
           [11, 20],
           [21, 31]])
    >>> n_bases
    array([10, 10, 11])
    >>> counts
    array([3, 4, 2])

    """

    # check inputs
    if not isinstance(pos, SortedIndex):
        pos = SortedIndex(pos, copy=False)
    is_accessible = asarray_ndim(is_accessible, 1, allow_none=True)

    # calculate mean pairwise difference
    mpd = mean_pairwise_difference(ac, fill=0)

    # sum differences in windows
    mpd_sum, windows, counts = windowed_statistic(
        pos, values=mpd, statistic=np.sum, size=size, start=start, stop=stop,
        step=step, windows=windows, fill=0
    )

    # calculate value per base
    pi, n_bases = per_base(mpd_sum, windows, is_accessible=is_accessible,
                           fill=fill)

    return pi, windows, n_bases, counts


def windowed_divergence(pos, ac1, ac2, size=None, start=None, stop=None,
                        step=None, windows=None, is_accessible=None,
                        fill=np.nan):
    """Estimate nucleotide divergence between two populations in windows
    over a single chromosome/contig.

    Parameters
    ----------

    pos : array_like, int, shape (n_items,)
        Variant positions, using 1-based coordinates, in ascending order.
    ac1 : array_like, int, shape (n_variants, n_alleles)
        Allele counts array for the first population.
    ac2 : array_like, int, shape (n_variants, n_alleles)
        Allele counts array for the second population.
    size : int, optional
        The window size (number of bases).
    start : int, optional
        The position at which to start (1-based).
    stop : int, optional
        The position at which to stop (1-based).
    step : int, optional
        The distance between start positions of windows. If not given,
        defaults to the window size, i.e., non-overlapping windows.
    windows : array_like, int, shape (n_windows, 2), optional
        Manually specify the windows to use as a sequence of (window_start,
        window_stop) positions, using 1-based coordinates. Overrides the
        size/start/stop/step parameters.
    is_accessible : array_like, bool, shape (len(contig),), optional
        Boolean array indicating accessibility status for all positions in the
        chromosome/contig.
    fill : object, optional
        The value to use where a window is completely inaccessible.

    Returns
    -------

    Dxy : ndarray, float, shape (n_windows,)
        Nucleotide divergence in each window.
    windows : ndarray, int, shape (n_windows, 2)
        The windows used, as an array of (window_start, window_stop) positions,
        using 1-based coordinates.
    n_bases : ndarray, int, shape (n_windows,)
        Number of (accessible) bases in each window.
    counts : ndarray, int, shape (n_windows,)
        Number of variants in each window.

    Examples
    --------

    Simplest case, two haplotypes in each population::

        >>> import allel
        >>> h = allel.HaplotypeArray([[0, 0, 0, 0],
        ...                           [0, 0, 0, 1],
        ...                           [0, 0, 1, 1],
        ...                           [0, 1, 1, 1],
        ...                           [1, 1, 1, 1],
        ...                           [0, 0, 1, 2],
        ...                           [0, 1, 1, 2],
        ...                           [0, 1, -1, -1],
        ...                           [-1, -1, -1, -1]])
        >>> ac1 = h.count_alleles(subpop=[0, 1])
        >>> ac2 = h.count_alleles(subpop=[2, 3])
        >>> pos = [2, 4, 7, 14, 15, 18, 19, 25, 27]
        >>> dxy, windows, n_bases, counts = windowed_divergence(
        ...     pos, ac1, ac2, size=10, start=1, stop=31
        ... )
        >>> dxy
        array([ 0.15 ,  0.225,  0.   ])
        >>> windows
        array([[ 1, 10],
               [11, 20],
               [21, 31]])
        >>> n_bases
        array([10, 10, 11])
        >>> counts
        array([3, 4, 2])

    """

    # check inputs
    pos = SortedIndex(pos, copy=False)
    is_accessible = asarray_ndim(is_accessible, 1, allow_none=True)

    # calculate mean pairwise divergence
    mpd = mean_pairwise_difference_between(ac1, ac2, fill=0)

    # sum in windows
    mpd_sum, windows, counts = windowed_statistic(
        pos, values=mpd, statistic=np.sum, size=size, start=start,
        stop=stop, step=step, windows=windows, fill=0
    )

    # calculate value per base
    dxy, n_bases = per_base(mpd_sum, windows, is_accessible=is_accessible,
                            fill=fill)

    return dxy, windows, n_bases, counts


def windowed_df(pos, ac1, ac2, size=None, start=None, stop=None, step=None,
                windows=None, is_accessible=None, fill=np.nan):
    """Calculate the density of fixed differences between two populations in
    windows over a single chromosome/contig.

    Parameters
    ----------

    pos : array_like, int, shape (n_items,)
        Variant positions, using 1-based coordinates, in ascending order.
    ac1 : array_like, int, shape (n_variants, n_alleles)
        Allele counts array for the first population.
    ac2 : array_like, int, shape (n_variants, n_alleles)
        Allele counts array for the second population.
    size : int, optional
        The window size (number of bases).
    start : int, optional
        The position at which to start (1-based).
    stop : int, optional
        The position at which to stop (1-based).
    step : int, optional
        The distance between start positions of windows. If not given,
        defaults to the window size, i.e., non-overlapping windows.
    windows : array_like, int, shape (n_windows, 2), optional
        Manually specify the windows to use as a sequence of (window_start,
        window_stop) positions, using 1-based coordinates. Overrides the
        size/start/stop/step parameters.
    is_accessible : array_like, bool, shape (len(contig),), optional
        Boolean array indicating accessibility status for all positions in the
        chromosome/contig.
    fill : object, optional
        The value to use where a window is completely inaccessible.

    Returns
    -------

    df : ndarray, float, shape (n_windows,)
        Per-base density of fixed differences in each window.
    windows : ndarray, int, shape (n_windows, 2)
        The windows used, as an array of (window_start, window_stop) positions,
        using 1-based coordinates.
    n_bases : ndarray, int, shape (n_windows,)
        Number of (accessible) bases in each window.
    counts : ndarray, int, shape (n_windows,)
        Number of variants in each window.

    See Also
    --------

    allel.model.locate_fixed_differences

    """

    # check inputs
    pos = SortedIndex(pos, copy=False)
    is_accessible = asarray_ndim(is_accessible, 1, allow_none=True)

    # locate fixed differences
    loc_df = locate_fixed_differences(ac1, ac2)

    # count number of fixed differences in windows
    n_df, windows, counts = windowed_statistic(
        pos, values=loc_df, statistic=np.count_nonzero, size=size, start=start,
        stop=stop, step=step, windows=windows, fill=0
    )

    # calculate value per base
    df, n_bases = per_base(n_df, windows, is_accessible=is_accessible,
                           fill=fill)

    return df, windows, n_bases, counts


# noinspection PyPep8Naming
def watterson_theta(pos, ac, start=None, stop=None,
                    is_accessible=None):
    """Calculate the value of Watterson's estimator over a given region.

    Parameters
    ----------

    pos : array_like, int, shape (n_items,)
        Variant positions, using 1-based coordinates, in ascending order.
    ac : array_like, int, shape (n_variants, n_alleles)
        Allele counts array.
    start : int, optional
        The position at which to start (1-based).
    stop : int, optional
        The position at which to stop (1-based).
    is_accessible : array_like, bool, shape (len(contig),), optional
        Boolean array indicating accessibility status for all positions in the
        chromosome/contig.

    Returns
    -------

    theta_hat_w : float
        Watterson's estimator (theta hat per base).

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
    ...                          [[0, 1], [-1, -1]],
    ...                          [[-1, -1], [-1, -1]]])
    >>> ac = g.count_alleles()
    >>> pos = [2, 4, 7, 14, 15, 18, 19, 25, 27]
    >>> theta_hat_w = allel.stats.watterson_theta(pos, ac, start=1, stop=31)
    >>> theta_hat_w
    0.10557184750733138

    """

    # check inputs
    if not isinstance(pos, SortedIndex):
        pos = SortedIndex(pos, copy=False)
    is_accessible = asarray_ndim(is_accessible, 1, allow_none=True)
    if not hasattr(ac, 'count_segregating'):
        ac = AlleleCountsArray(ac, copy=False)

    # deal with subregion
    if start is not None or stop is not None:
        loc = pos.locate_range(start, stop)
        pos = pos[loc]
        ac = ac[loc]
    if start is None:
        start = pos[0]
    if stop is None:
        stop = pos[-1]

    # count segregating variants
    S = ac.count_segregating()

    # assume number of chromosomes sampled is constant for all variants
    n = ac.sum(axis=1).max()

    # (n-1)th harmonic number
    a1 = np.sum(1 / np.arange(1, n))

    # calculate absolute value
    theta_hat_w_abs = S / a1

    # calculate value per base
    if is_accessible is None:
        n_bases = stop - start + 1
    else:
        n_bases = np.count_nonzero(is_accessible[start-1:stop])
    theta_hat_w = theta_hat_w_abs / n_bases

    return theta_hat_w


# noinspection PyPep8Naming
def windowed_watterson_theta(pos, ac, size=None, start=None, stop=None,
                             step=None, windows=None, is_accessible=None,
                             fill=np.nan):
    """Calculate the value of Watterson's estimator in windows over a single
    chromosome/contig.

    Parameters
    ----------

    pos : array_like, int, shape (n_items,)
        Variant positions, using 1-based coordinates, in ascending order.
    ac : array_like, int, shape (n_variants, n_alleles)
        Allele counts array.
    size : int, optional
        The window size (number of bases).
    start : int, optional
        The position at which to start (1-based).
    stop : int, optional
        The position at which to stop (1-based).
    step : int, optional
        The distance between start positions of windows. If not given,
        defaults to the window size, i.e., non-overlapping windows.
    windows : array_like, int, shape (n_windows, 2), optional
        Manually specify the windows to use as a sequence of (window_start,
        window_stop) positions, using 1-based coordinates. Overrides the
        size/start/stop/step parameters.
    is_accessible : array_like, bool, shape (len(contig),), optional
        Boolean array indicating accessibility status for all positions in the
        chromosome/contig.
    fill : object, optional
        The value to use where a window is completely inaccessible.

    Returns
    -------

    theta_hat_w : ndarray, float, shape (n_windows,)
        Watterson's estimator (theta hat per base).
    windows : ndarray, int, shape (n_windows, 2)
        The windows used, as an array of (window_start, window_stop) positions,
        using 1-based coordinates.
    n_bases : ndarray, int, shape (n_windows,)
        Number of (accessible) bases in each window.
    counts : ndarray, int, shape (n_windows,)
        Number of variants in each window.

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
    ...                          [[0, 1], [-1, -1]],
    ...                          [[-1, -1], [-1, -1]]])
    >>> ac = g.count_alleles()
    >>> pos = [2, 4, 7, 14, 15, 18, 19, 25, 27]
    >>> theta_hat_w, windows, n_bases, counts = allel.stats.windowed_watterson_theta(
    ...     pos, ac, size=10, start=1, stop=31
    ... )
    >>> theta_hat_w
    array([ 0.10909091,  0.16363636,  0.04958678])
    >>> windows
    array([[ 1, 10],
           [11, 20],
           [21, 31]])
    >>> n_bases
    array([10, 10, 11])
    >>> counts
    array([3, 4, 2])

    """  # flake8: noqa

    # check inputs
    if not isinstance(pos, SortedIndex):
        pos = SortedIndex(pos, copy=False)
    is_accessible = asarray_ndim(is_accessible, 1, allow_none=True)
    if not hasattr(ac, 'count_segregating'):
        ac = AlleleCountsArray(ac, copy=False)

    # locate segregating variants
    is_seg = ac.is_segregating()

    # count segregating variants in windows
    S, windows, counts = windowed_statistic(pos, is_seg,
                                            statistic=np.count_nonzero,
                                            size=size, start=start,
                                            stop=stop, step=step,
                                            windows=windows, fill=0)

    # assume number of chromosomes sampled is constant for all variants
    n = ac.sum(axis=1).max()

    # (n-1)th harmonic number
    a1 = np.sum(1 / np.arange(1, n))

    # absolute value of Watterson's theta
    theta_hat_w_abs = S / a1

    # theta per base
    theta_hat_w, n_bases = per_base(theta_hat_w_abs, windows=windows,
                                    is_accessible=is_accessible, fill=fill)

    return theta_hat_w, windows, n_bases, counts


# noinspection PyPep8Naming
def tajima_d(ac, pos=None, start=None, stop=None):
    """Calculate the value of Tajima's D over a given region.

    Parameters
    ----------
    ac : array_like, int, shape (n_variants, n_alleles)
        Allele counts array.
    pos : array_like, int, shape (n_items,), optional
        Variant positions, using 1-based coordinates, in ascending order.
    start : int, optional
        The position at which to start (1-based).
    stop : int, optional
        The position at which to stop (1-based).

    Returns
    -------
    D : float

    Examples
    --------

    >>> import allel
    >>> g = allel.GenotypeArray([[[0, 0], [0, 0]],
    ...                    [[0, 0], [0, 1]],
    ...                          [[0, 0], [1, 1]],
    ...                          [[0, 1], [1, 1]],
    ...                          [[1, 1], [1, 1]],
    ...                          [[0, 0], [1, 2]],
    ...                          [[0, 1], [1, 2]],
    ...                          [[0, 1], [-1, -1]],
    ...                          [[-1, -1], [-1, -1]]])
    >>> ac = g.count_alleles()
    >>> allel.stats.tajima_d(ac)
    3.1445848780213814
    >>> pos = [2, 4, 7, 14, 15, 18, 19, 25, 27]
    >>> allel.stats.tajima_d(ac, pos=pos, start=7, stop=25)
    3.8779735196179366

    """

    # check inputs
    if not hasattr(ac, 'count_segregating'):
        ac = AlleleCountsArray(ac, copy=False)

    # deal with subregion
    if pos is not None and (start is not None or stop is not None):
        if not isinstance(pos, SortedIndex):
            pos = SortedIndex(pos, copy=False)
        loc = pos.locate_range(start, stop)
        ac = ac[loc]

    # assume number of chromosomes sampled is constant for all variants
    n = ac.sum(axis=1).max()

    # count segregating variants
    S = ac.count_segregating()

    # (n-1)th harmonic number
    a1 = np.sum(1 / np.arange(1, n))

    # calculate Watterson's theta (absolute value)
    theta_hat_w_abs = S / a1

    # calculate mean pairwise difference
    mpd = mean_pairwise_difference(ac, fill=0)

    # calculate theta_hat pi (sum differences over variants)
    theta_hat_pi_abs = np.sum(mpd)

    # N.B., both theta estimates are usually divided by the number of
    # (accessible) bases but here we want the absolute difference
    d = theta_hat_pi_abs - theta_hat_w_abs

    # calculate the denominator (standard deviation)
    a2 = np.sum(1 / (np.arange(1, n)**2))
    b1 = (n + 1) / (3 * (n - 1))
    b2 = 2 * (n**2 + n + 3) / (9 * n * (n - 1))
    c1 = b1 - (1 / a1)
    c2 = b2 - ((n + 2) / (a1 * n)) + (a2 / (a1**2))
    e1 = c1 / a1
    e2 = c2 / (a1**2 + a2)
    d_stdev = np.sqrt((e1 * S) + (e2 * S * (S - 1)))

    # finally calculate Tajima's D
    D = d / d_stdev

    return D


# noinspection PyPep8Naming
def windowed_tajima_d(pos, ac, size=None, start=None, stop=None,
                      step=None, windows=None, fill=np.nan):
    """Calculate the value of Tajima's D in windows over a single
    chromosome/contig.

    Parameters
    ----------
    pos : array_like, int, shape (n_items,)
        Variant positions, using 1-based coordinates, in ascending order.
    ac : array_like, int, shape (n_variants, n_alleles)
        Allele counts array.
    size : int, optional
        The window size (number of bases).
    start : int, optional
        The position at which to start (1-based).
    stop : int, optional
        The position at which to stop (1-based).
    step : int, optional
        The distance between start positions of windows. If not given,
        defaults to the window size, i.e., non-overlapping windows.
    windows : array_like, int, shape (n_windows, 2), optional
        Manually specify the windows to use as a sequence of (window_start,
        window_stop) positions, using 1-based coordinates. Overrides the
        size/start/stop/step parameters.
    fill : object, optional
        The value to use where a window is completely inaccessible.

    Returns
    -------
    D : ndarray, float, shape (n_windows,)
        Tajima's D.
    windows : ndarray, int, shape (n_windows, 2)
        The windows used, as an array of (window_start, window_stop) positions,
        using 1-based coordinates.
    counts : ndarray, int, shape (n_windows,)
        Number of variants in each window.

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
    ...                          [[0, 1], [-1, -1]],
    ...                          [[-1, -1], [-1, -1]]])
    >>> ac = g.count_alleles()
    >>> pos = [2, 4, 7, 14, 15, 18, 19, 25, 27]
    >>> D, windows, counts = allel.stats.windowed_tajima_d(
    ...     pos, ac, size=10, start=1, stop=31
    ... )
    >>> D
    array([ 0.59158014,  2.93397641,  6.12372436])
    >>> windows
    array([[ 1, 10],
           [11, 20],
           [21, 31]])
    >>> counts
    array([3, 4, 2])

    """

    # check inputs
    if not isinstance(pos, SortedIndex):
        pos = SortedIndex(pos, copy=False)
    if not hasattr(ac, 'count_segregating'):
        ac = AlleleCountsArray(ac, copy=False)

    # assume number of chromosomes sampled is constant for all variants
    n = ac.sum(axis=1).max()

    # calculate constants
    a1 = np.sum(1 / np.arange(1, n))
    a2 = np.sum(1 / (np.arange(1, n)**2))
    b1 = (n + 1) / (3 * (n - 1))
    b2 = 2 * (n**2 + n + 3) / (9 * n * (n - 1))
    c1 = b1 - (1 / a1)
    c2 = b2 - ((n + 2) / (a1 * n)) + (a2 / (a1**2))
    e1 = c1 / a1
    e2 = c2 / (a1**2 + a2)

    # locate segregating variants
    is_seg = ac.is_segregating()

    # calculate mean pairwise difference
    mpd = mean_pairwise_difference(ac, fill=0)

    # define statistic to compute for each window
    # noinspection PyPep8Naming
    def statistic(w_is_seg, w_mpd):
        S = np.count_nonzero(w_is_seg)
        pi = np.sum(w_mpd)
        d = pi - (S / a1)
        d_stdev = np.sqrt((e1 * S) + (e2 * S * (S - 1)))
        wD = d / d_stdev
        return wD

    D, windows, counts = windowed_statistic(pos, values=(is_seg, mpd),
                                            statistic=statistic, size=size,
                                            start=start, stop=stop, step=step,
                                            windows=windows, fill=fill)

    return D, windows, counts


def moving_tajima_d(ac, size, start=0, stop=None, step=None):
    """Calculate the value of Tajima's D in moving windows of `size` variants.


    Parameters
    ----------
    ac : array_like, int, shape (n_variants, n_alleles)
        Allele counts array.
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
        Tajima's D.

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
    ...                          [[0, 1], [-1, -1]],
    ...                          [[-1, -1], [-1, -1]]])
    >>> ac = g.count_alleles()
    >>> D = allel.stats.moving_tajima_d(ac, size=3)
    >>> D
    array([ 0.59158014,  1.89305645,  5.79748537])

    """

    d = moving_statistic(values=ac, statistic=tajima_d, size=size, start=start, stop=stop,
                         step=step)
    return d
