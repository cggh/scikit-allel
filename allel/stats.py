# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np


from allel.model import SortedIndex
from allel.util import asarray_ndim, ignore_invalid, check_arrays_aligned


def moving_statistic(values, statistic, size, start=0, stop=None, step=None):
    """TODO doco

    """

    # determine step
    if stop is None:
        stop = len(values)
    if step is None:
        step = size

    # setup output
    out = []

    # iterate over windows
    for window_start in range(start, stop, step):

        # determine window stop
        window_stop = window_start + size
        if window_stop >= stop:
            # last window
            window_stop = stop
            last = True
        else:
            last = False

        # extract values for window
        window_values = values[window_start:window_stop]

        # compute statistic
        s = statistic(window_values)

        # store outputs
        out.append(s)

        if last:
            break

    # convert to arrays for output
    return np.array(out)


def windowed_count(pos, size, start=None, stop=None, step=None):
    """TODO doco

    """

    # assume sorted positions
    pos = SortedIndex(pos, copy=False)

    # determine start and stop positions
    if start is None:
        start = pos[0]
    if stop is None:
        stop = pos[-1]
    if step is None:
        step = size

    # setup outputs
    counts = []
    windows = []

    # iterate over windows
    for window_start in range(start, stop, step):

        # determine window stop
        window_stop = window_start + size
        if window_stop >= stop:
            # last window
            window_stop = stop
            last = True
        else:
            window_stop -= 1
            last = False

        # locate window
        try:
            loc = pos.locate_range(window_start, window_stop)
        except KeyError:
            n = 0
        else:
            n = loc.stop - loc.start

        # store outputs
        counts.append(n)
        windows.append([window_start, window_stop])

        if last:
            break

    # convert to arrays for output
    return np.array(counts), np.array(windows)


def windowed_statistic(pos, values, statistic, size, start=None, stop=None,
                       step=None, fill=np.nan):
    """TODO doco

    """

    # assume sorted positions
    pos = SortedIndex(pos, copy=False)

    # check lengths are equal
    if len(pos) != len(values):
        raise ValueError('arrays must be of equal length')

    # determine start and stop positions
    if start is None:
        start = pos[0]
    if stop is None:
        stop = pos[-1]
    if step is None:
        step = size

    # setup outputs
    out = []
    windows = []
    counts = []

    # iterate over windows
    for window_start in range(start, stop, step):

        # determine window stop
        window_stop = window_start + size
        if window_stop >= stop:
            # last window
            window_stop = stop
            last = True
        else:
            window_stop -= 1
            last = False

        # locate window
        try:
            loc = pos.locate_range(window_start, window_stop)
        except KeyError:
            n = 0
            s = fill
        else:
            n = loc.stop - loc.start
            # extract values for window
            window_values = values[loc]
            # compute statistic
            s = statistic(window_values)

        # store outputs
        out.append(s)
        windows.append([window_start, window_stop])
        counts.append(n)

        if last:
            break

    # convert to arrays for output
    return np.array(out), np.array(windows), np.array(counts)


def mean_pairwise_diversity(ac, an, fill=np.nan):
    """Calculate the mean number of pairwise differences between
    haplotypes for each variant.

    Parameters
    ----------

    ac : array_like, int, shape (n_variants, n_alleles)
        Allele counts array.
    an : array_like, int, shape (n_variants,)
        Allele number array.
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
    nucleotide diversity.

    Examples
    --------

    >>> import allel
    >>> h = allel.model.HaplotypeArray([[0, 0, 0, 0],
    ...                                 [0, 0, 0, 1],
    ...                                 [0, 0, 1, 1],
    ...                                 [0, 1, 1, 1],
    ...                                 [1, 1, 1, 1],
    ...                                 [0, 0, 1, 2],
    ...                                 [0, 1, 1, 2],
    ...                                 [0, 1, -1, -1]])
    >>> ac = h.allele_counts()
    >>> an = h.allele_number()
    >>> allel.stats.mean_pairwise_diversity(ac, an)
    array([ 0.        ,  0.5       ,  0.66666667,  0.5       ,  0.        ,
            0.83333333,  0.83333333,  1.        ])

    """

    # This function calculates the mean number of pairwise differences
    # between haplotypes within a single population, generalising to any number
    # of alleles.

    # check inputs
    ac = asarray_ndim(ac, 2)
    an = asarray_ndim(an, 1)

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


def _resize_dim2(a, l):
    newshape = a.shape[0], l
    b = np.zeros(newshape, dtype=a.dtype)
    b[:, :a.shape[1]] = a
    return b


def mean_pairwise_divergence(pop1_ac, pop2_ac, pop1_an, pop2_an, fill=np.nan):
    """TODO doco

    """

    # This function calculates the mean number of pairwise differences
    # between haplotypes from two different populations, generalising to any
    # number of alleles.

    # check inputs
    pop1_ac = asarray_ndim(pop1_ac, 2)
    pop1_an = asarray_ndim(pop1_an, 1)
    pop2_ac = asarray_ndim(pop2_ac, 2)
    pop2_an = asarray_ndim(pop2_an, 1)
    # check lengths match
    check_arrays_aligned(pop1_ac, pop1_an)
    check_arrays_aligned(pop1_ac, pop2_ac)
    check_arrays_aligned(pop1_ac, pop2_an)
    # ensure same number of alleles in both pops
    if pop1_ac.shape[1] < pop2_ac.shape[1]:
        pop1_ac = _resize_dim2(pop1_ac, pop2_ac.shape[1])
    elif pop2_ac.shape[1] < pop1_ac.shape[1]:
        pop2_ac = _resize_dim2(pop2_ac, pop1_ac.shape[1])

    # total number of pairwise comparisons for each variant
    n_pairs = pop1_an * pop2_an

    # number of pairwise comparisons where there is no difference:
    # sum of (pop1_ac * pop2_ac) for each allele (i.e., number of ways to
    # choose the same allele twice)
    n_same = np.sum(pop1_ac * pop2_ac, axis=1)

    # number of pairwise differences
    n_diff = n_pairs - n_same

    # mean number of pairwise differences, accounting for cases where
    # there are no pairs
    with ignore_invalid():
        mpd = np.where(n_pairs > 0, n_diff / n_pairs, fill)

    return mpd


def windowed_diversity(pos, ac, an, size, start=None, stop=None, step=None,
                       is_accessible=None, fill=np.nan):

    # check inputs
    pos = SortedIndex(pos, copy=False)
    is_accessible = asarray_ndim(is_accessible, 1, allow_none=True)

    # calculate mean pairwise diversity
    mpd = mean_pairwise_diversity(ac, an, fill=0)

    # sum in windows
    mpd_sum, windows, _ = windowed_statistic(pos, values=mpd, statistic=np.sum,
                                             size=size, start=start, stop=stop,
                                             step=step, fill=0)

    # TODO refactor per-base calculation code

    # calculate window sizes
    if is_accessible is None:
        # N.B., window stops are included
        n_bases = np.diff(windows, axis=1).reshape(-1) + 1
    else:
        pos_accessible = np.nonzero(is_accessible)[0] + 1  # use 1-based coords
        n_bases, _ = windowed_count(pos, size, start=start, stop=stop,
                                    step=step)

    with ignore_invalid():
        pi = np.where(n_bases > 0, mpd_sum / n_bases, fill)

    return pi, windows, n_bases


def windowed_divergence(pos, pop1_ac, pop2_ac, pop1_an, pop2_an, size,
                        start=None, stop=None, step=None,
                        is_accessible=None, fill=np.nan):

    # check inputs
    pos = SortedIndex(pos, copy=False)
    is_accessible = asarray_ndim(is_accessible, 1, allow_none=True)

    # calculate mean pairwise divergence
    mpd = mean_pairwise_divergence(pop1_ac, pop2_ac, pop1_an, pop2_an, fill=0)

    # sum in windows
    mpd_sum, windows, _ = windowed_statistic(pos, values=mpd, statistic=np.sum,
                                             size=size, start=start, stop=stop,
                                             step=step, fill=0)

    # TODO refactor per-base calculation code

    # calculate window sizes
    if is_accessible is None:
        # N.B., window stops are included
        n_bases = np.diff(windows, axis=1).reshape(-1) + 1
    else:
        pos_accessible = np.nonzero(is_accessible)[0] + 1  # use 1-based coords
        n_bases, _ = windowed_count(pos, size, start=start, stop=stop,
                                    step=step)

    with ignore_invalid():
        dxy = np.where(n_bases > 0, mpd_sum / n_bases, fill)

    return dxy, windows, n_bases
