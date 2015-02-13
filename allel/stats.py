# -*- coding: utf-8 -*-
"""
This module defines statistical functions for use with variant call data.

"""
from __future__ import absolute_import, print_function, division


import numpy as np
import scipy.stats


from allel.model import GenotypeArray, PositionIndex
from allel.util import ignore_invalid, check_arrays_aligned, asarray_ndim


def make_window_edges(pos, window, start=None, stop=None):
    region_start = pos.min() if start is None else start
    region_stop = pos.max() if stop is None else stop
    edges = np.arange(region_start, region_stop, window)
    if stop is None and edges[-1] < region_stop:
        # add one more window to ensure stop is included
        edges = np.append(edges, edges[-1] + window)
    elif stop is not None and edges[-1] < region_stop:
        # add one more window to ensure explicit stop is final edge
        edges = np.append(edges, stop)
    return edges


def windowed_statistic(pos, values, window, statistic, start=None, stop=None):
    """Compute a statistic in non-overlapping windows over a single
    chromosome/contig.

    Parameters
    ----------

    pos : array_like, int, shape (n_variants,)
        Positions array (1-based).
    values : array_like, shape (n_variants,)
        Values to be summarised within each window.
    window : int
        Window size.
    statistic : string or function
        Statistic to compute.
    start : int, optional
        Start position of first window (1-based).
    stop : int, optional
        Stop position of last window (1-based).

    Returns
    -------

    s : ndarray, shape (n_windows,)
        Computed statistic.
    edges : ndarray, shape (n_windows + 1,)
        Window edge positions.

    """

    # check inputs
    pos = np.asarray(pos)
    values = np.asarray(values)
    check_arrays_aligned(pos, values)

    # determine bin edges
    edges = make_window_edges(pos, window, start, stop)

    # compute statistic
    s, _, _ = scipy.stats.binned_statistic(pos, values, statistic=statistic,
                                           bins=edges)

    return s, edges


def windowed_nnz(pos, b, window, start=None, stop=None):
    """Count nonzero (i.e., True) elements in non-overlapping windows over a
    single chromosome/contig.

    Parameters
    ----------

    pos : array_like, int, shape (n_variants,)
        Positions array.
    b : array_like, bool, shape (n_variants,) or (n_variants, n_samples)
        Boolean array.
    window : int
        Window size.
    start : int, optional
        Start position of first window (1-based).
    stop : int, optional
        Stop position of last window (1-based).

    Returns
    -------

    counts : ndarray, int, shape (n_windows,) or (n_windows, n_samples)
        Counts array.
    edges : ndarray, shape (n_windows + 1,)
        Window edge positions.

    Examples
    --------

    >>> import allel
    >>> g = allel.GenotypeArray([[[0, 0], [0, 0]],
    ...                          [[0, 1], [0, 1]],
    ...                          [[1, 1], [1, 2]],
    ...                          [[2, 2], [-1, -1]]], dtype='i1')
    >>> pos = allel.PositionIndex([2, 14, 15, 27])
    >>> b = g.is_variant()
    >>> counts, edges = allel.windowed_nnz(pos, b, window=10)
    >>> edges
    array([ 2, 12, 22, 32])
    >>> counts
    array([0, 2, 1])
    >>> counts, edges = allel.windowed_nnz(pos, b, window=10, start=1,
    ...                                      stop=27)
    >>> edges
    array([ 1, 11, 21, 27])
    >>> counts
    array([0, 2, 1])
    >>> b = g.is_het()
    >>> counts, edges = allel.windowed_nnz(pos, b, window=10)
    >>> edges
    array([ 2, 12, 22, 32])
    >>> counts
    array([[0, 0],
           [1, 2],
           [0, 0]])

    """

    # check arguments
    pos = PositionIndex(pos)
    b = asarray_ndim(b, 1, 2)
    check_arrays_aligned(pos, b)

    # determine bin edges
    edges = make_window_edges(pos, window, start, stop)

    if b.ndim == 1:
        # 1D case
        pos_true = np.compress(b, pos)
        counts, _ = np.histogram(pos_true, bins=edges)

    else:
        # 2D case
        n_bins = len(edges) - 1
        n_samples = b.shape[1]
        counts = np.empty((n_bins, n_samples), dtype=int)
        for i in range(n_samples):
            pos_true = np.compress(b[:, i], pos)
            h, _ = np.histogram(pos_true, bins=edges)
            counts[:, i] = h

    return counts, edges


def per_base(edges, values, is_accessible=None, fill=np.nan):
    """Compute the per-base value of a windowed statistic, optionally taking
    into account genome accessibility.

    Parameters
    ----------

    edges : array_like, int, shape (n_windows + 1,)
        Window edge positions (1-based).
    values : array_like, shape (n_windows,)
        Values for each window.
    is_accessible : array_like, bool, shape (len(contig),), optional
        Accessibility mask. If provided, the size of each window will be
        calculated as the number of accessible positions, rather than the
        absolute window size.
    fill : optional
        Fill value to use if window size is zero.

    Returns
    -------

    m : array_like, shape (n_windows,)
        Values divided by corresponding window sizes.
    widths : array_like, int, shape (n_windows,)
        Number of accessible positions in each window.

    """

    # check inputs
    values = asarray_ndim(values, 1, 2)
    is_accessible = asarray_ndim(is_accessible, 1, allow_none=True)

    # determine window sizes
    if is_accessible is None:
        # assume all genome positions are accessible
        widths = np.diff(edges)
        # final bin includes right edge
        widths[-1] += 1

    else:
        # check accessibility array spans all windows
        if is_accessible.size < edges[-1]:
            raise ValueError('accessibility array does not span region')
        pos_accessible, = np.nonzero(is_accessible)
        # convert to 1-based coordinates
        pos_accessible += 1
        widths, _ = np.histogram(pos_accessible, bins=edges)

    if values.ndim == 2:
        # insert singleton dimension to enable broadcasting
        widths = widths[:, None]

    # calculate value per base
    with ignore_invalid():
        m = np.where(widths > 0, values / widths, fill)

    return m, widths[:]


def windowed_mean_per_base(pos, values, window, start=None, stop=None,
                           is_accessible=None, fill=np.nan):
    """Calculate the mean per genome position of the given values in
    non-overlapping windows over a single chromosome/contig.

    Parameters
    ----------

    pos : array_like, int, shape (n_variants,)
        Positions array.
    values : array_like, shape (n_variants,)
        Values to be summarised within each window.
    window : int
        Window size.
    start : int, optional
        Start position of first window (1-based).
    stop : int, optional
        Stop position of last window (1-based).
    is_accessible : array_like, bool, shape (len(contig),), optional
        Accessibility mask. If provided, the size of each window will be
        calculated as the number of accessible positions, rather than the
        absolute window size.
    fill : optional
        Fill value to use if window size is zero.

    Returns
    -------

    m : ndarray, shape (n_windows,)
        Mean per base of values in each window.
    edges : ndarray, shape (n_windows + 1,)
        Window edge positions.
    widths : array_like, int, shape (n_windows,)
        Number of accessible positions in each window.

    """

    s, edges = windowed_statistic(pos, values, window, statistic='sum',
                                  start=start, stop=stop)
    m, widths = per_base(edges, s, is_accessible=is_accessible, fill=fill)
    return m, edges, widths


def windowed_nnz_per_base(pos, b, window, start=None, stop=None,
                          is_accessible=None, fill=np.nan):
    """Calculate the per-base-pair density of nonzero (i.e., True) elements in
    non-overlapping windows over a single chromosome/contig.

    Parameters
    ----------

    pos : array_like, int, shape (n_variants,)
        Positions array.
    b : array_like, bool, shape (n_variants,) or (n_variants, n_samples)
        Boolean array.
    window : int
        Window size.
    start : int, optional
        Start position of first window (1-based).
    stop : int, optional
        Stop position of last window (1-based).
    is_accessible : array_like, bool, shape (len(contig),), optional
        Accessibility mask. If provided, the size of each window will be
        calculated as the number of accessible positions, rather than the
        absolute window size.
    fill : optional
        Fill value to use if window size is zero.

    Returns
    -------

    densities : ndarray, shape (n_windows,) or (n_windows, n_samples)
        Per base density of True values in each window.
    edges : ndarray, shape (n_windows + 1,)
        Window edge positions.
    counts : ndarray, int, shape (n_windows,) or (n_windows, n_samples)
        Counts array.
    widths : array_like, int, shape (n_windows,)
        Number of accessible positions in each window.

    Examples
    --------

    Assuming all positions are accessible::

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 0]],
        ...                          [[0, 1], [0, 1]],
        ...                          [[1, 1], [1, 2]],
        ...                          [[2, 2], [-1, -1]]], dtype='i1')
        >>> pos = allel.PositionIndex([2, 14, 15, 27])
        >>> b = g.is_variant()
        >>> densities, edges, counts, widths = allel.windowed_nnz_per_base(
        ...     pos, b, window=10
        ... )
        >>> edges
        array([ 2, 12, 22, 32])
        >>> widths
        array([10, 10, 11])
        >>> counts
        array([0, 2, 1])
        >>> densities
        array([ 0.        ,  0.2       ,  0.09090909])

    Density calculations can take into account the number of accessible
    positions within each window, e.g.::

        >>> is_accessible = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ...                           0, 1, 1, 1, 0, 0, 1, 1, 0, 0,
        ...                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ...                           dtype=bool)
        >>> densities, edges, counts, widths = allel.windowed_nnz_per_base(
        ...     pos, b, start=1, stop=31, window=10,
        ...     is_accessible=is_accessible, fill=np.nan
        ... )
        >>> edges
        array([ 1, 11, 21, 31])
        >>> widths
        array([ 0,  5, 11])
        >>> counts
        array([0, 2, 1])
        >>> densities
        array([        nan,  0.4     ,  0.09090909])

    """

    counts, edges = windowed_nnz(pos, b, window, start=start, stop=stop)
    densities, widths = per_base(edges, counts, is_accessible=is_accessible,
                                 fill=fill)
    return densities, edges, counts, widths


def windowed_nucleotide_diversity(g, pos, window, start=None, stop=None,
                                  is_accessible=None, fill=np.nan):
    """Calculate nucleotide diversity in non-overlapping windows over a
    single chromosome/contig.

    Parameters
    ----------

    g : array_like, int, shape (n_variants, n_samples, ploidy)
        Genotype array.
    pos : array_like, int, shape (n_variants,)
        Positions array.
    window : int
        Window size.
    start : int, optional
        Start position of first window (1-based).
    stop : int, optional
        Stop position of last window (1-based).
    is_accessible : array_like, bool, shape (len(contig),), optional
        Accessibility mask. If provided, the size of each window will be
        calculated as the number of accessible positions, rather than the
        absolute window size.
    fill : optional
        Fill value to use if window size is zero.

    Returns
    -------

    pi : ndarray, shape (n_windows,)
        Nucleotide diversity in each window.
    edges : ndarray, shape (n_windows + 1,)
        Window edge positions.
    widths : array_like, int, shape (n_windows,)
        Number of accessible positions in each window.

    """

    # check inputs
    g = GenotypeArray(g, copy=False)
    pos = PositionIndex(pos, copy=False)
    check_arrays_aligned(g, pos)

    # compute pairwise differences
    mpd = g.mean_pairwise_difference(fill=0)

    # mean per base
    pi, edges, widths = windowed_mean_per_base(
        pos, mpd, window, start=start, stop=stop, is_accessible=is_accessible,
        fill=fill
    )

    return pi, edges, widths
