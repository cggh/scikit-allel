# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np


from allel.model.ndarray import SortedIndex
from allel.util import asarray_ndim, ignore_invalid, check_equal_length


def moving_statistic(values, statistic, size, start=0, stop=None, step=None):
    """Calculate a statistic in a moving window over `values`.

    Parameters
    ----------

    values : array_like
        The data to summarise.
    statistic : function
        The statistic to compute within each window.
    size : int
        The window size (number of values).
    start : int, optional
        The index at which to start.
    stop : int, optional
        The index at which to stop.
    step : int, optional
        The distance between start positions of windows. If not given,
        defaults to the window size, i.e., non-overlapping windows.

    Returns
    -------

    out : ndarray, shape (n_windows,)

    Examples
    --------

    >>> import allel
    >>> values = [2, 5, 8, 16]
    >>> allel.stats.moving_statistic(values, np.sum, size=2)
    array([ 7, 24])
    >>> allel.stats.moving_statistic(values, np.sum, size=2, step=1)
    array([ 7, 13, 24])

    """

    windows = index_windows(values, size, start, stop, step)

    # setup output
    out = np.array([statistic(values[i:j]) for i, j in windows])

    return out


def moving_mean(values, size, start=0, stop=None, step=None):
    return moving_statistic(values, statistic=np.mean, size=size,
                            start=start, stop=stop, step=step)


def moving_std(values, size, start=0, stop=None, step=None):
    return moving_statistic(values, statistic=np.std, size=size,
                            start=start, stop=stop, step=step)


def moving_midpoint(values, size, start=0, stop=None, step=None):
    return moving_statistic(values, statistic=lambda v: (v[0] + v[-1])/2,
                            size=size, start=start, stop=stop, step=step)


def index_windows(values, size, start, stop, step):
    """Convenience function to construct windows for the
    :func:`moving_statistic` function.

    """

    # determine step
    if stop is None:
        stop = len(values)
    if step is None:
        # non-overlapping
        step = size

    # iterate over windows
    for window_start in range(start, stop, step):

        window_stop = window_start + size
        if window_stop > stop:
            # ensure all windows are equal sized
            raise StopIteration

        yield (window_start, window_stop)


def position_windows(pos, size, start, stop, step):
    """Convenience function to construct windows for the
    :func:`windowed_statistic` and :func:`windowed_count` functions.

    """
    last = False

    # determine start and stop positions
    if start is None:
        start = pos[0]
    if stop is None:
        stop = pos[-1]
    if step is None:
        # non-overlapping
        step = size

    windows = []
    for window_start in range(start, stop, step):

        # determine window stop
        window_stop = window_start + size
        if window_stop >= stop:
            # last window
            window_stop = stop
            last = True
        else:
            window_stop -= 1

        windows.append([window_start, window_stop])

        if last:
            break

    return np.asarray(windows)


def window_locations(pos, windows):
    """Locate indices in `pos` corresponding to the start and stop positions
    of `windows`.

    """
    start_locs = np.searchsorted(pos, windows[:, 0])
    stop_locs = np.searchsorted(pos, windows[:, 1], side='right')
    locs = np.column_stack((start_locs, stop_locs))
    return locs


def windowed_count(pos, size=None, start=None, stop=None, step=None,
                   windows=None):
    """Count the number of items in windows over a single chromosome/contig.

    Parameters
    ----------

    pos : array_like, int, shape (n_items,)
        The item positions in ascending order, using 1-based coordinates..
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

    Returns
    -------

    counts : ndarray, int, shape (n_windows,)
        The number of items in each window.
    windows : ndarray, int, shape (n_windows, 2)
        The windows used, as an array of (window_start, window_stop) positions,
        using 1-based coordinates.

    Notes
    -----

    The window stop positions are included within a window.

    The final window will be truncated to the specified stop position,
    and so may be smaller than the other windows.

    Examples
    --------

    Non-overlapping windows::

        >>> import allel
        >>> pos = [1, 7, 12, 15, 28]
        >>> counts, windows = allel.stats.windowed_count(pos, size=10)
        >>> counts
        array([2, 2, 1])
        >>> windows
        array([[ 1, 10],
               [11, 20],
               [21, 28]])

    Half-overlapping windows::

        >>> counts, windows = allel.stats.windowed_count(pos, size=10, step=5)
        >>> counts
        array([2, 3, 2, 0, 1])
        >>> windows
        array([[ 1, 10],
               [ 6, 15],
               [11, 20],
               [16, 25],
               [21, 28]])

    """

    # assume sorted positions
    if not isinstance(pos, SortedIndex):
        pos = SortedIndex(pos, copy=False)

    # setup windows
    if windows is None:
        windows = position_windows(pos, size, start, stop, step)
    else:
        windows = asarray_ndim(windows, 2)

    # find window locations
    locs = window_locations(pos, windows)

    # count number of items in each window
    counts = np.diff(locs, axis=1).reshape(-1)

    return counts, windows


def windowed_statistic(pos, values, statistic, size=None, start=None,
                       stop=None, step=None, windows=None, fill=np.nan):
    """Calculate a statistic from items in windows over a single
    chromosome/contig.

    Parameters
    ----------

    pos : array_like, int, shape (n_items,)
        The item positions in ascending order, using 1-based coordinates..
    values : array_like, int, shape (n_items,)
        The values to summarise. May also be a tuple of values arrays,
        in which case each array will be sliced and passed through to the
        statistic function as separate arguments.
    statistic : function
        The statistic to compute.
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
        The value to use where a window is empty, i.e., contains no items.

    Returns
    -------

    out : ndarray, shape (n_windows,)
        The value of the statistic for each window.
    windows : ndarray, int, shape (n_windows, 2)
        The windows used, as an array of (window_start, window_stop) positions,
        using 1-based coordinates.
    counts : ndarray, int, shape (n_windows,)
        The number of items in each window.

    Notes
    -----

    The window stop positions are included within a window.

    The final window will be truncated to the specified stop position,
    and so may be smaller than the other windows.

    Examples
    --------

    Count non-zero (i.e., True) items in non-overlapping windows::

        >>> import allel
        >>> pos = [1, 7, 12, 15, 28]
        >>> values = [True, False, True, False, False]
        >>> nnz, windows, counts = allel.stats.windowed_statistic(
        ...     pos, values, statistic=np.count_nonzero, size=10
        ... )
        >>> nnz
        array([1, 1, 0])
        >>> windows
        array([[ 1, 10],
               [11, 20],
               [21, 28]])
        >>> counts
        array([2, 2, 1])

    Compute a sum over items in half-overlapping windows::

        >>> values = [3, 4, 2, 6, 9]
        >>> x, windows, counts = allel.stats.windowed_statistic(
        ...     pos, values, statistic=np.sum, size=10, step=5, fill=0
        ... )
        >>> x
        array([ 7, 12,  8,  0,  9])
        >>> windows
        array([[ 1, 10],
               [ 6, 15],
               [11, 20],
               [16, 25],
               [21, 28]])
        >>> counts
        array([2, 3, 2, 0, 1])

    """

    # assume sorted positions
    if not isinstance(pos, SortedIndex):
        pos = SortedIndex(pos, copy=False)

    # check lengths are equal
    if isinstance(values, tuple):
        # assume multiple values arrays
        check_equal_length(pos, *values)
    else:
        # assume a single values array
        check_equal_length(pos, values)

    # setup windows
    if windows is None:
        windows = position_windows(pos, size, start, stop, step)
    else:
        windows = asarray_ndim(windows, 2)

    # find window locations
    locs = window_locations(pos, windows)

    # setup outputs
    out = []
    counts = []

    # iterate over windows
    for start_idx, stop_idx in locs:

        # calculate number of values in window
        n = stop_idx - start_idx

        if n == 0:
            # window is empty
            s = fill

        else:

            if isinstance(values, tuple):
                # assume multiple values arrays
                wv = [v[start_idx:stop_idx] for v in values]
                s = statistic(*wv)

            else:
                # assume a single values array
                wv = values[start_idx:stop_idx]
                s = statistic(wv)

        # store outputs
        out.append(s)
        counts.append(n)

    # convert to arrays for output
    return np.asarray(out), windows, np.asarray(counts)


def per_base(x, windows, is_accessible=None, fill=np.nan):
    """Calculate the per-base value of a windowed statistic.

    Parameters
    ----------

    x : array_like, shape (n_windows,)
        The statistic to average per-base.
    windows : array_like, int, shape (n_windows, 2)
        The windows used, as an array of (window_start, window_stop)
        positions using 1-based coordinates.
    is_accessible : array_like, bool, shape (len(contig),), optional
        Boolean array indicating accessibility status for all positions in the
        chromosome/contig.
    fill : object, optional
        Use this value where there are no accessible bases in a window.

    Returns
    -------

    y : ndarray, float, shape (n_windows,)
        The input array divided by the number of (accessible) bases in each
        window.
    n_bases : ndarray, int, shape (n_windows,)
        The number of (accessible) bases in each window

    """

    # calculate window sizes
    if is_accessible is None:
        # N.B., window stops are included
        n_bases = np.diff(windows, axis=1).reshape(-1) + 1
    else:
        n_bases = np.array([np.count_nonzero(is_accessible[i-1:j])
                            for i, j in windows])

    # deal with multidimensional x
    if x.ndim == 1:
        pass
    elif x.ndim == 2:
        n_bases = n_bases[:, None]
    else:
        raise NotImplementedError('only arrays of 1 or 2 dimensions supported')

    # calculate density per-base
    with ignore_invalid():
        y = np.where(n_bases > 0, x / n_bases, fill)

    # restore to 1-dimensional
    if n_bases.ndim > 1:
        n_bases = n_bases.reshape(-1)

    return y, n_bases


def equally_accessible_windows(is_accessible, size):
    """Create windows each containing the same number of accessible bases.

    Parameters
    ----------
    is_accessible : array_like, bool, shape (n_bases,)
        Array defining accessible status of all bases on a contig/chromosome.
    size : int
        Window size (number of accessible bases).

    Returns
    -------
    windows : ndarray, int, shape (n_windows, 2)
        Window start/stop positions (1-based).

    """
    pos_accessible, = np.nonzero(is_accessible)
    pos_accessible += 1  # convert to 1-based coordinates
    windows = moving_statistic(pos_accessible, lambda v: [v[0], v[-1]],
                               size=size)
    return windows
