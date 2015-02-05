# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np
from allel.compat import range


import allel.model


def windowed_count(pos, b, window, start=None, stop=None):
    """Count nonzero (i.e., True) elements in non-overlapping windows over a
    single chromosome or contig.

    Parameters
    ----------

    pos : array_like, int, shape (n_variants,)
        Positions array.
    b : array_like, bool, shape (n_variants,) or (n_variants, n_samples)
        Boolean array.
    window : int
        Window size.
    start : int, optional
        Start position.
    stop : int, optional
        Stop position.

    Returns
    -------

    counts : ndarray, int, shape (n_bins,) or (n_bins, n_samples)
        Counts array.
    edges : ndarray, int, shape (n_bins + 1,)
        Bin edges used for counting.

    Examples
    --------

    >>> import allel
    >>> g = allel.GenotypeArray([[[0, 0], [0, 0]],
    ...                          [[0, 1], [0, 1]],
    ...                          [[1, 1], [1, 2]],
    ...                          [[2, 2], [-1, -1]]], dtype='i1')
    >>> pos = allel.PositionIndex([2, 14, 15, 27])
    >>> b = g.is_variant()
    >>> counts, edges = allel.windowed_count(pos, b, window=10)
    >>> edges
    array([ 2, 12, 22, 32])
    >>> counts
    array([0, 2, 1])
    >>> counts, edges = allel.windowed_count(pos, b, window=10, start=1,
    ...                                      stop=27)
    >>> edges
    array([ 1, 11, 21, 27])
    >>> counts
    array([0, 2, 1])
    >>> b = g.is_het()
    >>> counts, edges = allel.windowed_count(pos, b, window=10)
    >>> edges
    array([ 2, 12, 22, 32])
    >>> counts
    array([[0, 0],
           [1, 2],
           [0, 0]])

    """

    # check arguments
    pos = allel.model.PositionIndex(pos)
    b = np.asarray(b)
    if pos.shape[0] != b.shape[0]:
        raise ValueError(
            'arrays do not have matching length for first '
            'dimension: pos %s, b %s'
            % (pos.shape[0], b.shape[0])
        )

    # determine bin edges
    bin_start = pos.min() if start is None else start
    bin_stop = pos.max() if stop is None else stop
    edges = np.arange(bin_start, bin_stop, window)
    if stop is None and edges[-1] < bin_stop:
        # add one more window to ensure stop is included
        edges = np.append(edges, edges[-1] + window)
    elif stop is not None and edges[-1] < bin_stop:
        # add one more window to ensure explicit stop is final edge
        edges = np.append(edges, stop)

    if b.ndim == 1:
        pos_incl = np.compress(b, pos)
        counts, _ = np.histogram(pos_incl, bins=edges)

    else:
        # assume 2D
        n_bins = len(edges) - 1
        n_samples = b.shape[1]
        counts = np.empty((n_bins, n_samples), dtype=int)
        for i in range(n_samples):
            pos_incl = np.compress(b[:, i], pos)
            h, _ = np.histogram(pos_incl, bins=edges)
            counts[:, i] = h

    return counts, edges


def windowed_density(pos, b, window, start=None, stop=None,
                     is_accessible=None, fill=0):
    """Calculate the per-base-pair density of nonzero (i.e., True) elements in
    non-overlapping windows over a single chromosome or contig.

    Parameters
    ----------

    pos : array_like, int, shape (n_variants,)
        Positions array.
    b : array_like, bool, shape (n_variants,) or (n_variants, n_samples)
        Boolean array.
    window : int
        Window size.
    start : int, optional
        Start position.
    stop : int, optional
        Stop position.
    is_accessible : array_like, bool, shape (len(contig),), optional
        Accessibility mask. If provided, the size of each window will be
        calculated as the number of accessible positions, rather than the
        absolute bin width.

    Returns
    -------

    densities : ndarray, int, shape (n_bins,) or (n_bins, n_samples)
        Densities array.
    counts : ndarray, int, shape (n_bins,) or (n_bins, n_samples)
        Counts array.
    widths : ndarray, int, shape (n_bins,)
        Size of each bin, taking accessibility into account.
    edges : ndarray, int, shape (n_bins + 1,)
        Bin edges used for counting.

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
        >>> densities, counts, widths, edges = allel.windowed_density(
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
        >>> densities, counts, widths, edges = allel.windowed_density(
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

    counts, edges = windowed_count(pos, b, window, start=start, stop=stop)

    # determine window sizes (i.e., bin widths)
    if is_accessible is None:
        # assume all genome positions are accessible
        widths = np.diff(edges)
        # final bin includes right edge
        widths[-1] += 1
    else:
        is_accessible = np.asarray(is_accessible)
        if is_accessible.ndim != 1:
            raise ValueError('expected 1 dimension, found %s' %
                             is_accessible.ndim)
        pos_accessible, = np.nonzero(is_accessible)
        # convert to 1-based coordinates
        pos_accessible += 1
        widths, _ = np.histogram(pos_accessible, bins=edges)

    if counts.ndim > 1:
        widths = widths[:, None]

    # calculate densities
    err = np.seterr(invalid='ignore')
    densities = np.where(widths > 0, counts / widths, fill)
    np.seterr(**err)

    return densities, counts, widths, edges
