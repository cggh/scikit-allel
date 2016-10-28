# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np


from allel.stats.window import windowed_statistic
from allel.util import asarray_ndim, ensure_square
from allel.chunked import get_blen_array


def rogers_huff_r(gn, fill=np.nan):
    """Estimate the linkage disequilibrium parameter *r* for each pair of
    variants using the method of Rogers and Huff (2008).

    Parameters
    ----------
    gn : array_like, int8, shape (n_variants, n_samples)
        Diploid genotypes at biallelic variants, coded as the number of
        alternate alleles per call (i.e., 0 = hom ref, 1 = het, 2 = hom alt).
    fill : float, optional
        Value to use where r cannot be calculated.

    Returns
    -------
    r : ndarray, float, shape (n_variants * (n_variants - 1) // 2,)
        Matrix in condensed form.

    Examples
    --------

    >>> import allel
    >>> g = allel.GenotypeArray([[[0, 0], [1, 1], [0, 0]],
    ...                          [[0, 0], [1, 1], [0, 0]],
    ...                          [[1, 1], [0, 0], [1, 1]],
    ...                          [[0, 0], [0, 1], [-1, -1]]], dtype='i1')
    >>> gn = g.to_n_alt(fill=-1)
    >>> gn
    array([[ 0,  2,  0],
           [ 0,  2,  0],
           [ 2,  0,  2],
           [ 0,  1, -1]], dtype=int8)
    >>> r = allel.stats.rogers_huff_r(gn)
    >>> r  # doctest: +ELLIPSIS
    array([ 1.        , -1.00000012,  1.        , -1.00000012,  1.        , -1.        ], ...
    >>> r ** 2  # doctest: +ELLIPSIS
    array([ 1.        ,  1.00000024,  1.        ,  1.00000024,  1.        ,  1.        ], ...
    >>> from scipy.spatial.distance import squareform
    >>> squareform(r ** 2)
    array([[ 0.        ,  1.        ,  1.00000024,  1.        ],
           [ 1.        ,  0.        ,  1.00000024,  1.        ],
           [ 1.00000024,  1.00000024,  0.        ,  1.        ],
           [ 1.        ,  1.        ,  1.        ,  0.        ]])

    """

    # check inputs
    gn = asarray_ndim(gn, 2, dtype='i1')

    # compute correlation coefficients
    from allel.opt.stats import gn_pairwise_corrcoef_int8
    r = gn_pairwise_corrcoef_int8(gn, fill)

    # convenience for singletons
    if r.size == 1:
        r = r[0]

    return r


def rogers_huff_r_between(gna, gnb, fill=np.nan):
    """Estimate the linkage disequilibrium parameter *r* for each pair of
    variants between the two input arrays, using the method of Rogers and
    Huff (2008).

    Parameters
    ----------
    gna, gnb : array_like, int8, shape (n_variants, n_samples)
        Diploid genotypes at biallelic variants, coded as the number of
        alternate alleles per call (i.e., 0 = hom ref, 1 = het, 2 = hom alt).
    fill : float, optional
        Value to use where r cannot be calculated.

    Returns
    -------
    r : ndarray, float, shape (m_variants, n_variants )
        Matrix in rectangular form.

    """

    # check inputs
    gna = asarray_ndim(gna, 2, dtype='i1')
    gnb = asarray_ndim(gnb, 2, dtype='i1')

    # compute correlation coefficients
    from allel.opt.stats import gn_pairwise2_corrcoef_int8
    r = gn_pairwise2_corrcoef_int8(gna, gnb, fill)

    # convenience for singletons
    if r.size == 1:
        r = r[0, 0]

    return r


def locate_unlinked(gn, size=100, step=20, threshold=.1, blen=None):
    """Locate variants in approximate linkage equilibrium, where r**2 is
    below the given `threshold`.

    Parameters
    ----------
    gn : array_like, int8, shape (n_variants, n_samples)
        Diploid genotypes at biallelic variants, coded as the number of
        alternate alleles per call (i.e., 0 = hom ref, 1 = het, 2 = hom alt).
    size : int
        Window size (number of variants).
    step : int
        Number of variants to advance to the next window.
    threshold : float
        Maximum value of r**2 to include variants.
    blen : int, optional
        Block length to use for chunked computation.

    Returns
    -------
    loc : ndarray, bool, shape (n_variants)
        Boolean array where True items locate variants in approximate
        linkage equilibrium.

    Notes
    -----
    The value of r**2 between each pair of variants is calculated using the
    method of Rogers and Huff (2008).

    """

    from allel.opt.stats import gn_locate_unlinked_int8

    # check inputs
    if not hasattr(gn, 'shape') or not hasattr(gn, 'dtype'):
        gn = np.asarray(gn, dtype='i1')
    if gn.ndim != 2:
        raise ValueError('gn must have two dimensions')

    # setup output
    loc = np.ones(gn.shape[0], dtype='u1')

    # compute in chunks to avoid loading big arrays into memory
    blen = get_blen_array(gn, blen)
    blen = max(blen, 10*size)  # avoid too small chunks
    n_variants = gn.shape[0]
    for i in range(0, n_variants, blen):
        # N.B., ensure overlap with next window
        j = min(n_variants, i+blen+size)
        gnb = np.asarray(gn[i:j], dtype='i1')
        locb = loc[i:j]
        gn_locate_unlinked_int8(gnb, locb, size, step, threshold)

    return loc.astype('b1')


def windowed_r_squared(pos, gn, size=None, start=None, stop=None, step=None,
                       windows=None, fill=np.nan, percentile=50):
    """Summarise linkage disequilibrium in windows over a single
    chromosome/contig.

    Parameters
    ----------
    pos : array_like, int, shape (n_items,)
        The item positions in ascending order, using 1-based coordinates..
    gn : array_like, int8, shape (n_variants, n_samples)
        Diploid genotypes at biallelic variants, coded as the number of
        alternate alleles per call (i.e., 0 = hom ref, 1 = het, 2 = hom alt).
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
    percentile : int or sequence of ints, optional
        The percentile or percentiles to calculate within each window.

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
    Linkage disequilibrium (r**2) is calculated using the method of Rogers
    and Huff (2008).

    See Also
    --------

    allel.stats.window.windowed_statistic

    """

    # define the statistic function
    if isinstance(percentile, (list, tuple)):
        fill = [fill for _ in percentile]

        def statistic(gnw):
            r_squared = rogers_huff_r(gnw) ** 2
            return [np.percentile(r_squared, p) for p in percentile]

    else:
        def statistic(gnw):
            r_squared = rogers_huff_r(gnw) ** 2
            return np.percentile(r_squared, percentile)

    return windowed_statistic(pos, gn, statistic, size, start=start,
                              stop=stop, step=step, windows=windows, fill=fill)


def plot_pairwise_ld(m, colorbar=True, ax=None, imshow_kwargs=None):
    """Plot a matrix of genotype linkage disequilibrium values between
    all pairs of variants.

    Parameters
    ----------
    m : array_like
        Array of linkage disequilibrium values in condensed form.
    colorbar : bool, optional
        If True, add a colorbar to the current figure.
    ax : axes, optional
        The axes on which to draw. If not provided, a new figure will be
        created.
    imshow_kwargs : dict-like, optional
        Additional keyword arguments passed through to
        :func:`matplotlib.pyplot.imshow`.

    Returns
    -------
    ax : axes
        The axes on which the plot was drawn.

    """

    import matplotlib.pyplot as plt

    # check inputs
    m_square = ensure_square(m)

    # blank out lower triangle and flip up/down
    m_square = np.tril(m_square)[::-1, :]

    # set up axes
    if ax is None:
        # make a square figure with enough pixels to represent each variant
        x = m_square.shape[0] / plt.rcParams['savefig.dpi']
        x = max(x, plt.rcParams['figure.figsize'][0])
        fig, ax = plt.subplots(figsize=(x, x))
        fig.tight_layout(pad=0)

    # setup imshow arguments
    if imshow_kwargs is None:
        imshow_kwargs = dict()
    imshow_kwargs.setdefault('interpolation', 'none')
    imshow_kwargs.setdefault('cmap', 'Greys')
    imshow_kwargs.setdefault('vmin', 0)
    imshow_kwargs.setdefault('vmax', 1)

    # plot as image
    im = ax.imshow(m_square, **imshow_kwargs)

    # tidy up
    ax.set_xticks([])
    ax.set_yticks([])
    for s in 'bottom', 'right':
        ax.spines[s].set_visible(False)
    if colorbar:
        plt.gcf().colorbar(im, shrink=.5, pad=0)

    return ax
