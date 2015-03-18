# -*- coding: utf-8 -*-
"""
Plotting functions for variant call data.

"""
from __future__ import absolute_import, print_function, division


import numpy as np


import allel.model
from allel.util import ensure_square


def variant_locator(pos, step=None, ax=None, start=None,
                    stop=None, flip=False, line_kwargs=None):
    """
    Plot lines indicating the physical genome location of variants from a
    single chromosome/contig. By default the top x axis is in variant index
    space, and the bottom x axis is in genome position space.

    Parameters
    ----------

    pos : array_like
        A sorted 1-dimensional array of genomic positions from a single
        chromosome/contig.
    step : int, optional
        Plot a line for every `step` variants.
    ax : axes, optional
        The axes on which to draw. If not provided, a new figure will be
        created.
    start : int, optional
        The start position for the region to draw.
    stop : int, optional
        The stop position for the region to draw.
    flip : bool, optional
        Flip the plot upside down.
    line_kwargs : dict-like
        Additional keyword arguments passed through to `plt.Line2D`.

    Returns
    -------

    ax : axes
        The axes on which the plot was drawn

    """

    import matplotlib.pyplot as plt

    # check inputs
    pos = allel.model.SortedIndex(pos, copy=False)

    # set up axes
    if ax is None:
        x = plt.rcParams['figure.figsize'][0]
        y = x / 7
        fig, ax = plt.subplots(figsize=(x, y))
        fig.tight_layout()

    # determine x axis limits
    if start is None:
        start = np.min(pos)
    if stop is None:
        stop = np.max(pos)
    loc = pos.locate_range(start, stop)
    pos = pos[loc]
    if step is None:
        step = len(pos) // 100
    ax.set_xlim(start, stop)

    # plot the lines
    if line_kwargs is None:
        line_kwargs = dict()
    # line_kwargs.setdefault('linewidth', .5)
    n_variants = len(pos)
    for i, p in enumerate(pos[::step]):
        xfrom = p
        xto = (
            start +
            ((i * step / n_variants) * (stop-start))
        )
        l = plt.Line2D([xfrom, xto], [0, 1], **line_kwargs)
        ax.add_line(l)

    # invert?
    if flip:
        ax.invert_yaxis()
        ax.xaxis.tick_top()
    else:
        ax.xaxis.tick_bottom()

    # tidy up
    ax.set_yticks([])
    ax.xaxis.set_tick_params(direction='out')
    for l in 'left', 'right':
        ax.spines[l].set_visible(False)

    return ax


def pairwise_distance(dist, labels=None, colorbar=True, ax=None,
                      imshow_kwargs=None):
    """Plot a pairwise distance matrix.

    Parameters
    ----------

    dist : array_like
        The distance matrix in condensed form.
    labels : sequence of strings, optional
        Sample labels for the axes.
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
        The axes on which the plot was drawn

    """

    import matplotlib.pyplot as plt

    # check inputs
    dist_square = ensure_square(dist)

    # set up axes
    if ax is None:
        # make a square figure
        x = plt.rcParams['figure.figsize'][0]
        fig, ax = plt.subplots(figsize=(x, x))
        fig.tight_layout()

    # setup imshow arguments
    if imshow_kwargs is None:
        imshow_kwargs = dict()
    imshow_kwargs.setdefault('interpolation', 'none')
    imshow_kwargs.setdefault('cmap', 'jet')
    imshow_kwargs.setdefault('vmin', np.min(dist))
    imshow_kwargs.setdefault('vmax', np.max(dist))

    # plot as image
    im = ax.imshow(dist_square, **imshow_kwargs)

    # tidy up
    if labels:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels, rotation=0)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    if colorbar:
        plt.gcf().colorbar(im, shrink=.5)

    return ax


def pairwise_ld(m, colorbar=True, ax=None, imshow_kwargs=None):
    """Plot a matrix of linkage disequilibrium values between pairs of
    variants.

    Parameters
    ----------

    m : array_like
        LD matrix in condensed form.
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
        The axes on which the plot was drawn

    """

    import matplotlib.pyplot as plt

    # check inputs
    m_square = ensure_square(m)

    # blank out lower triangle and flip up/down
    m_square = np.tril(m_square)[::-1, :]

    # set up axes
    if ax is None:
        # make a square figure
        x = plt.rcParams['figure.figsize'][0]
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
