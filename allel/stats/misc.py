# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np


from allel.model.ndarray import SortedIndex


def jackknife(values, statistic):
    """Estimate standard error for `statistic` computed over `values` using
    the jackknife.

    Parameters
    ----------
    values : array_like or tuple of array_like
        Input array, or tuple of input arrays.
    statistic : function
        The statistic to compute.

    Returns
    -------
    m : float
        Mean of jackknife values.
    se : float
        Estimate of standard error.
    vj : ndarray
        Statistic values computed for each jackknife iteration.

    """

    if isinstance(values, tuple):
        # multiple input arrays
        n = len(values[0])
        masked_values = [np.ma.asarray(v) for v in values]
        for m in masked_values:
            assert m.ndim == 1, 'only 1D arrays supported'
            assert m.shape[0] == n, 'input arrays not of equal length'
            m.mask = np.zeros(m.shape, dtype=bool)

    else:
        n = len(values)
        masked_values = np.ma.asarray(values)
        assert masked_values.ndim == 1, 'only 1D arrays supported'
        masked_values.mask = np.zeros(masked_values.shape, dtype=bool)

    # values of the statistic calculated in each jackknife iteration
    vj = list()

    for i in range(n):

        if isinstance(values, tuple):
            # multiple input arrays
            for m in masked_values:
                m.mask[i] = True
            x = statistic(*masked_values)
            for m in masked_values:
                m.mask[i] = False

        else:
            masked_values.mask[i] = True
            x = statistic(masked_values)
            masked_values.mask[i] = False

        vj.append(x)

    # convert to array for convenience
    vj = np.array(vj)

    # compute mean of jackknife values
    m = vj.mean()

    # compute standard error
    sv = ((n - 1) / n) * np.sum((vj - m) ** 2)
    se = np.sqrt(sv)

    return m, se, vj


def plot_variant_locator(pos, step=None, ax=None, start=None,
                         stop=None, flip=False,
                         line_kwargs=None):
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
    pos = SortedIndex(pos, copy=False)

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
