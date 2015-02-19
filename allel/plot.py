# -*- coding: utf-8 -*-
"""
Plotting functions for variant call data.

"""
from __future__ import absolute_import, print_function, division


import numpy as np


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
    import scipy.spatial

    # normalise inputs
    dist_square = scipy.spatial.distance.squareform(dist, force='tomatrix')

    # set up axes
    if ax is None:
        # make a square figure
        x = plt.rcParams['figure.figsize'][0]
        fig, ax = plt.subplots(figsize=(x, x))

    # setup imshow arguments
    if imshow_kwargs is None:
        imshow_kwargs = dict()
    imshow_kwargs.setdefault('interpolation', 'none')
    imshow_kwargs.setdefault('cmap', 'jet')
    # normalisation
    dist = scipy.spatial.distance.squareform(dist_square)
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
    if colorbar:
        plt.gcf().colorbar(im)

    return ax
