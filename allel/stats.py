# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np


import allel.model
import allel.util


def nucleotide_diversity(g, pos, window, start=None, stop=None,
                         is_accessible=None, fill=np.nan):
    """TODO

    """

    # check inputs
    g = allel.model.GenotypeArray(g, copy=False)
    pos = allel.model.PositionIndex(pos, copy=False)
    if g.n_variants != pos.size:
        raise ValueError('genotype and position arrays with unequal length')

    # compute pairwise differences
    mpd = g.mean_pairwise_difference(fill=0)

    # mean per base
    pi, edges, widths = allel.util.windowed_mean_per_base(pos, mpd, window,
                                                          start=start,
                                                          stop=stop,
                                                          is_accessible=is_accessible,
                                                          fill=fill)

    return pi, edges, widths
