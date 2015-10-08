# -*- coding: utf-8 -*-
"""
Plotting functions.

N.B., this module is deprecated and plotting functions have been moved into
various statistics modules.

"""
from __future__ import absolute_import, print_function, division


from allel.stats.misc import plot_variant_locator
from allel.stats.distance import plot_pairwise_distance
from allel.stats.ld import plot_pairwise_ld
from allel.stats.selection import plot_voight_painting


# plotting functions have been moved, keep these here for backwards
# compatibility
variant_locator = plot_variant_locator
pairwise_distance = plot_pairwise_distance
pairwise_ld = plot_pairwise_ld
voight_painting = plot_voight_painting
