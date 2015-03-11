# -*- coding: utf-8 -*-
# flake8: noqa
"""
This sub-package provides statistical functions for use with variant call data.

"""


from allel.stats.window import moving_statistic, windowed_count, \
    windowed_statistic, per_base

from allel.stats.diversity import mean_pairwise_diversity, \
    sequence_diversity, windowed_diversity, mean_pairwise_divergence, \
    sequence_divergence, windowed_divergence, \
    weir_cockerham_fst, hudson_fst

from allel.stats.distance import pairwise_distance, pairwise_dxy

from allel.stats.hw import heterozygosity_observed, heterozygosity_expected, \
    inbreeding_coefficient

from allel.stats.ld import rogers_huff_r, rogers_huff_r_between, \
    locate_unlinked
