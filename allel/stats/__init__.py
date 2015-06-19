# -*- coding: utf-8 -*-
# flake8: noqa
"""
This sub-package provides statistical functions for use with variant call data.

"""


from allel.stats.window import moving_statistic, windowed_count, \
    windowed_statistic, per_base

from allel.stats.diversity import mean_pairwise_difference, \
    sequence_diversity, windowed_diversity, mean_pairwise_difference_between, \
    sequence_divergence, windowed_divergence, windowed_df, watterson_theta, \
    windowed_watterson_theta, tajima_d, windowed_tajima_d

from allel.stats.fst import weir_cockerham_fst, hudson_fst, \
    windowed_weir_cockerham_fst, windowed_hudson_fst, patterson_fst, \
    windowed_patterson_fst, blockwise_weir_cockerham_fst, \
    blockwise_hudson_fst, blockwise_patterson_fst

from allel.stats.distance import pairwise_distance, pairwise_dxy, pcoa

from allel.stats.hw import heterozygosity_observed, heterozygosity_expected, \
    inbreeding_coefficient

from allel.stats.ld import rogers_huff_r, rogers_huff_r_between, \
    locate_unlinked

from allel.stats.decomposition import pca, randomized_pca

from allel.stats.preprocessing import StandardScaler, CenterScaler, \
    PattersonScaler

from allel.stats.admixture import patterson_f2, patterson_f3, patterson_d, \
    blockwise_patterson_f3, blockwise_patterson_d
