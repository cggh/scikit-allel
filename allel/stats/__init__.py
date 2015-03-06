# -*- coding: utf-8 -*-
"""
This sub-package provides statistical functions for use with variant call data.

"""
from __future__ import absolute_import, print_function, division


from allel.stats.window import moving_statistic, windowed_count, \
    windowed_statistic, per_base

from allel.stats.diversity import mean_pairwise_diversity, \
    sequence_diversity, windowed_diversity, mean_pairwise_divergence, \
    sequence_divergence, windowed_divergence

from allel.stats.distance import pairwise_distance, pairwise_dxy

from allel.stats.hw import heterozygosity_observed, heterozygosity_expected, \
    inbreeding_coefficient
