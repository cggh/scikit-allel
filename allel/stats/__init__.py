# -*- coding: utf-8 -*-
# flake8: noqa
"""
This sub-package provides statistical functions for use with variant call data.

"""


from allel.stats.window import moving_statistic, windowed_count, \
    windowed_statistic, per_base, equally_accessible_windows, moving_mean, \
    moving_std, moving_midpoint

from allel.stats.diversity import mean_pairwise_difference, \
    sequence_diversity, windowed_diversity, mean_pairwise_difference_between, \
    sequence_divergence, windowed_divergence, windowed_df, watterson_theta, \
    windowed_watterson_theta, tajima_d, windowed_tajima_d, moving_tajima_d

from allel.stats.fst import weir_cockerham_fst, hudson_fst, \
    windowed_weir_cockerham_fst, windowed_hudson_fst, patterson_fst, \
    windowed_patterson_fst, blockwise_weir_cockerham_fst, \
    blockwise_hudson_fst, blockwise_patterson_fst

from allel.stats.distance import pairwise_distance, pairwise_dxy, pcoa, \
    plot_pairwise_distance, condensed_coords, condensed_coords_between, \
    condensed_coords_within

from allel.stats.hw import heterozygosity_observed, heterozygosity_expected, \
    inbreeding_coefficient

from allel.stats.ld import rogers_huff_r, rogers_huff_r_between, \
    locate_unlinked, plot_pairwise_ld

from allel.stats.decomposition import pca, randomized_pca

from allel.stats.preprocessing import StandardScaler, CenterScaler, \
    PattersonScaler

from allel.stats.admixture import patterson_f2, patterson_f3, patterson_d, \
    blockwise_patterson_f3, blockwise_patterson_d

from allel.stats.selection import ehh_decay, voight_painting, xpehh, ihs, \
    plot_voight_painting, fig_voight_painting, plot_haplotype_frequencies, \
    plot_moving_haplotype_frequencies, haplotype_diversity, \
    moving_haplotype_diversity, garud_h, moving_garud_h, nsl, xpnsl, \
    standardize, standardize_by_allele_count, moving_delta_tajima_d

from allel.stats.sf import sfs, sfs_folded, sfs_scaled, sfs_folded_scaled, \
    joint_sfs, joint_sfs_folded, joint_sfs_scaled, joint_sfs_folded_scaled, \
    fold_sfs, fold_joint_sfs, scale_sfs, scale_sfs_folded, scale_joint_sfs, \
    scale_joint_sfs_folded, plot_sfs, plot_sfs_folded, plot_sfs_scaled, \
    plot_sfs_folded_scaled, plot_joint_sfs, plot_joint_sfs_folded, \
    plot_joint_sfs_scaled, plot_joint_sfs_folded_scaled

from allel.stats.misc import plot_variant_locator

from allel.stats.mendel import mendel_errors, paint_transmission, \
    phase_progeny_by_transmission, phase_parents_by_transmission, \
    phase_by_transmission

from allel.stats.roh import roh_mhmm
