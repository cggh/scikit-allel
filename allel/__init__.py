# -*- coding: utf-8 -*-
# flake8: noqa

from .model.ndarray import *
from .model.chunked import *
from .model.util import *
try:
    import dask
except ImportError:
    pass
else:
    from .model.dask import *

from .stats.window import moving_statistic, windowed_count, \
    windowed_statistic, per_base, equally_accessible_windows, moving_mean, \
    moving_std, moving_midpoint, index_windows, position_windows, window_locations

from .stats.diversity import mean_pairwise_difference, \
    sequence_diversity, windowed_diversity, mean_pairwise_difference_between, \
    sequence_divergence, windowed_divergence, windowed_df, watterson_theta, \
    windowed_watterson_theta, tajima_d, windowed_tajima_d, moving_tajima_d

from .stats.fst import weir_cockerham_fst, hudson_fst, \
    windowed_weir_cockerham_fst, windowed_hudson_fst, patterson_fst, \
    windowed_patterson_fst, blockwise_weir_cockerham_fst, \
    blockwise_hudson_fst, blockwise_patterson_fst, average_hudson_fst, \
    average_patterson_fst, average_weir_cockerham_fst, moving_hudson_fst, \
    moving_patterson_fst, moving_weir_cockerham_fst

from .stats.distance import pairwise_distance, pairwise_dxy, pcoa, \
    plot_pairwise_distance, condensed_coords, condensed_coords_between, \
    condensed_coords_within

from .stats.hw import heterozygosity_observed, heterozygosity_expected, \
    inbreeding_coefficient

from .stats.ld import rogers_huff_r, rogers_huff_r_between, \
    locate_unlinked, plot_pairwise_ld, windowed_r_squared

from .stats.decomposition import pca, randomized_pca

from .stats.preprocessing import StandardScaler, CenterScaler, PattersonScaler, get_scaler

from .stats.admixture import patterson_f2, patterson_f3, patterson_d, \
    blockwise_patterson_f3, blockwise_patterson_d, average_patterson_d, \
    average_patterson_f3, moving_patterson_d, moving_patterson_f3

from .stats.selection import ehh_decay, voight_painting, xpehh, ihs, \
    plot_voight_painting, fig_voight_painting, plot_haplotype_frequencies, \
    plot_moving_haplotype_frequencies, haplotype_diversity, \
    moving_haplotype_diversity, garud_h, moving_garud_h, nsl, xpnsl, \
    standardize, standardize_by_allele_count, moving_delta_tajima_d, pbs

from .stats.sf import sfs, sfs_folded, sfs_scaled, sfs_folded_scaled, \
    joint_sfs, joint_sfs_folded, joint_sfs_scaled, joint_sfs_folded_scaled, \
    fold_sfs, fold_joint_sfs, scale_sfs, scale_sfs_folded, scale_joint_sfs, \
    scale_joint_sfs_folded, plot_sfs, plot_sfs_folded, plot_sfs_scaled, \
    plot_sfs_folded_scaled, plot_joint_sfs, plot_joint_sfs_folded, \
    plot_joint_sfs_scaled, plot_joint_sfs_folded_scaled

from .stats.misc import plot_variant_locator, tabulate_state_transitions, \
    tabulate_state_blocks

from .stats.mendel import mendel_errors, paint_transmission, \
    phase_progeny_by_transmission, phase_parents_by_transmission, \
    phase_by_transmission, INHERIT_MISSING, INHERIT_NONPARENTAL, INHERIT_NONSEG_ALT, \
    INHERIT_NONSEG_REF, INHERIT_PARENT1, INHERIT_PARENT2, INHERIT_PARENT_MISSING, \
    INHERIT_UNDETERMINED

from .stats.roh import roh_mhmm, roh_poissonhmm

from .io.vcf_read import *
from .io.vcf_write import *
from .io.gff import *
from .io.fasta import *
from .io.util import *

from .util import hdf5_cache

from .version import version as __version__
