# -*- coding: utf-8 -*-
def test_public_api():
    # The idea of this test is to ensure that all functions we expect to be in the
    # public API under the correct namespace are indeed there.

    import allel

    # allel.model.ndarray
    assert callable(allel.GenotypeVector)
    assert callable(allel.GenotypeArray)
    assert callable(allel.HaplotypeArray)
    assert callable(allel.AlleleCountsArray)
    assert callable(allel.GenotypeAlleleCountsVector)
    assert callable(allel.GenotypeAlleleCountsArray)
    assert callable(allel.SortedIndex)
    assert callable(allel.UniqueIndex)
    assert callable(allel.SortedMultiIndex)
    assert callable(allel.VariantTable)
    assert callable(allel.FeatureTable)

    # allel.model.dask
    assert callable(allel.GenotypeDaskVector)
    assert callable(allel.GenotypeDaskArray)
    assert callable(allel.HaplotypeDaskArray)
    assert callable(allel.AlleleCountsDaskArray)
    assert callable(allel.GenotypeAlleleCountsDaskVector)
    assert callable(allel.GenotypeAlleleCountsDaskArray)

    # allel.model.chunked
    assert callable(allel.GenotypeChunkedArray)
    assert callable(allel.HaplotypeChunkedArray)
    assert callable(allel.AlleleCountsChunkedArray)
    assert callable(allel.GenotypeAlleleCountsChunkedArray)
    assert callable(allel.VariantChunkedTable)
    assert callable(allel.AlleleCountsChunkedTable)

    # allel.model.util
    assert callable(allel.create_allele_mapping)
    assert callable(allel.locate_fixed_differences)
    assert callable(allel.locate_private_alleles)
    assert callable(allel.sample_to_haplotype_selection)

    # allel.io.fasta
    assert callable(allel.write_fasta)

    # allel.io.gff
    assert callable(allel.iter_gff3)
    assert callable(allel.gff3_to_recarray)
    assert callable(allel.gff3_to_dataframe)
    assert callable(allel.gff3_parse_attributes)

    # allel.io.vcf_read
    assert callable(allel.read_vcf)
    assert callable(allel.vcf_to_npz)
    assert callable(allel.vcf_to_hdf5)
    assert callable(allel.vcf_to_zarr)
    assert callable(allel.iter_vcf_chunks)
    assert callable(allel.read_vcf_headers)
    assert callable(allel.vcf_to_dataframe)
    assert callable(allel.vcf_to_csv)
    assert callable(allel.vcf_to_recarray)

    # allel.io.vcf_write
    assert callable(allel.write_vcf)
    assert callable(allel.write_vcf_header)
    assert callable(allel.write_vcf_data)

    # allel.stats.admixture
    assert callable(allel.patterson_f2)
    assert callable(allel.patterson_f3)
    assert callable(allel.patterson_d)
    assert callable(allel.moving_patterson_f3)
    assert callable(allel.moving_patterson_d)
    assert callable(allel.average_patterson_f3)
    assert callable(allel.average_patterson_d)
    # backwards compatibility
    assert callable(allel.blockwise_patterson_f3)
    assert callable(allel.blockwise_patterson_d)

    # allel.stats.decomposition
    assert callable(allel.pca)
    assert callable(allel.randomized_pca)

    # allel.stats.distance
    assert callable(allel.pairwise_distance)
    assert callable(allel.pairwise_dxy)
    assert callable(allel.pcoa)
    assert callable(allel.condensed_coords)
    assert callable(allel.condensed_coords_within)
    assert callable(allel.condensed_coords_between)
    assert callable(allel.plot_pairwise_distance)

    # allel.stats.diversity
    assert callable(allel.mean_pairwise_difference)
    assert callable(allel.mean_pairwise_difference_between)
    assert callable(allel.sequence_diversity)
    assert callable(allel.sequence_divergence)
    assert callable(allel.windowed_diversity)
    assert callable(allel.windowed_divergence)
    assert callable(allel.windowed_df)
    assert callable(allel.watterson_theta)
    assert callable(allel.windowed_watterson_theta)
    assert callable(allel.tajima_d)
    assert callable(allel.windowed_tajima_d)
    assert callable(allel.moving_tajima_d)

    # allel.stats.fst
    assert callable(allel.weir_cockerham_fst)
    assert callable(allel.hudson_fst)
    assert callable(allel.patterson_fst)
    assert callable(allel.windowed_weir_cockerham_fst)
    assert callable(allel.windowed_hudson_fst)
    assert callable(allel.windowed_patterson_fst)
    assert callable(allel.moving_weir_cockerham_fst)
    assert callable(allel.moving_hudson_fst)
    assert callable(allel.moving_patterson_fst)
    assert callable(allel.average_weir_cockerham_fst)
    assert callable(allel.average_hudson_fst)
    assert callable(allel.average_patterson_fst)
    # backwards compatibility
    assert callable(allel.blockwise_weir_cockerham_fst)
    assert callable(allel.blockwise_hudson_fst)
    assert callable(allel.blockwise_patterson_fst)

    # allel.stats.hw
    assert callable(allel.heterozygosity_observed)
    assert callable(allel.heterozygosity_expected)
    assert callable(allel.inbreeding_coefficient)

    # allel.stats.ld
    assert callable(allel.rogers_huff_r)
    assert callable(allel.rogers_huff_r_between)
    assert callable(allel.locate_unlinked)
    assert callable(allel.windowed_r_squared)
    assert callable(allel.plot_pairwise_ld)

    # allel.stats.mendel
    assert callable(allel.mendel_errors)
    assert callable(allel.paint_transmission)
    assert callable(allel.phase_progeny_by_transmission)
    assert callable(allel.phase_parents_by_transmission)
    assert callable(allel.phase_by_transmission)

    # allel.stats.misc
    assert callable(allel.plot_variant_locator)
    assert callable(allel.tabulate_state_transitions)
    assert callable(allel.tabulate_state_blocks)

    # allel.stats.preprocessing
    assert callable(allel.get_scaler)
    assert callable(allel.StandardScaler)
    assert callable(allel.CenterScaler)
    assert callable(allel.PattersonScaler)

    # allel.stats.roh
    assert callable(allel.roh_mhmm)
    assert callable(allel.roh_poissonhmm)

    # allel.stats.roh
    assert callable(allel.ehh_decay)
    assert callable(allel.voight_painting)
    assert callable(allel.plot_voight_painting)
    assert callable(allel.fig_voight_painting)
    assert callable(allel.ihs)
    assert callable(allel.xpehh)
    assert callable(allel.nsl)
    assert callable(allel.xpnsl)
    assert callable(allel.haplotype_diversity)
    assert callable(allel.moving_haplotype_diversity)
    assert callable(allel.garud_h)
    assert callable(allel.moving_garud_h)
    assert callable(allel.plot_haplotype_frequencies)
    assert callable(allel.moving_delta_tajima_d)
    assert callable(allel.standardize)
    assert callable(allel.standardize_by_allele_count)
    assert callable(allel.pbs)

    # allel.stats.sf
    assert callable(allel.sfs)
    assert callable(allel.sfs_folded)
    assert callable(allel.sfs_scaled)
    assert callable(allel.sfs_folded_scaled)
    assert callable(allel.scale_sfs)
    assert callable(allel.scale_sfs_folded)
    assert callable(allel.joint_sfs)
    assert callable(allel.joint_sfs_folded)
    assert callable(allel.joint_sfs_scaled)
    assert callable(allel.joint_sfs_folded_scaled)
    assert callable(allel.scale_joint_sfs)
    assert callable(allel.scale_joint_sfs_folded)
    assert callable(allel.fold_sfs)
    assert callable(allel.plot_sfs)
    assert callable(allel.plot_sfs_folded)
    assert callable(allel.plot_sfs_scaled)
    assert callable(allel.plot_sfs_folded_scaled)
    assert callable(allel.plot_joint_sfs)
    assert callable(allel.plot_joint_sfs_folded)
    assert callable(allel.plot_joint_sfs_scaled)
    assert callable(allel.plot_joint_sfs_folded_scaled)

    # allel.stats.window
    assert callable(allel.moving_statistic)
    assert callable(allel.moving_mean)
    assert callable(allel.moving_std)
    assert callable(allel.moving_midpoint)
    assert callable(allel.index_windows)
    assert callable(allel.position_windows)
    assert callable(allel.window_locations)
    assert callable(allel.windowed_count)
    assert callable(allel.windowed_statistic)
    assert callable(allel.per_base)
    assert callable(allel.equally_accessible_windows)

    # allel.util
    assert callable(allel.hdf5_cache)
    # N.B., check this is not clobbered, see
    # https://github.com/cggh/scikit-allel/issues/163
    import allel.util
    assert callable(allel.util.hdf5_cache)
