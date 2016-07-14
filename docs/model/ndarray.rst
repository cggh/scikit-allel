In-memory data structures
=========================

.. automodule:: allel.model.ndarray

GenotypeArray
-------------

.. autoclass:: GenotypeArray

    .. autoattribute:: n_variants
    .. autoattribute:: n_samples
    .. autoattribute:: ploidy
    .. autoattribute:: mask
    .. automethod:: fill_masked
    .. automethod:: subset
    .. automethod:: is_called
    .. automethod:: is_missing
    .. automethod:: is_hom
    .. automethod:: is_hom_ref
    .. automethod:: is_hom_alt
    .. automethod:: is_het
    .. automethod:: is_call
    .. automethod:: count_called
    .. automethod:: count_missing
    .. automethod:: count_hom
    .. automethod:: count_hom_ref
    .. automethod:: count_hom_alt
    .. automethod:: count_het
    .. automethod:: count_call
    .. automethod:: count_alleles
    .. automethod:: count_alleles_subpops
    .. automethod:: map_alleles
    .. automethod:: to_haplotypes
    .. automethod:: to_n_ref
    .. automethod:: to_n_alt
    .. automethod:: to_allele_counts
    .. automethod:: to_packed
    .. automethod:: from_packed
    .. automethod:: to_sparse
    .. automethod:: from_sparse
    .. automethod:: to_gt
    .. automethod:: haploidify_samples
    .. automethod:: vstack
    .. automethod:: hstack


HaplotypeArray
--------------

.. autoclass:: HaplotypeArray

    .. autoattribute:: n_variants
    .. autoattribute:: n_haplotypes
    .. automethod:: subset
    .. automethod:: is_called
    .. automethod:: is_missing
    .. automethod:: is_ref
    .. automethod:: is_alt
    .. automethod:: is_call
    .. automethod:: count_called
    .. automethod:: count_missing
    .. automethod:: count_ref
    .. automethod:: count_alt
    .. automethod:: count_call
    .. automethod:: count_alleles
    .. automethod:: count_alleles_subpops
    .. automethod:: map_alleles
    .. automethod:: to_genotypes
    .. automethod:: to_sparse
    .. automethod:: from_sparse
    .. automethod:: prefix_argsort
    .. automethod:: distinct
    .. automethod:: distinct_counts
    .. automethod:: distinct_frequencies
    .. automethod:: vstack
    .. automethod:: hstack


AlleleCountsArray
-----------------

.. autoclass:: AlleleCountsArray

    .. autoattribute:: n_variants
    .. autoattribute:: n_alleles
    .. automethod:: max_allele
    .. automethod:: allelism
    .. automethod:: is_variant
    .. automethod:: is_non_variant
    .. automethod:: is_segregating
    .. automethod:: is_non_segregating
    .. automethod:: is_singleton
    .. automethod:: is_doubleton
    .. automethod:: is_biallelic
    .. automethod:: is_biallelic_01
    .. automethod:: count_variant
    .. automethod:: count_non_variant
    .. automethod:: count_segregating
    .. automethod:: count_non_segregating
    .. automethod:: count_singleton
    .. automethod:: count_doubleton
    .. automethod:: to_frequencies
    .. automethod:: map_alleles
    .. automethod:: vstack
    .. automethod:: hstack

VariantTable
------------

.. autoclass:: VariantTable

    .. autoattribute:: n_variants
    .. autoattribute:: names
    .. automethod:: eval
    .. automethod:: query
    .. automethod:: query_position
    .. automethod:: query_region
    .. automethod:: to_vcf

FeatureTable
------------

.. autoclass:: FeatureTable

    .. autoattribute:: n_features
    .. autoattribute:: names
    .. automethod:: eval
    .. automethod:: query
    .. automethod:: from_gff3
    .. automethod:: to_mask

SortedIndex
-----------

.. autoclass:: SortedIndex

    .. autoattribute:: is_unique
    .. automethod:: locate_key
    .. automethod:: locate_keys
    .. automethod:: locate_intersection
    .. automethod:: intersect
    .. automethod:: locate_range
    .. automethod:: intersect_range
    .. automethod:: locate_ranges
    .. automethod:: locate_intersection_ranges
    .. automethod:: intersect_ranges


UniqueIndex
-----------

.. autoclass:: UniqueIndex

    .. automethod:: locate_key
    .. automethod:: locate_keys
    .. automethod:: locate_intersection
    .. automethod:: intersect

SortedMultiIndex
----------------

.. autoclass:: SortedMultiIndex

    .. automethod:: locate_key
    .. automethod:: locate_range

Utility functions
-----------------

.. autofunction:: create_allele_mapping
.. autofunction:: locate_fixed_differences
.. autofunction:: locate_private_alleles
.. autofunction:: sample_to_haplotype_selection
