In-memory data structures
=========================

.. automodule:: allel.model.ndarray

GenotypeArray
-------------

.. autoclass:: GenotypeArray

    .. autoattribute:: n_variants
    .. autoattribute:: n_samples
    .. autoattribute:: ploidy
    .. autoattribute:: n_calls
    .. autoattribute:: n_allele_calls
    .. automethod:: count_alleles
    .. automethod:: count_alleles_subpops
    .. automethod:: to_packed
    .. automethod:: from_packed
    .. automethod:: to_sparse
    .. automethod:: from_sparse
    .. automethod:: haploidify_samples
    .. automethod:: subset

GenotypeVector
--------------

.. autoclass:: GenotypeVector

    .. autoattribute:: n_calls
    .. autoattribute:: ploidy
    .. autoattribute:: n_allele_calls

Genotypes
---------

Methods available on both :class:`GenotypeArray` and :class:`GenotypeVector` classes:

.. autoclass:: Genotypes

    .. autoattribute:: mask
    .. autoattribute:: is_phased
    .. automethod:: fill_masked
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
    .. automethod:: to_n_ref
    .. automethod:: to_n_alt
    .. automethod:: to_allele_counts
    .. automethod:: to_gt
    .. automethod:: map_alleles
    .. automethod:: compress
    .. automethod:: take
    .. automethod:: concatenate

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
    .. automethod:: compress
    .. automethod:: take
    .. automethod:: subset
    .. automethod:: concatenate


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
    .. automethod:: compress
    .. automethod:: take
    .. automethod:: concatenate

GenotypeAlleleCountsArray
-------------------------

.. autoclass:: GenotypeAlleleCountsArray

    .. autoattribute:: n_variants
    .. autoattribute:: n_samples
    .. autoattribute:: n_alleles
    .. automethod:: count_alleles
    .. automethod:: subset

GenotypeAlleleCountsVector
--------------------------

.. autoclass:: GenotypeAlleleCountsVector

    .. autoattribute:: n_calls
    .. autoattribute:: n_alleles

GenotypeAlleleCounts
--------------------

Methods available on both :class:`GenotypeAlleleCountsArray` and
:class:`GenotypeAlleleCountsVector` classes:

.. autoclass:: GenotypeAlleleCounts

    .. automethod:: is_called
    .. automethod:: is_missing
    .. automethod:: is_hom
    .. automethod:: is_hom_ref
    .. automethod:: is_hom_alt
    .. automethod:: is_het
    .. automethod:: compress
    .. automethod:: take
    .. automethod:: concatenate

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


SortedMultiIndex
----------------

.. autoclass:: SortedMultiIndex

    .. automethod:: locate_key
    .. automethod:: locate_range

UniqueIndex
-----------

.. autoclass:: UniqueIndex

    .. automethod:: locate_key
    .. automethod:: locate_keys
    .. automethod:: locate_intersection
    .. automethod:: intersect
