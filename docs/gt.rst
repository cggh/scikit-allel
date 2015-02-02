Genotype arrays - :mod:`allel.gt`
=================================

.. automodule:: allel.gt

Call matching functions
-----------------------

.. autofunction:: is_missing
.. autofunction:: is_called
.. autofunction:: is_hom
.. autofunction:: is_hom_ref
.. autofunction:: is_hom_alt
.. autofunction:: is_het
.. autofunction:: is_call

Genotype transformations
------------------------

.. autofunction:: to_haplotypes
.. autofunction:: from_haplotypes
.. autofunction:: to_n_alt
.. autofunction:: to_allele_counts
.. autofunction:: to_packed
.. autofunction:: from_packed
.. autofunction:: to_sparse
.. autofunction:: from_sparse

Allele frequency calculations
-----------------------------

.. autofunction:: max_allele
.. autofunction:: allelism
.. autofunction:: allele_number
.. autofunction:: allele_count
.. autofunction:: allele_frequency
.. autofunction:: allele_counts
.. autofunction:: allele_frequencies
.. autofunction:: is_variant
.. autofunction:: is_non_variant
.. autofunction:: is_segregating
.. autofunction:: is_non_segregating
.. autofunction:: is_singleton
.. autofunction:: is_doubleton

Counting functions
------------------

.. autofunction:: count
.. autofunction:: windowed_count
.. autofunction:: windowed_density

Plotting functions
------------------

.. autofunction:: plot_discrete_calldata
.. autofunction:: plot_continuous_calldata
.. autofunction:: plot_diploid_genotypes
.. autofunction:: plot_genotype_counts_by_sample
.. autofunction:: plot_genotype_counts_by_variant
.. autofunction:: plot_continuous_calldata_by_sample
.. autofunction:: plot_windowed_call_count
.. autofunction:: plot_windowed_call_density
