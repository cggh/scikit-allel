:mod:`allel.gt` - discrete genotype arrays
==========================================

.. automodule:: allel.gt

Finding and counting calls
--------------------------

.. autofunction:: is_called
.. autofunction:: is_missing
.. autofunction:: is_hom
.. autofunction:: is_het
.. autofunction:: is_hom_ref
.. autofunction:: is_hom_alt
.. autofunction:: count_called
.. autofunction:: count_missing
.. autofunction:: count_hom
.. autofunction:: count_het
.. autofunction:: count_hom_ref
.. autofunction:: count_hom_alt

Genotype transformations
------------------------

.. autofunction:: as_haplotypes
.. autofunction:: as_n_alt
.. autofunction:: as_012
.. autofunction:: as_allele_counts
.. autofunction:: pack_diploid
.. autofunction:: unpack_diploid

Allele frequencies
------------------

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
.. autofunction:: count_variant
.. autofunction:: count_non_variant
.. autofunction:: count_segregating
.. autofunction:: count_non_segregating
.. autofunction:: count_singleton
.. autofunction:: count_doubleton

Windowed genotype calculations
------------------------------

.. autofunction:: windowed_genotype_counts
.. autofunction:: windowed_genotype_density
.. autofunction:: windowed_genotype_rate

Plotting functions
------------------

.. autofunction:: plot_discrete_calldata
.. autofunction:: plot_continuous_calldata
.. autofunction:: plot_diploid_genotypes
.. autofunction:: plot_genotype_counts_by_sample
.. autofunction:: plot_genotype_counts_by_variant
.. autofunction:: plot_continuous_calldata_by_sample
.. autofunction:: plot_windowed_genotype_counts
.. autofunction:: plot_windowed_genotype_density
.. autofunction:: plot_windowed_genotype_rate
