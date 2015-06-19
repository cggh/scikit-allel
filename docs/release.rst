Release notes
=============

v0.15
-----

* Added functions to estimate Fst with standard error via a
  block-jackknife:
  :func:`allel.stats.fst.blockwise_weir_cockerham_fst`,
  :func:`allel.stats.fst.blockwise_hudson_fst`,
  :func:`allel.stats.fst.blockwise_patterson_fst`.

* Fixed a serious bug in :func:`allel.stats.fst.weir_cockerham_fst`
  related to incorrect estimation of heterozygosity, which manifested
  if the subpopulations being compared were not a partition of the
  total population (i.e., there were one or more samples in the
  genotype array that were not included in the subpopulations to
  compare).

* Added method :func:`allel.model.AlleleCountsArray.max_allele` to
  determine highest allele index for each variant.

* Changed first return value from admixture functions
  :func:`allel.stats.admixture.blockwise_patterson_f3` and
  :func:`allel.stats.admixture.blockwise_patterson_d` to return the
  estimator from the whole dataset.

* Added utility functions to the :mod:`allel.stats.distance` module
  for transforming coordinates between condensed and uncondensed
  forms of a distance matrix.

* Classes previously available from the `allel.model` and
  `allel.bcolz` modules are now aliased from the root :mod:`allel`
  module for convenience. These modules have been reorganised into an
  :mod:`allel.model` package with sub-modules
  :mod:`allel.model.ndarray` and :mod:`allel.model.bcolz`.

* All functions in the :mod:`allel.model.bcolz` module use cparams from
  input carray as default for output carray (convenient if you, e.g.,
  want to use zlib level 1 throughout).

* All classes in the :mod:`allel.model.ndarray` and
  :mod:`allel.model.bcolz` modules have changed the default value for
  the `copy` keyword argument to `False`. This means that **not**
  copying the input data, just wrapping it, is now the default
  behaviour.

* Fixed bug in :func:`GenotypeArray.to_gt` where maximum allele index
  is zero.

v0.14
-----

* Added a new module :mod:`allel.stats.admixture` with statistical
  tests for admixture between populations, implementing the f2, f3 and
  D statistics from Patterson (2012). Functions include
  :func:`allel.stats.admixture.blockwise_patterson_f3` and
  :func:`allel.stats.admixture.blockwise_patterson_d` which compute
  the f3 and D statistics respectively in blocks of a given number of
  variants and perform a block-jackknife to estimate the standard
  error.

v0.12
-----

* Added functions for principal components analysis of genotype
  data. Functions in the new module :mod:`allel.stats.decomposition`
  include :func:`allel.stats.decomposition.pca` to perform a PCA via
  full singular value decomposition, and
  :func:`allel.stats.decomposition.randomized_pca` which uses an
  approximate truncated singular value decomposition to speed up
  computation. In tests with real data the randomized PCA is around 5
  times faster and uses half as much memory as the conventional PCA,
  producing highly similar results.

* Added function :func:`allel.stats.distance.pcoa` for principal
  coordinate analysis (a.k.a. classical multi-dimensional scaling) of
  a distance matrix.

* Added new utility module :mod:`allel.stats.preprocessing` with
  classes for scaling genotype data prior to use as input for PCA or
  PCoA. By default the scaling (i.e., normalization) of
  Patterson (2006) is used with principal components analysis
  functions in the :mod:`allel.stats.decomposition` module. Scaling
  functions can improve the ability to resolve population structure
  via PCA or PCoA.

* Added method :func:`allel.model.GenotypeArray.to_n_ref`. Also added
  ``dtype`` argument to :func:`allel.model.GenotypeArray.to_n_ref()`
  and :func:`allel.model.GenotypeArray.to_n_alt()` methods to enable
  direct output as float arrays, which can be convenient if these
  arrays are then going to be scaled for use in PCA or PCoA.

* Added :attr:`allel.model.GenotypeArray.mask` property which can be
  set with a Boolean mask to filter genotype calls from genotype and
  allele counting operations. A similar property is available on the
  :class:`allel.bcolz.GenotypeCArray` class. Also added method
  :func:`allel.model.GenotypeArray.fill_masked` and similar method
  on the :class:`allel.bcolz.GenotypeCArray` class to fill masked
  genotype calls with a value (e.g., -1).

v0.11
-----

* Added functions for calculating Watterson's theta (proportional to
  the number of segregating variants):
  :func:`allel.stats.diversity.watterson_theta` for calculating over a
  given region, and
  :func:`allel.stats.diversity.windowed_watterson_theta` for
  calculating in windows over a chromosome/contig.

* Added functions for calculating Tajima's D statistic (balance
  between nucleotide diversity and number of segregating sites):
  :func:`allel.stats.diversity.tajima_d` for calculating over a given
  region and :func:`allel.stats.diversity.windowed_tajima_d` for
  calculating in windows over a chromosome/contig.

* Added :func:`allel.stats.diversity.windowed_df` for calculating the
  rate of fixed differences between two populations.

* Added function :func:`allel.model.locate_fixed_differences` for
  locating variants that are fixed for different alleles in two
  different populations.

* Added function :func:`allel.model.locate_private_alleles` for
  locating alleles and variants that are private to a single
  population.

v0.10
-----

* Added functions implementing the Weir and Cockerham (1984)
  estimators for F-statistics:
  :func:`allel.stats.fst.weir_cockerham_fst` and
  :func:`allel.stats.fst.windowed_weir_cockerham_fst`.

* Added functions implementing the Hudson (1992) estimator for Fst:
  :func:`allel.stats.fst.hudson_fst` and
  :func:`allel.stats.fst.windowed_hudson_fst`.

* Added new module :mod:`allel.stats.ld` with functions for
  calculating linkage disequilibrium estimators, including
  :func:`allel.stats.ld.rogers_huff_r` for pairwise variant LD
  calculation, :func:`allel.stats.ld.windowed_r_squared` for windowed
  LD calculations, and :func:`allel.stats.ld.locate_unlinked` for
  locating variants in approximate linkage equilibrium.

* Added function :func:`allel.plot.pairwise_ld` for visualising a
  matrix of linkage disequilbrium values between pairs of variants.

* Added function :func:`allel.model.create_allele_mapping` for
  creating a mapping of alleles into a different index system, i.e.,
  if you want 0 and 1 to represent something other than REF and ALT,
  e.g., ancestral and derived. Also added methods
  :func:`allel.model.GenotypeArray.map_alleles`,
  :func:`allel.model.HaplotypeArray.map_alleles` and
  :func:`allel.model.AlleleCountsArray.map_alleles` which will perform
  an allele transformation given an allele mapping.

* Added function :func:`allel.plot.variant_locator` ported across from
  anhima.

* Refactored the :mod:`allel.stats` module into a package with
  sub-modules for easier maintenance.

v0.9
----

* Added documentation for the functions
  :func:`allel.bcolz.carray_from_hdf5`,
  :func:`allel.bcolz.carray_to_hdf5`,
  :func:`allel.bcolz.ctable_from_hdf5_group`,
  :func:`allel.bcolz.ctable_to_hdf5_group`. 

* Refactoring of internals within the :mod:`allel.bcolz` module.

v0.8
----

* Added `subpop` argument to
  :func:`allel.model.GenotypeArray.count_alleles` and
  :func:`allel.model.HaplotypeArray.count_alleles` to enable count
  alleles within a sub-population without subsetting the array.

* Added functions
  :func:`allel.model.GenotypeArray.count_alleles_subpops` and
  :func:`allel.model.HaplotypeArray.count_alleles_subpops` to enable
  counting alleles in multiple sub-populations in a single pass over
  the array, without sub-setting.

* Added classes :class:`allel.model.FeatureTable` and
  :class:`allel.bcolz.FeatureCTable` for storing and querying data on
  genomic features (genes, etc.), with functions for parsing from a GFF3
  file.

* Added convenience function :func:`allel.stats.distance.pairwise_dxy`
  for computing a distance matrix using Dxy as the metric.

v0.7
----

* Added function :func:`allel.io.write_fasta` for writing a nucleotide
  sequence stored as a NumPy array out to a FASTA format file.

v0.6
----

* Added method :func:`allel.model.VariantTable.to_vcf` for writing a
  variant table to a VCF format file.
