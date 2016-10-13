Release notes
=============

Mendelian errors and phasing by transmission
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This release includes a new :mod:`allel.stats.mendel` module with functions to 
help with analysis of related individuals. The function 
:func:`allel.stats.mendel.mendel_errors` locates genotype calls within a trio 
or cross that are not consistent with Mendelian segregation of alleles. The 
function :func:`allel.stats.mendel.phase_by_transmission` will resolve unphased 
diploid genotypes into phased haplotypes for a trio or cross using Mendelian 
transmission rules. The function :func:`allel.stats.mendel.paint_transmission` 
can help with evaluating and visualizing the results of phasing a trio or 
cross.

v0.21.2
-------

This release resolves compatibility issues with Zarr version 2.1.

v0.21.1
-------

* Added parameter `min_maf` to :func:`allel.stats.selection.ihs` to skip IHS
  calculation for variants below a given minor allele frequency.
* Minor change to calculation of integrated haplotype homozygosity to enable
  values to be reported for first and last variants if `include_edges` is
  `True`.
* Minor change to :func:`allel.stats.selection.standardize_by_allele_count`
  to better handle missing values.

v0.21.0
-------

In this release the implementations of :func:`allel.stats.selection.ihs`
and :func:`allel.stats.selection.xpehh` selection statistics have been
reworked to address a number of issues:

* Both functions can now integrate over either a genetic map (via the
  `map_pos` parameter) or a physical map.
* Both functions now accept `max_gap` and `gap_scale` parameters to perform
  adjustments to integrated haplotype homozygosity where there are large
  gaps between variants, following the standard approach. Alternatively, if
  a map of genome accessibility is available, it may be provided via the
  `is_accessible` parameter, in which case the distance between variants
  will be scaled by the fraction of accessible bases between them.
* Both functions are now faster and can make use of multiple threads to
  further accelerate computation.
* Several bugs in the previous implementations of these functions have been
  fixed (`#91 <https://github.com/cggh/scikit-allel/issues/91>`_).
* New utility functions are provided for standardising selection scores,
  see :func:`allel.stats.selection.standardize_by_allele_count` (for use
  with IHS and NSL) and
  :func:`allel.stats.selection.standardize` (for use with XPEHH).

Other changes:

* Added functions :func:`allel.stats.diversity.moving_tajima_d` and
  :func:`allel.stats.selection.moving_delta_tajima_d`
  (`#81 <https://github.com/cggh/scikit-allel/issues/81>`_,
  `#70 <https://github.com/cggh/scikit-allel/issues/70>`_).
* Added functions :func:`allel.stats.fst.moving_weir_cockerham_fst`,
  :func:`allel.stats.fst.moving_hudson_fst`,
  :func:`allel.stats.fst.moving_patterson_fst`.
* Added functions :func:`allel.stats.admixture.moving_patterson_f3` and
  :func:`allel.stats.admixture.moving_patterson_d`.
* Renamed "blockwise" to "average" in function names in
  :mod:`allel.stats.fst` and :mod:`allel.stats.admixture` for clarity.
* Added convenience methods
  :func:`allel.model.ndarray.AlleleCountsArray.is_biallelic` and
  :func:`allel.model.ndarray.AlleleCountsArray.is_biallelic_01` for locating
  biallelic variants.
* Added support for `zarr <http://zarr.readthedocs.io>`_ in the
  :mod:`allel.chunked` module
  (`#101 <https://github.com/cggh/scikit-allel/issues/101>`_).
* Changed HDF5 default chunked storage to use gzip level 1 compression
  instead of no compression
  (`#100 <https://github.com/cggh/scikit-allel/issues/100>`_).
* Fixed bug in :func:`allel.stats.diversity.sequence_divergence`
  (`#75 <https://github.com/cggh/scikit-allel/issues/75>`_).
* Added workaround for chunked arrays if passed as arguments into numpy
  aggregation functions
  (`#66 <https://github.com/cggh/scikit-allel/issues/66>`_).
* Protect against invalid coordinates when mapping from square to condensed
  coords (`#83 <https://github.com/cggh/scikit-allel/issues/83>`_).
* Fixed bug in :func:`allel.stats.sf.plot_sfs_folded` and added docstrings
  for all plotting functions in :mod:`allel.stats.sf`
  (`#80 <https://github.com/cggh/scikit-allel/issues/80>`_).
* Fixed bug related to taking views of genotype and haplotype arrays
  (`#77 <https://github.com/cggh/scikit-allel/issues/77>`_).

v0.20.3
-------

* Fixed a bug in the `count_alleles()` methods on genotype and haplotype array
  classes that manifested if the `max_allele` argument was provided
  (`#59 <https://github.com/cggh/scikit-allel/issues/59>`_).
* Fixed a bug in Jupyter notebook `display` method for chunked tables
  (`#57 <https://github.com/cggh/scikit-allel/issues/57>`_).
* Fixed a bug in site frequency spectrum scaling functions
  (`#54 <https://github.com/cggh/scikit-allel/issues/54>`_).
* Changed behaviour of `subset` method on genotype and haplotype arrays to
  better infer argument types and handle None argument values
  (`#55 <https://github.com/cggh/scikit-allel/issues/55>`_).
* Changed table `eval` and `query` methods to make python the default for
  expression evaluation, because it is more expressive than numexpr
  (`#58 <https://github.com/cggh/scikit-allel/issues/58>`_).

v0.20.2
-------

* Changed :func:`allel.util.hdf5_cache` to resolve issues with hashing and
  argument order
  (`#51 <https://github.com/cggh/scikit-allel/issues/51>`_,
  `#52 <https://github.com/cggh/scikit-allel/issues/52>`_).

v0.20.1
-------

* Changed functions :func:`allel.stats.fst.weir_cockerham_fst` and
  :func:`allel.stats.ld.locate_unlinked` such that chunked implementations
  are now used by default, to avoid accidentally and unnecessarily loading
  very large arrays into memory
  (`#50 <https://github.com/cggh/scikit-allel/issues/50>`_).

v0.20.0
-------

* Added new :mod:`allel.model.dask` module, providing
  implementations of the genotype, haplotype and allele counts classes
  backed by `dask.array <http://dask.pydata.org/en/latest/array.html>`_
  (`#32 <https://github.com/cggh/scikit-allel/issues/32>`_).
* Released the GIL where possible in Cython optimised functions
  (`#43 <https://github.com/cggh/scikit-allel/issues/43>`_).
* Changed functions in :mod:`allel.stats.selection` that accept `min_ehh`
  argument, such that `min_ehh = None` should now be used to indicate that
  no minimum EHH threshold should be applied.

v0.19.0
-------

The major change in v0.19.0 is the addition of the new
:mod:`allel.model.chunked` module, which provides classes for variant
call data backed by chunked array storage (`#31
<https://github.com/cggh/scikit-allel/issues/31>`_). This is a
generalisation of the previously available :mod:`allel.model.bcolz` to
enable the use of both bcolz and HDF5 (via h5py) as backing
storage. The :mod:`allel.model.bcolz` module is now deprecated but
will be retained for backwargs compatibility until the next major
release.

Other changes:

* Added function for computing the number of segregating sites by length
  (nSl), a summary statistic comparing haplotype homozygosity between
  different alleles (similar to IHS), see :func:`allel.stats.selection.nsl`
  (`#40 <https://github.com/cggh/scikit-allel/issues/40>`_).
* Added functions for computing haplotype diversity, see
  :func:`allel.stats.selection.haplotype_diversity` and
  :func:`allel.stats.selection.moving_haplotype_diversity`
  (`#29 <https://github.com/cggh/scikit-allel/issues/29>`_).
* Added function
  :func:`allel.stats.selection.plot_moving_haplotype_frequencies` for
  visualising haplotype frequency spectra in moving windows over the genome
  (`#30 <https://github.com/cggh/scikit-allel/issues/30>`_).
* Added `vstack()` and `hstack()` methods to genotype and haplotype arrays to
  enable combining data from multiple arrays
  (`#21 <https://github.com/cggh/scikit-allel/issues/21>`_).
* Added convenience function
  :func:`allel.stats.window.equally_accessible_windows`
  (`#16 <https://github.com/cggh/scikit-allel/issues/16>`_).
* Added methods `from_hdf5_group()` and `to_hdf5_group()` to
  :class:`allel.model.ndarray.VariantTable`
  (`#26 <https://github.com/cggh/scikit-allel/issues/26>`_).
* Added :func:`allel.util.hdf5_cache` utility function.
* Modified functions in the :mod:`allel.stats.selection` module that depend
  on calculation of integrated haplotype homozygosity to return NaN when
  haplotypes do not decay below a specified threshold
  (`#39 <https://github.com/cggh/scikit-allel/issues/39>`_).
* Fixed missing return value in
  :func:`allel.stats.selection.plot_voight_painting`
  (`#23 <https://github.com/cggh/scikit-allel/issues/23>`_).
* Fixed return type from array reshape()
  (`#34 <https://github.com/cggh/scikit-allel/issues/34>`_).

Contributors: `alimanfoo <https://github.com/alimanfoo>`_,
`hardingnj <https://github.com/hardingnj>`_

v0.18.1
-------

* Minor change to the Garud H statistics to avoid raising an exception when
  the number of distinct haplotypes is very low
  (`#20 <https://github.com/cggh/scikit-allel/issues/20>`_).

v0.18.0
-------

* Added functions for computing H statistics for detecting signatures of soft
  sweeps, see :func:`allel.stats.selection.garud_h`,
  :func:`allel.stats.selection.moving_garud_h`,
  :func:`allel.stats.selection.plot_haplotype_frequencies`
  (`#19 <https://github.com/cggh/scikit-allel/issues/19>`_).
* Added function :func:`allel.stats.selection.fig_voight_painting` to paint
  both flanks either side of some variant under selection in a single figure
  (`#17 <https://github.com/cggh/scikit-allel/issues/17>`_).
* Changed return values from :func:`allel.stats.selection.voight_painting` to
  also return the indices used for sorting haplotypes by prefix
  (`#18 <https://github.com/cggh/scikit-allel/issues/18>`_).

v0.17.0
-------

* Added new module for computing and plotting site frequency spectra, see
  :mod:`allel.stats.sf`
  (`#12 <https://github.com/cggh/scikit-allel/issues/12>`_).
* All plotting functions have been moved into the appropriate stats module
  that they naturally correspond to. The :mod:`allel.plot` module is
  deprecated (`#13 <https://github.com/cggh/scikit-allel/issues/13>`_).
* Improved performance of carray and ctable loading from HDF5 with a
  condition (`#11 <https://github.com/cggh/scikit-allel/issues/11>`_).

v0.16.2
-------

* Fixed behaviour of take() method on compressed arrays when indices are not
  in increasing order
  (`#6 <https://github.com/cggh/scikit-allel/issues/6>`_).
* Minor change to scaler argument to PCA functions in
  :mod:`allel.stats.decomposition` to avoid confusion about when to fall
  back to default scaler
  (`#7 <https://github.com/cggh/scikit-allel/issues/7>`_).

v0.16.1
-------

* Added block-wise implementation to :func:`allel.stats.ld.locate_unlinked` so
  it can be used with compressed arrays as input.

v0.16.0
-------

* Added new selection module with functions for haplotype-based analyses of
  recent selection, see :mod:`allel.stats.selection`.

v0.15.2
-------

* Improved performance of :func:`allel.model.bcolz.carray_block_compress`,
  :func:`allel.model.bcolz.ctable_block_compress` and
  :func:`allel.model.bcolz.carray_block_subset` for very sparse selections.
* Fix bug in IPython HTML table captions.
* Fix bug in addcol() method on bcolz ctable wrappers.

v0.15.1
-------

* Fix missing package in setup.py.

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
