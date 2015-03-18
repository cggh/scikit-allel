Release notes
=============

v0.12
-----

* Added functions for principal components analysis of genotype
  data. Functions in the new module :mod:`allel.stats.decomposition`
  include :func:`allel.stats.decomposition.pca` to perform a PCA via
  full singular value decomposition, and
  :func:`allel.stats.decomposition.randomized_pca` which uses an
  approximate truncated singular value decomposition to speed up
  computation. In tests with real data the randomized PCA is around 5
  times faster and uses half as much memory as the conventional PCA.

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
