.. module:: allel

scikit-allel - Explore and analyse genetic variation
====================================================

This package provides utilities for exploratory analysis of large
scale genetic variation data. It is based on numpy_, scipy_ and other
established Python scientific libraries.

* Source: https://github.com/cggh/scikit-allel
* Documentation: http://scikit-allel.readthedocs.org/
* Download: https://pypi.python.org/pypi/scikit-allel
* Gitter: https://gitter.im/cggh/pygenomics

Please feel free to ask questions via
`cggh/pygenomics <https://gitter.im/cggh/pygenomics>`_  on Gitter. Release
announcements are posted to the
`cggh/pygenomics <https://gitter.im/cggh/pygenomics>`_ Gitter channel and the
`biovalidation mailing list <https://groups.google.com/forum/#!forum/biovalidation>`_.
If you find a bug or would like to suggest a feature, please `raise an issue
on GitHub <https://github.com/cggh/scikit-allel/issues/new>`_.

This site provides reference documentation for `scikit-allel`. For
worked examples with real data, see the following articles:

* `Introducing scikit-allel <http://alimanfoo.github.io/2015/09/15/introducing-scikit-allel.html>`_
* `A tour of scikit-allel <http://alimanfoo.github.io/2016/06/10/scikit-allel-tour.html>`_
* |Estimating FST|
* `Fast PCA <http://alimanfoo.github.io/2015/09/28/fast-pca.html>`_

.. |Estimating FST| raw:: HTML

    <a href="http://alimanfoo.github.io/2015/09/21/estimating-fst.html">Estimating F<sub>ST</sub></a>

If you would like to cite `scikit-allel` please use the DOI below.

.. image:: https://zenodo.org/badge/7890/cggh/scikit-allel.svg
   :target: https://zenodo.org/badge/latestdoi/7890/cggh/scikit-allel

Why "scikit-allel"?
-------------------

"`SciKits <http://www.scipy.org/scikits.html>`_" (short for SciPy Toolkits)
are add-on packages for SciPy, hosted and developed separately and
independently from the main SciPy distribution.

"Allel" (Greek ἀλλήλ) is the root of the word
"`allele <https://en.wikipedia.org/wiki/Allele>`_" short for "allelomorph", a
word coined by William Bateson to mean variant forms of a gene. Today we use
"allele" to mean any of the variant forms found at a site of genetic variation,
such as the different nucleotides observed at a single nucleotide polymorphism
(SNP).

Installation
------------

This package requires numpy_, scipy_, matplotlib_, seaborn_, pandas_,
scikit-learn_, h5py_, numexpr_, bcolz_, zarr_, dask_ and petl_. Please install
these dependencies first, then use pip to install scikit-allel::

    $ pip install -U scikit-allel

Contents
--------

.. toctree::
    :maxdepth: 2

    model
    stats
    io
    chunked
    util
    release

Acknowledgments
---------------

Development of this package is supported by the
`MRC Centre for Genomics and Global Health <http://www.cggh.org>`_.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _numpy: http://www.numpy.org
.. _scipy: http://www.scipy.org/
.. _matplotlib: http://matplotlib.org/
.. _seaborn: http://stanford.edu/~mwaskom/software/seaborn/
.. _pandas: http://pandas.pydata.org/
.. _scikit-learn: http://scikit-learn.org/
.. _h5py: http://www.h5py.org/
.. _numexpr: https://github.com/pydata/numexpr
.. _bcolz: http://bcolz.blosc.org/
.. _zarr: http://zarr.readthedocs.io/
.. _dask: http://dask.pydata.org/
.. _petl: http://petl.readthedocs.org/
