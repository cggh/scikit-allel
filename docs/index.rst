.. module:: allel

scikit-allel - Explore and analyse genetic variation
====================================================

This package provides utilities for exploratory analysis of large
scale genetic variation data. It is based on numpy_, scipy_ and other
general-purpose Python scientific libraries.

* Source: https://github.com/cggh/scikit-allel
* Documentation: http://scikit-allel.readthedocs.org/
* Download: https://pypi.python.org/pypi/scikit-allel
* Mailing list: https://groups.google.com/forum/#!forum/scikit-allel

Please feel free to ask questions via the `mailing list <https://groups.google.com/forum/#!forum/scikit-allel>`_.

If you find a bug or would like to suggest a feature, please `raise an issue
on GitHub <https://github.com/cggh/scikit-allel/issues/new>`_.

This site provides reference documentation for `scikit-allel`. For
worked examples with real data, see the following articles:

* `Selecting variants and samples <http://alimanfoo.github.io/2018/04/09/selecting-variants.html>`_
* `Extracting data from VCF files <http://alimanfoo.github.io/2017/06/14/read-vcf.html>`_
* `A tour of scikit-allel <http://alimanfoo.github.io/2016/06/10/scikit-allel-tour.html>`_
* |Estimating FST|
* `Fast PCA <http://alimanfoo.github.io/2015/09/28/fast-pca.html>`_
* `Mendelian transmission <http://alimanfoo.github.io/2017/02/14/mendelian-transmission.html>`_

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

Pre-built binaries are available for Windows, Mac and Linux, and can
be installed via conda::

    $ conda install -c conda-forge scikit-allel

This will install scikit-allel and all dependencies.

Alternatively, if you have a C compiler on your system, `scikit-allel`
can be installed via pip::

    $ pip install scikit-allel

This will install `scikit-allel` along with minimal dependencies
(numpy_, cython_ and dask_). Some features of `scikit-allel` require
optional third-party packages to be installed, including scipy_,
matplotlib_, seaborn_, pandas_, scikit-learn_, h5py_ and zarr_. To
install `scikit-allel` with all optional dependencies via pip, do::

    $ pip install scikit-allel[full]

If you have never installed Python before, you might find the
following article useful: `Installing Python for data analysis
<http://alimanfoo.github.io/2017/05/18/installing-python.html>`_

Contributing
------------

This is academic software, written in the cracks of free time between other commitments, by people
who are often learning as we code. We greatly appreciate bug reports, pull requests, and any other
feedback or advice. If you do find a bug, we'll do our best to fix it, but apologies in advance if
we are not able to respond quickly. If you are doing any serious work with this package, please do
not expect everything to work perfectly first time or be 100% correct. Treat everything with a
healthy dose of suspicion, and don't be afraid to dive into the source code if you have to. Pull
requests are always welcome.

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
.. _cython: https://cython.org/
