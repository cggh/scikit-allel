.. module:: allel

scikit-allel - Explore and analyse genetic variation
====================================================

This package provides utilities for exploratory analysis of large
scale genetic variation data. It is based on numpy_, scipy_ and other
established Python scientific libraries.

* GitHub repository: https://github.com/cggh/scikit-allel
* Documentation: http://scikit-allel.readthedocs.org/
* Download: https://pypi.python.org/pypi/scikit-allel

If you have any questions, find a bug, or would like to suggest a
feature, please `raise an issue on GitHub
<https://github.com/cggh/scikit-allel/issues/new>`_.

This site provides reference documentation for `scikit-allel`. For
worked examples with real data, see the following articles:

* `Introducing scikit-allel <http://alimanfoo.github.io/2015/09/15/introducing-scikit-allel.html>`_
* |Estimating FST|
* `Fast PCA <http://alimanfoo.github.io/2015/09/28/fast-pca.html>`_  

.. |Estimating FST| raw:: HTML

    <a href="http://alimanfoo.github.io/2015/09/21/estimating-fst.html">Estimating F<sub>ST</sub></a>	       
		     
Installation
------------

This package requires numpy_, scipy_, matplotlib_, seaborn_, pandas_,
scikit-learn_, h5py_, numexpr_, bcolz_ and petl_. Install these
dependencies first, then use pip to install scikit-allel::

    $ pip install -U scikit-allel

Contents
--------

.. toctree::
    :maxdepth: 3

    model
    stats
    io
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
.. _petl: http://petl.readthedocs.org/
