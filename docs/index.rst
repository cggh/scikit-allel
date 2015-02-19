.. module:: allel

scikit-allel - Explore and analyse genetic variation
====================================================

This package provides utility functions for working with large scale
genetic variation data using numpy_, scipy_ and other Python scientific
libraries.

This package is in an early stage of development, if you have any
questions please email Alistair Miles <alimanfoo@googlemail.com>.

* GitHub repository: https://github.com/cggh/scikit-allel 

Installation
------------

This package requires numpy_, scipy_, numexpr_ and bcolz_. Install these
dependencies first, then::

    $ pip install -U scikit-allel

Contents
--------

.. toctree::
    :maxdepth: 2

    model
    bcolz
    stats
    plot

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
.. _numexpr: https://github.com/pydata/numexpr
.. _bcolz: http://bcolz.blosc.org/
