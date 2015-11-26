# -*- coding: utf-8 -*-
"""This module provides alternative implementations of array and table
classes defined in the :mod:`allel.model.ndarray` module, using
`dask.array <http://dask.pydata.org/en/latest/array.html>`_ as the
computational engine.

Dask uses blocked algorithms and task scheduling to break up work into
smaller pieces, allowing computation over very large datasets. It also uses
lazy evaluation, meaning that multiple operations can be chained together
into a task graph, reducing total memory requirements for intermediate
results, and only the tasks required to generate the requested
part of the final data set will be executed.

This module is experimental, if you find a bug please `raise an issue on GitHub
<https://github.com/cggh/scikit-allel/issues/new>`_.

Currently this module currently requires a specific branch of Dask to be
installed::

    $ pip install git+https://github.com/mrocklin/dask.git@drop-new-axes

"""
from __future__ import absolute_import, print_function, division


import numpy as np
import dask.array as da


def get_chunks(data, chunks):
    """Try to guess a reasonable chunk shape to use for block-wise
    algorithms operating over `data`."""

    if chunks is None:

        if hasattr(data, 'chunklen') and hasattr(data, 'shape'):
            # bcolz carray, chunk first dimension only
            return (data.chunklen,) + data.shape[1:]

        elif hasattr(data, 'chunks') and hasattr(data, 'shape') and \
                len(data.chunks) == len(data.shape):
            # h5py dataset
            return data.chunks

        else:
            # fall back to something simple, ~1Mb chunks of first dimension
            row = np.asarray(data[0])
    else:
        return chunks

class GenotypeDaskArray(object):
    """TODO"""

    def __init__(self, data, chunks=None):
