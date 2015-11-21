# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import tempfile
import atexit
import os
import operator


import numpy as np
import h5py


from allel.compat import reduce
from .backend_base import Backend, get_column_names


def h5fmem(**kwargs):
    """Create an in-memory HDF5 file."""

    # need a file name even tho nothing is ever written
    fn = tempfile.mktemp()

    # file creation args
    kwargs['mode'] = 'w'
    kwargs['driver'] = 'core'
    kwargs['backing_store'] = False

    # open HDF5 file
    h5f = h5py.File(fn, **kwargs)

    return h5f


# noinspection PyShadowingBuiltins
def h5ftmp(**kwargs):
    """Create an HDF5 file backed by a temporary file."""

    # create temporary file name
    suffix = kwargs.pop('suffix', '.h5')
    prefix = kwargs.pop('prefix', 'scikit_allel_')
    dir = kwargs.pop('dir', None)
    fn = tempfile.mktemp(suffix=suffix, prefix=prefix, dir=dir)
    atexit.register(os.remove, fn)

    # file creation args
    kwargs['mode'] = 'w'

    # open HDF5 file
    h5f = h5py.File(fn, **kwargs)

    return h5f


def h5dmem(*args, **kwargs):
    """Create an in-memory HDF5 dataset."""

    # open HDF5 file
    h5f = h5fmem()

    # defaults for dataset creation
    kwargs.setdefault('chunks', True)
    if len(args) == 0 and 'name' not in kwargs:
        # default dataset name
        args = ('data',)

    # create dataset
    h5d = h5f.create_dataset(*args, **kwargs)

    return h5d


def h5dtmp(*args, **kwargs):
    """Create an HDF5 dataset backed by a temporary file."""

    # open HDF5 file
    h5f = h5ftmp()

    # defaults for dataset creation
    kwargs.setdefault('chunks', True)
    if len(args) == 0 and 'name' not in kwargs:
        # default dataset name
        args = ('data',)

    # create dataset
    h5d = h5f.create_dataset(*args, **kwargs)

    return h5d


class H5Backend(Backend):

    def __init(self, **kwargs):
        self.defaults = kwargs

    def create_h5f(self):
        pass

    def create_h5d(self, h5f, data, **kwargs):

        # set defaults
        kwargs.setdefault('name', 'data')
        for k, v in self.defaults.items():
            kwargs.setdefault(k, v)

        # by default, simple chunking across rows
        data = np.asarray(data)
        rowsize = data.dtype.itemsize * reduce(operator.mul, data.shape[1:], 1)
        chunklen = max(1, (2**16) // rowsize)
        chunks = (chunklen,) + data.shape[1:]
        kwargs.setdefault('chunks', chunks)

        # determine maxshape
        maxshape = (None,) + data.shape[1:]
        kwargs.setdefault('maxshape', maxshape)

        # set data
        kwargs['data'] = data

        return h5f.create_dataset(**kwargs)

    def create(self, data, expectedlen=None, **kwargs):
        # ignore expectedlen for now
        h5f = self.create_h5f()
        h5d = self.create_h5d(h5f, data, **kwargs)
        return h5d

    def create_table(self, data, expectedlen=None, **kwargs):
        # ignore expectedlen for now
        h5f = self.create_h5f()
        for n in get_column_names(data):
            self.create_h5d(h5f, data[n], name=n, **kwargs)
        return h5f

    def append(self, h5d, data):
        hl = len(h5d)
        dl = len(data)
        hln = hl + dl
        h5d.resize(hln, axis=0)
        h5d[hl:hln] = data
        return h5d

    def append_table(self, h5g, data):
        names = get_column_names(data)
        for n in names:
            self.append(h5g[n], data[n])
        return h5g


class H5memBackend(H5Backend):

    def __init__(self, **kwargs):
        self.defaults = kwargs

    def create_h5f(self):
        return h5fmem()


class H5tmpBackend(H5Backend):

    def __init__(self, **kwargs):
        self.defaults = kwargs

    def create_h5f(self):
        return h5ftmp()


h5mem_backend = H5memBackend()
h5mem_gzip1_backend = H5memBackend(compression='gzip', compression_opts=1)
h5tmp_backend = H5tmpBackend()
h5tmp_gzip1_backend = H5tmpBackend(compression='gzip', compression_opts=1)
