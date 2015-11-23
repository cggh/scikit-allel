# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import tempfile
import atexit
import operator
import os
from types import MethodType


import h5py


from allel.compat import reduce
from allel.model.chunked import util as _util


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


def h5ftmp(**kwargs):
    """Create an HDF5 file backed by a temporary file."""

    # create temporary file name
    suffix = kwargs.pop('suffix', '.h5')
    prefix = kwargs.pop('prefix', 'scikit_allel_')
    tempdir = kwargs.pop('dir', None)
    fn = tempfile.mktemp(suffix=suffix, prefix=prefix, dir=tempdir)
    atexit.register(os.remove, fn)

    # file creation args
    kwargs['mode'] = 'w'

    # open HDF5 file
    h5f = h5py.File(fn, **kwargs)

    return h5f


def _array_append(h5d, data):
    hl = len(h5d)
    dl = len(data)
    hln = hl + dl
    h5d.resize(hln, axis=0)
    h5d[hl:hln] = data


def _table_append(h5g, data):
    names, columns = _util.check_table_like(data, names=h5g.names)
    for n, c in zip(names, columns):
        h5d = h5g[n]
        _array_append(h5d, c)


class HDF5Storage(object):

    def __init(self, **kwargs):
        self.defaults = kwargs

    def create_h5f(self, **kwargs):
        pass

    def create_h5d(self, h5g, data=None, **kwargs):

        # set defaults
        kwargs.setdefault('name', 'data')
        for k, v in self.defaults.items():
            kwargs.setdefault(k, v)

        if data is not None:
            data = _util.ensure_array_like(data)

            # by default, simple chunking across rows
            rowsize = data.dtype.itemsize * reduce(operator.mul,
                                                   data.shape[1:], 1)
            # by default, 1Mb chunks
            chunklen = max(1, (2**20) // rowsize)
            chunks = (chunklen,) + data.shape[1:]
            kwargs.setdefault('chunks', chunks)

            # by default, can resize dim 0
            maxshape = (None,) + data.shape[1:]
            kwargs.setdefault('maxshape', maxshape)

            # set data
            kwargs['data'] = data

        h5d = h5g.create_dataset(**kwargs)
        # patch in append method
        h5d.append = MethodType(_array_append, h5d)
        return h5d

    # noinspection PyUnusedLocal
    def array(self, data, expectedlen=None, **kwargs):
        # ignore expectedlen for now
        data = _util.ensure_array_like(data)
        # use root group
        h5g, kwargs = self.create_h5f(**kwargs)
        h5d = self.create_h5d(h5g, data=data, **kwargs)
        return h5d

    # noinspection PyUnusedLocal
    def table(self, data, names=None, expectedlen=None, **kwargs):
        # ignore expectedlen for now
        names, columns = _util.check_table_like(data, names=names)
        # use root group
        h5g, kwargs = self.create_h5f(**kwargs)
        for n, c in zip(names, columns):
            self.create_h5d(h5g, data=c, name=n, **kwargs)
        # patch in append method
        h5g.append = MethodType(_table_append, h5g)
        # patch in names attribute
        h5g.names = names
        return h5g


class HDF5MemStorage(HDF5Storage):

    def __init__(self, **kwargs):
        self.defaults = kwargs

    def create_h5f(self, **kwargs):
        return h5fmem(), kwargs


class HDF5TmpStorage(HDF5Storage):

    def __init__(self, **kwargs):
        self.defaults = kwargs

    def create_h5f(self, **kwargs):
        suffix = kwargs.pop('suffix', '.h5')
        prefix = kwargs.pop('prefix', 'scikit_allel_')
        tempdir = kwargs.pop('dir', None)
        return h5ftmp(dir=tempdir, suffix=suffix, prefix=prefix), kwargs


hdf5mem_storage = HDF5MemStorage()
hdf5tmp_storage = HDF5TmpStorage()
hdf5mem_zlib1_storage = HDF5MemStorage(compression='gzip', compression_opts=1)
hdf5tmp_zlib1_storage = HDF5TmpStorage(compression='gzip', compression_opts=1)
_util.storage_registry['hdf5mem'] = hdf5mem_storage
_util.storage_registry['hdf5tmp'] = hdf5tmp_storage
_util.storage_registry['hdf5mem_zlib1'] = hdf5mem_zlib1_storage
_util.storage_registry['hdf5tmp_zlib1'] = hdf5tmp_zlib1_storage
