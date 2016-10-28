# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import tempfile
import atexit
import operator
import os
from types import MethodType


import h5py


from allel.compat import reduce
from allel.chunked import util as _util


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


def _dataset_append(h5d, data):
    hl = len(h5d)
    dl = len(data)
    hln = hl + dl
    h5d.resize(hln, axis=0)
    h5d[hl:hln] = data


def _table_append(h5g, data):
    names, columns = _util.check_table_like(data, names=h5g.names)
    for n, c in zip(names, columns):
        h5d = h5g[n]
        _dataset_append(h5d, c)


class HDF5Storage(object):
    """Storage layer using HDF5 dataset and group."""

    def __init__(self, **kwargs):
        self.defaults = kwargs

    def open_file(self, **kwargs):
        # override in sub-classes
        raise NotImplementedError('group must be provided')

    def create_dataset(self, h5g, data=None, expectedlen=None, **kwargs):

        # set defaults
        kwargs.setdefault('name', 'data')
        for k, v in self.defaults.items():
            kwargs.setdefault(k, v)

        # handle data
        if data is not None:
            data = _util.ensure_array_like(data)

            # by default, simple chunking across rows
            rowsize = data.dtype.itemsize * reduce(operator.mul,
                                                   data.shape[1:], 1)
            # 1Mb chunks
            chunklen = max(1, (2**20) // rowsize)
            if expectedlen is not None:
                # ensure chunks not bigger than expected length
                chunklen = min(chunklen, expectedlen)
            chunks = (chunklen,) + data.shape[1:]
            kwargs.setdefault('chunks', chunks)

            # by default, can resize dim 0
            maxshape = (None,) + data.shape[1:]
            kwargs.setdefault('maxshape', maxshape)

            # set data
            kwargs['data'] = data

        # create dataset
        h5d = h5g.create_dataset(**kwargs)

        return h5d

    # noinspection PyUnusedLocal
    def array(self, data, expectedlen=None, **kwargs):

        # setup
        data = _util.ensure_array_like(data)

        # obtain group
        h5g = kwargs.pop('group', None)
        if h5g is None:
            # open file, use root group
            h5g, kwargs = self.open_file(**kwargs)

        # create dataset
        h5d = self.create_dataset(h5g, data=data, expectedlen=expectedlen,
                                  **kwargs)

        # patch in append method
        h5d.append = MethodType(_dataset_append, h5d)

        return h5d

    # noinspection PyUnusedLocal
    def table(self, data, names=None, expectedlen=None, **kwargs):

        # setup
        names, columns = _util.check_table_like(data, names=names)

        # obtain group
        h5g = kwargs.pop('group', None)
        if h5g is None:
            # open file, use root group
            h5g, kwargs = self.open_file(**kwargs)

        # create columns
        for n, c in zip(names, columns):
            self.create_dataset(h5g, data=c, name=n, expectedlen=expectedlen,
                                **kwargs)

        # patch in append method
        h5g.append = MethodType(_table_append, h5g)

        # patch in names attribute
        h5g.names = names

        return h5g


class HDF5MemStorage(HDF5Storage):

    def open_file(self, **kwargs):
        return h5fmem(), kwargs


class HDF5TmpStorage(HDF5Storage):

    def open_file(self, **kwargs):
        suffix = kwargs.pop('suffix', '.h5')
        prefix = kwargs.pop('prefix', 'scikit_allel_')
        tempdir = kwargs.pop('dir', None)
        return h5ftmp(dir=tempdir, suffix=suffix, prefix=prefix), kwargs


hdf5_storage = HDF5Storage()
"""HDF5 storage with default parameters"""
hdf5mem_storage = HDF5MemStorage()
"""HDF5 in-memory storage with default compression"""
hdf5tmp_storage = HDF5TmpStorage()
"""HDF5 temporary file storage with default compression"""
hdf5_zlib1_storage = HDF5Storage(compression='gzip', compression_opts=1)
"""HDF5 storage with zlib level 1 compression"""
hdf5mem_zlib1_storage = HDF5MemStorage(compression='gzip', compression_opts=1)
"""HDF5 in-memory storage with zlib level 1 compression"""
hdf5tmp_zlib1_storage = HDF5TmpStorage(compression='gzip', compression_opts=1)
"""HDF5 temporary file storage with zlib level 1 compression"""
hdf5_lzf_storage = HDF5Storage(compression='lzf')
"""HDF5 storage with LZF compression"""
hdf5mem_lzf_storage = HDF5MemStorage(compression='lzf')
"""HDF5 in-memory storage with LZF compression"""
hdf5tmp_lzf_storage = HDF5TmpStorage(compression='lzf')
"""HDF5 temporary file storage with LZF compression"""

_util.storage_registry['hdf5'] = hdf5_storage
_util.storage_registry['hdf5mem'] = hdf5mem_storage
_util.storage_registry['hdf5tmp'] = hdf5tmp_storage
_util.storage_registry['hdf5_zlib1'] = hdf5_zlib1_storage
_util.storage_registry['hdf5mem_zlib1'] = hdf5mem_zlib1_storage
_util.storage_registry['hdf5tmp_zlib1'] = hdf5tmp_zlib1_storage
_util.storage_registry['hdf5_lzf'] = hdf5_lzf_storage
_util.storage_registry['hdf5mem_lzf'] = hdf5mem_lzf_storage
_util.storage_registry['hdf5tmp_lzf'] = hdf5tmp_lzf_storage
