# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import operator
import tempfile
import os
import atexit
import shutil


import zarr


from allel.chunked import util as _util
from allel.compat import reduce, zip


def default_chunks(data):
    # by default, simple chunking across rows
    rowsize = data.dtype.itemsize * reduce(operator.mul,
                                           data.shape[1:], 1)
    # 1Mb chunks
    chunklen = max(1, (2 ** 20) // rowsize)
    chunks = (chunklen,) + data.shape[1:]
    return chunks


class ZarrStorage(object):
    """Storage layer using Zarr."""

    def __init__(self, **kwargs):
        self.defaults = kwargs

    def _set_defaults(self, kwargs):

        # copy in master defaults
        for k, v in self.defaults.items():
            kwargs.setdefault(k, v)

        # general settings
        kwargs.setdefault('fill_value', 0)

        return kwargs

    def _create_array(self, data, **kwargs):

        # determine chunks
        chunks = default_chunks(data)
        kwargs.setdefault('chunks', chunks)

        # create array
        if 'path' in kwargs:
            kwargs['mode'] = 'w'
            kwargs['shape'] = data.shape
            # ensure dtype is specified
            dtype = kwargs.get('dtype', None)
            if not dtype:
                kwargs['dtype'] = data.dtype
            z = zarr.open(**kwargs)
            z[:] = data
        else:
            z = zarr.array(data, **kwargs)

        return z

    # noinspection PyUnusedLocal
    def array(self, data, expectedlen=None, **kwargs):
        # ignore expectedlen

        # setup
        data = _util.ensure_array_like(data)
        kwargs = self._set_defaults(kwargs)

        # create
        z = self._create_array(data, **kwargs)
        return z

    def table(self, data, names=None, expectedlen=None, **kwargs):
        # ignore expectedlen

        # setup
        names, columns = _util.check_table_like(data, names=names)
        kwargs = self._set_defaults(kwargs)
        path = kwargs.get('path', None)
        zcols = list()

        # create columns
        for n, c in zip(names, columns):
            col_kwargs = kwargs.copy()
            if path:
                col_kwargs['path'] = os.path.join(path, n)
            zcol = self._create_array(c, **col_kwargs)
            zcols.append(zcol)

        # create table
        ztbl = ZarrTable(names, zcols)
        return ztbl


class ZarrTable(object):

    def __init__(self, names, columns):
        self.names = names
        self.columns = columns

    def __getitem__(self, item):
        i = self.names.index(item)
        return self.columns[i]

    def append(self, data):
        _, columns = _util.check_table_like(data, names=self.names)
        for co, cn in zip(self.columns, columns):
            co.append(cn)


class ZarrMemStorage(ZarrStorage):

    # noinspection PyShadowingBuiltins
    def _set_defaults(self, kwargs):
        kwargs = super(ZarrMemStorage, self)._set_defaults(kwargs)
        if 'path' in kwargs:
            del kwargs['path']
        return kwargs


class ZarrTmpStorage(ZarrStorage):

    def _set_defaults(self, kwargs):
        kwargs = super(ZarrTmpStorage, self)._set_defaults(kwargs)
        suffix = kwargs.pop('suffix', '.zarr')
        prefix = kwargs.pop('prefix', 'scikit_allel_')
        tempdir = kwargs.pop('dir', None)
        path = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=tempdir)
        atexit.register(shutil.rmtree, path)
        kwargs['path'] = path
        kwargs['mode'] = 'w'
        return kwargs


zarr_storage = ZarrStorage()
"""zarr storage with default parameters"""
zarrmem_storage = ZarrMemStorage()
"""zarr in-memory storage with default compression"""
zarrtmp_storage = ZarrTmpStorage()
"""zarr temporary file storage with default compression"""

_util.storage_registry['zarr'] = zarr_storage
_util.storage_registry['zarrmem'] = zarrmem_storage
_util.storage_registry['zarrtmp'] = zarrtmp_storage
