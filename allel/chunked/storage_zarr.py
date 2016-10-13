# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import operator


import zarr
import zarr.util


from allel.chunked import util as _util
from allel.compat import zip, reduce


def default_chunks(data, expectedlen):
    # here we will only ever chunk first dimension
    rowsize = data.dtype.itemsize
    if data.ndim > 1:
        # pretend array is 1D
        rowsize *= reduce(operator.mul, data.shape[1:])
    if expectedlen is None:
        # default to 4M chunks of first dimension
        chunklen = 2**22 // rowsize
    else:
        # use zarr heuristics
        chunklen, = zarr.util.guess_chunks((expectedlen,), rowsize)
    if data.ndim > 1:
        chunks = (chunklen,) + data.shape[1:]
    else:
        chunks = chunklen,
    return chunks


class ZarrStorage(object):
    """Storage layer using Zarr."""

    def __init__(self, **kwargs):
        self.defaults = kwargs

    def _set_defaults(self, kwargs):

        # copy in master defaults
        for k, v in self.defaults.items():
            kwargs.setdefault(k, v)

        return kwargs

    # noinspection PyUnusedLocal
    def array(self, data, expectedlen=None, **kwargs):

        # setup
        data = _util.ensure_array_like(data)
        kwargs = self._set_defaults(kwargs)

        # determine chunks
        kwargs.setdefault('chunks', default_chunks(data, expectedlen))

        # create
        z = zarr.array(data, **kwargs)

        return z

    def table(self, data, names=None, expectedlen=None, **kwargs):

        # setup
        names, columns = _util.check_table_like(data, names=names)
        kwargs = self._set_defaults(kwargs)
        g = zarr.group(**kwargs)

        # create columns
        chunks = kwargs.get('chunks', None)
        for n, c in zip(names, columns):
            if chunks is None:
                chunks = default_chunks(c, expectedlen)
            g.array(name=n, data=c, chunks=chunks)

        # create table
        ztbl = ZarrTable(g, names=names)
        return ztbl


class ZarrTable(object):

    def __init__(self, grp, names=None):
        self.grp = grp
        available_names = sorted(grp.array_keys())
        if names is None:
            names = available_names
        else:
            for n in names:
                if n not in available_names:
                    raise ValueError('name not available: %s' % n)
        self.names = names

    def __getitem__(self, item):
        return self.grp[item]

    def append(self, data):
        names, columns = _util.check_table_like(data, names=self.names)
        for n, c in zip(names, columns):
            self.grp[n].append(c)


class ZarrMemStorage(ZarrStorage):

    # noinspection PyShadowingBuiltins
    def _set_defaults(self, kwargs):
        kwargs = super(ZarrMemStorage, self)._set_defaults(kwargs)
        kwargs.setdefault('store', zarr.DictStore())
        return kwargs


class ZarrTmpStorage(ZarrStorage):

    def _set_defaults(self, kwargs):
        kwargs = super(ZarrTmpStorage, self)._set_defaults(kwargs)
        suffix = kwargs.pop('suffix', '.zarr')
        prefix = kwargs.pop('prefix', 'scikit_allel_')
        # noinspection PyShadowingBuiltins
        dir = kwargs.pop('dir', None)
        kwargs.setdefault('store', zarr.TempStore(suffix=suffix,
                                                  prefix=prefix, dir=dir))
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
