# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import zarr


from allel.chunked import util as _util
from allel.compat import zip


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
        # ignore expectedlen

        # setup
        data = _util.ensure_array_like(data)
        kwargs = self._set_defaults(kwargs)

        # create
        z = zarr.array(data, **kwargs)

        return z

    def table(self, data, names=None, expectedlen=None, **kwargs):
        # ignore expectedlen

        # setup
        names, columns = _util.check_table_like(data, names=names)
        kwargs = self._set_defaults(kwargs)
        g = zarr.group(**kwargs)

        # create columns
        for n, c in zip(names, columns):
            g.array(name=n, data=c)

        # create table
        ztbl = ZarrTable(names, g)
        return ztbl


class ZarrTable(object):

    def __init__(self, names, columns):
        self.names = names
        self.columns = columns

    def __getitem__(self, item):
        return self.columns[item]

    def append(self, data):
        _, columns = _util.check_table_like(data, names=self.names)
        for n in self.names:
            self.columns[n].append(columns[n])


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
