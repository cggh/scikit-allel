# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import tempfile
import atexit
import shutil
from types import MethodType


import bcolz


from allel.chunked import util as _util


def _table_append(ctbl, data):

    if hasattr(data, 'keys') and callable(data.keys):
        # normalise dict-like data
        data = [data[n] for n in ctbl.names]

    ctbl.append_original(data)


class BcolzStorage(object):
    """Storage layer using bcolz carray and ctable."""

    def __init__(self, **kwargs):
        self.defaults = kwargs

    def _set_defaults(self, kwargs):
        for k, v in self.defaults.items():
            kwargs.setdefault(k, v)
        return kwargs

    def array(self, data, expectedlen=None, **kwargs):
        data = _util.ensure_array_like(data)
        kwargs = self._set_defaults(kwargs)
        return bcolz.carray(data, expectedlen=expectedlen, **kwargs)

    def table(self, data, names=None, expectedlen=None, **kwargs):
        names, columns = _util.check_table_like(data, names=names)
        kwargs = self._set_defaults(kwargs)
        ctbl = bcolz.ctable(columns, names=names, expectedlen=expectedlen,
                            **kwargs)
        # patch append method
        ctbl.append_original = ctbl.append
        ctbl.append = MethodType(_table_append, ctbl)
        return ctbl


class BcolzMemStorage(BcolzStorage):

    # noinspection PyShadowingBuiltins
    def _set_defaults(self, kwargs):
        for k, v in self.defaults.items():
            kwargs.setdefault(k, v)
        kwargs['rootdir'] = None
        return kwargs


class BcolzTmpStorage(BcolzStorage):

    def _set_defaults(self, kwargs):
        for k, v in self.defaults.items():
            kwargs.setdefault(k, v)
        suffix = kwargs.pop('suffix', '.bcolz')
        prefix = kwargs.pop('prefix', 'scikit_allel_')
        tempdir = kwargs.pop('dir', None)
        rootdir = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=tempdir)
        atexit.register(shutil.rmtree, rootdir)
        kwargs['rootdir'] = rootdir
        kwargs['mode'] = 'w'
        return kwargs


bcolz_storage = BcolzStorage()
"""bcolz storage with default parameters"""
bcolzmem_storage = BcolzMemStorage()
"""bcolz in-memory storage with default compression"""
bcolztmp_storage = BcolzTmpStorage()
"""bcolz temporary file storage with default compression"""
_zlib1 = bcolz.cparams(cname='zlib', clevel=1)
bcolz_zlib1_storage = BcolzStorage(cparams=_zlib1)
"""bcolz storage with zlib level 1 compression"""
bcolzmem_zlib1_storage = BcolzMemStorage(cparams=_zlib1)
"""bcolz in-memory storage with zlib level 1 compression"""
bcolztmp_zlib1_storage = BcolzTmpStorage(cparams=_zlib1)
"""bcolz temporary file storage with zlib level 1 compression"""

_util.storage_registry['bcolz'] = bcolz_storage
_util.storage_registry['bcolzmem'] = bcolzmem_storage
_util.storage_registry['bcolztmp'] = bcolztmp_storage
_util.storage_registry['bcolz_zlib1'] = bcolz_zlib1_storage
_util.storage_registry['bcolzmem_zlib1'] = bcolzmem_zlib1_storage
_util.storage_registry['bcolztmp_zlib1'] = bcolztmp_zlib1_storage
