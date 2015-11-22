# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import tempfile
import atexit
import shutil


import bcolz


from .backend_base import Backend


class BColzBackend(Backend):

    def __init__(self, **kwargs):
        self.defaults = kwargs

    def set_defaults(self, kwargs):
        for k, v in self.defaults.items():
            kwargs.setdefault(k, v)
        return kwargs

    def create(self, data, expectedlen=None, **kwargs):
        kwargs = self.set_defaults(kwargs)
        return bcolz.carray(data, expectedlen=expectedlen, **kwargs)

    def append(self, carr, data):
        carr.append(data)
        return carr

    def create_table(self, data, expectedlen=None, **kwargs):
        kwargs = self.set_defaults(kwargs)
        return bcolz.ctable(data, expectedlen=expectedlen, **kwargs)

    def append_table(self, ctbl, data):
        ctbl.append(data)
        return ctbl


# singleton instances for convenience
bcolz_backend = BColzBackend()
bcolz_gzip1_backend = BColzBackend(cparams=bcolz.cparams(cname='zlib',
                                                         clevel=1))


class BColzTmpBackend(BColzBackend):

    # noinspection PyShadowingBuiltins
    def set_defaults(self, kwargs):
        for k, v in self.defaults.items():
            kwargs.setdefault(k, v)
        suffix = kwargs.pop('suffix', '.bcolz')
        prefix = kwargs.pop('prefix', 'scikit_allel_')
        dir = kwargs.pop('dir', None)
        rootdir = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=dir)
        atexit.register(shutil.rmtree, rootdir)
        kwargs['rootdir'] = rootdir
        kwargs['mode'] = 'w'
        return kwargs


# singleton instance
bcolztmp_backend = BColzTmpBackend()
bcolztmp_gzip1_backend = BColzTmpBackend(cparams=bcolz.cparams(cname='zlib',
                                                               clevel=1))
