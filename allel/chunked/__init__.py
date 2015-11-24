# -*- coding: utf-8 -*-
# flake8: noqa
"""
This package provides an abstraction layer over generic chunked array storage
libraries. Currently HDF5 (via `h5py <http://www.h5py.org/>`_) and `bcolz
<http://bcolz.blosc.org>`_ are supported storage layers.

"""
from __future__ import absolute_import, print_function, division


from .util import *
from .storage_bcolz import *
from .storage_hdf5 import *
from .core import *


storage_registry['default'] = BcolzStorage()
