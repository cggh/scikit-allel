# -*- coding: utf-8 -*-
# flake8: noqa
"""
This package provides an abstraction layer over chunked array storage
libraries. Currently bcolz and h5py are supported storage layers.

"""
from __future__ import absolute_import, print_function, division


from .util import *
from .storage_bcolz import *
from .storage_hdf5 import *
from .core import *


storage_registry['default'] = BcolzStorage()
