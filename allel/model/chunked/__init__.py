# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


from .util import storage_registry
from .storage_bcolz import *
from .storage_hdf5 import *
from .core import *
from .ext import *


storage_registry['default'] = bcolzmem_storage
