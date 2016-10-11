# -*- coding: utf-8 -*-
# flake8: noqa
from __future__ import absolute_import, print_function, division


from . import ndarray
from .ndarray import *
from . import chunked
from .chunked import *
from . import util
from .util import create_allele_mapping, locate_fixed_differences, locate_private_alleles

# from allel.model import chunked

# experimental
# try:
#     import dask.array as _da
#     from allel.model import dask
# except ImportError:
#     pass

# deprecated
# try:
#     import bcolz as _bcolz
#     from allel.model import bcolz
# except ImportError:
#     pass
