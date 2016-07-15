# -*- coding: utf-8 -*-
# flake8: noqa


from allel import model
from allel import stats
from allel import plot
from allel import io
from allel import chunked
from allel import constants
from allel import util

# convenient shortcuts
from allel.model.ndarray import *
from allel.model.chunked import *

# experimental
try:
    import dask.array as _da
    from allel.model.dask import *
except ImportError:
    pass

# deprecated
try:
    import bcolz as _bcolz
    from allel.model.bcolz import *
except ImportError:
    pass

__version__ = '0.21.1'
