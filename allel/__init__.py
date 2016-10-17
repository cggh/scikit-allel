# -*- coding: utf-8 -*-
# flake8: noqa


from . import model
from .model.ndarray import *
from .model.chunked import *
from .model.util import *
try:
    import dask
except ImportError:
    pass
else:
    from .model.dask import *
from . import stats
from .stats import *
from . import plot
from . import io
from . import chunked
from . import constants
from . import util

__version__ = '1.0.0b4'
