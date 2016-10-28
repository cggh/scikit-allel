# -*- coding: utf-8 -*-
# flake8: noqa
from __future__ import absolute_import, print_function, division


from . import ndarray
from .ndarray import *
from . import chunked
from .chunked import *
from . import util
from .util import *

try:
    # dask support is optional
    # noinspection PyUnresolvedReferences
    import dask.array
except ImportError:
    pass
else:
    from . import dask
    from .dask import *
