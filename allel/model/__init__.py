# -*- coding: utf-8 -*-
# flake8: noqa
from __future__ import absolute_import, print_function, division


from allel.model import ndarray
from allel.model import chunked

# experimental
try:
    import dask.array as _da
    from allel.model import dask
except ImportError:
    pass

# deprecated
try:
    import bcolz as _bcolz
    from allel.model import bcolz
except ImportError:
    pass
