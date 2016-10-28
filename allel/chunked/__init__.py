# -*- coding: utf-8 -*-
# flake8: noqa
"""
This module provides an abstraction layer over generic chunked array storage
libraries. Currently HDF5 (via `h5py <http://www.h5py.org/>`_), `bcolz
<http://bcolz.blosc.org>`_ and `zarr <http://zarr.readthedocs.io>`_ are
supported storage layers.

Different storage configurations can be used with the functions and classes
defined below. Wherever a function or method takes a `storage` keyword
argument, the value of the argument will determine the storage used for the
output.

If `storage` is a string, it will be used to look up one of several predefined
storage configurations via the storage registry, which is a dictionary
located at `allel.chunked.storage_registry`. The default storage can be
changed globally by setting the value of the 'default' key in the storage
registry.

Alternatively, `storage` may be an instance of one of the storage classes
defined below, e.g., :class:`allel.chunked.storage_bcolz.BcolzMemStorage` or
:class:`allel.chunked.storage_hdf5.HDF5TmpStorage`, which allows custom
configuration of storage parameters such as compression type and level.

For example::

    >>> from allel import chunked
    >>> import numpy as np
    >>> a = np.arange(10000000)
    >>> chunked.copy(a)
    Array((10000000,), int64, chunks=(39063,), order=C)
      nbytes: 76.3M; nbytes_stored: 1.2M; ratio: 66.2; initialized: 256/256
      compressor: Blosc(cname='lz4', clevel=5, shuffle=1)
      store: DictStore
    >>> chunked.copy(a, storage='bcolzmem')
    carray((10000000,), int64)
      nbytes: 76.29 MB; cbytes: 1.85 MB; ratio: 41.19
      cparams := cparams(clevel=5, shuffle=1, cname='blosclz')
    [      0       1       2 ..., 9999997 9999998 9999999]
    >>> chunked.copy(a, storage='bcolztmp') # doctest: +ELLIPSIS
    carray((10000000,), int64)
      nbytes: 76.29 MB; cbytes: 1.85 MB; ratio: 41.19
      cparams := cparams(clevel=5, shuffle=1, cname='blosclz')
      rootdir := '/tmp/scikit_allel_....bcolz'
      mode    := 'w'
    [      0       1       2 ..., 9999997 9999998 9999999]
    >>> chunked.copy(a, storage='zarrmem')
    Array((10000000,), int64, chunks=(39063,), order=C)
      nbytes: 76.3M; nbytes_stored: 1.2M; ratio: 66.2; initialized: 256/256
      compressor: Blosc(cname='lz4', clevel=5, shuffle=1)
      store: DictStore
    >>> chunked.copy(a, storage='zarrtmp')
    Array((10000000,), int64, chunks=(39063,), order=C)
      nbytes: 76.3M; nbytes_stored: 1.2M; ratio: 66.2; initialized: 256/256
      compressor: Blosc(cname='lz4', clevel=5, shuffle=1)
      store: TempStore
    >>> chunked.copy(a, storage=chunked.BcolzStorage(cparams=bcolz.cparams(cname='lz4')))
    carray((10000000,), int64)
      nbytes: 76.29 MB; cbytes: 1.82 MB; ratio: 41.98
      cparams := cparams(clevel=5, shuffle=1, cname='lz4')
    [      0       1       2 ..., 9999997 9999998 9999999]
    >>> chunked.copy(a, storage='hdf5mem_zlib1')
    <HDF5 dataset "data": shape (10000000,), type "<i8">
    >>> chunked.copy(a, storage='hdf5tmp_zlib1')
    <HDF5 dataset "data": shape (10000000,), type "<i8">
    >>> import h5py
    >>> h5f = h5py.File('example.h5', mode='w')
    >>> h5g = h5f.create_group('test')
    >>> chunked.copy(a, storage='hdf5', group=h5g, name='data')
    <HDF5 dataset "data": shape (10000000,), type "<i8">
    >>> h5f['test/data']
    <HDF5 dataset "data": shape (10000000,), type "<i8">

"""
from __future__ import absolute_import, print_function, division


from .util import *
from .core import *

try:
    import h5py as _h5py
    from .storage_hdf5 import *
    storage_registry['default'] = hdf5mem_zlib1_storage
except ImportError:
    pass

try:
    import bcolz as _bcolz
    from .storage_bcolz import *
    storage_registry['default'] = bcolzmem_storage
except ImportError:
    pass

try:
    import zarr as _zarr
    from .storage_zarr import *
    storage_registry['default'] = zarrmem_storage
except ImportError:
    pass
