# -*- coding: utf-8 -*-
# flake8: noqa
"""
This module provides an abstraction layer over generic chunked array storage
libraries. Currently HDF5 (via `h5py <http://www.h5py.org/>`_) and
`zarr <http://zarr.readthedocs.io>`_ are supported.

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
defined below, e.g., :class:`allel.chunked.storage_zarr.ZarrMemStorage` or
:class:`allel.chunked.storage_hdf5.HDF5TmpStorage`, which allows custom
configuration of storage parameters such as compression type and level.

For example::

    >>> from allel import chunked
    >>> import numpy as np
    >>> a = np.arange(10000000)
    >>> chunked.copy(a)
    <zarr.core.Array (10000000,) int64>
    >>> chunked.copy(a, storage='zarrmem')
    <zarr.core.Array (10000000,) int64>
    >>> chunked.copy(a, storage='zarrtmp')
    <zarr.core.Array (10000000,) int64>
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
from .util import *
from .core import *

try:
    import h5py as _h5py
    from .storage_hdf5 import *
    storage_registry['default'] = hdf5mem_zlib1_storage
except ImportError:
    pass

try:
    import zarr as _zarr
    from .storage_zarr import *
    storage_registry['default'] = zarrmem_storage
except ImportError:
    pass
