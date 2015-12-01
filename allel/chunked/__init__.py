# -*- coding: utf-8 -*-
# flake8: noqa
"""
This module provides an abstraction layer over generic chunked array storage
libraries. Currently HDF5 (via `h5py <http://www.h5py.org/>`_) and `bcolz
<http://bcolz.blosc.org>`_ are supported storage layers.

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
    >>> import bcolz
    >>> a = bcolz.arange(100000)
    >>> a
    carray((100000,), int64)
      nbytes: 781.25 KB; cbytes: 269.83 KB; ratio: 2.90
      cparams := cparams(clevel=5, shuffle=True, cname='blosclz')
    [    0     1     2 ..., 99997 99998 99999]
    >>> chunked.copy(a)
    carray((100000,), int64)
      nbytes: 781.25 KB; cbytes: 269.83 KB; ratio: 2.90
      cparams := cparams(clevel=5, shuffle=True, cname='blosclz')
    [    0     1     2 ..., 99997 99998 99999]
    >>> chunked.copy(a, storage='bcolztmp') # doctest: +ELLIPSIS
    carray((100000,), int64)
      nbytes: 781.25 KB; cbytes: 269.83 KB; ratio: 2.90
      cparams := cparams(clevel=5, shuffle=True, cname='blosclz')
      rootdir := '/tmp/scikit_allel_...'
      mode    := 'w'
    [    0     1     2 ..., 99997 99998 99999]
    >>> chunked.copy(a, storage=chunked.BcolzStorage(cparams=bcolz.cparams(cname='lz4')))
    carray((100000,), int64)
      nbytes: 781.25 KB; cbytes: 269.52 KB; ratio: 2.90
      cparams := cparams(clevel=5, shuffle=True, cname='lz4')
    [    0     1     2 ..., 99997 99998 99999]
    >>> chunked.copy(a, storage='hdf5mem_zlib1')
    <HDF5 dataset "data": shape (100000,), type "<i8">
    >>> import h5py
    >>> h5f = h5py.File('example.h5', mode='w')
    >>> h5g = h5f.create_group('test')
    >>> chunked.copy(a, storage='hdf5', group=h5g, name='data')
    <HDF5 dataset "data": shape (100000,), type "<i8">
    >>> h5f['test/data']
    <HDF5 dataset "data": shape (100000,), type "<i8">

"""
from __future__ import absolute_import, print_function, division


from .util import *
from .core import *

try:
    import h5py as _h5py
    from .storage_hdf5 import *
    storage_registry['default'] = HDF5MemStorage()
except ImportError:
    pass

try:
    import bcolz as _bcolz
    from .storage_bcolz import *
    storage_registry['default'] = BcolzStorage()
except ImportError:
    pass
