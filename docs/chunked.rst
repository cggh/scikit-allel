Chunked storage
===============

.. automodule:: allel.chunked

Storage
-------

Different storage configurations can be used with the functions and classes
defined below. Wherever a function or method takes a `storage` keyword
argument, the value of the argument will determine the storage used for the
output.

If `storage` is a string, it will be used to look up one of several predefined
storage configurations via the storage registry, which is a dictionary
located at `allel.chunked.storage_registry`. The default storage can be
changed globally by setting the value of the 'default' key in the storage
registry.

Alternatively, storage may be an instance of one of the storage classes
defined below, e.g., :class:`allel.chunked.storage_bcolz.BcolzMemStorage` or
:class:`allel.chunked.storage_hdf5.HDF5TmpStorage`, which allows custom
configuration of storage parameters such as compression type and level.

For example::

    >>> import allel
    >>> import bcolz
    >>> a = bcolz.arange(100000)
    >>> a
    carray((100000,), int64)
      nbytes: 781.25 KB; cbytes: 269.83 KB; ratio: 2.90
      cparams := cparams(clevel=5, shuffle=True, cname='blosclz')
    [    0     1     2 ..., 99997 99998 99999]
    >>> allel.chunked.copy(a)
    carray((100000,), int64)
      nbytes: 781.25 KB; cbytes: 269.83 KB; ratio: 2.90
      cparams := cparams(clevel=5, shuffle=True, cname='blosclz')
    [    0     1     2 ..., 99997 99998 99999]
    >>> allel.chunked.copy(a, storage='bcolztmp')
    carray((100000,), int64)
      nbytes: 781.25 KB; cbytes: 269.83 KB; ratio: 2.90
      cparams := cparams(clevel=5, shuffle=True, cname='blosclz')
      rootdir := '/tmp/scikit_allel___rz4te1.bcolz'
      mode    := 'w'
    [    0     1     2 ..., 99997 99998 99999]
    >>> allel.chunked.copy(a, storage=allel.chunked.BcolzStorage(cparams=bcolz.cparams(cname='lz4')))
    carray((100000,), int64)
      nbytes: 781.25 KB; cbytes: 269.52 KB; ratio: 2.90
      cparams := cparams(clevel=5, shuffle=True, cname='lz4')
    [    0     1     2 ..., 99997 99998 99999]
    >>> allel.chunked.copy(a, storage='hdf5mem_zlib1')
    <HDF5 dataset "data": shape (100000,), type "<i8">
    >>> import h5py
    >>> h5f = h5py.File('example.h5', mode='w')
    >>> h5g = h5f.create_group('test')
    >>> allel.chunked.copy(a, storage='hdf5', group=h5g, name='data')
    <HDF5 dataset "data": shape (100000,), type "<i8">
    >>> h5f['test/data']
    <HDF5 dataset "data": shape (100000,), type "<i8">

bcolz
~~~~~

.. autoclass:: allel.chunked.storage_bcolz.BcolzStorage
.. autoclass:: allel.chunked.storage_bcolz.BcolzMemStorage
.. autoclass:: allel.chunked.storage_bcolz.BcolzTmpStorage

.. autodata:: allel.chunked.storage_bcolz.bcolz_storage
    :annotation: = 'bcolz'
.. autodata:: allel.chunked.storage_bcolz.bcolzmem_storage
    :annotation: = 'bcolzmem'
.. autodata:: allel.chunked.storage_bcolz.bcolztmp_storage
    :annotation: = 'bcolztmp'
.. autodata:: allel.chunked.storage_bcolz.bcolz_zlib1_storage
    :annotation: = 'bcolz_zlib1'
.. autodata:: allel.chunked.storage_bcolz.bcolzmem_zlib1_storage
    :annotation: = 'bcolzmem_zlib1'
.. autodata:: allel.chunked.storage_bcolz.bcolztmp_zlib1_storage
    :annotation: = 'bcolztmp_zlib1'

HDF5 (h5py)
~~~~~~~~~~~

.. autoclass:: allel.chunked.storage_hdf5.HDF5Storage
.. autoclass:: allel.chunked.storage_hdf5.HDF5MemStorage
.. autoclass:: allel.chunked.storage_hdf5.HDF5TmpStorage

.. autodata:: allel.chunked.storage_hdf5.hdf5_storage
    :annotation: = 'hdf5'
.. autodata:: allel.chunked.storage_hdf5.hdf5mem_storage
    :annotation: = 'hdf5mem'
.. autodata:: allel.chunked.storage_hdf5.hdf5tmp_storage
    :annotation: = 'hdf5tmp'
.. autodata:: allel.chunked.storage_hdf5.hdf5_zlib1_storage
    :annotation: = 'hdf5_zlib1'
.. autodata:: allel.chunked.storage_hdf5.hdf5mem_zlib1_storage
    :annotation: = 'hdf5mem_zlib1'
.. autodata:: allel.chunked.storage_hdf5.hdf5tmp_zlib1_storage
    :annotation: = 'hdf5tmp_zlib1'

.. autofunction:: allel.chunked.storage_hdf5.h5fmem
.. autofunction:: allel.chunked.storage_hdf5.h5ftmp

Functions
---------

.. autofunction:: allel.chunked.core.store
.. autofunction:: allel.chunked.core.copy
.. autofunction:: allel.chunked.core.apply
.. autofunction:: allel.chunked.core.areduce
.. autofunction:: allel.chunked.core.amax
.. autofunction:: allel.chunked.core.amin
.. autofunction:: allel.chunked.core.asum
.. autofunction:: allel.chunked.core.count_nonzero
.. autofunction:: allel.chunked.core.compress
.. autofunction:: allel.chunked.core.take
.. autofunction:: allel.chunked.core.subset
.. autofunction:: allel.chunked.core.hstack
.. autofunction:: allel.chunked.core.vstack
.. autofunction:: allel.chunked.core.binary_op
.. autofunction:: allel.chunked.core.copy_table
.. autofunction:: allel.chunked.core.compress_table
.. autofunction:: allel.chunked.core.take_table
.. autofunction:: allel.chunked.core.vstack_table
.. autofunction:: allel.chunked.core.eval_table

Classes
-------

.. autoclass:: allel.chunked.core.ChunkedArray
.. autoclass:: allel.chunked.core.ChunkedTable
