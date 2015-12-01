Chunked storage utilities
=========================

.. automodule:: allel.chunked

Storage
-------

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
.. autodata:: allel.chunked.storage_hdf5.hdf5_lzf_storage
    :annotation: = 'hdf5_lzf'
.. autodata:: allel.chunked.storage_hdf5.hdf5mem_lzf_storage
    :annotation: = 'hdf5mem_lzf'
.. autodata:: allel.chunked.storage_hdf5.hdf5tmp_lzf_storage
    :annotation: = 'hdf5tmp_lzf'

.. autofunction:: allel.chunked.storage_hdf5.h5fmem
.. autofunction:: allel.chunked.storage_hdf5.h5ftmp

Functions
---------

.. autofunction:: allel.chunked.core.store
.. autofunction:: allel.chunked.core.copy
.. autofunction:: allel.chunked.core.apply
.. autofunction:: allel.chunked.core.reduce_axis
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
