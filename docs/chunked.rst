Chunked storage utilities
=========================

.. automodule:: allel.chunked

Storage
-------

Zarr
~~~~

.. autoclass:: allel.chunked.storage_zarr.ZarrStorage
.. autoclass:: allel.chunked.storage_zarr.ZarrMemStorage
.. autoclass:: allel.chunked.storage_zarr.ZarrTmpStorage

.. autodata:: allel.chunked.storage_zarr.zarr_storage
    :annotation: = 'zarr'
.. autodata:: allel.chunked.storage_zarr.zarrmem_storage
    :annotation: = 'zarrmem'
.. autodata:: allel.chunked.storage_zarr.zarrtmp_storage
    :annotation: = 'zarrtmp'

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
.. autofunction:: allel.chunked.core.map_blocks
.. autofunction:: allel.chunked.core.reduce_axis
.. autofunction:: allel.chunked.core.amax
.. autofunction:: allel.chunked.core.amin
.. autofunction:: allel.chunked.core.asum
.. autofunction:: allel.chunked.core.count_nonzero
.. autofunction:: allel.chunked.core.compress
.. autofunction:: allel.chunked.core.take
.. autofunction:: allel.chunked.core.subset
.. autofunction:: allel.chunked.core.concatenate
.. autofunction:: allel.chunked.core.binary_op
.. autofunction:: allel.chunked.core.copy_table
.. autofunction:: allel.chunked.core.compress_table
.. autofunction:: allel.chunked.core.take_table
.. autofunction:: allel.chunked.core.concatenate_table
.. autofunction:: allel.chunked.core.eval_table

Classes
-------

.. autoclass:: allel.chunked.core.ChunkedArrayWrapper
.. autoclass:: allel.chunked.core.ChunkedTableWrapper
