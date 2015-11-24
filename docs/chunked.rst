Chunked storage
===============

.. automodule:: allel.chunked

Classes
-------

.. autoclass:: allel.chunked.core.ChunkedArray
.. autoclass:: allel.chunked.core.ChunkedTable

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

Storage
-------

bcolz
~~~~~

.. autoclass:: allel.chunked.storage_bcolz.BcolzStorage
.. autoclass:: allel.chunked.storage_bcolz.BcolzMemStorage
.. autoclass:: allel.chunked.storage_bcolz.BcolzTmpStorage
.. autodata:: allel.chunked.storage_bcolz.bcolz_storage

HDF5 (h5py)
~~~~~~~~~~~

.. autoclass:: allel.chunked.storage_hdf5.HDF5Storage
.. autoclass:: allel.chunked.storage_hdf5.HDF5MemStorage
.. autoclass:: allel.chunked.storage_hdf5.HDF5TmpStorage
.. autofunction:: allel.chunked.storage_hdf5.h5fmem
.. autofunction:: allel.chunked.storage_hdf5.h5ftmp
