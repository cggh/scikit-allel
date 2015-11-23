# -*- coding: utf-8 -*-
# flake8: noqa
from __future__ import absolute_import, print_function, division


from .util import storage_registry
from .storage_bcolz import BcolzStorage, BcolzMemStorage, BcolzTmpStorage, \
    bcolzmem_storage, bcolztmp_storage, bcolzmem_zlib1_storage, \
    bcolztmp_zlib1_storage
from .storage_hdf5 import HDF5Storage, HDF5MemStorage, HDF5TmpStorage, \
    hdf5mem_storage, hdf5tmp_storage, hdf5mem_zlib1_storage, \
    hdf5tmp_zlib1_storage
from .core import store, copy, apply, areduce, amax, amin, asum, \
    count_nonzero, compress, take, compress_table, take_table, subset, \
    hstack, vstack, vstack_table, binary_op, ChunkedArray, ChunkedTable
from .ext import GenotypeChunkedArray, HaplotypeChunkedArray, \
    AlleleCountsChunkedArray, VariantChunkedTable, FeatureChunkedTable, \
    AlleleCountsChunkedTable


storage_registry['default'] = BcolzMemStorage()
