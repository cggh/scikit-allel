# -*- coding: utf-8 -*-
# flake8: noqa
from __future__ import absolute_import, print_function, division


from .util import storage_registry
from .storage_bcolz import bcolzmem_storage, bcolztmp_storage, BcolzStorage,\
    BcolzMemStorage, BcolzTmpStorage
from .storage_hdf5 import hdf5mem_storage, hdf5tmp_storage, HDF5Storage, \
    HDF5MemStorage, HDF5TmpStorage
from .core import store, copy, apply, areduce, amax, amin, asum, \
    count_nonzero, compress, take, compress_table, take_table, subset, \
    hstack, vstack, vstack_table, binary_op, Array, Table
from .ext import GenotypeChunkedArray, HaplotypeChunkedArray, \
    AlleleCountsChunkedArray, VariantChunkedTable, FeatureChunkedTable, \
    AlleleCountsChunkedTable


storage_registry['default'] = bcolzmem_storage
