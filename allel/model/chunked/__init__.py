# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


from allel.model.chunked.backend_numpy import NumpyBackend, numpy_backend


from allel.model.chunked.backend_bcolz import BColzBackend, bcolz_backend, \
    bcolz_gzip1_backend, BColzTmpBackend, bcolztmp_backend, \
    bcolztmp_gzip1_backend


from allel.model.chunked.backend_h5py import H5memBackend, H5tmpBackend, \
    h5dmem, h5dtmp, h5fmem, h5ftmp, h5mem_backend, h5mem_gzip1_backend, \
    h5tmp_backend, h5tmp_gzip1_backend


from allel.model.chunked.model import ChunkedArray, GenotypeChunkedArray, \
    HaplotypeChunkedArray, AlleleCountsChunkedArray, ChunkedTable


__all__ = ['ChunkedArray', 'GenotypeChunkedArray', 'HaplotypeChunkedArray',
           'AlleleCountsChunkedArray', 'ChunkedTable']