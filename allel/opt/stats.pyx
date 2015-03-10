# -*- coding: utf-8 -*-
# cython: profile=False
from __future__ import absolute_import, print_function, division


import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport sqrt


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef inline double gn_corrcoef_int8(cnp.int8_t[:] gn0,
                                     cnp.int8_t[:] gn1,
                                     cnp.int8_t[:] gn0_sq,
                                     cnp.int8_t[:] gn1_sq):
    cdef cnp.int8_t x, y, xsq, ysq
    cdef int n
    cdef cnp.float32_t m0, m1, v0, v1, cov, r

    # initialise variables
    m0 = m1 = v0 = v1 = cov = n = 0

    # iterate over input vectors
    for i in range(gn0.shape[0]):
        x = gn0[i]
        y = gn1[i]
        # consider negative values as missing
        if x >= 0 and y >= 0:
            n += 1
            m0 += x
            m1 += y
            xsq = gn0_sq[i]
            ysq = gn1_sq[i]
            v0 += xsq
            v1 += ysq
            cov += x * y

    # compute mean, variance, covariance
    m0 /= n
    m1 /= n
    v0 /= n
    v1 /= n
    cov /= n
    cov -= m0 * m1
    v0 -= m0 * m0
    v1 -= m1 * m1

    # compute correlation coeficient
    if v0 == 0 or v1 == 0:
        r = np.nan
    else:
        r = cov / sqrt(v0 * v1)

    return r


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
def gn_pairwise_corrcoef_int8(cnp.int8_t[:, :] gn):
    cdef int i, j, k
    cdef cnp.float32_t r
    # correlation matrix in condensed form
    cdef cnp.float32_t[:] out
    cdef cnp.int8_t[:, :] gn_sq
    cdef cnp.int8_t[:] gn0, gn1, gn0_sq, gn1_sq

    # cache square calculation to improve performance
    gn_sq = np.power(gn, 2)

    # setup output array
    n = gn.shape[0]
    # number of distinct pairs
    n_pairs = n * (n - 1) // 2
    out = np.zeros((n_pairs,), dtype=np.float32)

    # iterate over distinct pairs
    k = 0
    for i in range(gn.shape[0]):
        for j in range(i+1, gn.shape[0]):
            gn0 = gn[i]
            gn1 = gn[j]
            gn0_sq = gn_sq[i]
            gn1_sq = gn_sq[j]
            r = gn_corrcoef_int8(gn0, gn1, gn0_sq, gn1_sq)
            out[k] = r
            k += 1

    return np.asarray(out)
