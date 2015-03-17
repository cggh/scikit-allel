# -*- coding: utf-8 -*-
# cython: profile=True
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
cpdef inline cnp.float32_t gn_corrcoef_int8(cnp.int8_t[:] gn0,
                                            cnp.int8_t[:] gn1,
                                            cnp.int8_t[:] gn0_sq,
                                            cnp.int8_t[:] gn1_sq,
                                            cnp.float32_t fill=np.nan):
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
        r = fill
    else:
        r = cov / sqrt(v0 * v1)

    return r


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
def gn_pairwise_corrcoef_int8(cnp.int8_t[:, :] gn, cnp.float32_t fill=np.nan):
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
            r = gn_corrcoef_int8(gn0, gn1, gn0_sq, gn1_sq, fill)
            out[k] = r
            k += 1

    return np.asarray(out)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
def gn_pairwise2_corrcoef_int8(cnp.int8_t[:, :] gna,
                               cnp.int8_t[:, :] gnb,
                               cnp.float32_t fill=np.nan):
    cdef int i, j, k
    cdef cnp.float32_t r
    # correlation matrix in condensed form
    cdef cnp.float32_t[:, :] out
    cdef cnp.int8_t[:, :] gna_sq, gnb_sq
    cdef cnp.int8_t[:] gn0, gn1, gn0_sq, gn1_sq

    # cache square calculation to improve performance
    gna_sq = np.power(gna, 2)
    gnb_sq = np.power(gnb, 2)

    # setup output array
    m = gna.shape[0]
    n = gnb.shape[0]
    out = np.zeros((m, n), dtype=np.float32)

    # iterate over distinct pairs
    for i in range(gna.shape[0]):
        for j in range(gnb.shape[0]):
            gn0 = gna[i]
            gn1 = gnb[j]
            gn0_sq = gna_sq[i]
            gn1_sq = gnb_sq[j]
            r = gn_corrcoef_int8(gn0, gn1, gn0_sq, gn1_sq, fill)
            out[i, j] = r

    return np.asarray(out)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
def gn_locate_unlinked_int8(cnp.int8_t[:, :] gn, int size, int step,
                            cnp.float32_t threshold):
    cdef cnp.uint8_t[:] loc
    cdef int window_start, window_stop, i, j
    cdef cnp.float32_t r_squared
    cdef cnp.int8_t[:, :] gn_sq
    cdef cnp.int8_t[:] gn0, gn1, gn0_sq, gn1_sq
    cdef int overlap = size - step
    cdef bint last

    # cache square calculation to improve performance
    gn_sq = np.power(gn, 2)

    # setup output
    loc = np.ones(gn.shape[0], dtype='u1')

    # setup intermediates
    last = False

    for window_start in range(0, gn.shape[0], step):

        # determine end of current window
        window_stop = window_start + size
        if window_stop > gn.shape[0]:
            window_stop = gn.shape[0]
            last = True

        if window_start == 0:
            # first window
            for i in range(window_start, window_stop):
                # only go further if still unlinked
                if loc[i]:
                    for j in range(i+1, window_stop):
                        # only go further if still unlinked
                        if loc[j]:
                            gn0 = gn[i]
                            gn1 = gn[j]
                            gn0_sq = gn_sq[i]
                            gn1_sq = gn_sq[j]
                            r_squared = gn_corrcoef_int8(gn0, gn1, gn0_sq, gn1_sq) ** 2
                            if r_squared > threshold:
                                loc[j] = 0

        else:
            # subsequent windows
            for i in range(window_start, window_stop):
                # only go further if still unlinked
                if loc[i]:
                    # don't recalculate anything from overlap with previous
                    # window
                    ii = max(i+1, window_start+overlap)
                    if ii < window_stop:
                        for j in range(ii, window_stop):
                            # only go further if still unlinked
                            if loc[j]:
                                gn0 = gn[i]
                                gn1 = gn[j]
                                gn0_sq = gn_sq[i]
                                gn1_sq = gn_sq[j]
                                r_squared = gn_corrcoef_int8(gn0, gn1, gn0_sq, gn1_sq) ** 2
                                if r_squared > threshold:
                                    loc[j] = 0

        if last:
            break

    return np.asarray(loc).view(dtype='b1')
