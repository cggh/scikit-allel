# -*- coding: utf-8 -*-
# cython: profile=True
from __future__ import absolute_import, print_function, division


import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef inline np.float32_t gn_corrcoef_int8(np.int8_t[:] gn0,
                                           np.int8_t[:] gn1,
                                           np.int8_t[:] gn0_sq,
                                           np.int8_t[:] gn1_sq,
                                           np.float32_t fill=np.nan):
    cdef:
        np.int8_t x, y, xsq, ysq
        Py_ssize_t i
        int n
        np.float32_t m0, m1, v0, v1, cov, r

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
def gn_pairwise_corrcoef_int8(np.int8_t[:, :] gn, np.float32_t fill=np.nan):
    cdef:
        Py_ssize_t i, j, k, n
        np.float32_t r
        # correlation matrix in condensed form
        np.float32_t[:] out
        np.int8_t[:, :] gn_sq
        np.int8_t[:] gn0, gn1, gn0_sq, gn1_sq

    # cache square calculation to improve performance
    gn_sq = np.power(gn, 2)

    # setup output array
    n = gn.shape[0]
    # number of distinct pairs
    n_pairs = n * (n - 1) // 2
    out = np.zeros(n_pairs, dtype=np.float32)

    # iterate over distinct pairs
    k = 0
    for i in range(n):
        for j in range(i+1, n):
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
def gn_pairwise2_corrcoef_int8(np.int8_t[:, :] gna,
                               np.int8_t[:, :] gnb,
                               np.float32_t fill=np.nan):
    cdef:
        Py_ssize_t i, j, k, m, n
        np.float32_t r
        # correlation matrix in condensed form
        np.float32_t[:, :] out
        np.int8_t[:, :] gna_sq, gnb_sq
        np.int8_t[:] gn0, gn1, gn0_sq, gn1_sq

    # cache square calculation to improve performance
    gna_sq = np.power(gna, 2)
    gnb_sq = np.power(gnb, 2)

    # setup output array
    m = gna.shape[0]
    n = gnb.shape[0]
    out = np.zeros((m, n), dtype=np.float32)

    # iterate over distinct pairs
    for i in range(m):
        for j in range(n):
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
def gn_locate_unlinked_int8(np.int8_t[:, :] gn, Py_ssize_t size,
                            Py_ssize_t step, np.float32_t threshold):
    cdef:
        np.uint8_t[:] loc
        Py_ssize_t window_start, window_stop, i, j
        np.float32_t r_squared
        np.int8_t[:, :] gn_sq
        np.int8_t[:] gn0, gn1, gn0_sq, gn1_sq
        int overlap = size - step
        bint last

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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef Py_ssize_t shared_prefix_length_int8(np.int8_t[:] a, np.int8_t[:] b):
    """Compute the length of the shared prefix between two arrays."""

    cdef:
        Py_ssize_t i, n

    # count up to the length of the shortest array
    n = min(a.shape[0], b.shape[0])

    # iterate until we find a difference
    for i in range(n):
        if a[i] != b[i]:
            return i

    # arrays are equal up to shared length
    return n


cpdef pairwise_shared_prefix_lengths_int8(np.int8_t[:, :] h):
    """Compute the length of the shared prefix between all pairs of
    columns in a 2-dimensional array."""

    cdef:
        Py_ssize_t i, j, k, n, n_pairs
        np.int32_t[:] lengths

    # initialise variables
    n = h.shape[1]
    n_pairs = (n * (n - 1)) // 2
    lengths = np.empty(n_pairs, dtype='i4')
    k = 0

    # iterate over pairs
    for i in range(n):
        for j in range(i+1, n):
            lengths[k] = shared_prefix_length_int8(h[:, i], h[:, j])
            k += 1

    return np.asarray(lengths)


cpdef neighbour_shared_prefix_lengths_int8(np.int8_t[:, :] h):
    """Compute the length of the shared prefix between neighbouring
    columns in a 2-dimensional array."""

    cdef:
        Py_ssize_t i, n
        np.int32_t[:] lengths

    # initialise variables
    n = h.shape[1]
    lengths = np.empty(n-1, dtype='i4')

    # iterate over columns
    for i in range(n-1):
        lengths[i] = shared_prefix_length_int8(h[:, i], h[:, i+1])

    return np.asarray(lengths)


cdef inline Py_ssize_t bisect_left_int8(np.int8_t[:] s, int x):
    """Optimized implementation of bisect_left."""
    cdef:
        Py_ssize_t l, u, m, v

    # initialise
    l = 0  # lower index
    u = s.shape[0]  # upper index

    # bisect
    while (u - l) > 1:
        m = (u + l) // 2
        v = s[m]
        if v >= x:
            u = m
        else:
            l = m

    # check boundary condition
    if s[l] >= x:
        return l

    return u


def prefix_sort(h):
    """Sort columns in the input array by prefix, i.e., lexical sort,
    using the first variant as the first key, then the second variant, etc."""
    lex = np.lexsort(h[::-1])
    h = np.take(h, lex, axis=1)
    return h


def paint_shared_prefixes_int8(np.int8_t[:, :] h):

    cdef:
        Py_ssize_t n_variants, n_haplotypes, pp_start, pp_stop, pp_size, n0, n1
        np.int32_t pp_color, next_color
        np.int32_t[:, :] painting
        np.int8_t[:] s

    # first sort columns in the input array by prefix
    h = prefix_sort(h)

    # initialise variables
    n_variants = h.shape[0]
    n_haplotypes = h.shape[1]
    prefixes = [(0, n_haplotypes, 1)]
    next_color = 2
    painting = np.zeros((n_variants, n_haplotypes), dtype='i4')

    # iterate over variants
    for i in range(n_variants):

        # setup for this iteration
        parent_prefixes = prefixes
        prefixes = list()

        if not parent_prefixes:
            # no more shared prefixes
            break

        # iterate over parent prefixes
        for pp_start, pp_stop, pp_color in parent_prefixes:
            pp_size = pp_stop - pp_start

            # find the split point
            s = h[i, pp_start:pp_stop]
            # number of reference alleles
            n0 = bisect_left_int8(s, 1)
            # number of alternate alleles
            n1 = pp_size - n0

            if n0 == 0 or n1 == 0:
                # no split, continue parent prefix
                painting[i, pp_start:pp_stop] = pp_color
                prefixes.append((pp_start, pp_stop, pp_color))

            elif n0 > n1:
                # ref is major, alt is minor
                painting[i, pp_start:pp_start+n0] = pp_color
                prefixes.append((pp_start, pp_start+n0, pp_color))
                if n1 > 1:
                    painting[i, pp_start+n0:pp_stop] = next_color
                    prefixes.append((pp_start+n0, pp_stop, next_color))
                    next_color += 1

            elif n1 > n0:
                # ref is minor, alt is major
                if n0 > 1:
                    painting[i, pp_start:pp_start+n0] = next_color
                    prefixes.append((pp_start, pp_start+n0, next_color))
                    next_color += 1
                painting[i, pp_start+n0:pp_stop] = pp_color
                prefixes.append((pp_start+n0, pp_stop, pp_color))

            elif n0 == n1 and n0 > 1:
                # same number of ref and alt alleles, arbitrarily pick ref as major
                painting[i, pp_start:pp_start+n0] = pp_color
                prefixes.append((pp_start, pp_start+n0, pp_color))
                painting[i, pp_start+n0:pp_stop] = next_color
                prefixes.append((pp_start+n0, pp_stop, next_color))
                next_color += 1

    return np.asarray(painting)
