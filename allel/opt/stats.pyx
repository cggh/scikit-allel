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
def gn_locate_unlinked_int8(np.int8_t[:, :] gn, np.uint8_t[:] loc,
                            Py_ssize_t size, Py_ssize_t step,
                            np.float32_t threshold):
    cdef:
        Py_ssize_t window_start, window_stop, i, j
        np.float32_t r_squared
        np.int8_t[:, :] gn_sq
        np.int8_t[:] gn0, gn1, gn0_sq, gn1_sq
        int overlap = size - step
        bint last

    # cache square calculation to improve performance
    gn_sq = np.power(gn, 2)

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


cpdef neighbour_shared_prefix_lengths_unsorted_int8(np.int8_t[:, :] h,
                                                    np.int64_t[:] indices):
    """Compute the length of the shared prefix between neighbouring
    columns in a 2-dimensional array."""

    cdef:
        Py_ssize_t i, n, ix, jx
        np.int32_t[:] lengths

    # initialise variables
    n = h.shape[1]
    lengths = np.empty(n-1, dtype='i4')

    # iterate over columns
    for i in range(n-1):
        ix = indices[i]
        jx = indices[i+1]
        lengths[i] = shared_prefix_length_int8(h[:, ix], h[:, jx])

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


def paint_shared_prefixes_int8(np.int8_t[:, :] h):
    """Paint each shared prefix with a different number. N.B., `h` must be
    already sorted by prefix.

    """

    cdef:
        Py_ssize_t n_variants, n_haplotypes, pp_start, pp_stop, pp_size, n0, n1
        np.int32_t pp_color, next_color
        np.int32_t[:, :] painting
        np.int8_t[:] s

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


from bisect import bisect_right


cdef inline np.float64_t ssl2ihh(ssl, pos, i, min_ehh):
    """Compute integrated haplotype homozygosity from shared suffix lengths."""

    n_pairs = ssl.shape[0]
    if n_pairs > 0:

        # compute EHH
        b = np.bincount(ssl)
        c = np.cumsum(b[::-1])[:-1]
        ehh = c / n_pairs

        # deal with minimum EHH
        if min_ehh > 0:
            ix = bisect_right(ehh, min_ehh)
            ehh = ehh[ix:]

        # compute variant spacing
        s = ehh.shape[0]
        # take absolute value because this might be a reverse scan
        g = np.abs(np.diff(pos[i-s+1:i+1]))

        # compute IHH via trapezoid rule
        ihh = np.sum(g * (ehh[:-1] + ehh[1:]) / 2)

    else:
        ihh = 0

    return ihh


def ihh_scan_int8(np.int8_t[:, :] h, pos, min_ehh=0):
    """Scan forwards over haplotypes, computing the integrated haplotype
    homozygosity backwards for each variant."""

    cdef:
        Py_ssize_t n_variants, n_haplotypes, n_pairs, i, j, k, p, s
        np.int32_t[:] ssl
        np.int8_t a1, a2
        np.float64_t[:] vihh
        np.float64_t ihh

    # initialise
    n_variants = h.shape[0]
    n_haplotypes = h.shape[1]
    n_pairs = (n_haplotypes * (n_haplotypes - 1)) // 2

    # shared suffix lengths between all pairs of haplotypes
    ssl = np.zeros(n_pairs, dtype='i4')

    # integrated haplotype homozygosity values for each variant
    vihh = np.empty(n_variants, dtype='f8')

    # iterate forward over variants
    for i in range(n_variants):

        # pairwise comparison of alleles between haplotypes to determine
        # shared suffix lengths
        # N.B., this is the critical performance section
        p = 0  # pair index
        for j in range(n_haplotypes):
            a1 = h[i, j]  # allele on first haplotype in pair
            for k in range(j+1, n_haplotypes):
                a2 = h[i, k]  # allele on second haplotype in pair
                # test for non-equal and non-missing alleles
                if (a1 != a2) and (a1 >= 0) and (a2 >= 0):
                    # break shared suffix, reset length to zero
                    ssl[p] = 0
                else:
                    # extend shared suffix
                    ssl[p] += 1
                # increment pair index
                p += 1

        # compute IHH from shared suffix lengths
        ihh = ssl2ihh(ssl, pos, i, min_ehh)
        vihh[i] = ihh

    return np.asarray(vihh)


cdef np.int32_t[:] tovector_int32(np.int32_t[:, :] m):
    cdef:
        Py_ssize_t n, n_pairs, i, j, k
        np.int32_t[:] v
    n = m.shape[0]
    n_pairs = (n * (n - 1)) // 2
    v = np.empty(n_pairs, dtype='i4')
    k = 0
    for i in range(n):
        for j in range(i+1, n):
            v[k] = m[i, j]
            k += 1
    return v


def ihh01_scan_int8(np.int8_t[:, :] h, pos, min_ehh=0):
    """Scan forwards over haplotypes, computing the integrated haplotype
    homozygosity backwards for each variant for the reference (0) and
    alternate (1) alleles separately."""

    cdef:
        Py_ssize_t n_variants, n_haplotypes, n_pairs, i, j, k, p, s
        np.int32_t[:, :] ssl
        np.int8_t a1, a2
        np.float64_t[:] vihh0, vihh1
        np.float64_t ihh0, ihh1
        np.uint8_t[:] loc0, loc1

    # initialise
    n_variants = h.shape[0]
    n_haplotypes = h.shape[1]
    # location of haplotypes carrying reference (0) allele
    loc0 = np.zeros(n_haplotypes, dtype='u1')
    # location of haplotypes carrying alternate (1) allele
    loc1 = np.zeros(n_haplotypes, dtype='u1')

    # shared suffix lengths between all pairs of haplotypes
    # N.B., this time we'll use a square matrix, because this makes
    # subsetting to ref-ref and alt-alt pairs easier and quicker further
    # down the line
    ssl = np.zeros((n_haplotypes, n_haplotypes), dtype='i4')

    # integrated haplotype homozygosity values for each variant
    vihh0 = np.empty(n_variants, dtype='f8')
    vihh1 = np.empty(n_variants, dtype='f8')

    # iterate forward over variants
    for i in range(n_variants):

        # pairwise comparison of alleles between haplotypes to determine
        # shared suffix lengths
        # N.B., this is the critical performance section
        loc0[:] = 0
        loc1[:] = 0
        for j in range(n_haplotypes):
            a1 = h[i, j]
            # keep track of which haplotypes carry which alleles
            if a1 == 0:
                loc0[j] = 1
            elif a1 == 1:
                loc1[j] = 1
            for k in range(j+1, n_haplotypes):
                a2 = h[i, k]
                # test for non-equal and non-missing alleles
                if (a1 != a2) and (a1 >= 0) and (a2 >= 0):
                    # break shared suffix, reset to zero
                    ssl[j, k] = 0
                else:
                    # extend shared suffix
                    ssl[j, k] += 1

        # locate 00 and 11 pairs
        l0 = np.asarray(loc0, dtype='b1')
        l1 = np.asarray(loc1, dtype='b1')
        ssl00 = tovector_int32(np.asarray(ssl).compress(l0, axis=0).compress(l0, axis=1))
        ssl11 = tovector_int32(np.asarray(ssl).compress(l1, axis=0).compress(l1, axis=1))

        # compute IHH from shared suffix lengths
        ihh0 = ssl2ihh(ssl00, pos, i, min_ehh)
        ihh1 = ssl2ihh(ssl11, pos, i, min_ehh)
        vihh0[i] = ihh0
        vihh1[i] = ihh1

    return np.asarray(vihh0), np.asarray(vihh1)
