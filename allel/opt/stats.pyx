# -*- coding: utf-8 -*-
# cython: profile=True
# cython: linetrace=False
# cython: binding=False
from __future__ import absolute_import, print_function, division


import numpy as np
cimport numpy as cnp
import cython
cimport cython
from libc.math cimport sqrt, fabs, fmin
from libc.stdlib cimport malloc, free
from libc.string cimport memset


# work around NAN undeclared in windows
cdef:
    cnp.float32_t nan32 = np.nan
    cnp.float64_t nan64 = np.nan


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef inline cnp.float32_t gn_corrcoef_int8(cnp.int8_t[:] gn0,
                                            cnp.int8_t[:] gn1,
                                            cnp.int8_t[:] gn0_sq,
                                            cnp.int8_t[:] gn1_sq,
                                            cnp.float32_t fill) nogil:
    cdef:
        cnp.int8_t x, y, xsq, ysq
        Py_ssize_t i
        int n
        cnp.float32_t m0, m1, v0, v1, cov, r

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
def gn_pairwise_corrcoef_int8(cnp.int8_t[:, :] gn not None,
                              cnp.float32_t fill=nan32):
    cdef:
        Py_ssize_t i, j, k, n
        cnp.float32_t r
        # correlation matrix in condensed form
        cnp.float32_t[:] out
        cnp.int8_t[:, :] gn_sq
        cnp.int8_t[:] gn0, gn1, gn0_sq, gn1_sq

    # cache square calculation to improve performance
    gn_sq = np.power(gn, 2)

    # setup output array
    n = gn.shape[0]
    # number of distinct pairs
    n_pairs = n * (n - 1) // 2
    out = np.zeros(n_pairs, dtype=np.float32)

    # iterate over distinct pairs
    with nogil:
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
def gn_pairwise2_corrcoef_int8(cnp.int8_t[:, :] gna not None,
                               cnp.int8_t[:, :] gnb not None,
                               cnp.float32_t fill=nan32):
    cdef:
        Py_ssize_t i, j, k, m, n
        cnp.float32_t r
        # correlation matrix in condensed form
        cnp.float32_t[:, :] out
        cnp.int8_t[:, :] gna_sq, gnb_sq
        cnp.int8_t[:] gn0, gn1, gn0_sq, gn1_sq

    # cache square calculation to improve performance
    gna_sq = np.power(gna, 2)
    gnb_sq = np.power(gnb, 2)

    # setup output array
    m = gna.shape[0]
    n = gnb.shape[0]
    out = np.zeros((m, n), dtype=np.float32)

    # iterate over distinct pairs
    with nogil:
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
def gn_locate_unlinked_int8(cnp.int8_t[:, :] gn not None,
                            cnp.uint8_t[:] loc not None,
                            Py_ssize_t size, Py_ssize_t step,
                            cnp.float32_t threshold):
    cdef:
        Py_ssize_t window_start, window_stop, i, j, n_variants
        cnp.float32_t r_squared
        cnp.int8_t[:, :] gn_sq
        cnp.int8_t[:] gn0, gn1, gn0_sq, gn1_sq
        int overlap = size - step
        bint last
        cnp.float32_t fill = nan32

    # cache square calculation to improve performance
    gn_sq = np.power(gn, 2)

    # setup
    n_variants = gn.shape[0]
    last = False

    for window_start in range(0, n_variants, step):
        with nogil:

            # determine end of current window
            window_stop = window_start + size
            if window_stop > n_variants:
                window_stop = n_variants
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
                                r_squared = gn_corrcoef_int8(gn0, gn1, gn0_sq,
                                                             gn1_sq, fill) ** 2
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
                                    r_squared = gn_corrcoef_int8(gn0, gn1,
                                                                 gn0_sq,
                                                                 gn1_sq,
                                                                 fill) ** 2
                                    if r_squared > threshold:
                                        loc[j] = 0

            if last:
                break


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef Py_ssize_t shared_prefix_length_int8(cnp.int8_t[:] a,
                                           cnp.int8_t[:] b) nogil:
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


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef pairwise_shared_prefix_lengths_int8(cnp.int8_t[:, :] h):
    """Compute the length of the shared prefix between all pairs of
    columns in a 2-dimensional array."""

    cdef:
        Py_ssize_t i, j, k, n, n_pairs
        cnp.int32_t[:] lengths

    # initialise variables
    n = h.shape[1]
    n_pairs = (n * (n - 1)) // 2
    lengths = np.empty(n_pairs, dtype='i4')
    k = 0

    # iterate over pairs
    with nogil:
        for i in range(n):
            for j in range(i+1, n):
                lengths[k] = shared_prefix_length_int8(h[:, i], h[:, j])
                k += 1

    return np.asarray(lengths)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef neighbour_shared_prefix_lengths_int8(cnp.int8_t[:, :] h):
    """Compute the length of the shared prefix between neighbouring
    columns in a 2-dimensional array."""

    cdef:
        Py_ssize_t i, n
        cnp.int32_t[:] lengths

    # initialise variables
    n = h.shape[1]
    lengths = np.empty(n-1, dtype='i4')

    # iterate over columns
    with nogil:
        for i in range(n-1):
            lengths[i] = shared_prefix_length_int8(h[:, i], h[:, i+1])

    return np.asarray(lengths)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef neighbour_shared_prefix_lengths_unsorted_int8(cnp.int8_t[:, :] h,
                                                    cnp.int64_t[:] indices):
    """Compute the length of the shared prefix between neighbouring
    columns in a 2-dimensional array."""

    cdef:
        Py_ssize_t i, n, ix, jx
        cnp.int32_t[:] lengths

    # initialise variables
    n = h.shape[1]
    lengths = np.empty(n-1, dtype='i4')

    # iterate over columns
    with nogil:
        for i in range(n-1):
            ix = indices[i]
            jx = indices[i+1]
            lengths[i] = shared_prefix_length_int8(h[:, ix], h[:, jx])

    return np.asarray(lengths)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef inline Py_ssize_t bisect_left_int8(cnp.int8_t[:] s, int x) nogil:
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


@cython.boundscheck(False)
@cython.wraparound(False)
def paint_shared_prefixes_int8(cnp.int8_t[:, :] h not None):
    """Paint each shared prefix with a different number. N.B., `h` must be
    already sorted by prefix.

    """

    cdef:
        Py_ssize_t n_variants, n_haplotypes, pp_start, pp_stop, pp_size, n0, n1
        cnp.int32_t pp_color, next_color
        cnp.int32_t[:, :] painting
        cnp.int8_t[:] s

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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef cnp.float64_t ssl2ihh(cnp.int32_t[:] ssl,
                            cnp.int32_t l_max,
                            Py_ssize_t variant_idx,
                            cnp.float64_t[:] gaps,
                            cnp.float64_t min_ehh=0,
                            bint include_edges=False) nogil:
    """Compute integrated haplotype homozygosity from shared suffix lengths.

    Parameters
    ----------
    ssl : ndarray, int32, shape (n_pairs,)
        Shared suffix lengths between all haplotype pairs.
    l_max : int
        Largest value within `ssl`.
    variant_idx : int
        Current variant index.
    gaps : ndarray, float64, shape (n_variants - 1,)
        Gaps between variants.
    min_ehh : float
        Minimum EHH below which IHH computation will be truncated.
    include_edges : bool
        If True, report results for variants where EHH does not fall below the
        specified minimum before reaching the contig end.

    Returns
    -------
    ihh : float
        Integrated haplotype homozygosity.

    """

    cdef:
        Py_ssize_t i, j, gap_idx, n_pairs
        cnp.int32_t l
        cnp.float64_t ehh_prv, ehh_cur, ihh, ret, gap, n_pairs_ident
        int *hh_breaks

    # initialize
    n_pairs = ssl.shape[0]

    # only compute if at least 1 pair
    if n_pairs > 0:

        # initialise
        ihh = 0

        # find breaks in haplotype homozygosity via bincount
        hh_breaks = <int *>malloc((l_max + 1) * sizeof(int))

        try:

            # do bincount
            memset(hh_breaks, 0, (l_max + 1) * sizeof(int))
            for j in range(n_pairs):
                l = ssl[j]
                hh_breaks[l] += 1

            # initialise EHH
            n_pairs_ident = n_pairs - hh_breaks[0]
            ehh_prv = n_pairs_ident / n_pairs

            # edge case - check if haplotype homozygosity is already below
            # min_ehh
            if ehh_prv <= min_ehh:
                return 0

            # iterate backwards over variants
            for i in range(1, variant_idx + 1):

                # compute current EHH
                n_pairs_ident -= hh_breaks[i]
                ehh_cur = n_pairs_ident / n_pairs

                # determine gap width
                gap_idx = variant_idx - i
                gap = gaps[gap_idx]

                # handle very long gaps
                if gap < 0:
                    return nan64

                # accumulate IHH
                ihh += gap * (ehh_cur + ehh_prv) / 2

                # check if we've reached minimum EHH
                if ehh_cur <= min_ehh:
                    return ihh

                # move on
                ehh_prv = ehh_cur

        finally:
            # clean up
            free(hh_breaks)

        # if we get this far, EHH never decayed below min_ehh
        if include_edges:
            return ihh

    return nan64


@cython.boundscheck(False)
@cython.wraparound(False)
def ihh_scan_int8(cnp.int8_t[:, :] h,
                  cnp.float64_t[:] gaps,
                  cnp.float64_t min_ehh=0,
                  bint include_edges=False):
    """Scan forwards over haplotypes, computing the integrated haplotype
    homozygosity backwards for each variant."""

    cdef:
        Py_ssize_t n_variants, n_haplotypes, n_pairs, i, j, k, u, s
        cnp.int32_t[:] ssl
        cnp.int32_t l, l_max
        cnp.int8_t a1, a2
        cnp.float64_t[:] vihh
        cnp.float64_t ihh

    n_variants = h.shape[0]
    # initialise
    n_haplotypes = h.shape[1]
    n_pairs = (n_haplotypes * (n_haplotypes - 1)) // 2

    # shared suffix lengths between all pairs of haplotypes
    ssl = np.zeros(n_pairs, dtype='i4')

    # integrated haplotype homozygosity values for each variant
    vihh = np.empty(n_variants, dtype='f8')

    with nogil:

        # iterate forward over variants
        for i in range(n_variants):
            u = 0  # pair index
            l_max = 0

            # pairwise comparison of alleles between haplotypes to determine
            # shared suffix lengths
            for j in range(n_haplotypes):
                a1 = h[i, j]  # allele on first haplotype in pair
                for k in range(j+1, n_haplotypes):
                    a2 = h[i, k]  # allele on second haplotype in pair
                    # test for non-equal and non-missing alleles
                    if (a1 != a2) and (a1 >= 0) and (a2 >= 0):
                        # break shared suffix, reset length to zero
                        l = 0
                    else:
                        # extend shared suffix
                        l = ssl[u] + 1
                    ssl[u] = l
                    # increment pair index
                    u += 1
                    # update max l
                    if l > l_max:
                        l_max = l

            # compute IHH from shared suffix lengths
            ihh = ssl2ihh(ssl, l_max, i, gaps,
                          min_ehh=min_ehh,
                          include_edges=include_edges)
            vihh[i] = ihh

    return np.asarray(vihh)


@cython.boundscheck(False)
@cython.wraparound(False)
def nsl_scan_int8(cnp.int8_t[:, :] h):
    """Scan forwards over haplotypes, computing NSL backwards for each variant."""

    cdef:
        Py_ssize_t n_variants, n_haplotypes, n_pairs, i, j, k, u, s
        cnp.int32_t[:] ssl
        cnp.int64_t ssl_sum
        cnp.int32_t l
        cnp.int8_t a1, a2
        cnp.float64_t[:] vnsl
        cnp.float64_t nsl

    # initialise
    n_variants = h.shape[0]
    n_haplotypes = h.shape[1]
    n_pairs = (n_haplotypes * (n_haplotypes - 1)) // 2

    # shared suffix lengths between all pairs of haplotypes
    ssl = np.zeros(n_pairs, dtype='i4')

    # NSL values for each variant
    vnsl = np.empty(n_variants, dtype='f8')

    with nogil:

        # iterate forward over variants
        for i in range(n_variants):
            u = 0  # pair index
            ssl_sum = 0

            # pairwise comparison of alleles between haplotypes to determine
            # shared suffix lengths
            for j in range(n_haplotypes):
                a1 = h[i, j]  # allele on first haplotype in pair
                for k in range(j+1, n_haplotypes):
                    a2 = h[i, k]  # allele on second haplotype in pair
                    # test for non-equal and non-missing alleles
                    if (a1 != a2) and (a1 >= 0) and (a2 >= 0):
                        # break shared suffix, reset length to zero
                        l = 0
                    else:
                        # extend shared suffix
                        l = ssl[u] + 1
                    ssl[u] = l
                    ssl_sum += l
                    # increment pair index
                    u += 1

            # compute nsl from shared suffix lengths
            nsl = ssl_sum / u
            vnsl[i] = nsl

    return np.asarray(vnsl)


@cython.boundscheck(False)
@cython.wraparound(False)
def ssl01_scan_int8(cnp.int8_t[:, :] h, stat, **kwargs):
    """Scan forwards over haplotypes, computing a summary statistic derived
    from the pairwise shared suffix lengths for each variant, for the
    reference (0) and alternate (1) alleles separately."""

    cdef:
        Py_ssize_t n_variants, n_haplotypes, n_pairs, i, j, k, u, u00, u11
        cnp.int32_t l
        cnp.int32_t[:] ssl, ssl00, ssl11
        cnp.int8_t a1, a2
        cnp.float64_t[:] vstat0, vstat1

    # initialise
    n_variants = h.shape[0]
    n_haplotypes = h.shape[1]

    # shared suffix lengths between all pairs of haplotypes
    n_pairs = (n_haplotypes * (n_haplotypes - 1)) // 2
    ssl = np.zeros(n_pairs, dtype='i4')
    ssl00 = np.zeros(n_pairs, dtype='i4')
    ssl11 = np.zeros(n_pairs, dtype='i4')

    # statistic values for each variant
    vstat0 = np.empty(n_variants, dtype='f8')
    vstat1 = np.empty(n_variants, dtype='f8')

    # iterate forward over variants
    for i in range(n_variants):

        # pairwise comparison of alleles between haplotypes to determine
        # shared suffix lengths
        with nogil:
            u = u00 = u11 = 0
            for j in range(n_haplotypes):
                a1 = h[i, j]
                for k in range(j+1, n_haplotypes):
                    a2 = h[i, k]
                    if a1 < 0 or a2 < 0:
                        # missing allele, assume sharing continues
                        l = ssl[u] + 1
                        ssl[u] = l
                    elif a1 == a2 == 0:
                        l = ssl[u] + 1
                        ssl[u] = l
                        ssl00[u00] = l
                        u00 += 1
                    elif a1 == a2 == 1:
                        l = ssl[u] + 1
                        ssl[u] = l
                        ssl11[u11] = l
                        u11 += 1
                    else:
                        # break shared suffix, reset to zero
                        ssl[u] = 0
                    u += 1

        # compute statistic from shared suffix lengths
        stat00 = stat(np.asarray(ssl00[:u00]), i, **kwargs)
        stat11 = stat(np.asarray(ssl11[:u11]), i, **kwargs)
        vstat0[i] = stat00
        vstat1[i] = stat11

    return np.asarray(vstat0), np.asarray(vstat1)


@cython.boundscheck(False)
@cython.wraparound(False)
def ihh01_scan_int8(cnp.int8_t[:, :] h,
                    cnp.float64_t[:] gaps,
                    cnp.float64_t min_ehh=0,
                    cnp.float64_t min_maf=0,
                    bint include_edges=False):
    """Scan forwards over haplotypes, computing a summary statistic derived
    from the pairwise shared suffix lengths for each variant, for the
    reference (0) and alternate (1) alleles separately."""

    cdef:
        Py_ssize_t n_variants, n_haplotypes, n_pairs, i, j, k, u, u00, u11, \
            c0, c1
        cnp.int32_t l, l_max_00, l_max_11
        cnp.int32_t[:] ssl, ssl00, ssl11
        cnp.int8_t a1, a2
        cnp.float64_t[:] vstat0, vstat1
        cnp.float64_t maf

    # initialise
    n_variants = h.shape[0]
    n_haplotypes = h.shape[1]

    # shared suffix lengths between all pairs of haplotypes
    n_pairs = (n_haplotypes * (n_haplotypes - 1)) // 2
    ssl = np.zeros(n_pairs, dtype='i4')
    ssl00 = np.zeros(n_pairs, dtype='i4')
    ssl11 = np.zeros(n_pairs, dtype='i4')

    # statistic values for each variant
    vstat0 = np.empty(n_variants, dtype='f8')
    vstat1 = np.empty(n_variants, dtype='f8')

    with nogil:

        # iterate forward over variants
        for i in range(n_variants):
            u = u00 = u11 = c0 = c1 = 0
            l_max_00 = l_max_11 = 0

            # pairwise comparison of alleles between haplotypes to determine
            # shared suffix lengths
            for j in range(n_haplotypes):
                a1 = h[i, j]
                if a1 == 0:
                    c0 += 1
                elif a1 == 1:
                    c1 += 1
                for k in range(j+1, n_haplotypes):
                    a2 = h[i, k]
                    if a1 < 0 or a2 < 0:
                        # missing allele, assume sharing continues
                        l = ssl[u] + 1
                        ssl[u] = l
                    elif a1 == a2 == 0:
                        l = ssl[u] + 1
                        ssl[u] = l
                        ssl00[u00] = l
                        u00 += 1
                        if l > l_max_00:
                            l_max_00 = l
                    elif a1 == a2 == 1:
                        l = ssl[u] + 1
                        ssl[u] = l
                        ssl11[u11] = l
                        u11 += 1
                        if l > l_max_11:
                            l_max_11 = l
                    else:
                        # break shared suffix, reset to zero
                        ssl[u] = 0
                    u += 1

            # compute minor allele frequency
            if c0 < c1:
                maf = c0 / (c0 + c1)
            else:
                maf = c1 / (c0 + c1)

            if maf < min_maf:
                # minor allele frequency below cutoff, don't bother to compute
                vstat0[i] = nan64
                vstat1[i] = nan64

            else:
                # compute statistic from shared suffix lengths
                vstat0[i] = ssl2ihh(ssl00[:u00], l_max_00, i, gaps,
                                    min_ehh=min_ehh,
                                    include_edges=include_edges)
                vstat1[i] = ssl2ihh(ssl11[:u11], l_max_11, i, gaps,
                                    min_ehh=min_ehh,
                                    include_edges=include_edges)

    return np.asarray(vstat0), np.asarray(vstat1)


@cython.boundscheck(False)
@cython.wraparound(False)
def nsl01_scan_int8(cnp.int8_t[:, :] h):
    """Scan forwards over haplotypes, computing the number of segregating
    sites by length backwards for each variant for the reference (0) and
    alternate (1) alleles separately."""

    cdef:
        Py_ssize_t n_variants, n_haplotypes, n_pairs, i, j, k, u, u00, u11
        cnp.int32_t l
        cnp.int32_t[:] ssl
        cnp.int64_t ssl00_sum, ssl11_sum
        cnp.int8_t a1, a2
        cnp.float64_t[:] vstat0, vstat1

    # initialise
    n_variants = h.shape[0]
    n_haplotypes = h.shape[1]

    # shared suffix lengths between all pairs of haplotypes
    n_pairs = (n_haplotypes * (n_haplotypes - 1)) // 2
    ssl = np.zeros(n_pairs, dtype='i4')

    # statistic values for each variant
    vstat0 = np.empty(n_variants, dtype='f8')
    vstat1 = np.empty(n_variants, dtype='f8')

    with nogil:

        # iterate forward over variants
        for i in range(n_variants):
            u = u00 = u11 = 0
            ssl00_sum = ssl11_sum = 0

            # pairwise comparison of alleles between haplotypes to determine
            # shared suffix lengths
            for j in range(n_haplotypes):
                a1 = h[i, j]
                for k in range(j+1, n_haplotypes):
                    a2 = h[i, k]
                    if a1 < 0 or a2 < 0:
                        # missing allele, assume sharing continues
                        l = ssl[u] + 1
                    elif a1 == a2 == 0:
                        l = ssl[u] + 1
                        ssl00_sum += l
                        u00 += 1
                    elif a1 == a2 == 1:
                        l = ssl[u] + 1
                        ssl11_sum += l
                        u11 += 1
                    else:
                        # break shared suffix, reset to zero
                        l = 0
                    ssl[u] = l
                    u += 1

            if u00 > 0:
                vstat0[i] = ssl00_sum / u00
            else:
                vstat0[i] = nan64
            if u11 > 0:
                vstat1[i] = ssl11_sum / u11
            else:
                vstat1[i] = nan64

    return np.asarray(vstat0), np.asarray(vstat1)


@cython.boundscheck(False)
@cython.wraparound(False)
def phase_progeny_by_transmission_int8(cnp.int8_t[:, :, :] g):
    # N.B., here we will modify g in-place

    cdef:
        Py_ssize_t n_variants, n_samples, n_progeny, i, j, max_allele
        cnp.uint8_t[:, :] is_phased
        cnp.int8_t a1, a2, ma1, ma2, pa1, pa2
        cnp.uint8_t[:] mac, pac

    # guard conditions
    assert g.shape[2] == 2

    n_variants = g.shape[0]
    n_samples = g.shape[1]
    n_progeny = n_samples - 2
    max_allele = np.max(g)

    # setup intermediates
    mac = np.zeros(max_allele + 1, dtype='u1')  # maternal allele counts
    pac = np.zeros(max_allele + 1, dtype='u1')  # paternal allele counts

    # setup outputs
    is_phased = np.zeros((n_variants, n_samples), dtype='u1')

    # iterate over variants
    for i in range(n_variants):

        # access parental genotypes
        ma1 = g[i, 0, 0]  # maternal allele 1
        ma2 = g[i, 0, 1]  # maternal allele 2
        pa1 = g[i, 1, 0]  # paternal allele 1
        pa2 = g[i, 1, 1]  # paternal allele 2

        # check for any missing calls in parents
        if ma1 < 0 or ma2 < 0 or pa1 < 0 or pa2 < 0:
            continue

        # parental allele counts
        mac[:] = 0  # reset to zero
        pac[:] = 0  # reset to zero
        mac[ma1] = 1
        mac[ma2] = 1
        pac[pa1] = 1
        pac[pa2] = 1

        # iterate over progeny
        for j in range(2, n_progeny + 2):

            # access progeny alleles
            a1 = g[i, j, 0]
            a2 = g[i, j, 1]

            if a1 < 0 or a2 < 0:  # child is missing
                continue

            elif a1 == a2:  # child is homozygous

                if mac[a1] > 0 and pac[a1] > 0:  # Mendelian consistent
                    # trivially phase the child
                    is_phased[i, j] = 1

            else:  # child is heterozygous

                if mac[a1] > 0 and pac[a1] == 0 and pac[a2] > 0:
                    # allele 1 is unique to mother, no need to swap
                    is_phased[i, j] = 1

                elif mac[a2] > 0 and pac[a2] == 0 and pac[a1] > 0:
                    # allele 2 is unique to mother, swap child alleles
                    g[i, j, 0] = a2
                    g[i, j, 1] = a1
                    is_phased[i, j] = 1

                elif pac[a1] > 0 and mac[a1] == 0 and mac[a2] > 0:
                    # allele 1 is unique to father, swap child alleles
                    g[i, j, 0] = a2
                    g[i, j, 1] = a1
                    is_phased[i, j] = 1

                elif pac[a2] > 0 and mac[a2] == 0 and mac[a1] > 0:
                    # allele 2 is unique to father, no need to swap
                    is_phased[i, j] = 1

    return is_phased


@cython.boundscheck(False)
@cython.wraparound(False)
def phase_parents_by_transmission_int8(cnp.int8_t[:, :, :] g,
                                       cnp.uint8_t[:, :] is_phased,
                                       Py_ssize_t window_size):
    # N.B., here we will modify g and is_phased in-place

    cdef:
        Py_ssize_t i, parent, ii, n_variants, n_samples, keep, flip, n_inf
        cnp.int8_t a1, a2, max_allele, pa1, pa2, x, y
        cnp.uint32_t[:] block_start
        cnp.uint32_t[:] n_progeny_phased
        cnp.uint32_t[:, :] linkage

    # guard conditions
    assert g.shape[2] == 2
    assert g.shape[0] == is_phased.shape[0]
    assert g.shape[1] == is_phased.shape[1]

    # setup intermediates
    n_variants = g.shape[0]
    n_samples = g.shape[1]
    max_allele = np.max(g)
    linkage = np.zeros((max_allele + 1, max_allele + 1), dtype='u4')
    n_progeny_phased = np.sum(is_phased[:, 2:], axis=1).astype('u4')

    # iterate over variants
    for i in range(n_variants):

        if n_progeny_phased[i] == 0:
            # no progeny genotypes phased, cannot phase parent
            continue

        # iterate over parents
        for parent in range(2):

            if is_phased[i, parent]:
                # parent already phased somehow, not expected but skip anyway
                continue

            # access parent's alleles
            a1 = g[i, parent, 0]
            a2 = g[i, parent, 1]

            if a1 < 0 or a2 < 0:
                # missing call, skip
                continue

            elif a1 == a2:
                # parent is homozygous, trivially phase
                is_phased[i, parent] = 1

            elif n_progeny_phased[i] > 0:
                # parent is het and some progeny are phased, so should be
                # able to phase parent

                # setup accumulators for evidence on whether to swap alleles
                keep = flip = 0

                # keep track of how many informative variants are visited
                n_inf = 0

                # setup index for back-tracking
                ii = i - 1

                # look back and collect linkage evidence from previous variants
                while ii >= 0 and n_inf < window_size:

                    # access alleles for previous variant
                    pa1 = g[ii, parent, 0]
                    pa2 = g[ii, parent, 1]

                    if (is_phased[ii, parent] and
                            (pa1 != pa2) and
                            (n_progeny_phased[ii] > 0)):

                        # variant is phase informative, accumulate
                        n_inf += 1

                        # collect linkage information
                        linkage[:, :] = 0
                        for j in range(2, n_samples):
                            if is_phased[ii, j] and is_phased[i, j]:
                                x = g[ii, j, parent]
                                y = g[i, j, parent]
                                linkage[x, y] += 1

                        # accumulate evidence
                        keep += linkage[pa1, a1] + linkage[pa2, a2]
                        flip += linkage[pa1, a2] + linkage[pa2, a1]

                    ii -= 1

                # make a decision
                if n_inf == 0:
                    # no previous informative variants, start of data,
                    # phase arbitrarily
                    is_phased[i, parent] = 1

                elif keep > flip:
                    is_phased[i, parent] = 1

                elif flip > keep:
                    is_phased[i, parent] = 1
                    g[i, parent, 0] = a2
                    g[i, parent, 1] = a1
