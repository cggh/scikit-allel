# -*- coding: utf-8 -*-
# cython: profile=False
from __future__ import absolute_import, print_function, division


import numpy as np
cimport numpy as cnp
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
def genotype_pack_diploid(cnp.int8_t[:, :, :] g):

    n_variants = g.shape[0]
    n_samples = g.shape[1]

    # counting variables
    cdef Py_ssize_t i, j

    # first and second alleles from genotype
    cdef cnp.int8_t a1, a2

    # packed genotype
    cdef cnp.uint8_t p

    # create output array
    cdef cnp.uint8_t[:, :] packed = np.empty((n_variants, n_samples),
                                             dtype='u1')

    for i in range(n_variants):
        for j in range(n_samples):
            a1 = g[i, j, 0]
            a2 = g[i, j, 1]

            # add 1 to handle missing alleles coded as -1
            a1 += 1
            a2 += 1

            # left shift first allele by 4 bits
            a1 <<= 4

            # mask left-most 4 bits to ensure second allele doesn't clash with
            # first allele
            a2 &= 15

            # pack the alleles into a single byte
            p = a1 | a2

            # rotate round so that hom ref calls are encoded as 0, better for
            # sparse matrices
            p -= 17

            # assign to output array
            packed[i, j] = p

    return np.asarray(packed)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
def genotype_unpack_diploid(cnp.uint8_t[:, :] packed):

    n_variants = packed.shape[0]
    n_samples = packed.shape[1]

    # counting variables
    cdef Py_ssize_t i, j

    # first and second alleles from genotype
    cdef cnp.int8_t a1, a2

    # packed genotype
    cdef cnp.uint8_t p

    # create output array
    cdef cnp.int8_t[:, :, :] g = np.empty((n_variants, n_samples, 2),
                                          dtype='i1')

    for i in range(n_variants):
        for j in range(n_samples):
            p = packed[i, j]

            # rotate back round so missing calls are encoded as 0
            p += 17

            # right shift 4 bits to extract first allele
            a1 = p >> 4

            # mask left-most 4 bits to extract second allele
            a2 = p & 15

            # subtract 1 to restore coding of missing alleles as -1
            a1 -= 1
            a2 -= 1

            # assign to output array
            g[i, j, 0] = a1
            g[i, j, 1] = a2

    return np.asarray(g)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
def haplotype_int8_count_alleles(cnp.int8_t[:, :] h, max_allele):
    cdef cnp.int32_t[:, :] ac
    cdef cnp.int8_t allele
    cdef Py_ssize_t i, j

    # initialise output array
    ac = np.zeros((h.shape[0], max_allele + 1), dtype='i4')

    # iterate over variants
    for i in range(h.shape[0]):
        # iterate over haplotypes
        for j in range(h.shape[1]):
            allele = h[i, j]
            if allele >= 0:
                ac[i, allele] += 1

    return np.asarray(ac)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
def haplotype_int8_count_alleles_subpop(cnp.int8_t[:, :] h,
                                        cnp.int8_t max_allele,
                                        cnp.int64_t[:] subpop):
    cdef cnp.int32_t[:, :] ac
    cdef cnp.int8_t allele
    cdef Py_ssize_t i, j
    cdef cnp.int64_t idx

    # initialise output array
    ac = np.zeros((h.shape[0], max_allele + 1), dtype='i4')

    # iterate over variants
    for i in range(h.shape[0]):
        # iterate over haplotypes
        for j in range(subpop.shape[0]):
            idx = subpop[j]
            allele = h[i, idx]
            if allele >= 0:
                ac[i, allele] += 1

    return np.asarray(ac)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
def genotype_int8_count_alleles(cnp.int8_t[:, :, :] g, int max_allele):
    cdef cnp.int32_t[:, :] ac
    cdef cnp.int8_t allele
    cdef Py_ssize_t i, j, k

    # initialise output array
    ac = np.zeros((g.shape[0], max_allele + 1), dtype='i4')

    # iterate over variants
    for i in range(g.shape[0]):
        # iterate over samples
        for j in range(g.shape[1]):
            # iterate over alleles
            for k in range(g.shape[2]):
                allele = g[i, j, k]
                if allele >= 0:
                    ac[i, allele] += 1

    return np.asarray(ac)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
def genotype_int8_count_alleles_masked(cnp.int8_t[:, :, :] g,
                                       cnp.uint8_t[:, :] mask,
                                       int max_allele):
    cdef cnp.int32_t[:, :] ac
    cdef cnp.int8_t allele
    cdef Py_ssize_t i, j, k

    # initialise output array
    ac = np.zeros((g.shape[0], max_allele + 1), dtype='i4')

    # iterate over variants
    for i in range(g.shape[0]):
        # iterate over samples
        for j in range(g.shape[1]):
            # deal with mask
            if not mask[i, j]:
                # iterate over alleles
                for k in range(g.shape[2]):
                    allele = g[i, j, k]
                    if allele >= 0:
                        ac[i, allele] += 1

    return np.asarray(ac)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
def genotype_int8_count_alleles_subpop(cnp.int8_t[:, :, :] g,
                                       cnp.int8_t max_allele,
                                       cnp.int64_t[:] subpop):
    cdef cnp.int32_t[:, :] ac
    cdef cnp.int8_t allele
    cdef Py_ssize_t i, j, k
    cdef cnp.int64_t idx

    # initialise output array
    ac = np.zeros((g.shape[0], max_allele + 1), dtype='i4')

    # iterate over variants
    for i in range(g.shape[0]):
        # iterate over samples
        for j in range(subpop.shape[0]):
            idx = subpop[j]
            for k in range(g.shape[2]):
                allele = g[i, idx, k]
                if allele >= 0:
                    ac[i, allele] += 1

    return np.asarray(ac)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
def genotype_int8_count_alleles_subpop_masked(cnp.int8_t[:, :, :] g,
                                              cnp.uint8_t[:, :] mask,
                                              cnp.int8_t max_allele,
                                              cnp.int64_t[:] subpop):
    cdef cnp.int32_t[:, :] ac
    cdef cnp.int8_t allele
    cdef Py_ssize_t i, j, k
    cdef cnp.int64_t idx

    # initialise output array
    ac = np.zeros((g.shape[0], max_allele + 1), dtype='i4')

    # iterate over variants
    for i in range(g.shape[0]):
        # iterate over samples
        for j in range(subpop.shape[0]):
            idx = subpop[j]
            # deal with mask
            if not mask[i, idx]:
                for k in range(g.shape[2]):
                    allele = g[i, idx, k]
                    if allele >= 0:
                        ac[i, allele] += 1

    return np.asarray(ac)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
def haplotype_int8_map_alleles(cnp.int8_t[:, :] h,
                               cnp.int8_t[:, :] mapping,
                               copy=True):
    cdef Py_ssize_t i, j, m
    cdef cnp.int8_t allele
    cdef cnp.int8_t[:, :] ho
    if copy:
        ho = h.copy()
    else:
        ho = h
    m = mapping.shape[1]

    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            allele = h[i, j]
            if 0 <= allele < m:
                ho[i, j] = mapping[i, allele]
            else:
                ho[i, j] = -1

    return np.asarray(ho)
