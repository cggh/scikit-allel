# -*- coding: utf-8 -*-
# cython: profile=False
# cython: linetrace=False
# cython: binding=False
from __future__ import absolute_import, print_function, division


import numpy as np
cimport numpy as cnp
import cython
cimport cython


ctypedef fused integral_t:
    short
    int
    long
    cnp.int8_t
    cnp.int16_t
    cnp.int32_t
    cnp.int64_t
    cnp.uint8_t
    cnp.uint16_t
    cnp.uint32_t
    cnp.uint64_t


@cython.boundscheck(False)
@cython.wraparound(False)
def genotype_array_pack_diploid(integral_t[:, :, :] g not None):

    cdef:
        # counting variables
        Py_ssize_t i, j, n_variants, n_samples
        # first and second alleles from genotype
        integral_t a1, a2
        # packed genotype
        cnp.uint8_t p
        # create output array
        cnp.uint8_t[:, :] packed

    # setup
    n_variants = g.shape[0]
    n_samples = g.shape[1]
    packed = np.empty((n_variants, n_samples), dtype='u1')

    # main work loop
    with nogil:
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
def genotype_array_unpack_diploid(cnp.uint8_t[:, :] packed not None):

    cdef:
        # counting variables
        Py_ssize_t i, j, n_variants, n_samples
        # first and second alleles
        cnp.int8_t a1, a2
        # packed genotype
        cnp.uint8_t p
        # output
        cnp.int8_t[:, :, :] g

    # setup
    n_variants = packed.shape[0]
    n_samples = packed.shape[1]
    g = np.empty((n_variants, n_samples, 2), dtype='i1')

    # main work loop
    with nogil:
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
def haplotype_array_count_alleles(integral_t[:, :] h not None, integral_t max_allele):
    cdef cnp.int32_t[:, :] ac
    cdef integral_t allele
    cdef Py_ssize_t i, j, n_variants, n_haplotypes

    # setup
    n_variants = h.shape[0]
    n_haplotypes = h.shape[1]
    ac = np.zeros((n_variants, max_allele + 1), dtype='i4')

    # main work loop
    with nogil:
        # iterate over variants
        for i in range(n_variants):
            # iterate over haplotypes
            for j in range(n_haplotypes):
                allele = h[i, j]
                if 0 <= allele <= max_allele:
                    ac[i, allele] += 1

    return np.asarray(ac)


@cython.boundscheck(False)
@cython.wraparound(False)
def haplotype_array_count_alleles_subpop(integral_t[:, :] h not None,
                                         integral_t max_allele,
                                         cnp.int64_t[:] subpop not None):
    cdef:
        cnp.int32_t[:, :] ac
        integral_t allele
        Py_ssize_t i, j, n_variants, n_haplotypes
        cnp.int64_t idx

    # setup
    n_variants = h.shape[0]
    n_haplotypes = subpop.shape[0]
    ac = np.zeros((n_variants, max_allele + 1), dtype='i4')

    # main work loop
    with nogil:
        # iterate over variants
        for i in range(n_variants):
            # iterate over haplotypes
            for j in range(n_haplotypes):
                idx = subpop[j]
                allele = h[i, idx]
                if 0 <= allele <= max_allele:
                    ac[i, allele] += 1

    return np.asarray(ac)


@cython.boundscheck(False)
@cython.wraparound(False)
def genotype_array_count_alleles(integral_t[:, :, :] g not None,
                                 integral_t max_allele):
    cdef:
        cnp.int32_t[:, :] ac
        integral_t allele
        Py_ssize_t i, j, k, n_variants, n_samples, ploidy

    # setup
    n_variants = g.shape[0]
    n_samples = g.shape[1]
    ploidy = g.shape[2]
    ac = np.zeros((n_variants, max_allele + 1), dtype='i4')

    # main work loop
    with nogil:
        # iterate over variants
        for i in range(n_variants):
            # iterate over samples
            for j in range(n_samples):
                # iterate over alleles
                for k in range(ploidy):
                    allele = g[i, j, k]
                    if 0 <= allele <= max_allele:
                        ac[i, allele] += 1

    return np.asarray(ac)


@cython.boundscheck(False)
@cython.wraparound(False)
def genotype_array_count_alleles_masked(integral_t[:, :, :] g not None,
                                        cnp.uint8_t[:, :] mask not None,
                                        integral_t max_allele):
    cdef:
        cnp.int32_t[:, :] ac
        integral_t allele
        Py_ssize_t i, j, k, n_variants, n_samples, ploidy

    # setup
    n_variants = g.shape[0]
    n_samples = g.shape[1]
    ploidy = g.shape[2]
    ac = np.zeros((g.shape[0], max_allele + 1), dtype='i4')

    # main work loop
    with nogil:
        # iterate over variants
        for i in range(n_variants):
            # iterate over samples
            for j in range(n_samples):
                # deal with mask
                if not mask[i, j]:
                    # iterate over alleles
                    for k in range(ploidy):
                        allele = g[i, j, k]
                        if 0 <= allele <= max_allele:
                            ac[i, allele] += 1

    return np.asarray(ac)


@cython.boundscheck(False)
@cython.wraparound(False)
def genotype_array_count_alleles_subpop(integral_t[:, :, :] g not None,
                                        integral_t max_allele,
                                        cnp.int64_t[:] subpop not None):
    cdef:
        cnp.int32_t[:, :] ac
        integral_t allele
        Py_ssize_t i, j, k, n_variants, n_samples, ploidy
        cnp.int64_t idx

    # setup
    n_variants = g.shape[0]
    n_samples = subpop.shape[0]
    ploidy = g.shape[2]
    ac = np.zeros((g.shape[0], max_allele + 1), dtype='i4')

    # main work loop
    with nogil:
        # iterate over variants
        for i in range(n_variants):
            # iterate over samples
            for j in range(n_samples):
                idx = subpop[j]
                for k in range(ploidy):
                    allele = g[i, idx, k]
                    if 0 <= allele <= max_allele:
                        ac[i, allele] += 1

    return np.asarray(ac)


@cython.boundscheck(False)
@cython.wraparound(False)
def genotype_array_count_alleles_subpop_masked(integral_t[:, :, :] g not None,
                                               cnp.uint8_t[:, :] mask not None,
                                               integral_t max_allele,
                                               cnp.int64_t[:] subpop not None):
    cdef:
        cnp.int32_t[:, :] ac
        integral_t allele
        Py_ssize_t i, j, k, n_variants, n_samples, ploidy
        cnp.int64_t idx

    # setup
    n_variants = g.shape[0]
    n_samples = subpop.shape[0]
    ploidy = g.shape[2]
    ac = np.zeros((n_variants, max_allele + 1), dtype='i4')

    # main work loop
    with nogil:
        # iterate over variants
        for i in range(n_variants):
            # iterate over samples
            for j in range(n_samples):
                idx = subpop[j]
                # deal with mask
                if not mask[i, idx]:
                    for k in range(ploidy):
                        allele = g[i, idx, k]
                        if 0 <= allele <= max_allele:
                            ac[i, allele] += 1

    return np.asarray(ac)


@cython.boundscheck(False)
@cython.wraparound(False)
def haplotype_array_map_alleles(integral_t[:, :] h not None,
                                integral_t[:, :] mapping not None,
                                copy=True):
    cdef:
        Py_ssize_t i, j, m, n_variants, n_haplotypes
        integral_t allele
        integral_t[:, :] ho

    # setup
    n_variants = h.shape[0]
    n_haplotypes = h.shape[1]
    if copy:
        ho = h.copy()
    else:
        ho = h
    m = mapping.shape[1]

    # main work loop
    with nogil:
        for i in range(n_variants):
            for j in range(n_haplotypes):
                allele = h[i, j]
                if 0 <= allele < m:
                    ho[i, j] = mapping[i, allele]
                else:
                    ho[i, j] = -1

    return np.asarray(ho)
