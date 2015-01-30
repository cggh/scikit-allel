# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np
cimport numpy as cnp


def pack_diploid(cnp.int8_t[:, :, :] g):

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


def unpack_diploid(cnp.uint8_t[:, :] packed):

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
