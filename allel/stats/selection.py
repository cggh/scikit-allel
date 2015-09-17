# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np


from allel.model.ndarray import HaplotypeArray
from allel.opt.stats import pairwise_shared_prefix_lengths_int8, \
    paint_shared_prefixes_int8


def ehh_decay(h, truncate=False):
    """Compute the decay of extended haplotype homozygosity (EHH)
    moving away from the first variant.

    Parameters
    ----------
    h : array_like, int, shape (n_variants, n_haplotypes)
        Haplotype array.
    truncate : bool, optional
        If True, the return array will exclude trailing zeros.

    Returns
    -------
    ehh : ndarray, float, shape (n_variants, )
        EHH at successive variants from the first variant.

    """

    # check inputs
    # N.B., ensure int8 so we can use cython optimisation
    h = HaplotypeArray(np.asarray(h, dtype='i1'), copy=False)
    if h.max() > 1:
        raise NotImplementedError('only biallelic variants are supported')
    if h.min() < 0:
        raise NotImplementedError('missing calls are not supported')

    # initialise
    n_variants = h.n_variants  # number of rows, i.e., variants
    n_haplotypes = h.n_haplotypes  # number of columns, i.e., haplotypes
    n_pairs = (n_haplotypes * (n_haplotypes - 1)) // 2

    # compute the shared prefix length between all pairs of haplotypes
    spl = pairwise_shared_prefix_lengths_int8(h)

    # compute EHH by counting the number of shared prefixes extending beyond
    # each variant
    minlength = None if truncate else n_variants + 1
    b = np.bincount(spl, minlength=minlength)
    c = np.cumsum(b[::-1])[:-1]
    ehh = (c / n_pairs)[::-1]

    return ehh


def voight_painting(h):
    """Paint haplotypes, assigning a unique integer to each shared haplotype
    prefix.

    Parameters
    ----------
    h : array_like, int, shape (n_variants, n_haplotypes)
        Haplotype array.

    Returns
    -------
    painting : ndarray, int, shape (n_variants, n_haplotypes)
        Painting array.

    """

    # check inputs
    # N.B., ensure int8 so we can use cython optimisation
    h = HaplotypeArray(np.asarray(h, dtype='i1'), copy=False)
    if h.max() > 1:
        raise NotImplementedError('only biallelic variants are supported')
    if h.min() < 0:
        raise NotImplementedError('missing calls are not supported')

    return paint_shared_prefixes_int8(h)
