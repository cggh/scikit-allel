# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np


HAPLOID = 1
DIPLOID = 2
DIM_VARIANTS = 0
DIM_SAMPLES = 1
DIM_PLOIDY = 2


class ArgumentError(Exception):
    pass


def check_genotype_array(g, allow_none=False):

    if g is None:
        if allow_none:
            return g
        else:
            raise ArgumentError('genotype array required')
        
    # ensure we have a numpy array
    g = np.asarray(g)

    # check dimensionality
    if g.ndim == 2:
        # assume haploid
        ploidy = HAPLOID
    elif g.ndim == 3:
        ploidy = g.shape[2]
        if ploidy == HAPLOID:
            # drop empty ploidy dimension
            g = g[:, :, 0]
    else:
        raise ArgumentError('expected 2 or 3 dimensions, found %s' % g.ndim)

    return g, ploidy


def check_haplotype_array(h, allow_none=False):

    if h is None:
        if allow_none:
            return h
        else:
            raise ArgumentError('haplotype array required')

    # ensure we have a numpy array
    h = np.asarray(h)

    # check dimensionality
    if h.ndim != 2:
        raise ArgumentError('expected 2 dimensions, found %s' % h.ndim)

    return h


def check_boolean_array(b, allow_none=False):

    if b is None:
        if allow_none:
            return b
        else:
            raise ArgumentError('boolean array required')

    # ensure we have a numpy boolean array
    b = np.asarray(b).view(dtype='b1')

    # check dimensionality
    if b.ndim not in {1, 2}:
        raise ArgumentError('expected 1 or 2 dimensions, found %s' % b.ndim)

    return b


def check_pos_array(pos, allow_none=False):

    if pos is None:
        if allow_none:
            return pos
        else:
            raise ArgumentError('position array required')

    # ensure we have a numpy array
    pos = np.asarray(pos)

    # check dimensionality
    if pos.ndim != 1:
        raise ArgumentError('expected one dimension, found %s' % pos.ndim)

    # check positions are sorted
    if np.any(np.diff(pos) < 0):
        raise ArgumentError('array is not sorted')

    return pos


def check_axis(axis, allow_none=False):
    if axis is None:
        if allow_none:
            return None
        else:
            raise ArgumentError('axis required')
    elif axis == 'variants':
        return DIM_VARIANTS
    elif axis == 'samples':
        return DIM_SAMPLES
    elif axis == 'ploidy':
        return DIM_PLOIDY
    elif axis in {0, 1, 2}:
        return axis
    elif isinstance(axis, (list, tuple)):
        return tuple(check_axis(a) for a in axis)
    else:
        raise ArgumentError('invalid axis: %r' % axis)


def check_allele(allele, allow_none=False):
    if allele is None:
        if allow_none:
            return None
        else:
            raise ArgumentError('allele required')
    elif allele in {'ref', 'reference'}:
        return 0
    elif allele in {'alt', 'alternate'}:
        return 1
    elif isinstance(allele, int):
        return allele
    else:
        raise ArgumentError('invalid allele: %r' % allele)


def check_alleles(alleles, allow_none=False):
    if alleles is None:
        if allow_none:
            return None
        else:
            raise ArgumentError('alleles required')
    elif isinstance(alleles, (tuple, list)):
        return tuple(check_allele(a) for a in alleles)
    else:
        raise ArgumentError('invalid alleles: %r' % alleles)


def check_ploidy(ploidy, allow_none=False):
    if ploidy is None:
        if allow_none:
            return None
        else:
            raise ArgumentError('ploidy required')
    elif ploidy == 'haploid':
        return HAPLOID
    elif ploidy == 'diploid':
        return DIPLOID
    elif isinstance(ploidy, int) and ploidy > 0:
        return ploidy
    else:
        raise ArgumentError('invalid ploidy: %r' % ploidy)
