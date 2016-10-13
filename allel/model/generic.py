# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


# third party imports
import numpy as np


# internal imports
from allel.util import contains_newaxis


def index_genotype_vector(g, item, cls):

    # apply indexing operation on underlying values
    out = g.values[item]

    # decide whether to wrap the result
    wrap = (
        hasattr(out, 'ndim') and out.ndim == 2 and  # dimensionality preserved
        out.shape[1] == g.shape[1] and  # ploidy preserved
        not contains_newaxis(item)
    )

    if wrap:
        out = cls(out)
        if g.mask is not None:
            out.mask = g.mask[item]
        if g.is_phased is not None:
            out.is_phased = g.is_phased[item]

    return out


def index_genotype_array(g, item, array_cls, vector_cls):

    # apply indexing operation to underlying values
    out = g.values[item]

    # decide whether to wrap the output, if so how
    wrap_array = (
        hasattr(out, 'ndim') and out.ndim == 3 and  # dimensionality preserved
        out.shape[2] == g.shape[2] and  # ploidy preserved
        not contains_newaxis(item)
    )
    wrap_vector = (
        # single row selection
        isinstance(item, int) or (
            # other way to make a single row selection
            isinstance(item, tuple) and len(item) == 2 and
            isinstance(item[0], int) and
            isinstance(item[1], (slice, list, np.ndarray, type(Ellipsis)))
        ) or (
            # single column selection
            isinstance(item, tuple) and len(item) == 2 and
            isinstance(item[0], (slice, list, np.ndarray)) and
            isinstance(item[1], int)
        )
    )

    if wrap_array:
        out = array_cls(out)
    if wrap_vector:
        out = vector_cls(out)
    if wrap_array or wrap_vector:
        if g.mask is not None:
            out.mask = g.mask[item]
        if g.is_phased is not None:
            out.is_phased = g.is_phased[item]

    return out


def index_genotype_ac_vector(g, item, cls):

    # apply indexing operation on underlying values
    out = g.values[item]

    # decide whether to wrap the result
    wrap = (
        hasattr(out, 'ndim') and out.ndim == 2 and  # dimensionality preserved
        out.shape[1] == g.shape[1] and  # alleles preserved
        not contains_newaxis(item)
    )

    if wrap:
        out = cls(out)

    return out


def index_genotype_ac_array(g, item, array_cls, vector_cls):

    # apply indexing operation to underlying values
    out = g.values[item]

    # decide whether to wrap the output, if so how
    wrap_array = (
        hasattr(out, 'ndim') and out.ndim == 3 and  # dimensionality preserved
        out.shape[2] == g.shape[2] and  # alleles preserved
        not contains_newaxis(item)
    )
    wrap_vector = (
        # single row selection
        isinstance(item, int) or (
            # other way to make a single row selection
            isinstance(item, tuple) and len(item) == 2 and
            isinstance(item[0], int) and
            isinstance(item[1], (slice, list, np.ndarray, type(Ellipsis)))
        ) or (
            # single column selection
            isinstance(item, tuple) and len(item) == 2 and
            isinstance(item[0], (slice, list, np.ndarray)) and
            isinstance(item[1], int)
        )
    )

    if wrap_array:
        out = array_cls(out)
    if wrap_vector:
        out = vector_cls(out)

    return out


def index_haplotype_array(h, item, cls):

    # apply indexing operation on underlying values
    out = h.values[item]

    # decide whether to wrap the result as HaplotypeArray
    wrap_array = (
        hasattr(out, 'ndim') and out.ndim == 2 and  # dimensionality preserved
        not contains_newaxis(item)
    )

    if wrap_array:
        out = cls(out)

    return out


def index_allele_counts_array(ac, item, cls):

    # apply indexing operation on underlying values
    out = ac.values[item]

    # decide whether to wrap the result as HaplotypeArray
    wrap_array = (
        hasattr(out, 'ndim') and out.ndim == 2 and  # dimensionality preserved
        ac.shape[1] == out.shape[1] and  # number of alleles preserved
        not contains_newaxis(item)
    )

    if wrap_array:
        out = cls(out)

    return out


def compress_genotypes(g, condition, axis, wrap_axes, cls, compress, **kwargs):

    # apply compress operation on the underlying values
    out = compress(condition, g.values, axis=axis, **kwargs)

    if axis in wrap_axes:
        out = cls(out)
        if g.mask is not None:
            out.mask = compress(condition, g.mask, axis=axis, **kwargs)
        if g.is_phased is not None:
            out.is_phased = compress(condition, g.is_phased, axis=axis, **kwargs)

    return out


def take_genotypes(g, indices, axis, wrap_axes, cls, take, **kwargs):

    # apply compress operation on the underlying values
    out = take(g.values, indices, axis=axis, **kwargs)

    if axis in wrap_axes:
        out = cls(out)
        if g.mask is not None:
            out.mask = take(g.mask, indices, axis=axis, **kwargs)
        if g.is_phased is not None:
            out.is_phased = take(g.is_phased, indices, axis=axis, **kwargs)

    return out


def concatenate_genotypes(g, others, axis, wrap_axes, cls, concatenate, **kwargs):
    if not isinstance(others, (tuple, list)):
        others = others,

    # apply the concatenate operation on the underlying values
    tup = (g.values,) + tuple(o.values for o in others)
    out = concatenate(tup, axis=axis, **kwargs)

    if axis in wrap_axes:
        out = cls(out)
        if g.mask is not None:
            tup = (g.mask,) + tuple(o.mask for o in others)
            out.mask = concatenate(tup, axis=axis, **kwargs)
        if g.is_phased is not None:
            tup = (g.is_phased,) + tuple(o.is_phased for o in others)
            out.is_phased = concatenate(tup, axis=axis, **kwargs)

    return out


def subset_genotype_array(g, sel0, sel1, cls, subset, **kwargs):

    # apply the subset operation
    out = subset(g.values, sel0, sel1, **kwargs)

    # wrap the output
    out = cls(out)
    if g.mask is not None:
        out.mask = subset(g.mask, sel0, sel1, **kwargs)
    if g.is_phased is not None:
        out.is_phased = subset(g.is_phased, sel0, sel1, **kwargs)

    return out


def compress_haplotype_array(h, condition, axis, cls, compress, **kwargs):
    out = compress(condition, h.values, axis=axis, **kwargs)
    return cls(out)


def take_haplotype_array(h, indices, axis, cls, take, **kwargs):
    out = take(h.values, indices, axis=axis, **kwargs)
    return cls(out)


def subset_haplotype_array(h, sel0, sel1, cls, subset, **kwargs):
    out = subset(h.values, sel0, sel1, **kwargs)
    return cls(out)


def concatenate_haplotype_array(h, others, axis, cls, concatenate, **kwargs):
    if not isinstance(others, (tuple, list)):
        others = others,
    tup = (h.values,) + tuple(o.values for o in others)
    out = concatenate(tup, axis=axis, **kwargs)
    out = cls(out)
    return out


def compress_allele_counts_array(ac, condition, axis, cls, compress, **kwargs):
    out = compress(condition, ac.values, axis=axis, **kwargs)
    if axis == 0:
        out = cls(out)
    return out


def take_allele_counts_array(ac, indices, axis, cls, take, **kwargs):
    out = take(ac.values, indices, axis=axis, **kwargs)
    if axis == 0:
        out = cls(out)
    return out


def concatenate_allele_counts_array(ac, others, axis, cls, concatenate, **kwargs):
    if not isinstance(others, (tuple, list)):
        others = others,
    tup = (ac.values,) + tuple(o.values for o in others)
    out = concatenate(tup, axis=axis, **kwargs)
    if axis == 0:
        out = cls(out)
    return out


def compress_genotype_ac(g, condition, axis, wrap_axes, cls, compress, **kwargs):
    out = compress(condition, g.values, axis=axis, **kwargs)
    if axis in wrap_axes:
        out = cls(out)
    return out


def take_genotype_ac(g, indices, axis, wrap_axes, cls, take, **kwargs):
    out = take(g.values, indices, axis=axis, **kwargs)
    if axis in wrap_axes:
        out = cls(out)
    return out


def concatenate_genotype_ac(g, others, axis, wrap_axes, cls, concatenate, **kwargs):
    if not isinstance(others, (tuple, list)):
        others = others,
    tup = (g.values,) + tuple(o.values for o in others)
    out = concatenate(tup, axis=axis, **kwargs)
    if axis in wrap_axes:
        out = cls(out)
    return out


def subset_genotype_ac_array(g, sel0, sel1, cls, subset, **kwargs):
    out = subset(g.values, sel0, sel1, **kwargs)
    out = cls(out)
    return out
