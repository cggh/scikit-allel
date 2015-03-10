# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


from allel.util import asarray_ndim


def rogers_huff_r(gn):
    """TODO

    """

    # check inputs
    gn = asarray_ndim(gn, 2, dtype='i1')

    # compute correlation coefficients
    from allel.opt.stats import gn_pairwise_corrcoef_int8
    r = gn_pairwise_corrcoef_int8(gn)

    # convenience for singletons
    if r.size == 1:
        r = r[0]

    return r
