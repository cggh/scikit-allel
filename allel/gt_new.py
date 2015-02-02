# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np


def _normalise_genotype_data(obj):
    obj = np.asarray(obj)

    # check dtype
    if obj.dtype.kind not in 'ui':
        raise TypeError('integer dtype required')

    # check dimensionality
    if obj.ndim not in {2, 3}:
        raise ValueError('array with 2 or 3 dimensions required')

    # determine ploidy
    if obj.ndim == 2:
        ploidy = 1
    elif obj.ndim == 3:
        ploidy = obj.shape[2]
        if ploidy == 1:
            # drop extra ploidy dimension
            obj = obj[:, :, 0]

    return obj, ploidy


class GenotypeArray(np.ndarray):

    def __new__(cls, data):
        obj, ploidy = _normalise_genotype_data(data)
        obj = obj.view(cls)
        obj.ploidy = ploidy
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        obj, ploidy = _normalise_genotype_data(obj)
        self.ploidy = ploidy
