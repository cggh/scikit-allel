# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np


def _check_genotype_array_data(obj):
    obj = np.asarray(obj)

    # check dtype
    if obj.dtype.kind not in 'ui':
        raise TypeError('integer dtype required')

    # check dimensionality
    if obj.ndim != 3:
        raise TypeError('array with 3 dimensions required')

    return obj


class GenotypeArray(np.ndarray):
    """TODO

    """

    def __new__(cls, data):
        obj = _check_genotype_array_data(data)
        obj = obj.view(cls)
        return obj

    # noinspection PyMethodMayBeStatic
    def __array_finalize__(self, obj):

        # called after constructor
        if obj is None:
            return

        # called after slice (new-from-template)
        if isinstance(obj, GenotypeArray):
            return

        # called after view
        _check_genotype_array_data(obj)

    def __getslice__(self, *args, **kwargs):
        s = np.ndarray.__getslice__(self, *args, **kwargs)
        if s.ndim == 3:
            return s
        else:
            # slice with reduced dimensionality returns plain ndarray
            return s.view(np.ndarray)

    def __getitem__(self, *args, **kwargs):
        s = np.ndarray.__getitem__(self, *args, **kwargs)
        if s.ndim == 3:
            return s
        else:
            # slice with reduced dimensionality returns plain ndarray
            return s.view(np.ndarray)

    @property
    def n_variants(self):
        return self.shape[0]

    @property
    def n_samples(self):
        return self.shape[1]

    @property
    def ploidy(self):
        return self.shape[2]

    def __repr__(self):
        s = super(GenotypeArray, self).__repr__()
        return s[:-1] + ', n_variants=%s, n_samples=%s, ploidy=%s)' % \
                        self.shape
