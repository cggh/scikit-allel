# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division

import numpy as np
import dask.array as da

from allel.compat import text_type


# from allel.util import asarray_ndim


def get_scaler(scaler, copy, ploidy):
    # normalise strings to lower case
    if isinstance(scaler, text_type):
        scaler = scaler.lower()
    if scaler == 'patterson':
        return PattersonScaler(copy=copy, ploidy=ploidy)
    elif scaler == 'standard':
        return StandardScaler(copy=copy)
    elif hasattr(scaler, 'fit'):
        return scaler
    elif scaler in ['center', 'centre'] or scaler is None:
        return CenterScaler(copy=copy)
    else:
        raise ValueError('unrecognised scaler: %s' % scaler)


class StandardScaler(object):

    def __init__(self, copy=True):
        self.copy = copy
        self.mean_ = None
        self.std_ = None

    def fit(self, gn):
        # find mean and scaling factor
        if type(gn) is da.Array:
            self.mean_ = da.mean(gn, axis=1, keepdims=True)
            self.std_ = da.std(gn, axis=1, keepdims=True)
        else:
            self.mean_ = np.mean(gn, axis=1, keepdims=True)
            self.std_ = np.std(gn, axis=1, keepdims=True)

        return self

    def transform(self, gn, copy=None):

        # check inputs
        if not gn.dtype.kind == 'f':
            gn = gn.astype('f2')

        # center
        gn -= self.mean_

        # scale
        gn /= self.std_

        return gn

    def fit_transform(self, gn, copy=None):
        self.fit(gn)
        return self.transform(gn, copy=copy)


class CenterScaler(object):

    def __init__(self, copy=True):
        self.copy = copy
        self.mean_ = None
        self.std_ = None

    def fit(self, gn):
        # find mean and scaling factor
        if type(gn) is da.Array:
            self.mean_ = da.mean(gn, axis=1, keepdims=True)
        else:
            self.mean_ = np.mean(gn, axis=1, keepdims=True)

        return self

    def transform(self, gn, copy=None):
        # check inputs
        if not gn.dtype.kind == 'f':
            gn = gn.astype('f2')

        # center
        gn -= self.mean_

        return gn

    def fit_transform(self, gn, copy=None):
        self.fit(gn)
        return self.transform(gn, copy=copy)


class PattersonScaler(object):

    def __init__(self, copy=True, ploidy=2):
        self.copy = copy
        self.ploidy = ploidy
        self.mean_ = None
        self.std_ = None

    def fit(self, gn):
        # find mean
        if type(gn) is da.Array:
            self.mean_ = da.mean(gn, axis=1, keepdims=True)
        else:
            self.mean_ = np.mean(gn, axis=1, keepdims=True)

        # find scaling factor
        p = self.mean_ / self.ploidy
        if type(gn) is da.Array:
            self.std_ = da.sqrt(p * (1 - p))
        else:
            self.std_ = np.sqrt(p * (1 - p))

        return self

    def transform(self, gn, copy=None):
        # check inputs
        if not gn.dtype.kind == 'f':
            gn = gn.astype('f2')

        # center
        gn -= self.mean_

        # scale
        gn /= self.std_

        return gn

    def fit_transform(self, gn, copy=None):
        self.fit(gn)
        return self.transform(gn, copy=copy)
