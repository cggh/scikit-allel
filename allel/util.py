# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


from contextlib import contextmanager


import numpy as np


@contextmanager
def ignore_invalid():
    err = np.seterr(invalid='ignore')
    try:
        yield
    finally:
        np.seterr(**err)


def asarray_ndim(a, *ndims, **kwargs):
    allow_none = kwargs.get('allow_none', False)
    if a is None and allow_none:
        return None
    a = np.asarray(a)
    if a.ndim not in ndims:
        raise ValueError('invalid number of dimensions: %s' % a.ndim)
    return a


def check_arrays_aligned(a, b):
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            'arrays do not have matching length for first dimension: %s, %s'
            % (a.shape[0], b.shape[0])
        )
