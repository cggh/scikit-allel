# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


from contextlib import contextmanager


import numpy as np
from scipy.spatial.distance import squareform


@contextmanager
def ignore_invalid():
    err = np.seterr(invalid='ignore')
    try:
        yield
    finally:
        np.seterr(**err)


def check_array_like(a, *ndims, **kwargs):
    if not hasattr(a, 'ndim'):
        cls = kwargs.pop('default', np.asarray)
        a = cls(a, **kwargs)
    if a.ndim not in ndims:
        raise ValueError('invalid number of dimensions: %s' % a.ndim)


def asarray_ndim(a, *ndims, **kwargs):
    allow_none = kwargs.pop('allow_none', False)
    kwargs.setdefault('copy', False)
    if a is None and allow_none:
        return None
    a = np.array(a, **kwargs)
    if a.ndim not in ndims:
        raise ValueError('invalid number of dimensions: %s' % a.ndim)
    return a


def check_dim0_aligned(a, *others):
    for b in others:
        if b.shape[0] != a.shape[0]:
            raise ValueError(
                'arrays do not have matching length for first dimension'
            )


def check_equal_length(a, *others):
    l = len(a)
    for b in others:
        if len(b) != l:
            raise ValueError('sequences do not have matching length')


def resize_dim1(a, l, fill=0):
    if a.shape[1] < l:
        newshape = a.shape[0], l
        b = np.zeros(newshape, dtype=a.dtype)
        if fill != 0:
            b.fill(fill)
        b[:, :a.shape[1]] = a
        return b
    else:
        return a


def ensure_dim1_aligned(*arrays, **kwargs):
    fill = kwargs.get('fill', 0)
    l = max(a.shape[1] for a in arrays)
    arrays = [resize_dim1(a, l, fill=fill) for a in arrays]
    return arrays


def ensure_square(dist):
    dist = asarray_ndim(dist, 1, 2)
    if dist.ndim == 1:
        dist = squareform(dist)
    else:
        if dist.shape[0] != dist.shape[1]:
            raise ValueError('distance matrix is not square')
    return dist
