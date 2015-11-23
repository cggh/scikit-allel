# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np


from allel.compat import string_types


storage_registry = dict()


def get_storage(storage=None):
    if storage is None:
        return storage_registry['default']
    elif isinstance(storage, string_types):
        # normalise storage name
        storage = str(storage).lower()
        return storage_registry[storage]
    else:
        # assume custom instance
        return storage


def check_equal_length(*sequences):
    s = sequences[0]
    for t in sequences[1:]:
        if len(t) != len(s):
            raise ValueError('lengths do not match')


def is_array_like(a):
    return hasattr(a, 'shape') and hasattr(a, 'dtype')


def ensure_array_like(a, **kwargs):
    ndim = kwargs.get('ndim', None)
    if not is_array_like(a):
        a = np.asarray(a)
    if ndim is not None and len(a.shape) != ndim:
        raise ValueError(
            'expected array-like with %s dimensions, found %s' %
            (ndim, len(a.shape))
        )
    return a


def check_table_like(data, names=None):

    if isinstance(data, (list, tuple)):
        # sequence of columns
        if names is None:
            names = ['f%d' % i for i in range(len(data))]
        else:
            if len(names) != len(data):
                raise ValueError('bad number of column names')
        columns = list(data)

    elif hasattr(data, 'names'):
        # bcolz ctable or similar
        if names is None:
            names = list(data.names)
        columns = [data[n] for n in names]

    elif hasattr(data, 'keys') and callable(data.keys):
        # dict, h5py Group or similar
        if names is None:
            names = list(data.keys())
        columns = [data[n] for n in names]

    elif hasattr(data, 'dtype') and hasattr(data.dtype, 'names'):
        # numpy recarray or similar
        if names is None:
            names = list(data.dtype.names)
        columns = [data[n] for n in names]

    else:
        raise ValueError('invalid data: %r' % data)

    columns = [ensure_array_like(c) for c in columns]
    check_equal_length(*columns)
    return names, columns


def get_blen_array(data, blen=None):
    """Try to guess a reasonable block length to use for block-wise iteration
    over `data`."""

    if blen is None:

        if hasattr(data, 'chunklen'):
            # bcolz carray
            return data.chunklen

        elif hasattr(data, 'chunks') and hasattr(data, 'shape') and \
                len(data.chunks) == len(data.shape):
            # h5py dataset
            return data.chunks[0]

        else:
            # fall back to something simple, ~1Mb chunks
            row = np.asanyarray(data[0])
            return max(1, (2**20) // row.nbytes)

    else:
        return blen


def get_blen_table(data, blen=None):
    if blen is None:
        _, columns = check_table_like(data)
        return min(get_blen_array(c) for c in columns)
    else:
        return blen
