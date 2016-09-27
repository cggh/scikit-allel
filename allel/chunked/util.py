# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import operator


import numpy as np


from allel.compat import string_types, reduce


storage_registry = dict()


def get_storage(storage=None):
    if storage is None:
        try:
            return storage_registry['default']
        except KeyError:
            raise RuntimeError('no default storage available; is either h5py '
                               'or bcolz installed?')

    elif isinstance(storage, string_types):
        # normalise storage name
        storage = str(storage).lower()
        try:
            return storage_registry[storage]
        except KeyError:
            raise RuntimeError('storage not recognised: %r' % storage)

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
            names = sorted(data.keys())
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

        elif hasattr(data, 'chunks') and \
                hasattr(data, 'shape') and \
                hasattr(data.chunks, '__len__') and \
                hasattr(data.shape, '__len__') and \
                len(data.chunks) == len(data.shape):
            # something like h5py dataset
            return data.chunks[0]

        else:
            # fall back to something simple, ~1Mb chunks
            row = np.asarray(data[0])
            return max(1, (2**20) // row.nbytes)

    else:
        return blen


def get_blen_table(data, blen=None):
    if blen is None:
        _, columns = check_table_like(data)
        return max(get_blen_array(c) for c in columns)
    else:
        return blen


def human_readable_size(size):
    if size < 2**10:
        return "%s" % size
    elif size < 2**20:
        return "%.1fK" % (size / float(2**10))
    elif size < 2**30:
        return "%.1fM" % (size / float(2**20))
    elif size < 2**40:
        return "%.1fG" % (size / float(2**30))
    else:
        return "%.1fT" % (size / float(2**40))


def get_nbytes(data):
    if hasattr(data, 'nbytes'):
        return data.nbytes
    elif is_array_like(data):
        return reduce(operator.mul, data.shape) * data.dtype.itemsize
    else:
        return None


# noinspection PyProtectedMember
def get_cbytes(data):
    if hasattr(data, 'cbytes'):
        return data.cbytes
    elif hasattr(data, 'nbytes_stored'):
        return data.nbytes_stored
    elif hasattr(data, '_id') and hasattr(data._id, 'get_storage_size'):
        return data._id.get_storage_size()
    else:
        return None


def get_compression(data):
    if hasattr(data, 'cparams'):
        return 'blosc'
    elif hasattr(data, 'compression'):
        return data.compression
    elif hasattr(data, 'compressor'):
        # zarr 2
        return data.compressor.codec_id
    else:
        return None


def get_compression_opts(data):
    if hasattr(data, 'cparams'):
        return data.cparams
    elif hasattr(data, 'compression_opts'):
        return data.compression_opts
    elif hasattr(data, 'compressor'):
        # zarr 2
        config = data.compressor.get_config()
        del config['id']
        return config
    else:
        return None


def get_shuffle(data):
    if hasattr(data, 'cparams'):
        return data.cparams.shuffle
    elif hasattr(data, 'shuffle'):
        return data.shuffle
    else:
        return None


def get_chunks(data):
    if hasattr(data, 'chunklen'):
        # bcolz carray
        return (data.chunklen,) + data.shape[1:]
    elif hasattr(data, 'chunks') and \
            hasattr(data, 'shape') and \
            hasattr(data.chunks, '__len__') and \
            hasattr(data.shape, '__len__') and \
            len(data.chunks) == len(data.shape):
        # something like h5py dataset
        return data.chunks
    else:
        return None
