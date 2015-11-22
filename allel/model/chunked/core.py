# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np


from allel.model.chunked.storage_bcolz import bcolzmem_storage, \
    bcolztmp_storage
from allel.compat import string_types


default_storage = bcolzmem_storage


storage_registry = {
    'bcolzmem': bcolzmem_storage,
    'bcolztmp': bcolztmp_storage,
    'hdf5mem': hdf5mem_storage,
    'hdf5tmp': hdf5tmp_storage,
}


def _get_storage(storage=None):
    if storage is None:
        return default_storage
    elif isinstance(storage, string_types):
        # normalise storage name
        storage = str(storage).lower()
        return storage_registry[storage]
    else:
        # assume custom instance
        return storage


def _get_blen(data, blen=None):
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
            # fall back to something simple, ~64k chunks
            row = np.asanyarray(data[0])
            return max(1, (2**16) // row.nbytes)

    else:
        return blen


def _check_equal_length(*sequences):
    s = sequences[0]
    for t in sequences[1:]:
        if len(t) != len(s):
            raise ValueError('lengths do not match')


def _get_column_names(data):
    if hasattr(data, 'names'):
        return data.names
    elif hasattr(data, 'keys'):
        return list(data.keys())
    elif hasattr(data, 'dtype') and hasattr(data.dtype, 'names') and \
            data.dtype.names:
        return data.dtype.names
    else:
        raise ValueError('could not get column names')


def _get_blen_table(data, blen=None):
    if blen is None:
        return min(_get_blen(data[n]) for n in _get_column_names(data))
    else:
        return blen


def store(data, arr, start=0, stop=None, offset=0, blen=None):
    """Copy `data` block-wise into `arr`."""

    # init
    blen = _get_blen(data, blen)

    # check arguments
    if stop is None:
        stop = len(data)
    else:
        stop = min(stop, len(data))
    length = stop - start
    if length < 0:
        raise ValueError('invalid stop/start')

    # copy block-wise
    for i in range(start, stop, blen):
        j = min(i+blen, stop)
        l = j-i
        arr[offset:offset+l] = data[i:j]
        offset += l


def copy(data, start=0, stop=None, blen=None, storage=None, **kwargs):
    """Copy `data` block-wise into a new array."""

    # init
    storage = _get_storage(storage)
    blen = _get_blen(data, blen)

    # check arguments
    if stop is None:
        stop = len(data)
    else:
        stop = min(stop, len(data))
    length = stop - start
    if length < 0:
        raise ValueError('invalid stop/start')

    # copy block-wise
    out = None
    for i in range(start, stop, blen):
        j = min(i+blen, stop)
        block = np.asanyarray(data[i:j])
        if out is None:
            out = storage.array(block, expectedlen=length, **kwargs)
        else:
            out.append(block)

    return out


def apply(data, f, blen=None, storage=None, create='array', **kwargs):

    # init
    storage = _get_storage(storage)
    if isinstance(data, tuple):
        blen = min(_get_blen(d, blen) for d in data)
    else:
        blen = _get_blen(data, blen)
    if isinstance(data, tuple):
        _check_equal_length(*data)
        length = len(data[0])
    else:
        length = len(data)

    # block-wise iteration
    out = None
    for i in range(0, length, blen):
        j = min(i+blen, length)

        # obtain blocks
        if isinstance(data, tuple):
            blocks = [np.asanyarray(d[i:j]) for d in data]
        else:
            blocks = np.asanyarray(data[i:j]),

        # map
        res = f(*blocks)

        # store
        if out is None:
            out = getattr(storage, create)(res, expectedlen=length, **kwargs)
        else:
            out.append(res)

    return out


def reduce(data, reducer, block_reducer, mapper=None, axis=None,
           blen=None, storage=None, **kwargs):

    # init
    storage = _get_storage(storage)
    blen = _get_blen(data, blen)
    length = len(data)
    # normalise axis arg
    if isinstance(axis, int):
        axis = (axis,)

    if axis is None or 0 in axis:
        # two-step reduction
        out = None
        for i in range(0, length, blen):
            j = min(i+blen, length)
            block = np.asanyarray(data[i:j])
            if mapper:
                block = mapper(block)
            res = reducer(block, axis=axis)
            if out is None:
                out = res
            else:
                out = block_reducer(out, res)
        if np.isscalar(out):
            return out
        elif len(out.shape) == 0:
            return out[()]
        else:
            return storage.array(out, **kwargs)

    else:
        # first dimension is preserved, no need to reduce blocks
        out = None
        for i in range(0, length, blen):
            j = min(i+blen, length)
            block = np.asanyarray(data[i:j])
            if mapper:
                block = mapper(block)
            r = reducer(block, axis=axis)
            if out is None:
                out = storage.array(r, expectedlen=length, **kwargs)
            else:
                out.append(r)
        return out


def amax(data, axis=None, mapper=None, blen=None, storage=None, **kwargs):
    return reduce(data, axis=axis, reducer=np.amax,
                  block_reducer=np.maximum, mapper=mapper,
                  blen=blen, storage=storage, **kwargs)


def amin(data, axis=None, mapper=None, blen=None, storage=None, **kwargs):
    return reduce(data, axis=axis, reducer=np.amin,
                  block_reducer=np.minimum, mapper=mapper,
                  blen=blen, storage=storage, **kwargs)


# noinspection PyShadowingBuiltins
def sum(data, axis=None, mapper=None, blen=None, storage=None, **kwargs):
    return reduce(data, axis=axis, reducer=np.sum,
                  block_reducer=np.add, mapper=mapper,
                  blen=blen, storage=storage, **kwargs)


def count_nonzero(data, mapper=None, blen=None, storage=None, **kwargs):
    return reduce(data, reducer=np.count_nonzero,
                  block_reducer=np.add, mapper=mapper,
                  blen=blen, storage=storage, **kwargs)


def compress(data, condition, axis=0, blen=None, storage=None, **kwargs):

    # init
    storage = _get_storage(storage)
    blen = _get_blen(data, blen)
    length = len(data)
    nnz = count_nonzero(condition)

    if axis == 0:
        _check_equal_length(data, condition)

        # block iteration
        out = None
        for i in range(0, length, blen):
            j = min(i+blen, length)
            bcond = np.asanyarray(condition[i:j])
            # don't access any data unless we have to
            if np.any(bcond):
                block = np.asanyarray(data[i:j])
                res = np.compress(bcond, block, axis=0)
                if out is None:
                    out = storage.array(res, expectedlen=nnz, **kwargs)
                else:
                    out.append(out, res)
        return out

    elif axis == 1:

        # block iteration
        out = None
        condition = np.asanyarray(condition)
        for i in range(0, length, blen):
            j = min(i+blen, length)
            block = np.asanyarray(data[i:j])
            res = np.compress(condition, block, axis=1)
            if out is None:
                out = storage.array(res, expectedlen=length, **kwargs)
            else:
                out.append(out, res)

        return out

    else:
        raise NotImplementedError('axis not supported: %s' % axis)


def take(data, indices, axis=0, blen=None, storage=None, **kwargs):

    # init
    length = len(data)

    if axis == 0:

        # check that indices are strictly increasing
        indices = np.asanyarray(indices)
        if np.any(indices[1:] <= indices[:-1]):
            raise NotImplementedError(
                'indices must be strictly increasing'
            )

        # implement via compress()
        condition = np.zeros((length,), dtype=bool)
        condition[indices] = True
        return compress(data, condition, axis=0, blen=blen, storage=storage,
                        **kwargs)

    elif axis == 1:

        # init
        storage = _get_storage(storage)
        blen = _get_blen(data, blen)

        # block iteration
        out = None
        for i in range(0, length, blen):
            j = min(i+blen, length)
            block = np.asanyarray(data[i:j])
            res = np.take(block, indices, axis=1)
            if out is None:
                out = storage.array(res, expectedlen=length, **kwargs)
            else:
                out.append(out, res)
        return out

    else:
        raise NotImplementedError('axis not supported: %s' % axis)


def compress_table(data, condition, blen=None, storage=None, **kwargs):

    # init
    storage = _get_storage(storage)
    blen = _get_blen_table(data, blen)
    names = _get_column_names(data)
    col0 = data[names[0]]
    length = len(col0)
    _check_equal_length(col0, condition)
    nnz = count_nonzero(condition)

    # block iteration
    out = None
    for i in range(0, length, blen):
        j = min(i+blen, length)
        bcond = np.asanyarray(condition[i:j])
        # don't access any data unless we have to
        if np.any(bcond):
            blocks = [np.asanyarray(data[n][i:j]) for n in names]
            res = [np.compress(bcond, block, axis=0) for block in blocks]
            if out is None:
                out = storage.table(res, names=names, expectedlen=nnz,
                                    **kwargs)
            else:
                out.append(out, res)
    return out


def take_table(data, indices, blen=None, storage=None, **kwargs):

    # check inputs
    names = _get_column_names(data)
    col0 = data[names[0]]
    length = len(col0)

    # check that indices are strictly increasing
    indices = np.asanyarray(indices)
    if np.any(indices[1:] <= indices[:-1]):
        raise NotImplementedError(
            'indices must be strictly increasing'
        )

    # implement via compress()
    condition = np.zeros((length,), dtype=bool)
    condition[indices] = True
    return compress_table(data, condition, blen=blen, storage=storage,
                          **kwargs)
