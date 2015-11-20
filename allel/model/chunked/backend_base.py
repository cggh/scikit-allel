# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import operator
import sys


import numpy as np


from allel.model.ndarray import subset


def check_equal_length(*sequences):
    s = sequences[0]
    for t in sequences[1:]:
        if len(t) != len(s):
            raise ValueError('lengths do not match')


def get_chunklen(data):
    """Try to guess a reasonable chunk length to use for block-wise iteration
    over `data`."""

    if hasattr(data, 'chunklen'):
        # bcolz carray
        return data.chunklen

    elif hasattr(data, 'chunks') and hasattr(data, 'shape') and \
            len(data.chunks) == len(data.shape):
        # h5py dataset
        return data.chunks[0]

    else:
        # fall back to something simple, ~64k chunks
        row = np.asarray(data[0])
        return max(1, (2**16) // row.nbytes)


def get_column_names(data):
    if hasattr(data, 'names'):
        return data.names
    elif hasattr(data, 'keys'):
        return list(data.keys())
    else:
        raise ValueError('could not get column names')


def get_chunklen_table(data):
    return min(get_chunklen(data[n]) for n in get_column_names(data))


class Backend(object):

    def create(self, data, expectedlen=None, **kwargs):
        pass

    def append(self, charr, data):
        pass

    def create_table(self, data, expectedlen=None, **kwargs):
        pass

    def append_table(self, chtbl, data):
        pass

    def copy(self, data, start=0, stop=None, blen=None, **kwargs):
        """Copy `data` block-wise into a new array."""

        # check arguments
        if stop is None:
            stop = len(data)
        else:
            stop = min(stop, len(data))
        length = stop - start
        if length < 0:
            raise ValueError('invalid stop/start')

        # block size for iteration
        if blen is None:
            blen = get_chunklen(data)

        # copy block-wise
        out = None
        for i in range(start, stop, blen):
            j = min(i+blen, stop)
            block = np.asarray(data[i:j])
            if out is None:
                out = self.create(block, expectedlen=length, **kwargs)
            else:
                out = self.append(out, block)

        return out

    def store(self, data, sink, start=0, stop=None, offset=0, blen=None):
        """Copy `data` block-wise into `sink`."""

        # check arguments
        if stop is None:
            stop = len(data)
        else:
            stop = min(stop, len(data))
        length = stop - start
        if length < 0:
            raise ValueError('invalid stop/start')

        # block size for iteration
        if blen is None:
            blen = get_chunklen(sink)

        # copy block-wise
        for i in range(start, stop, blen):
            j = min(i+blen, stop)
            l = j-i
            sink[offset:offset+l] = data[i:j]
            offset += l

    def reduce_axis(self, data, reducer, block_reducer, mapper=None,
                    axis=None, blen=None, **kwargs):

        # check arguments
        length = len(data)

        # block size for iteration
        if blen is None:
            blen = get_chunklen(data)

        # normalise axis arg
        if isinstance(axis, int):
            axis = (axis,)

        if axis is None or 0 in axis:
            # two-step reduction
            out = None
            for i in range(0, length, blen):
                j = min(i+blen, length)
                block = np.asarray(data[i:j])
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
                return self.create(out, **kwargs)

        else:
            # first dimension is preserved, no need to reduce blocks
            out = None
            for i in range(0, length, blen):
                j = min(i+blen, length)
                block = np.asarray(data[i:j])
                if mapper:
                    block = mapper(block)
                r = reducer(block, axis=axis)
                if out is None:
                    out = self.create(r, expectedlen=length, **kwargs)
                else:
                    out = self.append(out, r)
            return out

    def amax(self, data, axis=None, mapper=None, blen=None, **kwargs):
        return self.reduce_axis(data, axis=axis, reducer=np.amax,
                                block_reducer=np.maximum, mapper=mapper,
                                blen=blen, **kwargs)

    def amin(self, data, axis=None, mapper=None, blen=None, **kwargs):
        return self.reduce_axis(data, axis=axis, reducer=np.amin,
                                block_reducer=np.minimum, mapper=mapper,
                                blen=blen, **kwargs)

    def sum(self, data, axis=None, mapper=None, blen=None, **kwargs):
        return self.reduce_axis(data, axis=axis, reducer=np.sum,
                                block_reducer=np.add, mapper=mapper,
                                blen=blen, **kwargs)

    def count_nonzero(self, data, mapper=None, blen=None, **kwargs):
        return self.reduce_axis(data, reducer=np.count_nonzero,
                                block_reducer=np.add, mapper=mapper,
                                blen=blen, **kwargs)

    def map_blocks(self, data, mapper, blen=None, **kwargs):

        # check inputs
        if isinstance(data, tuple):
            check_equal_length(*data)
            length = len(data[0])
        else:
            length = len(data)

        # block size for iteration
        if blen is None:
            if isinstance(data, tuple):
                blen = min(get_chunklen(a) for a in data)
            else:
                blen = get_chunklen(data)

        # block-wise iteration
        out = None
        for i in range(0, length, blen):
            j = min(i+blen, length)

            # obtain blocks
            if isinstance(data, tuple):
                blocks = [np.asarray(a[i:j]) for a in data]
            else:
                blocks = np.asarray(data[i:j]),

            # map
            res = mapper(*blocks)

            # store
            if out is None:
                out = self.create(res, expectedlen=length, **kwargs)
            else:
                out = self.append(out, res)

        return out

    def map_blocks_into_table(self, data, mapper, blen=None, **kwargs):

        # check inputs
        if isinstance(data, tuple):
            check_equal_length(*data)
            length = len(data[0])
        else:
            length = len(data)

        # block size for iteration
        if blen is None:
            if isinstance(data, tuple):
                blen = min(get_chunklen(a) for a in data)
            else:
                blen = get_chunklen(data)

        # block-wise iteration
        out = None
        for i in range(0, length, blen):
            j = min(i+blen, length)

            # obtain blocks
            if isinstance(data, tuple):
                blocks = [np.asarray(a[i:j]) for a in data]
            else:
                blocks = np.asarray(data[i:j]),

            # map
            res = mapper(*blocks)

            # store
            if out is None:
                out = self.create_table(res, expectedlen=length, **kwargs)
            else:
                out = self.append_table(out, res)

        return out

    def compress(self, data, condition, axis=0, blen=None, **kwargs):

        # check inputs
        length = len(data)
        cond_nnz = self.count_nonzero(condition)

        # block size for iteration
        if blen is None:
            blen = get_chunklen(data)

        if axis == 0:
            check_equal_length(data, condition)

            # block iteration
            out = None
            for i in range(0, length, blen):
                j = min(i+blen, length)
                bcond = np.asarray(condition[i:j])
                # don't bother doing anything unless we have to
                n = np.count_nonzero(bcond)
                if n:
                    block = np.asarray(data[i:j])
                    res = np.compress(bcond, block, axis=0)
                    if out is None:
                        out = self.create(res, expectedlen=cond_nnz, **kwargs)
                    else:
                        out = self.append(out, res)
            return out

        elif axis == 1:

            # block iteration
            out = None
            condition = np.asarray(condition)
            for i in range(0, length, blen):
                j = min(i+blen, length)
                block = np.asarray(data[i:j])
                res = np.compress(condition, block, axis=1)
                if out is None:
                    out = self.create(res, expectedlen=length, **kwargs)
                else:
                    out = self.append(out, res)

            return out

        else:
            raise NotImplementedError('axis not supported: %s' % axis)

    def compress_table(self, data, condition, blen=None, **kwargs):

        # check inputs
        length = len(data)
        check_equal_length(data, condition)
        cond_nnz = self.count_nonzero(condition)

        # block size for iteration
        if blen is None:
            blen = get_chunklen_table(data)

        # block iteration
        out = None
        for i in range(0, length, blen):
            j = min(i+blen, length)
            bcond = np.asarray(condition[i:j])
            # don't bother doing anything unless we have to
            n = np.count_nonzero(bcond)
            if n:
                block = np.asarray(data[i:j])
                res = np.compress(bcond, block, axis=0)
                if out is None:
                    out = self.create_table(res, expectedlen=cond_nnz,
                                            **kwargs)
                else:
                    out = self.append_table(out, res)
        return out

    def take(self, data, indices, axis=0, blen=None, **kwargs):

        # check inputs
        length = len(data)
        indices = np.asarray(indices)

        # block size for iteration
        if blen is None:
            blen = get_chunklen(data)

        if axis == 0:

            # check that indices are strictly increasing
            if np.any(indices[1:] <= indices[:-1]):
                raise NotImplementedError(
                    'indices must be strictly increasing'
                )

            # implement via compress()
            condition = np.zeros((length,), dtype=bool)
            condition[indices] = True
            return self.compress(data, condition, axis=0, **kwargs)

        elif axis == 1:

            # block iteration
            out = None
            for i in range(0, length, blen):
                j = min(i+blen, length)
                block = np.asarray(data[i:j])
                res = np.take(block, indices, axis=1)
                if out is None:
                    out = self.create(res, expectedlen=length, **kwargs)
                else:
                    out = self.append(out, res)
            return out

        else:
            raise NotImplementedError('axis not supported: %s' % axis)

    def take_table(self, data, indices, blen=None, **kwargs):

        # check inputs
        length = len(data)
        indices = np.asarray(indices)

        # check that indices are strictly increasing
        if np.any(indices[1:] <= indices[:-1]):
            raise NotImplementedError(
                'indices must be strictly increasing'
            )

        # implement via compress()
        condition = np.zeros((length,), dtype=bool)
        condition[indices] = True
        return self.compress_table(data, condition, blen=blen, **kwargs)

    def subset(self, data, sel0, sel1, blen=None, **kwargs):

        # check inputs
        length = len(data)
        sel0 = np.asarray(sel0)
        sel1 = np.asarray(sel1)

        # ensure boolean array for dim 0
        if sel0.shape[0] < length:
            # assume indices, convert to boolean condition
            tmp = np.zeros(length, dtype=bool)
            tmp[sel0] = True
            sel0 = tmp

        # ensure indices for dim 1
        if sel1.shape[0] == data.shape[1]:
            # assume boolean condition, convert to indices
            sel1 = np.nonzero(sel1)[0]

        # block size for iteration
        if blen is None:
            blen = get_chunklen(data)

        # build output
        sel0_nnz = self.count_nonzero(sel0)
        out = None
        for i in range(0, length, blen):
            j = min(i+blen, length)
            bsel0 = sel0[i:j]
            # don't bother doing anything unless we have to
            n = np.count_nonzero(bsel0)
            if n:
                block = np.asarray(data[i:j])
                res = subset(block, bsel0, sel1)
                if out is None:
                    out = self.create(res, expectedlen=sel0_nnz, **kwargs)
                else:
                    out = self.append(out, res)

        return out
