# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np
import bcolz
import h5py
import tempfile


def h5dmem(*args, **kwargs):

    # need a file name even tho nothing is ever written
    fn = tempfile.mktemp()

    # default file creation args (allow user to override)
    backing_store = kwargs.pop('backing_store', False)
    block_size = kwargs.pop('block_size', 2**16)

    # open HDF5 file
    h5f = h5py.File(fn, mode='w', driver='core', backing_store=backing_store,
                    block_size=block_size)

    # defaults for dataset creation
    kwargs.setdefault('chunks', True)
    if kwargs['chunks']:
        kwargs.setdefault('compression', 'gzip')
        kwargs.setdefault('shuffle', False)
    if len(args) == 0 and 'name' not in kwargs:
        # default dataset name
        args = ('data',)

    # create dataset
    h5d = h5f.create_dataset(*args, **kwargs)

    return h5d


def h5dtmp(*args, **kwargs):

    # create temporary file name
    suffix = kwargs.pop('suffix', '')
    prefix = kwargs.pop('prefix', 'tmp')
    dir = kwargs.pop('dir', None)
    fn = tempfile.mktemp(suffix=suffix, prefix=prefix, dir=dir)

    # open HDF5 file
    h5f = h5py.File(fn, mode='w')

    # defaults for dataset creation
    kwargs.setdefault('chunks', True)
    if kwargs['chunks']:
        kwargs.setdefault('compression', 'gzip')
        kwargs.setdefault('shuffle', False)
    if len(args) == 0 and 'name' not in kwargs:
        # default dataset name
        args = ('data',)

    # create dataset
    h5d = h5f.create_dataset(*args, **kwargs)

    return h5d


def check_array_like(a):
    if not hasattr(a, 'shape') or not hasattr(a, 'dtype'):
        raise ValueError(
            'expected array-like with shape and dtype, found %r' % a
        )


class NumpyBackend(object):

    @classmethod
    def zeros(cls, shape, **kwargs):
        return np.zeros(shape, **kwargs)

    @classmethod
    def store(cls, data, **kwargs):
        return np.asarray(data, **kwargs)

    @classmethod
    def max(cls, charr, axis=None, **kwargs):

        # determine block size
        blen = kwargs.pop('blen', charr.chunklen)

        # initialise output variable
        out = None

        # initialise output defaults
        kwargs.setdefault('dtype', charr.dtype)

        # absolute maximum
        if axis is None:
            for i in range(0, charr.shape[0], blen):
                block = charr[i:i+blen]
                m = np.max(block)
                if out is None:
                    out = m
                else:
                    out = max(m, out)
            return out

        # maximum over first dimension
        elif axis == 0 or axis == (0, 2):
            for i in range(0, charr.shape[0], blen):
                block = charr[i:i+blen]
                m = np.max(block, axis=axis)
                if out is None:
                    out = m
                else:
                    out = np.where(m > out, m, out)
            return cls.store(out, **kwargs)

        # maximum over second dimension
        elif axis == 1 or axis == (1, 2):
            out = cls.zeros((charr.shape[0],), **kwargs)
            for i in range(0, charr.shape[0], blen):
                block = charr[i:i+blen]
                out[i:i+blen] = np.max(block, axis=axis)
            return out

        else:
            raise NotImplementedError('axis not supported: %s' % axis)


class BColzBackend(NumpyBackend):

    @classmethod
    def zeros(cls, shape, **kwargs):
        return bcolz.zeros(shape, **kwargs)

    @classmethod
    def store(cls, data, **kwargs):
        return bcolz.carray(data, **kwargs)


class H5dtmpBackend(NumpyBackend):

    @classmethod
    def zeros(cls, shape, **kwargs):
        return h5dtmp(shape=shape, fillvalue=0, **kwargs)

    @classmethod
    def store(cls, data, **kwargs):
        return h5dtmp(data=data, **kwargs)


class H5dmemBackend(NumpyBackend):

    @classmethod
    def zeros(cls, shape, **kwargs):
        return h5dmem(shape=shape, fillvalue=0, **kwargs)

    @classmethod
    def store(cls, data, **kwargs):
        return h5dmem(data=data, **kwargs)


default_backend = BColzBackend


def get_backend(out_flavour=None):
    if out_flavour is None:
        return default_backend
    else:
        # normalise out_flavour
        out_flavour = str(out_flavour).lower()
        if out_flavour in ['numpy', 'ndarray', 'np']:
            return NumpyBackend
        elif out_flavour in ['bcolz', 'carray']:
            return BColzBackend
        elif out_flavour in ['hdf5', 'h5py', 'h5dtmp']:
            return H5dtmpBackend
        elif out_flavour in ['h5dmem']:
            return H5dmemBackend
        else:
            raise ValueError('unknown flavour: %s' % out_flavour)


class ChunkedArrayWrapper(object):

    def __init__(self, data, chunklen=None):
        check_array_like(data)
        if chunklen is None:
            if hasattr(data, 'chunklen'):
                # bcolz carray
                chunklen = data.chunklen
            elif hasattr(data, 'chunks'):
                # h5py dataset
                chunklen = data.chunks[0]
            else:
                raise ValueError('could not determine chunk length')
        self.data = data
        self.chunklen = chunklen

    def __getitem__(self, *args):
        return self.data.__getitem__(*args)

    def __setitem__(self, key, value):
        return self.data.__setitem__(key, value)

    def __getattr__(self, item):
        return getattr(self.data, item)

    def __array__(self):
        return self.data[:]

    def __repr__(self):
        return '%s(%r)' % (type(self), self.data)

    def __len__(self):
        return len(self.carr)

    def max(self, axis=None, out_flavour=None, **kwargs):
        backend = get_backend(out_flavour)
        return backend.max(self, axis=axis, **kwargs)
