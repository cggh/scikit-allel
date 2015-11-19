# -*- coding: utf-8 -*-
"""TODO

"""
from __future__ import absolute_import, print_function, division


import shutil
import atexit
import os
import numpy as np
import bcolz
import h5py
import tempfile
import operator
from collections import namedtuple
from allel.compat import reduce, string_types, copy_method_doc, integer_types
from allel.util import check_dim0_aligned, asarray_ndim, check_same_ndim, \
    check_dim_aligned

from allel.model.ndarray import GenotypeArray, subset, HaplotypeArray, \
    AlleleCountsArray, recarray_to_html_str, recarray_display


# noinspection PyShadowingBuiltins
def h5dmem(*args, **kwargs):
    """Create an in-memory HDF5 dataset, by default chunked and gzip
    compressed.

    All arguments are passed through to the h5py create_dataset() function.

    """

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
    if len(args) == 0 and 'name' not in kwargs:
        # default dataset name
        args = ('data',)

    # create dataset
    h5d = h5f.create_dataset(*args, **kwargs)

    return h5d


# noinspection PyShadowingBuiltins
def h5dtmp(*args, **kwargs):
    """Create an HDF5 dataset backed by a temporary file, by default chunked
    and gzip compressed.

    All arguments are passed through to the h5py create_dataset() function.

    """

    # create temporary file name
    suffix = kwargs.pop('suffix', '.h5')
    prefix = kwargs.pop('prefix', 'scikit_allel_')
    dir = kwargs.pop('dir', None)
    fn = tempfile.mktemp(suffix=suffix, prefix=prefix, dir=dir)
    atexit.register(os.remove, fn)

    # open HDF5 file
    h5f = h5py.File(fn, mode='w')

    # defaults for dataset creation
    kwargs.setdefault('chunks', True)
    if len(args) == 0 and 'name' not in kwargs:
        # default dataset name
        args = ('data',)

    # create dataset
    h5d = h5f.create_dataset(*args, **kwargs)

    return h5d


def is_array_like(a):
    return hasattr(a, 'shape') and hasattr(a, 'dtype')


def check_array_like(a, ndim=None):
    if isinstance(a, (tuple, list)):
        for x in a:
            check_array_like(x)
    else:
        if not is_array_like(a):
            raise ValueError(
                'expected array-like with shape and dtype, found %r' % a
            )
        if ndim is not None and len(a.shape) != ndim:
            raise ValueError(
                'expected array-like with %s dimensions, found %s' %
                (ndim, len(a.shape))
            )


def get_chunklen(a):
    check_array_like(a)
    if hasattr(a, 'chunklen'):
        # bcolz carray
        return a.chunklen
    elif hasattr(a, 'chunks') and len(a.chunks) == len(a.shape):
        # h5py dataset
        return a.chunks[0]
    else:
        # do something vaguely sensible - ~64k blocks
        rowsize = a.dtype.itemsize * reduce(operator.mul, a.shape)
        return max(1, (2**16) // rowsize)


class Backend(object):

    def empty(self, shape, **kwargs):
        pass

    def create(self, data, expectedlen=None, **kwargs):
        pass

    def append(self, charr, data):
        pass

    # noinspection PyMethodMayBeStatic
    def store(self, source, sink, start=0, stop=None, offset=0, blen=None):

        # check arguments
        check_array_like(source)
        check_array_like(sink)
        if stop is None:
            stop = source.shape[0]
        length = stop - start
        if length < 0:
            raise ValueError('invalid start/stop indices')
        if sink.shape[0] < (offset + length):
            raise ValueError('sink is too short')

        # determine block size
        if blen is None:
            blen = get_chunklen(sink)

        # copy block-wise
        for i in range(start, stop, blen):
            j = min(i+blen, stop)
            l = j-i
            sink[offset:offset+l] = source[i:j]
            offset += l

    def copy(self, charr, start=0, stop=None, blen=None, **kwargs):

        # check arguments
        check_array_like(charr)
        if stop is None:
            stop = charr.shape[0]
        length = stop - start

        # initialise block size for iteration
        if blen is None:
            blen = get_chunklen(charr)

        # copy block-wise
        out = None
        for i in range(start, stop, blen):
            j = min(i+blen, stop)
            block = charr[i:j]
            if out is None:
                out = self.create(block, expectedlen=length, **kwargs)
            else:
                out = self.append(out, block)

        return out

    def reduce_axis(self, charr, reducer, block_reducer, mapper=None,
                    axis=None, **kwargs):

        # check arguments
        check_array_like(charr)
        length = charr.shape[0]

        # determine block size for iteration
        blen = kwargs.pop('blen', get_chunklen(charr))

        # normalise axis argument
        if isinstance(axis, int):
            axis = (axis,)

        if axis is None or 0 in axis:
            out = None
            for i in range(0, length, blen):
                j = min(i+blen, length)
                block = charr[i:j]
                if mapper:
                    block = mapper(block)
                r = reducer(block, axis=axis)
                if i == 0:
                    out = r
                else:
                    out = block_reducer(out, r)
            if np.isscalar(out):
                return out
            elif len(out.shape) == 0:
                # slightly weird case where array is returned
                return out[()]
            else:
                return self.create(out, **kwargs)

        else:

            # initialise output
            out = None

            # block iteration
            for i in range(0, length, blen):
                j = min(i+blen, length)
                block = charr[i:j]
                if mapper:
                    block = mapper(block)
                r = reducer(block, axis=axis)
                if out is None:
                    out = self.create(r, expectedlen=length, **kwargs)
                else:
                    out = self.append(out, r)
                # no need for block_reducer

            return out

    def amax(self, charr, axis=None, mapper=None, **kwargs):
        return self.reduce_axis(charr, axis=axis, reducer=np.amax,
                                block_reducer=np.maximum, mapper=mapper,
                                **kwargs)

    def amin(self, charr, axis=None, mapper=None, **kwargs):
        return self.reduce_axis(charr, axis=axis, reducer=np.amin,
                                block_reducer=np.minimum, mapper=mapper,
                                **kwargs)

    def sum(self, charr, axis=None, mapper=None, **kwargs):
        return self.reduce_axis(charr, axis=axis, reducer=np.sum,
                                block_reducer=np.add, mapper=mapper, **kwargs)

    def count_nonzero(self, charr, mapper=None, **kwargs):
        return self.reduce_axis(charr, reducer=np.count_nonzero,
                                block_reducer=np.add, mapper=mapper, **kwargs)

    def map_blocks(self, domain, mapper, blen=None, **kwargs):
        """N.B., assumes mapper will preserve leading dimension."""

        # check inputs
        check_array_like(domain)
        if isinstance(domain, tuple):
            check_dim0_aligned(domain)
            length = domain[0].shape[0]
        else:
            length = domain.shape[0]

        # determine block size for iteration
        if blen is None:
            if isinstance(domain, tuple):
                blen = min(get_chunklen(a) for a in domain)
            else:
                blen = get_chunklen(domain)

        # block-wise iteration
        out = None
        for i in range(0, length, blen):
            j = min(i+blen, length)

            # slice domain
            if isinstance(domain, tuple):
                blocks = [a[i:j] for a in domain]
            else:
                blocks = domain[i:j],

            # map
            res = mapper(*blocks)

            # store
            if out is None:
                out = self.create(res, expectedlen=length, **kwargs)
            else:
                out = self.append(out, res)

        return out

    def dict_map_blocks(self, domain, mapper, blen=None, **kwargs):
        """N.B., assumes mapper will preserve leading dimension."""

        # check inputs
        check_array_like(domain)
        if isinstance(domain, tuple):
            check_dim0_aligned(domain)
            length = domain[0].shape[0]
        else:
            length = domain.shape[0]

        # determine block size for iteration
        if blen is None:
            if isinstance(domain, tuple):
                blen = min(get_chunklen(a) for a in domain)
            else:
                blen = get_chunklen(domain)

        # block-wise iteration
        out = None
        for i in range(0, length, blen):
            j = min(i+blen, length)

            # slice domain
            if isinstance(domain, tuple):
                blocks = [a[i:j] for a in domain]
            else:
                blocks = domain[i:j],

            # map
            res = mapper(*blocks)

            # create
            if out is None:
                out = dict()
                for k, v in res.items():
                    out[k] = self.create(v, expectedlen=length, **kwargs)
            else:
                for k, v in res.items():
                    out = self.append(out[k], v)

        return out

    def compress(self, charr, condition, axis=0, **kwargs):

        # check inputs
        check_array_like(charr)
        length = charr.shape[0]
        if not is_array_like(condition):
            condition = np.asarray(condition)
        check_array_like(condition, 1)
        cond_nnz = self.count_nonzero(condition)

        # determine block size for iteration
        blen = kwargs.pop('blen', get_chunklen(charr))

        if axis == 0:
            check_dim0_aligned(charr, condition)

            # block iteration
            out = None
            for i in range(0, length, blen):
                j = min(i+blen, length)
                bcond = condition[i:j]
                # don't bother doing anything unless we have to
                n = np.count_nonzero(bcond)
                if n:
                    block = charr[i:j]
                    res = np.compress(bcond, block, axis=0)
                    if out is None:
                        out = self.create(res, expectedlen=cond_nnz, **kwargs)
                    else:
                        out = self.append(out, res)
            return out

        elif axis == 1:
            if condition.shape[0] != charr.shape[1]:
                raise ValueError('length of condition must match length of '
                                 'second dimension; expected %s, found %s' %
                                 (charr.shape[1], condition.size))

            # block iteration
            out = None
            for i in range(0, length, blen):
                j = min(i+blen, length)
                block = charr[i:j]
                res = np.compress(condition, block, axis=1)
                if out is None:
                    out = self.create(res, expectedlen=length, **kwargs)
                else:
                    out = self.append(out, res)

            return out

        else:
            raise NotImplementedError('axis not supported: %s' % axis)

    # noinspection PyTypeChecker
    def take(self, charr, indices, axis=0, **kwargs):

        # check inputs
        check_array_like(charr)
        length = charr.shape[0]
        indices = asarray_ndim(indices, 1)

        # determine block size for iteration
        blen = kwargs.pop('blen', get_chunklen(charr))

        if axis == 0:

            # check that indices are strictly increasing
            if np.any(indices[1:] <= indices[:-1]):
                raise NotImplementedError(
                    'indices must be strictly increasing'
                )
            # implement via compress()
            condition = np.zeros((length,), dtype=bool)
            condition[indices] = True
            return self.compress(charr, condition, axis=0, **kwargs)

        elif axis == 1:

            # block iteration
            out = None
            for i in range(0, length, blen):
                j = min(i+blen, length)
                block = charr[i:j]
                res = np.take(block, indices, axis=1)
                if out is None:
                    out = self.create(res, expectedlen=length, **kwargs)
                else:
                    out = self.append(out, res)

            return out

        else:
            raise NotImplementedError('axis not supported: %s' % axis)

    # noinspection PyUnresolvedReferences
    def subset(self, charr, sel0, sel1, **kwargs):

        # check inputs
        check_array_like(charr)
        if len(charr.shape) < 2:
            raise ValueError('expected array-like with at least 2 dimensions')
        length = charr.shape[0]
        sel0 = asarray_ndim(sel0, 1)
        sel1 = asarray_ndim(sel1, 1)

        # determine block size for iteration
        blen = kwargs.pop('blen', get_chunklen(charr))

        # ensure boolean array for dim 0
        if sel0.shape[0] < length:
            # assume indices, convert to boolean condition
            tmp = np.zeros(length, dtype=bool)
            tmp[sel0] = True
            sel0 = tmp

        # ensure indices for dim 1
        if sel1.shape[0] == charr.shape[1]:
            # assume boolean condition, convert to indices
            sel1 = np.nonzero(sel1)[0]

        # build output
        sel0_nnz = self.count_nonzero(sel0)
        out = None
        for i in range(0, length, blen):
            j = min(i+blen, length)
            bsel0 = sel0[i:j]
            # don't bother doing anything unless we have to
            n = np.count_nonzero(bsel0)
            if n:
                block = charr[i:j]
                res = subset(block, bsel0, sel1)
                if out is None:
                    out = self.create(res, expectedlen=sel0_nnz, **kwargs)
                else:
                    out = self.append(out, res)

        return out

    def hstack(self, tup, **kwargs):

        # check inputs
        if not isinstance(tup, (tuple, list)):
            raise ValueError('expected tuple or list, found %r' % tup)
        if len(tup) < 2:
            raise ValueError('expected two or more arrays to stack')
        check_dim0_aligned(*tup)
        check_same_ndim(*tup)

        def mapper(*blocks):
            return np.hstack(blocks)

        return self.map_blocks(tup, mapper, **kwargs)

    def vstack(self, tup, blen=None, **kwargs):

        # check inputs
        if not isinstance(tup, (tuple, list)):
            raise ValueError('expected tuple or list, found %r' % tup)
        if len(tup) < 2:
            raise ValueError('expected two or more arrays to stack')
        check_same_ndim(*tup)
        for i in range(1, len(tup[0].shape)):
            check_dim_aligned(i, *tup)

        # set block size to use
        if blen is None:
            blen = min([get_chunklen(a) for a in tup])

        # build output
        expectedlen = sum(a.shape[0] for a in tup)
        out = None
        for a in tup:
            for i in range(0, a.shape[0], blen):
                j = min(i+blen, a.shape[0])
                block = a[i:j]
                if out is None:
                    out = self.create(block, expectedlen=expectedlen, **kwargs)
                else:
                    out = self.append(out, block)
        return out

    def op_scalar(self, charr, op, other, **kwargs):

        # check inputs
        check_array_like(charr)
        if not np.isscalar(other):
            raise ValueError('expected scalar')

        def mapper(block):
            return op(block, other)

        return self.map_blocks(charr, mapper, **kwargs)


class NumpyBackend(Backend):
    """Reference implementation, will not be efficient."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def empty(self, shape, **kwargs):
        return np.empty(shape, **kwargs)

    def create(self, data, expectedlen=None, **kwargs):
        return np.array(data, **kwargs)

    def append(self, arr, data):
        return np.append(arr, data, axis=0)


# singleton instance
numpy_backend = NumpyBackend()


class BColzBackend(Backend):

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def empty(self, shape, **kwargs):
        for k, v in self.kwargs.items():
            kwargs.setdefault(k, v)
        return bcolz.zeros(shape, **kwargs)

    def create(self, data, expectedlen=None, **kwargs):
        for k, v in self.kwargs.items():
            kwargs.setdefault(k, v)
        return bcolz.carray(data, expectedlen=expectedlen, **kwargs)

    def append(self, carr, data):
        carr.append(data)
        return carr


# singleton instance
bcolz_backend = BColzBackend()
bcolz_gzip1_backend = BColzBackend(cparams=bcolz.cparams(cname='zlib',
                                                         clevel=1))


class BColzTmpBackend(Backend):

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    # noinspection PyShadowingBuiltins
    def empty(self, shape, **kwargs):
        for k, v in self.kwargs.items():
            kwargs.setdefault(k, v)
        suffix = kwargs.pop('suffix', '.bcolz')
        prefix = kwargs.pop('prefix', 'scikit_allel_')
        dir = kwargs.pop('dir', None)
        rootdir = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=dir)
        atexit.register(shutil.rmtree, rootdir)
        kwargs['rootdir'] = rootdir
        kwargs['mode'] = 'w'
        return bcolz.zeros(shape, **kwargs)

    # noinspection PyShadowingBuiltins
    def create(self, data, expectedlen=None, **kwargs):
        for k, v in self.kwargs.items():
            kwargs.setdefault(k, v)
        suffix = kwargs.pop('suffix', '.bcolz')
        prefix = kwargs.pop('prefix', 'scikit_allel_')
        dir = kwargs.pop('dir', None)
        rootdir = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=dir)
        atexit.register(shutil.rmtree, rootdir)
        kwargs['rootdir'] = rootdir
        kwargs['mode'] = 'w'
        return bcolz.carray(data, expectedlen=expectedlen, **kwargs)

    def append(self, carr, data):
        carr.append(data)
        return carr


# singleton instance
bcolztmp_backend = BColzTmpBackend()
bcolztmp_gzip1_backend = BColzTmpBackend(cparams=bcolz.cparams(cname='zlib',
                                                               clevel=1))


class H5Backend(Backend):

    def append(self, h5d, data):
        hl = h5d.shape[0]
        dl = data.shape[0]
        hln = hl + dl
        h5d.resize(hln, axis=0)
        h5d[hl:hln] = data
        return h5d


class H5tmpBackend(H5Backend):

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def empty(self, shape, **kwargs):
        for k, v in self.kwargs.items():
            kwargs.setdefault(k, v)
        return h5dtmp(shape=shape, **kwargs)

    def create(self, data, expectedlen=None, **kwargs):
        # ignore expectedlen argument
        for k, v in self.kwargs.items():
            kwargs.setdefault(k, v)
        if not is_array_like(data):
            data = np.asarray(data)
        maxshape = (None,) + data.shape[1:]
        return h5dtmp(data=data, maxshape=maxshape, **kwargs)


# singleton instance
h5tmp_backend = H5tmpBackend()
h5tmp_gzip1_backend = H5tmpBackend(compression='gzip', compression_opts=1)


class H5memBackend(H5Backend):

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def empty(self, shape, **kwargs):
        for k, v in self.kwargs.items():
            kwargs.setdefault(k, v)
        return h5dmem(shape=shape, **kwargs)

    def create(self, data, expectedlen=None, **kwargs):
        # ignore expectedlen argument
        for k, v in self.kwargs.items():
            kwargs.setdefault(k, v)
        if not is_array_like(data):
            data = np.asarray(data)
        maxshape = (None,) + data.shape[1:]
        return h5dmem(data=data, maxshape=maxshape, **kwargs)


# singleton instance
h5mem_backend = H5memBackend()
h5mem_gzip1_backend = H5memBackend(compression='gzip', compression_opts=1)


# set default
default_backend = bcolz_backend


def get_backend(backend=None):
    if backend is None:
        return default_backend
    elif isinstance(backend, string_types):
        # normalise backend
        backend = str(backend).lower()
        if backend in ['numpy', 'ndarray', 'np']:
            return numpy_backend
        elif backend in ['bcolz', 'carray']:
            return bcolz_backend
        elif backend in ['bcolztmp', 'carraytmp']:
            return bcolztmp_backend
        elif backend in ['bcolz_gzip1', 'bcolz_zlib1', 'carray_gzip1',
                         'carray_zlib1']:
            return bcolz_gzip1_backend
        elif backend in ['bcolztmp_gzip1', 'bcolztmp_zlib1',
                         'carraytmp_gzip1', 'carraytmp_zlib1']:
            return bcolztmp_gzip1_backend
        elif backend in ['hdf5', 'h5py', 'h5dtmp', 'h5tmp']:
            return h5tmp_backend
        elif backend in ['h5dmem', 'h5mem']:
            return h5mem_backend
        elif backend in ['h5tmp_gzip1', 'h5dtmp_gzip1', 'h5tmp_zlib1',
                         'h5dtmp_gzip1']:
            return h5tmp_gzip1_backend
        elif backend in ['h5mem_gzip1', 'h5dmem_gzip1', 'h5mem_zlib1',
                         'h5mem_gzip1']:
            return h5mem_gzip1_backend
        else:
            raise ValueError('unknown backend: %s' % backend)
    elif isinstance(backend, Backend):
        # custom backend
        return backend
    else:
        raise ValueError('expected None, string or Backend, found: %r'
                         % backend)


class ChunkedArray(object):

    def __init__(self, data):
        check_array_like(data)
        self.data = data

    def __getitem__(self, *args):
        return self.data.__getitem__(*args)

    def __setitem__(self, key, value):
        return self.data.__setitem__(key, value)

    def __getattr__(self, item):
        return getattr(self.data, item)

    def __array__(self):
        return self.data[:]

    def __repr__(self):
        return '%s(%s, %s, %s.%s)' % \
               (type(self).__name__, str(self.shape), str(self.dtype),
                type(self.data).__module__, type(self.data).__name__)

    def __str__(self):
        return str(self.data)

    def __len__(self):
        return len(self.data)

    @property
    def ndim(self):
        return len(self.shape)

    def store(self, sink, offset=0, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))
        backend.store(self, sink, offset=offset, **kwargs)

    def copy(self, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))
        out = backend.copy(self, **kwargs)
        return type(self)(out)

    def max(self, axis=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))
        return backend.amax(self, axis=axis, **kwargs)

    def min(self, axis=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))
        return backend.amin(self, axis=axis, **kwargs)

    def sum(self, axis=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))
        return backend.sum(self, axis=axis, **kwargs)

    def op_scalar(self, op, other, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))
        out = backend.op_scalar(self, op, other, **kwargs)
        return ChunkedArray(out)

    def __eq__(self, other, **kwargs):
        return self.op_scalar(operator.eq, other, **kwargs)

    def __ne__(self, other, **kwargs):
        return self.op_scalar(operator.ne, other, **kwargs)

    def __lt__(self, other, **kwargs):
        return self.op_scalar(operator.lt, other, **kwargs)

    def __gt__(self, other, **kwargs):
        return self.op_scalar(operator.gt, other, **kwargs)

    def __le__(self, other, **kwargs):
        return self.op_scalar(operator.le, other, **kwargs)

    def __ge__(self, other, **kwargs):
        return self.op_scalar(operator.ge, other, **kwargs)

    def __add__(self, other, **kwargs):
        return self.op_scalar(operator.add, other, **kwargs)

    def __floordiv__(self, other, **kwargs):
        return self.op_scalar(operator.floordiv, other, **kwargs)

    def __mod__(self, other, **kwargs):
        return self.op_scalar(operator.mod, other, **kwargs)

    def __mul__(self, other, **kwargs):
        return self.op_scalar(operator.mul, other, **kwargs)

    def __pow__(self, other, **kwargs):
        return self.op_scalar(operator.pow, other, **kwargs)

    def __sub__(self, other, **kwargs):
        return self.op_scalar(operator.sub, other, **kwargs)

    def __truediv__(self, other, **kwargs):
        return self.op_scalar(operator.truediv, other, **kwargs)

    def compress(self, condition, axis=0, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))
        out = backend.compress(self.data, condition, axis=axis, **kwargs)
        return type(self)(out)

    def take(self, indices, axis=0, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))
        out = backend.take(self.data, indices, axis=axis, **kwargs)
        return type(self)(out)

    def subset(self, sel0, sel1, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))
        out = backend.subset(self.data, sel0, sel1, **kwargs)
        return type(self)(out)

    def hstack(self, *others, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))
        tup = (self,) + others
        out = backend.hstack(tup, **kwargs)
        return type(self)(out)

    def vstack(self, *others, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))
        tup = (self,) + others
        out = backend.vstack(tup, **kwargs)
        return type(self)(out)


class GenotypeChunkedArray(ChunkedArray):
    """TODO

    """

    def __init__(self, data):
        self.check_input_data(data)
        super(GenotypeChunkedArray, self).__init__(data)
        self._mask = None

    @staticmethod
    def check_input_data(data):
        check_array_like(data, 3)
        # check dtype
        if data.dtype.kind not in 'ui':
            raise TypeError('integer dtype required')

    def __getitem__(self, *args):
        out = self.data.__getitem__(*args)
        if is_array_like(out) \
                and len(self.shape) == len(out.shape) \
                and self.shape[2] == out.shape[2]:
            # dimensionality and ploidy preserved
            out = GenotypeArray(out)
            if self.mask is not None:
                # attempt to slice mask too
                m = self.mask.__getitem__(*args)
                out.mask = m
        return out

    def _repr_html_(self):
        return self[:6].to_html_str(caption=repr(self))

    @property
    def n_variants(self):
        return self.shape[0]

    @property
    def n_samples(self):
        return self.shape[1]

    @property
    def ploidy(self):
        return self.shape[2]

    @property
    def n_calls(self):
        """Total number of genotype calls (n_variants * n_samples)."""
        return self.shape[0] * self.shape[1]

    @property
    def n_allele_calls(self):
        """Total number of allele calls (n_variants * n_samples * ploidy)."""
        return self.shape[0] * self.shape[1] * self.shape[2]

    @property
    def mask(self):
        if hasattr(self, '_mask'):
            return self._mask
        else:
            return None

    @mask.setter
    def mask(self, mask):

        # check input
        if not is_array_like(mask):
            mask = np.asarray(mask)
        check_array_like(mask, 2)
        if mask.shape != self.shape[:2]:
            raise ValueError('mask has incorrect shape')

        # store
        self._mask = mask

    # noinspection PyTypeChecker
    def fill_masked(self, value=-1, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.fill_masked(value=value)

        out = backend.map_blocks(self, mapper, **kwargs)
        return GenotypeChunkedArray(out)

    # noinspection PyTypeChecker
    def subset(self, sel0, sel1, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))
        out = backend.subset(self, sel0, sel1, **kwargs)
        g = GenotypeChunkedArray(out)
        if self.mask is not None:
            mask = backend.subset(self.mask, sel0, sel1, **kwargs)
            g.mask = mask
        return g

    def is_called(self, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_called()

        out = backend.map_blocks(self, mapper, **kwargs)
        return ChunkedArray(out)

    def is_missing(self, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_missing()

        out = backend.map_blocks(self, mapper, **kwargs)
        return ChunkedArray(out)

    def is_hom(self, allele=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_hom(allele=allele)

        out = backend.map_blocks(self, mapper, **kwargs)
        return ChunkedArray(out)

    def is_hom_ref(self, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_hom_ref()

        out = backend.map_blocks(self, mapper, **kwargs)
        return ChunkedArray(out)

    def is_hom_alt(self, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_hom_alt()

        out = backend.map_blocks(self, mapper, **kwargs)
        return ChunkedArray(out)

    def is_het(self, allele=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_het(allele=allele)

        out = backend.map_blocks(self, mapper, **kwargs)
        return ChunkedArray(out)

    def is_call(self, call, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_call(call)

        out = backend.map_blocks(self, mapper, **kwargs)
        return ChunkedArray(out)

    def count_called(self, axis=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_called()

        out = backend.sum(self, axis=axis, mapper=mapper, **kwargs)
        return out

    def count_missing(self, axis=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_missing()

        out = backend.sum(self, axis=axis, mapper=mapper, **kwargs)
        return out

    def count_hom(self, allele=None, axis=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_hom(allele=allele)

        out = backend.sum(self, axis=axis, mapper=mapper, **kwargs)
        return out

    def count_hom_ref(self, axis=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_hom_ref()

        out = backend.sum(self, axis=axis, mapper=mapper, **kwargs)
        return out

    def count_hom_alt(self, axis=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_hom_alt()

        out = backend.sum(self, axis=axis, mapper=mapper, **kwargs)
        return out

    def count_het(self, axis=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_het()

        out = backend.sum(self, axis=axis, mapper=mapper, **kwargs)
        return out

    def count_call(self, call, axis=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_call(call)

        out = backend.sum(self, axis=axis, mapper=mapper, **kwargs)
        return out

    def to_haplotypes(self, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.to_haplotypes()

        out = backend.map_blocks(self, mapper, **kwargs)
        # TODO wrap with HaplotypeChunkedArray
        return ChunkedArray(out)

    def to_n_ref(self, fill=0, dtype='i1', **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.to_n_ref(fill=fill, dtype=dtype)

        out = backend.map_blocks(self, mapper, dtype=dtype, **kwargs)
        return ChunkedArray(out)

    def to_n_alt(self, fill=0, dtype='i1', **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.to_n_alt(fill=fill, dtype=dtype)

        out = backend.map_blocks(self, mapper, dtype=dtype, **kwargs)
        return ChunkedArray(out)

    def to_allele_counts(self, alleles=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        # determine alleles to count
        if alleles is None:
            m = self.max()
            alleles = list(range(m+1))

        def mapper(block):
            return block.to_allele_counts(alleles)

        out = backend.map_blocks(self, mapper, **kwargs)
        return ChunkedArray(out)

    def to_packed(self, boundscheck=True, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        if self.ploidy != 2:
            raise ValueError('can only pack diploid calls')

        if boundscheck:
            amx = self.max()
            if amx > 14:
                raise ValueError('max allele for packing is 14, found %s'
                                 % amx)
            amn = self.min()
            if amn < -1:
                raise ValueError('min allele for packing is -1, found %s'
                                 % amn)

        def mapper(block):
            return block.to_packed(boundscheck=False)

        out = backend.map_blocks(self, mapper, **kwargs)
        return ChunkedArray(out)

    # noinspection PyTypeChecker
    @staticmethod
    def from_packed(packed, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        # check input
        check_array_like(packed)

        def mapper(block):
            return GenotypeArray.from_packed(block)

        out = backend.map_blocks(packed, mapper, **kwargs)
        return GenotypeChunkedArray(out)

    def count_alleles(self, max_allele=None, subpop=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        # if max_allele not specified, count all alleles
        if max_allele is None:
            max_allele = self.max()

        def mapper(block):
            return block.count_alleles(max_allele=max_allele, subpop=subpop)

        out = backend.map_blocks(self, mapper, **kwargs)
        return AlleleCountsChunkedArray(out)

    def count_alleles_subpops(self, subpops, max_allele=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        if max_allele is None:
            max_allele = self.max()

        def mapper(block):
            return block.count_alleles_subpops(subpops, max_allele=max_allele)

        out = backend.dict_map_blocks(self, mapper, **kwargs)
        for k, v in out.items():
            out[k] = AlleleCountsChunkedArray(v)
        return out

    def to_gt(self, phased=False, max_allele=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        if max_allele is None:
            max_allele = self.max()

        def mapper(block):
            return block.to_gt(phased=phased, max_allele=max_allele)

        out = backend.map_blocks(self, mapper, **kwargs)
        return ChunkedArray(out)

    # noinspection PyTypeChecker
    def map_alleles(self, mapping, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        # check inputs
        check_array_like(mapping)
        check_dim0_aligned(self, mapping)

        # define mapping function
        def mapper(block, bmapping):
            return block.map_alleles(bmapping, copy=False)

        # execute map
        domain = (self, mapping)
        kwargs.setdefault('dtype', self.dtype)  # TODO needed?
        out = backend.map_blocks(domain, mapper, **kwargs)
        return GenotypeChunkedArray(out)


# copy docstrings
copy_method_doc(GenotypeChunkedArray.fill_masked, GenotypeArray.fill_masked)
copy_method_doc(GenotypeChunkedArray.subset, GenotypeArray.subset)
copy_method_doc(GenotypeChunkedArray.is_called, GenotypeArray.is_called)
copy_method_doc(GenotypeChunkedArray.is_missing, GenotypeArray.is_missing)
copy_method_doc(GenotypeChunkedArray.is_hom, GenotypeArray.is_hom)
copy_method_doc(GenotypeChunkedArray.is_hom_ref, GenotypeArray.is_hom_ref)
copy_method_doc(GenotypeChunkedArray.is_hom_alt, GenotypeArray.is_hom_alt)
copy_method_doc(GenotypeChunkedArray.is_het, GenotypeArray.is_het)
copy_method_doc(GenotypeChunkedArray.is_call, GenotypeArray.is_call)
copy_method_doc(GenotypeChunkedArray.to_haplotypes,
                GenotypeArray.to_haplotypes)
copy_method_doc(GenotypeChunkedArray.to_n_ref, GenotypeArray.to_n_ref)
copy_method_doc(GenotypeChunkedArray.to_n_alt, GenotypeArray.to_n_alt)
copy_method_doc(GenotypeChunkedArray.to_allele_counts,
                GenotypeArray.to_allele_counts)
copy_method_doc(GenotypeChunkedArray.to_packed, GenotypeArray.to_packed)
GenotypeChunkedArray.from_packed.__doc__ = GenotypeArray.from_packed.__doc__
copy_method_doc(GenotypeChunkedArray.count_alleles,
                GenotypeArray.count_alleles)
copy_method_doc(GenotypeChunkedArray.count_alleles_subpops,
                GenotypeArray.count_alleles_subpops)
copy_method_doc(GenotypeChunkedArray.to_gt, GenotypeArray.to_gt)
copy_method_doc(GenotypeChunkedArray.map_alleles, GenotypeArray.map_alleles)
copy_method_doc(GenotypeChunkedArray.hstack, GenotypeArray.hstack)
copy_method_doc(GenotypeChunkedArray.vstack, GenotypeArray.vstack)


class HaplotypeChunkedArray(ChunkedArray):
    """TODO

    """

    def __init__(self, data):
        self.check_input_data(data)
        super(HaplotypeChunkedArray, self).__init__(data)

    @staticmethod
    def check_input_data(data):
        check_array_like(data, 2)
        if data.dtype.kind not in 'ui':
            raise TypeError('integer dtype required')

    def __getitem__(self, *args):
        out = self.data.__getitem__(*args)
        if is_array_like(out) and len(self.shape) == len(out.shape):
            out = HaplotypeArray(out)
        return out

    def _repr_html_(self):
        return self[:6].to_html_str(caption=repr(self))

    @property
    def n_variants(self):
        """Number of variants (length of first array dimension)."""
        return self.data.shape[0]

    @property
    def n_haplotypes(self):
        """Number of haplotypes (length of second array dimension)."""
        return self.data.shape[1]

    def to_genotypes(self, ploidy, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        # check ploidy is compatible
        if (self.n_haplotypes % ploidy) > 0:
            raise ValueError('incompatible ploidy')

        # build output
        def mapper(block):
            return block.to_genotypes(ploidy)

        out = backend.map_blocks(self, mapper, **kwargs)
        return GenotypeChunkedArray(out)

    def is_called(self, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))
        return backend.op_scalar(self, operator.ge, 0, **kwargs)

    def is_missing(self, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))
        return backend.op_scalar(self, operator.lt, 0, **kwargs)

    def is_ref(self, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))
        return backend.op_scalar(self, operator.eq, 0, **kwargs)

    def is_alt(self, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))
        return backend.op_scalar(self, operator.gt, 0, **kwargs)

    def is_call(self, allele, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))
        return backend.op_scalar(self, operator.eq, allele, **kwargs)

    def count_called(self, axis=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_called()

        out = backend.sum(self, axis=axis, mapper=mapper, **kwargs)
        return out

    def count_missing(self, axis=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_missing()

        out = backend.sum(self, axis=axis, mapper=mapper, **kwargs)
        return out

    def count_ref(self, axis=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_ref()

        out = backend.sum(self, axis=axis, mapper=mapper, **kwargs)
        return out

    def count_alt(self, axis=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_alt()

        out = backend.sum(self, axis=axis, mapper=mapper, **kwargs)
        return out

    def count_call(self, allele, axis=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_call(allele)

        out = backend.sum(self, axis=axis, mapper=mapper, **kwargs)
        return out

    def count_alleles(self, max_allele=None, subpop=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        # if max_allele not specified, count all alleles
        if max_allele is None:
            max_allele = self.max()

        def mapper(block):
            return block.count_alleles(max_allele=max_allele, subpop=subpop)

        out = backend.map_blocks(self, mapper, **kwargs)
        return AlleleCountsChunkedArray(out)

    def count_alleles_subpops(self, subpops, max_allele=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        if max_allele is None:
            max_allele = self.max()

        def mapper(block):
            return block.count_alleles_subpops(subpops, max_allele=max_allele)

        out = backend.dict_map_blocks(self, mapper, **kwargs)
        for k, v in out.items():
            out[k] = AlleleCountsChunkedArray(v)
        return out

    # noinspection PyTypeChecker
    def map_alleles(self, mapping, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        # check inputs
        check_array_like(mapping)
        check_dim0_aligned(self, mapping)

        # define mapping function
        def mapper(block, bmapping):
            return block.map_alleles(bmapping, copy=False)

        # execute map
        domain = (self, mapping)
        kwargs.setdefault('dtype', self.dtype)  # TODO needed?
        out = backend.map_blocks(domain, mapper, **kwargs)
        return HaplotypeChunkedArray(out)


# copy docstrings
copy_method_doc(HaplotypeChunkedArray.to_genotypes,
                HaplotypeArray.to_genotypes)
copy_method_doc(HaplotypeChunkedArray.count_alleles,
                HaplotypeArray.count_alleles)
copy_method_doc(HaplotypeChunkedArray.count_alleles_subpops,
                HaplotypeArray.count_alleles_subpops)
copy_method_doc(HaplotypeChunkedArray.map_alleles, HaplotypeArray.map_alleles)


class AlleleCountsChunkedArray(ChunkedArray):

    def __init__(self, data):
        self.check_input_data(data)
        super(AlleleCountsChunkedArray, self).__init__(data)

    @staticmethod
    def check_input_data(data):
        check_array_like(data, 2)
        if data.dtype.kind not in 'ui':
            raise TypeError('integer dtype required')

    def __getitem__(self, *args):
        out = self.data.__getitem__(*args)
        if is_array_like(out) and len(self.shape) == len(out.shape) and \
                out.shape[1] == self.shape[1]:
            out = AlleleCountsArray(out)
        return out

    def _repr_html_(self):
        return self[:6].to_html_str(caption=repr(self))

    @property
    def n_variants(self):
        """Number of variants (length of first array dimension)."""
        return self.shape[0]

    @property
    def n_alleles(self):
        """Number of alleles (length of second array dimension)."""
        return self.shape[1]

    def to_frequencies(self, fill=np.nan, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.to_frequencies(fill=fill)

        out = backend.map_blocks(self, mapper, **kwargs)
        return ChunkedArray(out)

    def allelism(self, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.allelism()

        out = backend.map_blocks(self, mapper, **kwargs)
        return ChunkedArray(out)

    def max_allele(self, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.max_allele()

        out = backend.map_blocks(self, mapper, **kwargs)
        return ChunkedArray(out)

    def is_variant(self, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_variant()

        out = backend.map_blocks(self, mapper, **kwargs)
        return ChunkedArray(out)

    def is_non_variant(self, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_non_variant()

        out = backend.map_blocks(self, mapper, **kwargs)
        return ChunkedArray(out)

    def is_segregating(self, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_segregating()

        out = backend.map_blocks(self, mapper, **kwargs)
        return ChunkedArray(out)

    def is_non_segregating(self, allele=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_non_segregating(allele=allele)

        out = backend.map_blocks(self, mapper, **kwargs)
        return ChunkedArray(out)

    def is_singleton(self, allele=1, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_singleton(allele=allele)

        out = backend.map_blocks(self, mapper, **kwargs)
        return ChunkedArray(out)

    def is_doubleton(self, allele=1, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_doubleton(allele=allele)

        out = backend.map_blocks(self, mapper, **kwargs)
        return ChunkedArray(out)

    def count_variant(self, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_variant()

        out = backend.sum(self, mapper=mapper, **kwargs)
        return out

    def count_non_variant(self, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_non_variant()

        out = backend.sum(self, mapper=mapper, **kwargs)
        return out

    def count_segregating(self, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_segregating()

        out = backend.sum(self, mapper=mapper, **kwargs)
        return out

    def count_non_segregating(self, allele=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_non_segregating(allele=allele)

        out = backend.sum(self, mapper=mapper, **kwargs)
        return out

    def count_singleton(self, allele=1, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_singleton(allele=allele)

        out = backend.sum(self, mapper=mapper, **kwargs)
        return out

    def count_doubleton(self, allele=1, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_doubleton(allele=allele)

        out = backend.sum(self, mapper=mapper, **kwargs)
        return out

    def map_alleles(self, mapping, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        # check inputs
        check_array_like(mapping)
        check_dim0_aligned(self, mapping)

        # define mapping function
        def mapper(block, bmapping):
            return block.map_alleles(bmapping)

        # execute map
        domain = (self, mapping)
        kwargs.setdefault('dtype', self.dtype)  # TODO needed?
        out = backend.map_blocks(domain, mapper, **kwargs)
        return AlleleCountsChunkedArray(out)


copy_method_doc(AlleleCountsChunkedArray.allelism, AlleleCountsArray.allelism)
copy_method_doc(AlleleCountsChunkedArray.max_allele,
                AlleleCountsArray.max_allele)
copy_method_doc(AlleleCountsChunkedArray.map_alleles,
                AlleleCountsArray.map_alleles)


def check_table_like(data, names=None):
    if names is None:
        if hasattr(data, 'keys'):
            # h5py group, pandas dataframe, or dictionary
            names = list(data.keys())
        elif hasattr(data, 'names'):
            # bcolz ctable
            names = list(data.names)
        else:
            raise ValueError('could not determine column names')
    if len(names) < 1:
        raise ValueError('at least one column name is required')
    # should raise KeyError if name not present
    cols = [data[n] for n in names]
    check_array_like(cols)
    check_dim0_aligned(*cols)
    return names


class ChunkedTable(object):

    def __init__(self, data, names=None):
        self.data = data
        self.names = check_table_like(data, names)
        self.rowcls = namedtuple('row', self.names)

    def __getitem__(self, item):

        if isinstance(item, string_types):
            # return column
            return ChunkedArray(self.data[item])

        elif isinstance(item, integer_types):
            # return row as tuple
            return self.rowcls(*(self.data[n][item] for n in self.names))

        elif isinstance(item, slice):
            # load into numpy structured array
            if item.start is None:
                start = 0
            else:
                start = item.start
            if item.stop is None:
                stop = self.shape[0]
            else:
                stop = item.stop
            if item.step is None:
                step = 1
            else:
                step = item.step
            outshape = (stop - start) // step
            out = np.empty(outshape, dtype=self.dtype)
            for n in self.names:
                out[n] = self.data[n][item]
            return out.view(np.recarray)

        elif isinstance(item, (list, tuple)) and \
                all([isinstance(i, string_types) for i in item]):
            # assume names of columns, return table
            return ChunkedTable(self.data, names=item)

        else:
            raise NotImplementedError('unsupported item: %r' % item)

    def __getattr__(self, item):
        if item in self.names:
            return self.data[item]
        else:
            raise AttributeError(item)

    def __repr__(self):
        return '%s(%s, %s.%s)' % \
               (type(self).__name__, self.shape[0],
                type(self.data).__module__, type(self.data).__name__)

    def _repr_html_(self):
        caption = repr(self)
        ra = self[:6]
        return recarray_to_html_str(ra, limit=5, caption=caption)

    def display(self, limit, **kwargs):
        kwargs.setdefault('caption', repr(self))
        ra = self[:limit+1]
        return recarray_display(ra, limit=limit, **kwargs)

    @property
    def shape(self):
        return self.data[self.names[0]].shape[:1]

    @property
    def dtype(self):
        l = []
        for n in self.names:
            c = self.data[n]
            # Need to account for multidimensional columns
            t = (n, c.dtype) if len(c.shape) == 1 else \
                (n, c.dtype, c.shape[1:])
            l.append(t)
        return np.dtype(l)

    # TODO compress
    # TODO take
    # TODO eval
    # TODO query
    # TODO __len__
    # TODO ndim
    # TODO addcol (and __setitem__?)
    # TODO delcol (and __delitem__?)
    # TODO store
    # TODO copy


# TODO write table classes
## ChunkedTable
## VariantChunkedTable
## FeatureChunkedTable
