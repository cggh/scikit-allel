# -*- coding: utf-8 -*-
"""This module provides alternative implementations of array
classes defined in the :mod:`allel.model.ndarray` module, using
`bcolz <http://bcolz.blosc.org>`_ compressed arrays instead of numpy
arrays for data storage.

.. note::

    Please note this module is now deprecated and will be removed in a
    future release. It has been superseded by the
    :mod:`allel.model.chunked` module which supports both bcolz and
    HDF5 as the underlying storage layer.

"""
from __future__ import absolute_import, print_function, division


import operator
import itertools
from allel.compat import range, copy_method_doc


import numpy as np
import bcolz


from allel.model.ndarray import GenotypeArray, HaplotypeArray, \
    AlleleCountsArray, SortedIndex, SortedMultiIndex, subset, VariantTable, \
    FeatureTable, recarray_to_html_str, recarray_display
from allel.constants import DIM_PLOIDY
from allel.util import asarray_ndim, check_dim0_aligned
from allel.io import write_vcf_header, write_vcf_data, iter_gff3


__all__ = ['GenotypeCArray', 'HaplotypeCArray', 'AlleleCountsCArray',
           'VariantCTable', 'FeatureCTable', 'AlleleCountsCTable']


def ensure_carray(a, *ndims, **kwargs):
    if isinstance(a, CArrayWrapper):
        a = a.carr
    elif not isinstance(a, bcolz.carray):
        a = bcolz.carray(a, **kwargs)
    if a.ndim not in ndims:
        raise ValueError('invalid number of dimensions: %s' % a.ndim)
    return a


def carray_block_map(domain, f, out=None, blen=None, wrap=None, **kwargs):

    # determine expected output length
    if isinstance(domain, tuple):
        dim0len = domain[0].shape[0]
        kwargs.setdefault('cparams', getattr(domain[0], 'cparams', None))
    else:
        dim0len = domain.shape[0]
        kwargs.setdefault('cparams', getattr(domain, 'cparams', None))
    kwargs.setdefault('expectedlen', dim0len)

    # determine block size for iteration
    if blen is None:
        if isinstance(domain, tuple):
            blen = domain[0].chunklen
        else:
            blen = domain.chunklen

    # block-wise iteration
    for i in range(0, dim0len, blen):

        # slice domain
        if isinstance(domain, tuple):
            args = [a[i:i+blen] for a in domain]
        else:
            args = domain[i:i+blen],

        # map block
        res = f(*args)

        # create or append
        if out is None:
            out = bcolz.carray(res, **kwargs)
        else:
            out.append(res)

    if wrap is not None:
        out = wrap(out, copy=False)
    return out


def carray_block_sum(carr, axis=None, blen=None, transform=None):
    if blen is None:
        blen = carr.chunklen

    if axis is None:
        out = 0
        for i in range(0, carr.shape[0], blen):
            block = carr[i:i+blen]
            if transform:
                block = transform(block)
            out += np.sum(block)
        return out

    elif axis == 0 or axis == (0, 2):
        out = np.zeros((carr.shape[1],), dtype=int)
        for i in range(0, carr.shape[0], blen):
            block = carr[i:i+blen]
            if transform:
                block = transform(block)
            out += np.sum(block, axis=0)
        return out

    elif axis == 1 or axis == (1, 2):
        out = np.zeros((carr.shape[0],), dtype=int)
        for i in range(0, carr.shape[0], blen):
            block = carr[i:i+blen]
            if transform:
                block = transform(block)
            out[i:i+blen] += np.sum(block, axis=1)
        return out

    else:
        raise NotImplementedError('axis not supported: %s' % axis)


def carray_block_max(carr, axis=None, blen=None):
    if blen is None:
        blen = carr.chunklen
    out = None

    if axis is None:
        for i in range(0, carr.shape[0], blen):
            block = carr[i:i+blen]
            m = np.max(block)
            if out is None:
                out = m
            else:
                out = m if m > out else out
        return out

    elif axis == 0 or axis == (0, 2):
        for i in range(0, carr.shape[0], blen):
            block = carr[i:i+blen]
            m = np.max(block, axis=axis)
            if out is None:
                out = m
            else:
                out = np.where(m > out, m, out)
        return out

    elif axis == 1 or axis == (1, 2):
        out = np.zeros((carr.shape[0],), dtype=int)
        for i in range(0, carr.shape[0], blen):
            block = carr[i:i+blen]
            out[i:i+blen] = np.max(block, axis=axis)
        return out

    else:
        raise NotImplementedError('axis not supported: %s' % axis)


def carray_block_min(carr, axis=None, blen=None):
    if blen is None:
        blen = carr.chunklen
    out = None

    if axis is None:
        for i in range(0, carr.shape[0], blen):
            block = carr[i:i+blen]
            m = np.min(block)
            if out is None:
                out = m
            else:
                out = m if m < out else out
        return out

    elif axis == 0 or axis == (0, 2):
        for i in range(0, carr.shape[0], blen):
            block = carr[i:i+blen]
            m = np.min(block, axis=axis)
            if out is None:
                out = m
            else:
                out = np.where(m < out, m, out)
        return out

    elif axis == 1 or axis == (1, 2):
        out = np.zeros((carr.shape[0],), dtype=int)
        for i in range(0, carr.shape[0], blen):
            block = carr[i:i+blen]
            out[i:i+blen] = np.min(block, axis=axis)
        return out

    else:
        raise NotImplementedError('axis not supported: %s' % axis)


def carray_block_compress(carr, condition, axis, blen=None, **kwargs):
    if blen is None:
        blen = carr.chunklen

    # check inputs
    condition = asarray_ndim(condition, 1)

    # output defaults
    kwargs.setdefault('dtype', carr.dtype)
    kwargs.setdefault('cparams', getattr(carr, 'cparams', None))

    if axis == 0:
        if condition.size != carr.shape[0]:
            raise ValueError('length of condition must match length of '
                             'first dimension; expected %s, found %s' %
                             (carr.shape[0], condition.size))

        # setup output
        kwargs.setdefault('expectedlen', np.count_nonzero(condition))
        out = bcolz.zeros((0,) + carr.shape[1:], **kwargs)

        # build output
        for i in range(0, carr.shape[0], blen):
            bcond = condition[i:i+blen]
            # don't bother decompressing the block unless we have to
            if np.any(bcond):
                block = carr[i:i+blen]
                out.append(np.compress(bcond, block, axis=0))

        return out

    elif axis == 1:
        if condition.size != carr.shape[1]:
            raise ValueError('length of condition must match length of '
                             'second dimension; expected %s, found %s' %
                             (carr.shape[1], condition.size))

        # setup output
        kwargs.setdefault('expectedlen', carr.shape[0])
        out = bcolz.zeros((0, np.count_nonzero(condition)) + carr.shape[2:],
                          **kwargs)

        # build output
        for i in range(0, carr.shape[0], blen):
            block = carr[i:i+blen]
            out.append(np.compress(condition, block, axis=1))

        return out

    else:
        raise NotImplementedError('axis not supported: %s' % axis)


def carray_block_take(carr, indices, axis, blen=None, **kwargs):
    if blen is None:
        blen = carr.chunklen

    # check inputs
    indices = asarray_ndim(indices, 1)

    if axis == 0:
        # check if indices are ordered
        if np.any(indices[1:] <= indices[:-1]):
            raise ValueError('indices must be strictly increasing')
        condition = np.zeros((carr.shape[0],), dtype=bool)
        condition[indices] = True
        return carray_block_compress(carr, condition, axis=0,
                                     blen=blen, **kwargs)

    elif axis == 1:

        # setup output
        kwargs.setdefault('dtype', carr.dtype)
        kwargs.setdefault('cparams', getattr(carr, 'cparams', None))
        kwargs.setdefault('expectedlen', carr.shape[0])
        out = bcolz.zeros((0, len(indices)) + carr.shape[2:],
                          **kwargs)

        # build output
        for i in range(0, carr.shape[0], blen):
            block = carr[i:i+blen]
            out.append(np.take(block, indices, axis=1))

        return out

    else:
        raise NotImplementedError('axis not supported: %s' % axis)


def ctable_block_compress(ctbl, condition, blen=None, **kwargs):
    if blen is None:
        blen = min(ctbl[col].chunklen for col in ctbl.cols)

    # check inputs
    condition = asarray_ndim(condition, 1)

    if condition.size != ctbl.shape[0]:
        raise ValueError('length of condition must match length of '
                         'table; expected %s, found %s' %
                         (ctbl.shape[0], condition.size))

    # setup output
    kwargs.setdefault('expectedlen', np.count_nonzero(condition))
    kwargs.setdefault('cparams', getattr(ctbl, 'cparams', None))
    out = None

    # build output
    for i in range(0, ctbl.shape[0], blen):
        bcond = condition[i:i+blen]
        # don't bother decompressing the block unless we have to
        if np.any(bcond):
            block = ctbl[i:i+blen]
            res = np.compress(bcond, block, axis=0)
            if out is None:
                out = bcolz.ctable(res, **kwargs)
            else:
                out.append(res)

    return out


def ctable_block_take(ctbl, indices, **kwargs):
    indices = asarray_ndim(indices, 1)
    # check if indices are ordered
    if np.any(indices[1:] <= indices[:-1]):
        raise ValueError('indices must be strictly increasing')
    condition = np.zeros((ctbl.shape[0],), dtype=bool)
    condition[indices] = True
    return ctable_block_compress(ctbl, condition, **kwargs)


def carray_block_subset(carr, sel0, sel1, blen=None, **kwargs):
    if blen is None:
        blen = carr.chunklen

    # check inputs
    sel0 = asarray_ndim(sel0, 1, allow_none=True)
    sel1 = asarray_ndim(sel1, 1, allow_none=True)

    # ensure boolean array for dim 0
    if sel0 is not None and sel0.dtype.kind != 'b':
        tmp = np.zeros((carr.shape[0],), dtype=bool)
        tmp[sel0] = True
        sel0 = tmp

    # ensure indices for dim 1
    if sel1 is not None and sel1.dtype.kind == 'b':
        sel1 = np.nonzero(sel1)[0]

    # shortcuts
    if sel0 is None and sel1 is None:
        return carr.copy(**kwargs)
    elif sel1 is None:
        return carray_block_compress(carr, sel0, axis=0, blen=blen, **kwargs)
    elif sel0 is None:
        return carray_block_take(carr, sel1, axis=1, blen=blen, **kwargs)

    # setup output
    kwargs.setdefault('dtype', carr.dtype)
    kwargs.setdefault('expectedlen', np.count_nonzero(sel0))
    kwargs.setdefault('cparams', getattr(carr, 'cparams', None))
    out = bcolz.zeros((0, sel1.size) + carr.shape[2:], **kwargs)

    # build output
    for i in range(0, carr.shape[0], blen):
        block = carr[i:i+blen]
        bsel0 = sel0[i:i+blen]
        # don't bother decompressing the block unless we have to
        if np.any(bsel0):
            out.append(subset(block, bsel0, sel1))

    return out


def carray_block_hstack(tup, blen=None, **kwargs):
    assert isinstance(tup, (tuple, list)), \
        'first argument must be tuple or list'
    assert len(tup) > 1, 'provide two or more arrays to stack'
    a = tup[0]
    for x in tup[1:]:
        assert x.shape[0] == a.shape[0], \
            'arrays must have equal size for first dimension'
        assert len(x.shape) == len(a.shape), \
            'arrays must have same dimensionality'

    # set block size to use
    if blen is None:
        blen = min([x.chunklen for x in tup])

    # output defaults
    kwargs.setdefault('dtype', a.dtype)
    kwargs.setdefault('cparams', getattr(a, 'cparams', None))
    kwargs.setdefault('expectedlen', a.shape[0])

    # setup output
    out = None

    # build output
    for i in range(0, a.shape[0], blen):
        block = np.hstack(tuple(x[i:i+blen] for x in tup))
        if out is None:
            out = bcolz.carray(block, **kwargs)
        else:
            out.append(block)

    return out


def carray_block_vstack(tup, blen=None, **kwargs):
    assert isinstance(tup, (tuple, list)), \
        'first argument must be tuple or list'
    assert len(tup) > 1, 'provide two or more arrays to stack'
    a = tup[0]
    for x in tup[1:]:
        assert x.shape[1:] == a.shape[1:], \
            'arrays must have equal size for trailing dimensions'
        assert len(x.shape) == len(a.shape), \
            'arrays must have same dimensionality'

    # set block size to use
    if blen is None:
        blen = min([x.chunklen for x in tup])

    # output defaults
    kwargs.setdefault('dtype', a.dtype)
    kwargs.setdefault('cparams', getattr(a, 'cparams', None))
    kwargs.setdefault('expectedlen', sum(x.shape[0] for x in tup))

    # setup output
    out = None

    # build output
    for a in tup:
        for i in range(0, a.shape[0], blen):
            block = a[i:i+blen]
            if out is None:
                out = bcolz.carray(block, **kwargs)
            else:
                out.append(block)

    return out


def carray_from_hdf5(*args, **kwargs):
    """Load a bcolz carray from an HDF5 dataset.

    Either provide an h5py dataset as a single positional argument,
    or provide two positional arguments giving the HDF5 file path and the
    dataset node path within the file.

    The following optional parameters may be given. Any other keyword
    arguments are passed through to the bcolz.carray constructor.

    Parameters
    ----------
    start : int, optional
        Index to start loading from.
    stop : int, optional
        Index to finish loading at.
    condition : array_like, bool, optional
        A 1-dimensional boolean array of the same length as the first
        dimension of the dataset to load, indicating a selection of rows to
        load.
    blen : int, optional
        Block size to use when loading.

    """

    import h5py

    h5f = None

    if len(args) == 1:
        dataset = args[0]

    elif len(args) == 2:
        file_path, node_path = args
        h5f = h5py.File(file_path, mode='r')
        try:
            dataset = h5f[node_path]
        except:
            h5f.close()
            raise

    else:
        raise ValueError('bad arguments; expected dataset or (file_path, '
                         'node_path), found %s' % repr(args))

    try:

        if not isinstance(dataset, h5py.Dataset):
            raise ValueError('expected dataset, found %r' % dataset)

        length = dataset.shape[0]
        start = kwargs.pop('start', 0)
        stop = kwargs.pop('stop', length)
        condition = kwargs.pop('condition', None)
        condition = asarray_ndim(condition, 1, allow_none=True)
        blen = kwargs.pop('blen', None)

        # setup output data
        if condition is None:
            expectedlen = (stop - start)
        else:
            if condition.size != length:
                raise ValueError('length of condition does not match length '
                                 'of dataset')
            expectedlen = np.count_nonzero(condition[start:stop])
        kwargs.setdefault('expectedlen', expectedlen)
        kwargs.setdefault('dtype', dataset.dtype)
        carr = bcolz.zeros((0,) + dataset.shape[1:], **kwargs)

        # determine block size
        if blen is None:
            if hasattr(dataset, 'chunks'):
                # use input chunk length
                blen = dataset.chunks[0]
            else:
                # use output chunk length
                blen = carr.chunklen

        # load block-wise
        for i in range(start, stop, blen):
            j = min(i + blen, stop)
            if condition is not None:
                bcnd = condition[i:j]
                # only decompress block if necessary
                if np.any(bcnd):
                    block = dataset[i:j]
                    block = np.compress(bcnd, block, axis=0)
                    carr.append(block)
            else:
                block = dataset[i:j]
                carr.append(block)

        return carr

    finally:
        if h5f is not None:
            h5f.close()


def carray_to_hdf5(carr, parent, name, **kwargs):
    """Write a bcolz carray to an HDF5 dataset.

    Parameters
    ----------
    carr : bcolz.carray
        Data to write.
    parent : string or h5py group
        Parent HDF5 file or group. If a string, will be treated as HDF5 file
        name.
    name : string
        Name or path of dataset to write data into.
    kwargs : keyword arguments
        Passed through to h5py require_dataset() function.

    Returns
    -------
    h5d : h5py dataset

    """

    import h5py

    h5f = None

    if isinstance(parent, str):
        h5f = h5py.File(parent, mode='a')
        parent = h5f

    try:

        kwargs.setdefault('chunks', True)  # auto-chunking
        kwargs.setdefault('dtype', carr.dtype)
        kwargs.setdefault('compression', 'gzip')
        h5d = parent.require_dataset(name, shape=carr.shape, **kwargs)

        blen = carr.chunklen
        for i in range(0, carr.shape[0], blen):
            h5d[i:i+blen] = carr[i:i+blen]

        return h5d

    finally:
        if h5f is not None:
            h5f.close()


def ctable_from_hdf5_group(*args, **kwargs):
    """Load a bcolz ctable from columns stored as separate datasets with an
    HDF5 group.

    Either provide an h5py group as a single positional argument,
    or provide two positional arguments giving the HDF5 file path and the
    group node path within the file.

    The following optional parameters may be given. Any other keyword
    arguments are passed through to the bcolz.carray constructor.

    Parameters
    ----------
    start : int, optional
        Index to start loading from.
    stop : int, optional
        Index to finish loading at.
    condition : array_like, bool, optional
        A 1-dimensional boolean array of the same length as the columns of the
        table to load, indicating a selection of rows to load.
    blen : int, optional
        Block size to use when loading.

    """

    import h5py

    h5f = None

    if len(args) == 1:
        group = args[0]

    elif len(args) == 2:
        file_path, node_path = args
        h5f = h5py.File(file_path, mode='r')
        try:
            group = h5f[node_path]
        except:
            h5f.close()
            raise

    else:
        raise ValueError('bad arguments; expected group or (file_path, '
                         'node_path), found %s' % repr(args))

    try:

        if not isinstance(group, h5py.Group):
            raise ValueError('expected group, found %r' % group)

        # determine dataset names to load
        available_dataset_names = [n for n in group.keys()
                                   if isinstance(group[n], h5py.Dataset)]
        names = kwargs.pop('names', available_dataset_names)
        for n in names:
            if n not in set(group.keys()):
                raise ValueError('name not found: %s' % n)
            if not isinstance(group[n], h5py.Dataset):
                raise ValueError('name does not refer to a dataset: %s, %r'
                                 % (n, group[n]))

        # check datasets are aligned
        datasets = [group[n] for n in names]
        length = datasets[0].shape[0]
        for d in datasets[1:]:
            if d.shape[0] != length:
                raise ValueError('datasets must be of equal length')

        # determine start and stop parameters for load
        start = kwargs.pop('start', 0)
        stop = kwargs.pop('stop', length)
        blen = kwargs.pop('blen', None)
        condition = kwargs.pop('condition', None)
        condition = asarray_ndim(condition, 1, allow_none=True)

        # setup output data
        if condition is None:
            expectedlen = (stop - start)
        else:
            if condition.size != length:
                raise ValueError('length of condition does not match length '
                                 'of datasets')
            expectedlen = np.count_nonzero(condition[start:stop])
        kwargs.setdefault('expectedlen', expectedlen)
        if blen is None:
            # use smallest input chunk length
            blen = min([d.chunks[0] for d in datasets if hasattr(d, 'chunks')])
        ctbl = None

        # load block-wise
        for i in range(start, stop, blen):
            j = min(i + blen, stop)

            if condition is not None:
                bcnd = condition[i:j]
                # only decompress block if necessary
                if np.any(bcnd):
                    blocks = [d[i:j] for d in datasets]
                    blocks = [np.compress(bcnd, block, axis=0)
                              for block in blocks]
                else:
                    blocks = None
            else:
                blocks = [d[i:j] for d in datasets]

            if blocks:
                if ctbl is None:
                    ctbl = bcolz.ctable(blocks, names=names, **kwargs)
                else:
                    ctbl.append(blocks)

        return ctbl

    finally:
        if h5f is not None:
            h5f.close()


def ctable_to_hdf5_group(ctbl, parent, name, **kwargs):
    """Write each column in a bcolz ctable to a dataset in an HDF5 group.

    Parameters
    ----------

    parent : string or h5py group
        Parent HDF5 file or group. If a string, will be treated as HDF5 file
        name.
    name : string
        Name or path of group to write data into.
    kwargs : keyword arguments
        Passed through to h5py require_dataset() function.

    Returns
    -------

    h5g : h5py group

    """

    import h5py

    h5f = None

    if isinstance(parent, str):
        h5f = h5py.File(parent, mode='a')
        parent = h5f

    try:

        h5g = parent.require_group(name)
        for col in ctbl.cols:
            carray_to_hdf5(ctbl[col], h5g, col, **kwargs)

        return h5g

    finally:
        if h5f is not None:
            h5f.close()


class CArrayWrapper(object):

    def __init__(self, data=None, copy=False, **kwargs):
        if copy or not isinstance(data, bcolz.carray):
            carr = bcolz.carray(data, **kwargs)
        else:
            carr = data
        self.carr = carr

    def __getitem__(self, *args):
        return self.carr.__getitem__(*args)

    def __setitem__(self, key, value):
        return self.carr.__setitem__(key, value)

    def __getattr__(self, item):
        return getattr(self.carr, item)

    def __array__(self, *args):
        return self.carr[:]

    def __repr__(self):
        s = repr(self.carr)
        s = type(self).__name__ + s[6:]
        return s

    def __len__(self):
        return len(self.carr)

    @classmethod
    def open(cls, rootdir, mode='r'):
        cobj = bcolz.open(rootdir, mode=mode)
        if isinstance(cobj, bcolz.carray):
            return cls(cobj, copy=False)
        else:
            raise ValueError('rootdir does not contain a carray')

    def max(self, axis=None, blen=None, **kwargs):
        # ignore any other kwargs
        return carray_block_max(self.carr, axis=axis, blen=blen)

    def min(self, axis=None, blen=None, **kwargs):
        # ignore any other kwargs
        return carray_block_min(self.carr, axis=axis, blen=blen)

    def sum(self, axis=None, blen=None, **kwargs):
        # ignore any other kwargs
        return carray_block_sum(self.carr, axis=axis, blen=blen)

    def op_scalar(self, op, other, **kwargs):
        if not np.isscalar(other):
            raise NotImplementedError('only supported for scalars')

        # build output
        def f(block):
            return op(block, other)
        out = carray_block_map(self.carr, f, wrap=CArrayWrapper, **kwargs)

        return out

    def __eq__(self, other):
        return self.op_scalar(operator.eq, other)

    def __ne__(self, other):
        return self.op_scalar(operator.ne, other)

    def __lt__(self, other):
        return self.op_scalar(operator.lt, other)

    def __gt__(self, other):
        return self.op_scalar(operator.gt, other)

    def __le__(self, other):
        return self.op_scalar(operator.le, other)

    def __ge__(self, other):
        return self.op_scalar(operator.ge, other)

    def __add__(self, other):
        return self.op_scalar(operator.add, other)

    def __floordiv__(self, other):
        return self.op_scalar(operator.floordiv, other)

    def __mod__(self, other):
        return self.op_scalar(operator.mod, other)

    def __mul__(self, other):
        return self.op_scalar(operator.mul, other)

    def __pow__(self, other):
        return self.op_scalar(operator.pow, other)

    def __sub__(self, other):
        return self.op_scalar(operator.sub, other)

    def __truediv__(self, other):
        return self.op_scalar(operator.truediv, other)

    def compress(self, condition, axis=0, **kwargs):
        carr = carray_block_compress(self.carr, condition, axis, **kwargs)
        return type(self)(carr, copy=False)

    def take(self, indices, axis=0, **kwargs):
        carr = carray_block_take(self.carr, indices, axis, **kwargs)
        return type(self)(carr, copy=False)

    def hstack(self, *others, **kwargs):
        tup = (self,) + others
        carr = carray_block_hstack(tup, **kwargs)
        return type(self)(carr, copy=False)

    def vstack(self, *others, **kwargs):
        tup = (self,) + others
        carr = carray_block_vstack(tup, **kwargs)
        return type(self)(carr, copy=False)

    def copy(self, *args, **kwargs):
        carr = self.carr.copy(*args, **kwargs)
        return type(self)(carr, copy=False)

    @classmethod
    def from_hdf5(cls, *args, **kwargs):
        carr = carray_from_hdf5(*args, **kwargs)
        return cls(carr, copy=False)

    def to_hdf5(self, parent, name, **kwargs):
        return carray_to_hdf5(self.carr, parent, name, **kwargs)


# N.B., class method
CArrayWrapper.from_hdf5.__func__.__doc__ = carray_from_hdf5.__doc__
copy_method_doc(CArrayWrapper.to_hdf5, carray_to_hdf5)


class GenotypeCArray(CArrayWrapper):
    """Alternative implementation of the
    :class:`allel.model.ndarray.GenotypeArray` class, using a
    :class:`bcolz.carray` as the backing store.

    Parameters
    ----------
    data : array_like, int, shape (n_variants, n_samples, ploidy), optional
        Data to initialise the array with. May be a bcolz carray, which will
        not be copied if copy=False. May also be None, in which case rootdir
        must be provided (disk-based array).
    copy : bool, optional
        If True, copy the input data into a new bcolz carray.
    **kwargs : keyword arguments
        Passed through to the bcolz carray constructor.

    Examples
    --------

    Instantiate a compressed genotype array from existing data::

        >>> import allel
        >>> g = allel.GenotypeCArray([[[0, 0], [0, 1]],
        ...                           [[0, 1], [1, 1]],
        ...                           [[0, 2], [-1, -1]]], dtype='i1')
        >>> g
        GenotypeCArray((3, 2, 2), int8)
          nbytes := 12; cbytes := 16.00 KB; ratio: 0.00
          cparams := cparams(clevel=5, shuffle=1, cname='lz4', quantize=0)
          chunklen := 4096; chunksize: 16384; blocksize: 0
        [[[ 0  0]
          [ 0  1]]
         [[ 0  1]
          [ 1  1]]
         [[ 0  2]
          [-1 -1]]]

    Obtain a numpy ndarray from a compressed array by slicing::

        >>> g[:]
        GenotypeArray((3, 2, 2), dtype=int8)
        [[[ 0  0]
          [ 0  1]]
         [[ 0  1]
          [ 1  1]]
         [[ 0  2]
          [-1 -1]]]

    Build incrementally::

        >>> import bcolz
        >>> data = bcolz.zeros((0, 2, 2), dtype='i1')
        >>> data.append([[0, 0], [0, 1]])
        >>> data.append([[0, 1], [1, 1]])
        >>> data.append([[0, 2], [-1, -1]])
        >>> g = allel.GenotypeCArray(data)
        >>> g
        GenotypeCArray((3, 2, 2), int8)
          nbytes := 12; cbytes := 16.00 KB; ratio: 0.00
          cparams := cparams(clevel=5, shuffle=1, cname='lz4', quantize=0)
          chunklen := 4096; chunksize: 16384; blocksize: 0
        [[[ 0  0]
          [ 0  1]]
         [[ 0  1]
          [ 1  1]]
         [[ 0  2]
          [-1 -1]]]

    Load from HDF5::

        >>> import h5py
        >>> with h5py.File('test1.h5', mode='w') as h5f:
        ...     h5f.create_dataset('genotype',
        ...                        data=[[[0, 0], [0, 1]],
        ...                              [[0, 1], [1, 1]],
        ...                              [[0, 2], [-1, -1]]],
        ...                        dtype='i1',
        ...                        chunks=(2, 2, 2))
        ...
        <HDF5 dataset "genotype": shape (3, 2, 2), type "|i1">
        >>> g = allel.GenotypeCArray.from_hdf5('test1.h5', 'genotype')
        >>> g
        GenotypeCArray((3, 2, 2), int8)
          nbytes := 12; cbytes := 16.00 KB; ratio: 0.00
          cparams := cparams(clevel=5, shuffle=1, cname='lz4', quantize=0)
          chunklen := 4096; chunksize: 16384; blocksize: 0
        [[[ 0  0]
          [ 0  1]]
         [[ 0  1]
          [ 1  1]]
         [[ 0  2]
          [-1 -1]]]

    Note that methods of this class will return bcolz carrays rather than
    numpy ndarrays where possible. E.g.::

        >>> g.take([0, 2], axis=0)
        GenotypeCArray((2, 2, 2), int8)
          nbytes := 8; cbytes := 16.00 KB; ratio: 0.00
          cparams := cparams(clevel=5, shuffle=1, cname='lz4', quantize=0)
          chunklen := 4096; chunksize: 16384; blocksize: 0
        [[[ 0  0]
          [ 0  1]]
         [[ 0  2]
          [-1 -1]]]
        >>> g.is_called()
        CArrayWrapper((3, 2), bool)
          nbytes := 6; cbytes := 16.00 KB; ratio: 0.00
          cparams := cparams(clevel=5, shuffle=1, cname='lz4', quantize=0)
          chunklen := 8192; chunksize: 16384; blocksize: 0
        [[ True  True]
         [ True  True]
         [ True False]]
        >>> g.to_haplotypes()
        HaplotypeCArray((3, 4), int8)
          nbytes := 12; cbytes := 16.00 KB; ratio: 0.00
          cparams := cparams(clevel=5, shuffle=1, cname='lz4', quantize=0)
          chunklen := 4096; chunksize: 16384; blocksize: 0
        [[ 0  0  0  1]
         [ 0  1  1  1]
         [ 0  2 -1 -1]]
        >>> g.count_alleles()
        AlleleCountsCArray((3, 3), int32)
          nbytes := 36; cbytes := 16.00 KB; ratio: 0.00
          cparams := cparams(clevel=5, shuffle=1, cname='lz4', quantize=0)
          chunklen := 1365; chunksize: 16380; blocksize: 0
        [[3 1 0]
         [1 3 0]
         [1 0 1]]

    """

    @staticmethod
    def check_input_data(obj):

        # check dtype
        if obj.dtype.kind not in 'ui':
            raise TypeError('integer dtype required')

        # check dimensionality
        if hasattr(obj, 'ndim'):
            ndim = obj.ndim
        else:
            ndim = len(obj.shape)
        if ndim != 3:
            raise TypeError('array with 3 dimensions required')

        # check length of ploidy dimension
        if obj.shape[DIM_PLOIDY] == 1:
            raise ValueError('use HaplotypeCArray for haploid calls')

    def __init__(self, data=None, copy=False, **kwargs):
        super(GenotypeCArray, self).__init__(data=data, copy=copy, **kwargs)
        # check late to avoid creating an intermediate numpy array
        self.check_input_data(self.carr)

    def __getitem__(self, *args):
        out = self.carr.__getitem__(*args)
        if hasattr(out, 'ndim') \
                and out.ndim == 3 \
                and self.shape[-1] == out.shape[-1]:
            # dimensionality and ploidy preserved
            out = GenotypeArray(out, copy=False)
            if self.mask is not None:
                # attempt to slice mask
                m = self.mask.__getitem__(*args)
                out.mask = m
        return out

    def _repr_html_(self):
        g = self[:6]
        caption = ' '.join(repr(self).split('\n')[:3])
        return g.to_html_str(caption=caption)

    @property
    def n_variants(self):
        return self.carr.shape[0]

    @property
    def n_samples(self):
        return self.carr.shape[1]

    @property
    def ploidy(self):
        return self.carr.shape[2]

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
        """A boolean mask associated with this genotype array, indicating
        genotype calls that should be filtered (i.e., excluded) from
        genotype and allele counting operations.

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeCArray([[[0, 0], [0, 1]],
        ...                           [[0, 1], [1, 1]],
        ...                           [[0, 2], [-1, -1]]], dtype='i1')
        >>> g.count_called()
        5
        >>> g.count_alleles()
        AlleleCountsCArray((3, 3), int32)
          nbytes := 36; cbytes := 16.00 KB; ratio: 0.00
          cparams := cparams(clevel=5, shuffle=1, cname='lz4', quantize=0)
          chunklen := 1365; chunksize: 16380; blocksize: 0
        [[3 1 0]
         [1 3 0]
         [1 0 1]]
        >>> mask = [[True, False], [False, True], [False, False]]
        >>> g.mask = mask
        >>> g.mask
        carray((3, 2), bool)
          nbytes := 6; cbytes := 16.00 KB; ratio: 0.00
          cparams := cparams(clevel=5, shuffle=1, cname='lz4', quantize=0)
          chunklen := 8192; chunksize: 16384; blocksize: 0
        [[ True False]
         [False  True]
         [False False]]
        >>> g.count_called()
        3
        >>> g.count_alleles()
        AlleleCountsCArray((3, 3), int32)
          nbytes := 36; cbytes := 16.00 KB; ratio: 0.00
          cparams := cparams(clevel=5, shuffle=1, cname='lz4', quantize=0)
          chunklen := 1365; chunksize: 16380; blocksize: 0
        [[1 1 0]
         [1 1 0]
         [1 0 1]]

        Notes
        -----

        This is a lightweight genotype call mask and **not** a mask in the
        sense of a numpy masked array. This means that the mask will only be
        taken into account by the genotype and allele counting methods of this
        class, and is ignored by any of the generic methods on the ndarray
        class or by any numpy ufuncs.

        Note also that the mask may not survive any slicing, indexing or
        other subsetting procedures (e.g., call to compress() or take()).
        I.e., the mask will have to be similarly indexed then reapplied. The
        only exceptions are simple slicing operations that preserve the
        dimensionality and ploidy of the array, and the subset() method,
        both of which **will** preserve the mask if present.

        """
        if hasattr(self, '_mask'):
            return self._mask
        else:
            return None

    @mask.setter
    def mask(self, mask):

        # check input
        mask = ensure_carray(mask, 2)
        if mask.shape != self.shape[:2]:
            raise ValueError('mask has incorrect shape')

        # store
        self._mask = mask

    def fill_masked(self, value=-1, mask=None, copy=True, **kwargs):

        # determine mask
        if mask is None and self.mask is None:
            raise ValueError('no mask found')
        mask = mask if mask is not None else self.mask
        mask = asarray_ndim(mask, 2)
        if mask.shape != self.shape[:2]:
            raise ValueError('mask has incorrect shape')

        # setup output
        if copy:
            out = None
            kwargs.setdefault('expectedlen', self.shape[0])
        else:
            out = self.carr

        # determine block length for iteration
        blen = kwargs.pop('blen', self.carr.chunklen)

        # block-wise iteration
        for i in range(0, self.shape[0], blen):
            block = self[i:i+blen]
            bmask = mask[i:i+blen]
            block = block.fill_masked(value=value, mask=bmask, copy=True)
            if copy:
                if out is None:
                    out = bcolz.carray(block, **kwargs)
                else:
                    out.append(block)
            else:
                out[i:i+blen] = block

        return GenotypeCArray(out, copy=False)

    def subset(self, sel0=None, sel1=None, **kwargs):
        carr = carray_block_subset(self.carr, sel0, sel1, **kwargs)
        g = GenotypeCArray(carr, copy=False)
        if self.mask is not None:
            mask = carray_block_subset(self.mask, sel0, sel1)
            g.mask = mask
        return g

    def is_called(self, **kwargs):
        def f(block):
            return block.is_called()
        return carray_block_map(self, f, wrap=CArrayWrapper, **kwargs)

    def is_missing(self, **kwargs):
        def f(block):
            return block.is_missing()
        return carray_block_map(self, f, wrap=CArrayWrapper, **kwargs)

    def is_hom(self, allele=None, **kwargs):
        def f(block):
            return block.is_hom(allele=allele)
        return carray_block_map(self, f, wrap=CArrayWrapper, **kwargs)

    def is_hom_ref(self, **kwargs):
        def f(block):
            return block.is_hom_ref()
        return carray_block_map(self, f, wrap=CArrayWrapper, **kwargs)

    def is_hom_alt(self, **kwargs):
        def f(block):
            return block.is_hom_alt()
        return carray_block_map(self, f, wrap=CArrayWrapper, **kwargs)

    def is_het(self, allele=None, **kwargs):
        def f(block):
            return block.is_het(allele=allele)
        return carray_block_map(self, f, wrap=CArrayWrapper, **kwargs)

    def is_call(self, call, **kwargs):
        def f(block):
            return block.is_call(call)
        return carray_block_map(self, f, wrap=CArrayWrapper, **kwargs)

    def count_called(self, axis=None):
        def f(block):
            return block.is_called()
        return carray_block_sum(self, axis=axis, transform=f)

    def count_missing(self, axis=None):
        def f(block):
            return block.is_missing()
        return carray_block_sum(self, axis=axis, transform=f)

    def count_hom(self, allele=None, axis=None):
        def f(block):
            return block.is_hom(allele=allele)
        return carray_block_sum(self, axis=axis, transform=f)

    def count_hom_ref(self, axis=None):
        def f(block):
            return block.is_hom_ref()
        return carray_block_sum(self, axis=axis, transform=f)

    def count_hom_alt(self, axis=None):
        def f(block):
            return block.is_hom_alt()
        return carray_block_sum(self, axis=axis, transform=f)

    def count_het(self, allele=None, axis=None):
        def f(block):
            return block.is_het(allele=allele)
        return carray_block_sum(self, axis=axis, transform=f)

    def count_call(self, call, axis=None):
        def f(block):
            return block.is_call(call=call)
        return carray_block_sum(self, axis=axis, transform=f)

    def to_haplotypes(self, **kwargs):

        # Unfortunately this cannot be implemented as a lightweight view,
        # so we have to copy.

        # build output
        def f(block):
            return block.to_haplotypes()
        out = carray_block_map(self, f, wrap=HaplotypeCArray, **kwargs)
        return out

    def to_n_ref(self, fill=0, dtype='i1', **kwargs):
        def f(block):
            return block.to_n_ref(fill=fill, dtype=dtype)
        return carray_block_map(self, f, wrap=CArrayWrapper, dtype=dtype,
                                **kwargs)

    def to_n_alt(self, fill=0, dtype='i1', **kwargs):
        def f(block):
            return block.to_n_alt(fill=fill, dtype=dtype)
        return carray_block_map(self, f, wrap=CArrayWrapper, dtype=dtype,
                                **kwargs)

    def to_allele_counts(self, alleles=None, **kwargs):

        # determine alleles to count
        if alleles is None:
            m = self.max()
            alleles = list(range(m+1))

        # build output
        def f(block):
            return block.to_allele_counts(alleles)

        return carray_block_map(self, f, wrap=CArrayWrapper, **kwargs)

    def to_packed(self, boundscheck=True, **kwargs):

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

        # build output
        def f(block):
            return block.to_packed(boundscheck=False)

        return carray_block_map(self, f, wrap=CArrayWrapper, **kwargs)

    @staticmethod
    def from_packed(packed, **kwargs):

        # check input
        if not isinstance(packed, (np.ndarray, bcolz.carray)):
            packed = np.asarray(packed)

        # set up output
        kwargs.setdefault('dtype', 'i1')
        kwargs.setdefault('expectedlen', packed.shape[0])
        out = bcolz.zeros((0, packed.shape[1], 2), **kwargs)
        blen = out.chunklen

        # build output
        def f(block):
            return GenotypeArray.from_packed(block)
        out = carray_block_map(packed, f, out=out, blen=blen,
                               wrap=GenotypeCArray)

        return out

    def count_alleles(self, max_allele=None, subpop=None, **kwargs):

        # if max_allele not specified, count all alleles
        if max_allele is None:
            max_allele = self.max()

        def f(block):
            return block.count_alleles(max_allele=max_allele, subpop=subpop)

        out = carray_block_map(self, f, wrap=AlleleCountsCArray, **kwargs)

        return out

    def count_alleles_subpops(self, subpops, max_allele=None, **kwargs):

        # if max_allele not specified, count all alleles
        if max_allele is None:
            max_allele = self.max()

        # setup intermediates
        names = sorted(subpops.keys())
        subpops = [subpops[n] for n in names]

        # setup output table
        kwargs['names'] = names  # override to ensure correct order
        kwargs.setdefault('expectedlen', self.shape[0])
        acs_ctbl = None

        # determine block size for iteration
        blen = kwargs.pop('blen', None)
        if blen is None:
            blen = self.carr.chunklen

        # block iteration
        for i in range(0, self.shape[0], blen):
            block = self[i:i+blen]
            cols = [block.count_alleles(max_allele=max_allele, subpop=subpop)
                    for subpop in subpops]
            if acs_ctbl is None:
                acs_ctbl = bcolz.ctable(cols, **kwargs)
            else:
                acs_ctbl.append(cols)

        return AlleleCountsCTable(acs_ctbl, copy=False)

    def to_gt(self, phased=False, max_allele=None, **kwargs):
        if max_allele is None:
            max_allele = self.max()

        def f(block):
            return block.to_gt(phased=phased, max_allele=max_allele)
        return carray_block_map(self, f, wrap=CArrayWrapper, **kwargs)

    def map_alleles(self, mapping, **kwargs):

        # check inputs
        mapping = asarray_ndim(mapping, 2)
        check_dim0_aligned(self, mapping)

        # setup output
        kwargs.setdefault('dtype', self.carr.dtype)

        # define mapping function
        def f(block, bmapping):
            return block.map_alleles(bmapping, copy=False)

        # execute map
        domain = (self, mapping)
        out = carray_block_map(domain, f, wrap=GenotypeCArray, **kwargs)

        return out


# copy docstrings
copy_method_doc(GenotypeCArray.fill_masked, GenotypeArray.fill_masked)
copy_method_doc(GenotypeCArray.subset, GenotypeArray.subset)
copy_method_doc(GenotypeCArray.is_called, GenotypeArray.is_called)
copy_method_doc(GenotypeCArray.is_missing, GenotypeArray.is_missing)
copy_method_doc(GenotypeCArray.is_hom, GenotypeArray.is_hom)
copy_method_doc(GenotypeCArray.is_hom_ref, GenotypeArray.is_hom_ref)
copy_method_doc(GenotypeCArray.is_hom_alt, GenotypeArray.is_hom_alt)
copy_method_doc(GenotypeCArray.is_het, GenotypeArray.is_het)
copy_method_doc(GenotypeCArray.is_call, GenotypeArray.is_call)
copy_method_doc(GenotypeCArray.to_haplotypes, GenotypeArray.to_haplotypes)
copy_method_doc(GenotypeCArray.to_n_ref, GenotypeArray.to_n_ref)
copy_method_doc(GenotypeCArray.to_n_alt, GenotypeArray.to_n_alt)
copy_method_doc(GenotypeCArray.to_allele_counts,
                GenotypeArray.to_allele_counts)
copy_method_doc(GenotypeCArray.to_packed, GenotypeArray.to_packed)
GenotypeCArray.from_packed.__doc__ = GenotypeArray.from_packed.__doc__
copy_method_doc(GenotypeCArray.count_alleles, GenotypeArray.count_alleles)
copy_method_doc(GenotypeCArray.count_alleles_subpops,
                GenotypeArray.count_alleles_subpops)
copy_method_doc(GenotypeCArray.to_gt, GenotypeArray.to_gt)
copy_method_doc(GenotypeCArray.map_alleles, GenotypeArray.map_alleles)
copy_method_doc(GenotypeCArray.hstack, GenotypeArray.hstack)
copy_method_doc(GenotypeCArray.vstack, GenotypeArray.vstack)


class HaplotypeCArray(CArrayWrapper):
    """Alternative implementation of the
    :class:`allel.model.ndarray.HaplotypeArray` class, using a
    :class:`bcolz.carray` as the backing store.

    Parameters
    ----------

    data : array_like, int, shape (n_variants, n_haplotypes), optional
        Data to initialise the array with. May be a bcolz carray, which will
        not be copied if copy=False. May also be None, in which case rootdir
        must be provided (disk-based array).
    copy : bool, optional
        If True, copy the input data into a new bcolz carray.
    **kwargs : keyword arguments
        Passed through to the bcolz carray constructor.

    """

    @staticmethod
    def check_input_data(obj):

        # check dtype
        if obj.dtype.kind not in 'ui':
            raise TypeError('integer dtype required')

        # check dimensionality
        if hasattr(obj, 'ndim'):
            ndim = obj.ndim
        else:
            ndim = len(obj.shape)
        if ndim != 2:
            raise TypeError('array with 2 dimensions required')

    def __init__(self, data=None, copy=False, **kwargs):
        super(HaplotypeCArray, self).__init__(data=data, copy=copy, **kwargs)
        self.check_input_data(self.carr)

    def __getitem__(self, *args):
        out = self.carr.__getitem__(*args)
        if hasattr(out, 'ndim') and out.ndim == 2:
            out = HaplotypeArray(out, copy=False)
        return out

    def _repr_html_(self):
        h = self[:6]
        caption = ' '.join(repr(self).split('\n')[:3])
        return h.to_html_str(caption=caption)

    @property
    def n_variants(self):
        """Number of variants (length of first array dimension)."""
        return self.carr.shape[0]

    @property
    def n_haplotypes(self):
        """Number of haplotypes (length of second array dimension)."""
        return self.carr.shape[1]

    def subset(self, sel0=None, sel1=None, **kwargs):
        data = carray_block_subset(self.carr, sel0, sel1, **kwargs)
        return HaplotypeCArray(data, copy=False)

    def to_genotypes(self, ploidy, **kwargs):
        # This cannot be implemented as a lightweight view,
        # so we have to copy.

        # check ploidy is compatible
        if (self.n_haplotypes % ploidy) > 0:
            raise ValueError('incompatible ploidy')

        # build output
        def f(block):
            return block.to_genotypes(ploidy)
        out = carray_block_map(self, f, wrap=GenotypeCArray, **kwargs)

        return out

    def is_called(self, **kwargs):
        return self.op_scalar(operator.ge, 0, **kwargs)

    def is_missing(self, **kwargs):
        return self.op_scalar(operator.lt, 0, **kwargs)

    def is_ref(self, **kwargs):
        return self.op_scalar(operator.eq, 0, **kwargs)

    def is_alt(self, **kwargs):
        return self.op_scalar(operator.gt, 0, **kwargs)

    def is_call(self, allele, **kwargs):
        return self.op_scalar(operator.eq, allele, **kwargs)

    def count_called(self, axis=None):
        def f(block):
            return block.is_called()
        return carray_block_sum(self, axis=axis, transform=f)

    def count_missing(self, axis=None):
        def f(block):
            return block.is_missing()
        return carray_block_sum(self, axis=axis, transform=f)

    def count_ref(self, axis=None):
        def f(block):
            return block.is_ref()
        return carray_block_sum(self, axis=axis, transform=f)

    def count_alt(self, axis=None):
        def f(block):
            return block.is_alt()
        return carray_block_sum(self, axis=axis, transform=f)

    def count_call(self, allele, axis=None):
        def f(block):
            return block.is_call(allele=allele)
        return carray_block_sum(self, axis=axis, transform=f)

    def count_alleles(self, max_allele=None, subpop=None, **kwargs):

        # if max_allele not specified, count all alleles
        if max_allele is None:
            max_allele = self.max()

        def f(block):
            return block.count_alleles(max_allele=max_allele, subpop=subpop)

        out = carray_block_map(self, f, wrap=AlleleCountsCArray,
                               **kwargs)

        return out

    def count_alleles_subpops(self, subpops, max_allele=None, **kwargs):

        # if max_allele not specified, count all alleles
        if max_allele is None:
            max_allele = self.max()

        # setup output table
        names = sorted(subpops.keys())
        subpops = [subpops[n] for n in names]
        kwargs['names'] = names  # override to ensure correct order
        kwargs.setdefault('expectedlen', self.shape[0])
        acs_ctbl = None

        # determine block size for iteration
        blen = kwargs.pop('blen', None)
        if blen is None:
            blen = self.carr.chunklen

        # block iteration
        for i in range(0, self.shape[0], blen):
            block = self[i:i+blen]
            cols = [block.count_alleles(max_allele=max_allele, subpop=subpop)
                    for subpop in subpops]
            if acs_ctbl is None:
                acs_ctbl = bcolz.ctable(cols, **kwargs)
            else:
                acs_ctbl.append(cols)

        # wrap for convenience
        return AlleleCountsCTable(acs_ctbl, copy=False)

    def map_alleles(self, mapping, **kwargs):

        # check inputs
        mapping = asarray_ndim(mapping, 2)
        check_dim0_aligned(self, mapping)

        # setup output
        kwargs.setdefault('dtype', self.carr.dtype)

        # define mapping function
        def f(block, bmapping):
            return block.map_alleles(bmapping, copy=False)

        # execute map
        domain = (self, mapping)
        out = carray_block_map(domain, f, wrap=HaplotypeCArray, **kwargs)

        return out


# copy docstrings
copy_method_doc(HaplotypeCArray.subset, HaplotypeArray.subset)
copy_method_doc(HaplotypeCArray.to_genotypes, HaplotypeArray.to_genotypes)
copy_method_doc(HaplotypeCArray.count_alleles, HaplotypeArray.count_alleles)
copy_method_doc(HaplotypeCArray.count_alleles_subpops,
                HaplotypeArray.count_alleles_subpops)
copy_method_doc(HaplotypeCArray.map_alleles, HaplotypeArray.map_alleles)
copy_method_doc(HaplotypeCArray.hstack, HaplotypeArray.hstack)
copy_method_doc(HaplotypeCArray.vstack, HaplotypeArray.vstack)


class AlleleCountsCArray(CArrayWrapper):
    """Alternative implementation of the
    :class:`allel.model.ndarray.AlleleCountsArray` class, using a
    :class:`bcolz.carray` as the backing store.

    Parameters
    ----------

    data : array_like, int, shape (n_variants, n_alleles), optional
        Data to initialise the array with. May be a bcolz carray, which will
        not be copied if copy=False. May also be None, in which case rootdir
        must be provided (disk-based array).
    copy : bool, optional
        If True, copy the input data into a new bcolz carray.
    **kwargs : keyword arguments
        Passed through to the bcolz carray constructor.

    """

    @staticmethod
    def check_input_data(obj):

        # check dtype
        if obj.dtype.kind not in 'ui':
            raise TypeError('integer dtype required')

        # check dimensionality
        if hasattr(obj, 'ndim'):
            ndim = obj.ndim
        else:
            ndim = len(obj.shape)
        if ndim != 2:
            raise TypeError('array with 2 dimensions required')

    def __init__(self, data=None, copy=False, **kwargs):
        super(AlleleCountsCArray, self).__init__(data=data, copy=copy,
                                                 **kwargs)
        # check late to avoid creating an intermediate numpy array
        self.check_input_data(self.carr)

    def __getitem__(self, *args):
        out = self.carr.__getitem__(*args)
        if hasattr(out, 'ndim') \
                and out.ndim == 2 \
                and out.shape[1] == self.n_alleles:
            # wrap only if number of alleles is preserved
            out = AlleleCountsArray(out, copy=False)
        return out

    def _repr_html_(self):
        ac = self[:6]
        caption = ' '.join(repr(self).split('\n')[:3])
        return ac.to_html_str(caption=caption)

    @property
    def n_variants(self):
        """Number of variants (length of first array dimension)."""
        return self.carr.shape[0]

    @property
    def n_alleles(self):
        """Number of alleles (length of second array dimension)."""
        return self.carr.shape[1]

    def compress(self, condition, axis=0, **kwargs):
        carr = carray_block_compress(self.carr, condition, axis, **kwargs)
        if carr.shape[1] == self.shape[1]:
            # alleles preserved, safe to wrap
            return AlleleCountsCArray(carr, copy=False)
        else:
            return CArrayWrapper(carr)

    def take(self, indices, axis=0, **kwargs):
        carr = carray_block_take(self.carr, indices, axis, **kwargs)
        if carr.shape[1] == self.shape[1]:
            # alleles preserved, safe to wrap
            return AlleleCountsCArray(carr, copy=False)
        else:
            return CArrayWrapper(carr)

    def to_frequencies(self, fill=np.nan, **kwargs):
        def f(block):
            return block.to_frequencies(fill=fill)
        return carray_block_map(self, f, wrap=CArrayWrapper, **kwargs)

    def allelism(self, **kwargs):
        def f(block):
            return block.allelism()
        return carray_block_map(self, f, wrap=CArrayWrapper, **kwargs)

    def max_allele(self, **kwargs):
        def f(block):
            return block.max_allele()
        return carray_block_map(self, f, wrap=CArrayWrapper, **kwargs)

    def is_variant(self, **kwargs):
        def f(block):
            return block.is_variant()
        return carray_block_map(self, f, wrap=CArrayWrapper, **kwargs)

    def is_non_variant(self, **kwargs):
        def f(block):
            return block.is_non_variant()
        return carray_block_map(self, f, wrap=CArrayWrapper, **kwargs)

    def is_segregating(self, **kwargs):
        def f(block):
            return block.is_segregating()
        return carray_block_map(self, f, wrap=CArrayWrapper, **kwargs)

    def is_non_segregating(self, allele=None, **kwargs):
        def f(block):
            return block.is_non_segregating(allele=allele)
        return carray_block_map(self, f, wrap=CArrayWrapper, **kwargs)

    def is_singleton(self, allele=1, **kwargs):
        def f(block):
            return block.is_singleton(allele=allele)
        return carray_block_map(self, f, wrap=CArrayWrapper, **kwargs)

    def is_doubleton(self, allele=1, **kwargs):
        def f(block):
            return block.is_doubleton(allele=allele)
        return carray_block_map(self, f, wrap=CArrayWrapper, **kwargs)

    def is_biallelic(self, **kwargs):
        def f(block):
            return block.is_biallelic()
        return carray_block_map(self, f, wrap=CArrayWrapper, **kwargs)

    def is_biallelic_01(self, min_mac=None, **kwargs):
        def f(block):
            return block.is_biallelic_01(min_mac=min_mac)
        return carray_block_map(self, f, wrap=CArrayWrapper, **kwargs)

    def count_variant(self):
        return carray_block_sum(self.is_variant())

    def count_non_variant(self):
        return carray_block_sum(self.is_non_variant())

    def count_segregating(self):
        return carray_block_sum(self.is_segregating())

    def count_non_segregating(self, allele=None):
        return carray_block_sum(self.is_non_segregating(allele=allele))

    def count_singleton(self, allele=1):
        return carray_block_sum(self.is_singleton(allele=allele))

    def count_doubleton(self, allele=1):
        return carray_block_sum(self.is_doubleton(allele=allele))

    def map_alleles(self, mapping, **kwargs):

        # check inputs
        mapping = asarray_ndim(mapping, 2)
        check_dim0_aligned(self, mapping)

        # setup output
        kwargs.setdefault('dtype', self.carr.dtype)

        # define mapping function
        def f(block, bmapping):
            return block.map_alleles(bmapping)

        # execute map
        domain = (self, mapping)
        out = carray_block_map(domain, f, wrap=AlleleCountsCArray, **kwargs)

        return out


# copy docstrings
copy_method_doc(AlleleCountsCArray.to_frequencies,
                AlleleCountsArray.to_frequencies)
copy_method_doc(AlleleCountsCArray.allelism, AlleleCountsArray.allelism)
copy_method_doc(AlleleCountsCArray.max_allele, AlleleCountsArray.max_allele)
copy_method_doc(AlleleCountsCArray.is_variant, AlleleCountsArray.is_variant)
copy_method_doc(AlleleCountsCArray.is_non_variant,
                AlleleCountsArray.is_non_variant)
copy_method_doc(AlleleCountsCArray.is_segregating,
                AlleleCountsArray.is_segregating)
copy_method_doc(AlleleCountsCArray.is_non_segregating,
                AlleleCountsArray.is_non_segregating)
copy_method_doc(AlleleCountsCArray.is_singleton,
                AlleleCountsArray.is_singleton)
copy_method_doc(AlleleCountsCArray.is_doubleton,
                AlleleCountsArray.is_doubleton)
copy_method_doc(AlleleCountsCArray.is_biallelic,
                AlleleCountsArray.is_biallelic)
copy_method_doc(AlleleCountsCArray.is_biallelic_01,
                AlleleCountsArray.is_biallelic_01)
copy_method_doc(AlleleCountsCArray.map_alleles, AlleleCountsArray.map_alleles)


class CTableWrapper(object):

    ndarray_cls = None

    def __array__(self):
        return self.ctbl[:]

    def __getitem__(self, item):
        res = self.ctbl[item]
        if isinstance(res, np.ndarray) \
                and res.dtype.names \
                and self.ndarray_cls is not None:
            # noinspection PyCallingNonCallable
            return self.ndarray_cls(res, copy=False)
        if isinstance(res, bcolz.ctable):
            return type(self)(res, copy=False)
        return res

    def __getattr__(self, item):
        if hasattr(self.ctbl, item):
            return getattr(self.ctbl, item)
        elif item in self.names:
            return self.ctbl[item]
        else:
            raise AttributeError(item)

    def __repr__(self):
        s = repr(self.ctbl)
        s = type(self).__name__ + s[6:]
        return s

    def __len__(self):
        return len(self.ctbl)

    def _repr_html_(self):
        caption = ' '.join(repr(self).split('\n')[:3])
        return ctable_to_html_str(self, limit=5, caption=caption)

    def display(self, limit, **kwargs):
        caption = ' '.join(repr(self).split('\n')[:3])
        kwargs.setdefault('caption', caption)
        return ctable_display(self, limit=limit, **kwargs)

    @classmethod
    def open(cls, rootdir, mode='r'):
        cobj = bcolz.open(rootdir, mode=mode)
        if isinstance(cobj, bcolz.ctable):
            return cls(cobj, copy=False)
        else:
            raise ValueError('rootdir does not contain a ctable')

    @property
    def names(self):
        return tuple(self.ctbl.names)

    def compress(self, condition, **kwargs):
        ctbl = ctable_block_compress(self.ctbl, condition, **kwargs)
        return type(self)(ctbl, copy=False)

    def take(self, indices, **kwargs):
        ctbl = ctable_block_take(self.ctbl, indices, **kwargs)
        return type(self)(ctbl, copy=False)

    def eval(self, expression, vm='python', **kwargs):
        return self.ctbl.eval(expression, vm=vm, **kwargs)

    def query(self, expression, vm='python'):
        condition = self.eval(expression, vm=vm)
        return self.compress(condition)

    @classmethod
    def from_hdf5_group(cls, *args, **kwargs):
        ctbl = ctable_from_hdf5_group(*args, **kwargs)
        return cls(ctbl, copy=False)

    def to_hdf5_group(self, parent, name, **kwargs):
        return ctable_to_hdf5_group(self.ctbl, parent, name, **kwargs)


# N.B., class method
CTableWrapper.from_hdf5_group.__func__.__doc__ = ctable_from_hdf5_group.__doc__
copy_method_doc(CTableWrapper.to_hdf5_group, ctable_to_hdf5_group)


class VariantCTable(CTableWrapper):
    """Alternative implementation of the
    :class:`allel.model.ndarray.VariantTable` class, using a
    :class:`bcolz.ctable` as the backing store.

    Parameters
    ----------

    data : tuple or list of column objects, optional
        The list of column data to build the ctable object. This can also be a
        pure NumPy structured array. May also be a bcolz ctable, which will
        not be copied if copy=False. May also be None, in which case rootdir
        must be provided (disk-based array).
    copy : bool, optional
        If True, copy the input data into a new bcolz ctable.
    index : string or pair of strings, optional
        If a single string, name of column to use for a sorted index. If a
        pair of strings, name of columns to use for a sorted multi-index.
    **kwargs : keyword arguments
        Passed through to the bcolz ctable constructor.

    Examples
    --------

    Instantiate from existing data::

        >>> import allel
        >>> chrom = [b'chr1', b'chr1', b'chr2', b'chr2', b'chr3']
        >>> pos = [2, 7, 3, 9, 6]
        >>> dp = [35, 12, 78, 22, 99]
        >>> qd = [4.5, 6.7, 1.2, 4.4, 2.8]
        >>> ac = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]
        >>> vt = allel.VariantCTable([chrom, pos, dp, qd, ac],
        ...                           names=['CHROM', 'POS', 'DP', 'QD', 'AC'],
        ...                           index=('CHROM', 'POS'))
        >>> vt
        VariantCTable((5,), [('CHROM', 'S4'), ('POS', '<i8'), ('DP', '<i8'), ('QD', '<f8'), ('AC', '<i8', (2,))])
          nbytes: 220; cbytes: 80.00 KB; ratio: 0.00
          cparams := cparams(clevel=5, shuffle=1, cname='lz4', quantize=0)
        [(b'chr1', 2, 35, 4.5, [1, 2]) (b'chr1', 7, 12, 6.7, [3, 4])
         (b'chr2', 3, 78, 1.2, [5, 6]) (b'chr2', 9, 22, 4.4, [7, 8])
         (b'chr3', 6, 99, 2.8, [9, 10])]

    Slicing rows returns :class:`allel.model.ndarray.VariantTable`::

        >>> vt[:2]
        VariantTable((2,), dtype=(numpy.record, [('CHROM', 'S4'), ('POS', '<i8'), ('DP', '<i8'), ('QD', '<f8'), ('AC', '<i8', (2,))]))
        [(b'chr1', 2, 35, 4.5, array([1, 2])) (b'chr1', 7, 12, 6.7, array([3, 4]))]

    Accessing columns returns :class:`allel.model.bcolz.VariantCTable`::

        >>> vt[['DP', 'QD']]
        VariantCTable((5,), [('DP', '<i8'), ('QD', '<f8')])
          nbytes: 80; cbytes: 32.00 KB; ratio: 0.00
          cparams := cparams(clevel=5, shuffle=1, cname='lz4', quantize=0)
        [(35, 4.5) (12, 6.7) (78, 1.2) (22, 4.4) (99, 2.8)]

    Use the index to locate variants:

        >>> loc = vt.index.locate_range(b'chr2', 1, 10)
        >>> vt[loc]
        VariantTable((2,), dtype=(numpy.record, [('CHROM', 'S4'), ('POS', '<i8'), ('DP', '<i8'), ('QD', '<f8'), ('AC', '<i8', (2,))]))
        [(b'chr2', 3, 78, 1.2, array([5, 6])) (b'chr2', 9, 22, 4.4, array([7, 8]))]

    """  # noqa

    ndarray_cls = VariantTable

    def __init__(self, data=None, copy=False, index=None, **kwargs):
        if copy or not isinstance(data, bcolz.ctable):
            ctbl = bcolz.ctable(data, **kwargs)
        else:
            ctbl = data
        object.__setattr__(self, 'ctbl', ctbl)
        # initialise index
        self.set_index(index)

    def set_index(self, index):
        if index is None:
            pass
        elif isinstance(index, str):
            index = SortedIndex(self.ctbl[index][:], copy=False)
        elif isinstance(index, (tuple, list)) and len(index) == 2:
            index = SortedMultiIndex(self.ctbl[index[0]][:],
                                     self.ctbl[index[1]][:],
                                     copy=False)
        else:
            raise ValueError('invalid index argument, expected string or '
                             'pair of strings, found %s' % repr(index))
        object.__setattr__(self, 'index', index)

    @property
    def n_variants(self):
        return len(self.ctbl)

    def to_vcf(self, path, rename=None, number=None, description=None,
               fill=None, blen=None, write_header=True):
        with open(path, 'w') as vcf_file:
            if write_header:
                write_vcf_header(vcf_file, self, rename=rename, number=number,
                                 description=description)
            if blen is None:
                blen = min(self.ctbl[col].chunklen for col in self.ctbl.cols)
            for i in range(0, self.ctbl.shape[0], blen):
                block = self.ctbl[i:i+blen]
                write_vcf_data(vcf_file, block, rename=rename, fill=fill)


class FeatureCTable(CTableWrapper):
    """Alternative implementation of the
    :class:`allel.model.ndarray.FeatureTable` class, using a
    :class:`bcolz.ctable` as the backing store.

    Parameters
    ----------

    data : tuple or list of column objects, optional
        The list of column data to build the ctable object. This can also be a
        pure NumPy structured array. May also be a bcolz ctable, which will
        not be copied if copy=False. May also be None, in which case rootdir
        must be provided (disk-based array).
    copy : bool, optional
        If True, copy the input data into a new bcolz ctable.
    index : pair or triplet of strings, optional
        Names of columns to use for positional index, e.g., ('start',
        'stop') if table contains 'start' and 'stop' columns and records
        from a single chromosome/contig, or ('seqid', 'start', 'end') if table
        contains records from multiple chromosomes/contigs.
    **kwargs : keyword arguments
        Passed through to the bcolz ctable constructor.

    """

    ndarray_cls = FeatureTable

    def __init__(self, data=None, copy=False, **kwargs):
        if copy or not isinstance(data, bcolz.ctable):
            ctbl = bcolz.ctable(data, **kwargs)
        else:
            ctbl = data
        object.__setattr__(self, 'ctbl', ctbl)
        # TODO initialise interval index
        # self.set_index(index)

    @property
    def n_features(self):
        return len(self.ctbl)

    def to_mask(self, size, start_name='start', stop_name='end'):
        """Construct a mask array where elements are True if the fall within
        features in the table.

        Parameters
        ----------

        size : int
            Size of chromosome/contig.
        start_name : string, optional
            Name of column with start coordinates.
        stop_name : string, optional
            Name of column with stop coordinates.

        Returns
        -------

        mask : ndarray, bool

        """
        m = np.zeros(size, dtype=bool)
        for start, stop in self[[start_name, stop_name]]:
            m[start-1:stop] = True
        return m

    @staticmethod
    def from_gff3(path, attributes=None, region=None,
                  score_fill=-1, phase_fill=-1, attributes_fill=b'.',
                  dtype=None, **kwargs):
        """Read a feature table from a GFF3 format file.

        Parameters
        ----------

        path : string
            File path.
        attributes : list of strings, optional
            List of columns to extract from the "attributes" field.
        region : string, optional
            Genome region to extract. If given, file must be position
            sorted, bgzipped and tabix indexed. Tabix must also be installed
            and on the system path.
        score_fill : object, optional
            Value to use where score field has a missing value.
        phase_fill : object, optional
            Value to use where phase field has a missing value.
        attributes_fill : object or list of objects, optional
            Value(s) to use where attribute field(s) have a missing value.
        dtype : numpy dtype, optional
            Manually specify a dtype.

        Returns
        -------

        ft : FeatureCTable

        """

        # setup iterator
        recs = iter_gff3(path, attributes=attributes, region=region,
                         score_fill=score_fill, phase_fill=phase_fill,
                         attributes_fill=attributes_fill)

        # determine dtype from sample of initial records
        if dtype is None:
            names = 'seqid', 'source', 'type', 'start', 'end', 'score', \
                    'strand', 'phase'
            if attributes is not None:
                names += tuple(attributes)
            recs_sample = list(itertools.islice(recs, 1000))
            a = np.rec.array(recs_sample, names=names)
            dtype = a.dtype
            recs = itertools.chain(recs_sample, recs)

        # set ctable defaults
        kwargs.setdefault('expectedlen', 200000)

        # initialise ctable
        ctbl = bcolz.ctable(np.array([], dtype=dtype), **kwargs)

        # determine block size to read
        blen = min(ctbl[col].chunklen for col in ctbl.cols)

        # read block-wise
        block = list(itertools.islice(recs, 0, blen))
        while block:
            a = np.array(block, dtype=dtype)
            ctbl.append(a)
            block = list(itertools.islice(recs, 0, blen))

        ft = FeatureCTable(ctbl, copy=False)
        return ft


class AlleleCountsCTable(CTableWrapper):

    def __init__(self, data=None, copy=False, **kwargs):
        if copy or not isinstance(data, bcolz.ctable):
            ctbl = bcolz.ctable(data, **kwargs)
        else:
            ctbl = data
        object.__setattr__(self, 'ctbl', ctbl)

    def __getitem__(self, item):
        o = super(AlleleCountsCTable, self).__getitem__(item)
        if isinstance(o, bcolz.carray):
            return AlleleCountsCArray(o, copy=False)
        return o


def ctable_to_html_str(ctbl, limit=5, caption=None):
    ra = ctbl[:limit+1]
    return recarray_to_html_str(ra, limit=limit, caption=caption)


def ctable_display(ctbl, limit=5, caption=None, **kwargs):
    ra = ctbl[:limit+1]
    return recarray_display(ra, limit=limit, caption=caption, **kwargs)
