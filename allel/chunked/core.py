# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import operator
from collections import namedtuple


import numpy as np


from allel.compat import string_types, integer_types, range
from allel.chunked import util as _util
from allel.abc import ArrayWrapper, DisplayAsTable
from allel.model.ndarray import subset as _numpy_subset, NumpyRecArrayWrapper


def store(data, arr, start=0, stop=None, offset=0, blen=None):
    """Copy `data` block-wise into `arr`."""

    # setup
    blen = _util.get_blen_array(data, blen)
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


def copy(data, start=0, stop=None, blen=None, storage=None, create='array',
         **kwargs):
    """Copy `data` block-wise into a new array."""

    # setup
    storage = _util.get_storage(storage)
    blen = _util.get_blen_array(data, blen)
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
        block = data[i:j]
        if out is None:
            out = getattr(storage, create)(block, expectedlen=length, **kwargs)
        else:
            out.append(block)

    return out


def copy_table(tbl, start=0, stop=None, blen=None, storage=None,
               create='table', **kwargs):
    """Copy `tbl` block-wise into a new table."""

    # setup
    names, columns = _util.check_table_like(tbl)
    storage = _util.get_storage(storage)
    blen = _util.get_blen_table(tbl, blen)
    if stop is None:
        stop = len(columns[0])
    else:
        stop = min(stop, len(columns[0]))
    length = stop - start
    if length < 0:
        raise ValueError('invalid stop/start')

    # copy block-wise
    out = None
    for i in range(start, stop, blen):
        j = min(i+blen, stop)
        res = [c[i:j] for c in columns]
        if out is None:
            out = getattr(storage, create)(res, names=names,
                                           expectedlen=length, **kwargs)
        else:
            out.append(res)

    return out


def apply(data, f, blen=None, storage=None, create='array', **kwargs):
    """Apply function `f` block-wise over `data`."""

    # setup
    storage = _util.get_storage(storage)
    if isinstance(data, tuple):
        blen = max(_util.get_blen_array(d, blen) for d in data)
    else:
        blen = _util.get_blen_array(data, blen)
    if isinstance(data, tuple):
        _util.check_equal_length(*data)
        length = len(data[0])
    else:
        length = len(data)

    # block-wise iteration
    out = None
    for i in range(0, length, blen):
        j = min(i+blen, length)

        # obtain blocks
        if isinstance(data, tuple):
            blocks = [d[i:j] for d in data]
        else:
            blocks = [data[i:j]]

        # map
        res = f(*blocks)

        # store
        if out is None:
            out = getattr(storage, create)(res, expectedlen=length, **kwargs)
        else:
            out.append(res)

    return out


def reduce_axis(data, reducer, block_reducer, mapper=None, axis=None,
                blen=None, storage=None, create='array', **kwargs):
    """Apply an operation to `data` that reduces over one or more axes."""

    # setup
    storage = _util.get_storage(storage)
    blen = _util.get_blen_array(data, blen)
    length = len(data)
    # normalise axis arg
    if isinstance(axis, int):
        axis = (axis,)

    # deal with 'out' kwarg if supplied, can arise if a chunked array is
    # passed as an argument to numpy.sum(), see also
    # https://github.com/cggh/scikit-allel/issues/66
    kwarg_out = kwargs.pop('out', None)
    if kwarg_out is not None:
        raise ValueError('keyword argument "out" is not supported')

    if axis is None or 0 in axis:
        # two-step reduction
        out = None
        for i in range(0, length, blen):
            j = min(i+blen, length)
            block = data[i:j]
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
            return getattr(storage, create)(out, **kwargs)

    else:
        # first dimension is preserved, no need to reduce blocks
        out = None
        for i in range(0, length, blen):
            j = min(i+blen, length)
            block = data[i:j]
            if mapper:
                block = mapper(block)
            r = reducer(block, axis=axis)
            if out is None:
                out = getattr(storage, create)(r, expectedlen=length, **kwargs)
            else:
                out.append(r)
        return out


def amax(data, axis=None, mapper=None, blen=None, storage=None,
         create='array', **kwargs):
    """Compute the maximum value."""
    return reduce_axis(data, axis=axis, reducer=np.amax,
                       block_reducer=np.maximum, mapper=mapper,
                       blen=blen, storage=storage, create=create, **kwargs)


def amin(data, axis=None, mapper=None, blen=None, storage=None,
         create='array', **kwargs):
    """Compute the minimum value."""
    return reduce_axis(data, axis=axis, reducer=np.amin,
                       block_reducer=np.minimum, mapper=mapper,
                       blen=blen, storage=storage, create=create, **kwargs)


# noinspection PyShadowingBuiltins
def asum(data, axis=None, mapper=None, blen=None, storage=None,
         create='array', **kwargs):
    """Compute the sum."""
    return reduce_axis(data, axis=axis, reducer=np.sum,
                       block_reducer=np.add, mapper=mapper,
                       blen=blen, storage=storage, create=create, **kwargs)


def count_nonzero(data, mapper=None, blen=None, storage=None,
                  create='array', **kwargs):
    """Count the number of non-zero elements."""
    return reduce_axis(data, reducer=np.count_nonzero,
                       block_reducer=np.add, mapper=mapper,
                       blen=blen, storage=storage, create=create, **kwargs)


def compress(condition, data, axis=0, blen=None, storage=None, create='array', **kwargs):
    """Return selected slices of an array along given axis."""

    # setup
    storage = _util.get_storage(storage)
    blen = _util.get_blen_array(data, blen)
    length = len(data)
    nnz = count_nonzero(condition)

    if axis == 0:
        _util.check_equal_length(data, condition)

        # block iteration
        out = None
        for i in range(0, length, blen):
            j = min(i+blen, length)
            bcond = np.asarray(condition[i:j])
            # don't access any data unless we have to
            if np.any(bcond):
                block = np.asarray(data[i:j])
                res = np.compress(bcond, block, axis=0)
                if out is None:
                    out = getattr(storage, create)(res, expectedlen=nnz,
                                                   **kwargs)
                else:
                    out.append(res)
        return out

    elif axis == 1:

        # block iteration
        out = None
        condition = np.asanyarray(condition)
        for i in range(0, length, blen):
            j = min(i+blen, length)
            block = np.asarray(data[i:j])
            res = np.compress(condition, block, axis=1)
            if out is None:
                out = getattr(storage, create)(res, expectedlen=length,
                                               **kwargs)
            else:
                out.append(res)

        return out

    else:
        raise NotImplementedError('axis not supported: %s' % axis)


def take(data, indices, axis=0, blen=None, storage=None,
         create='array', **kwargs):
    """Take elements from an array along an axis."""

    # setup
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
        return compress(condition, data, axis=0, blen=blen, storage=storage,
                        create=create, **kwargs)

    elif axis == 1:

        # setup
        storage = _util.get_storage(storage)
        blen = _util.get_blen_array(data, blen)

        # block iteration
        out = None
        for i in range(0, length, blen):
            j = min(i+blen, length)
            block = data[i:j]
            res = np.take(block, indices, axis=1)
            if out is None:
                out = getattr(storage, create)(res, expectedlen=length,
                                               **kwargs)
            else:
                out.append(res)
        return out

    else:
        raise NotImplementedError('axis not supported: %s' % axis)


def compress_table(condition, tbl, blen=None, storage=None, create='table', **kwargs):
    """Return selected rows of a table."""

    # setup
    storage = _util.get_storage(storage)
    names, columns = _util.check_table_like(tbl)
    blen = _util.get_blen_table(tbl, blen)
    _util.check_equal_length(columns[0], condition)
    length = len(columns[0])
    nnz = count_nonzero(condition)

    # block iteration
    out = None
    for i in range(0, length, blen):
        j = min(i+blen, length)
        bcond = condition[i:j]
        # don't access any data unless we have to
        if np.any(bcond):
            bcolumns = [c[i:j] for c in columns]
            res = [np.compress(bcond, c, axis=0) for c in bcolumns]
            if out is None:
                out = getattr(storage, create)(res, names=names,
                                               expectedlen=nnz, **kwargs)
            else:
                out.append(res)
    return out


def take_table(tbl, indices, blen=None, storage=None, create='table',
               **kwargs):
    """Return selected rows of a table."""

    # setup
    names, columns = _util.check_table_like(tbl)
    length = len(columns[0])

    # check that indices are strictly increasing
    indices = np.asanyarray(indices)
    if np.any(indices[1:] <= indices[:-1]):
        raise NotImplementedError(
            'indices must be strictly increasing'
        )

    # implement via compress()
    condition = np.zeros((length,), dtype=bool)
    condition[indices] = True
    return compress_table(condition, tbl, blen=blen, storage=storage,
                          create=create, **kwargs)


def subset(data, sel0=None, sel1=None, blen=None, storage=None, create='array',
           **kwargs):
    """Return selected rows and columns of an array."""

    # TODO refactor sel0 and sel1 normalization with ndarray.subset

    # setup
    storage = _util.get_storage(storage)
    blen = _util.get_blen_array(data, blen)
    length = len(data)
    if sel0 is not None:
        sel0 = np.asanyarray(sel0)
    if sel1 is not None:
        sel1 = np.asanyarray(sel1)

    # ensure boolean array for dim 0
    if sel0 is not None and sel0.dtype.kind != 'b':
        # assume indices, convert to boolean condition
        tmp = np.zeros(length, dtype=bool)
        tmp[sel0] = True
        sel0 = tmp

    # ensure indices for dim 1
    if sel1 is not None and sel1.dtype.kind == 'b':
        # assume boolean condition, convert to indices
        sel1, = np.nonzero(sel1)

    # shortcuts
    if sel0 is None and sel1 is None:
        return copy(data, blen=blen, storage=storage, create=create, **kwargs)
    elif sel1 is None:
        return compress(sel0, data, axis=0, blen=blen, storage=storage,
                        create=create, **kwargs)
    elif sel0 is None:
        return take(data, sel1, axis=1, blen=blen, storage=storage,
                    create=create, **kwargs)

    # build output
    sel0_nnz = count_nonzero(sel0)
    out = None
    for i in range(0, length, blen):
        j = min(i+blen, length)
        bsel0 = sel0[i:j]
        # don't access data unless we have to
        if np.any(bsel0):
            block = data[i:j]
            res = _numpy_subset(block, bsel0, sel1)
            if out is None:
                out = getattr(storage, create)(res, expectedlen=sel0_nnz,
                                               **kwargs)
            else:
                out.append(res)

    return out


def concatenate_table(tup, blen=None, storage=None, create='table', **kwargs):
    """Stack tables in sequence vertically (row-wise)."""

    # setup
    storage = _util.get_storage(storage)
    if not isinstance(tup, (tuple, list)):
        raise ValueError('expected tuple or list, found %r' % tup)
    if len(tup) < 2:
        raise ValueError('expected two or more tables to stack')

    # build output
    expectedlen = sum(len(t) for t in tup)
    out = None
    tnames = None
    for tdata in tup:
        tblen = _util.get_blen_table(tdata, blen)
        tnames, tcolumns = _util.check_table_like(tdata, names=tnames)
        tlen = len(tcolumns[0])
        for i in range(0, tlen, tblen):
            j = min(i+tblen, tlen)
            bcolumns = [c[i:j] for c in tcolumns]
            if out is None:
                out = getattr(storage, create)(bcolumns, names=tnames,
                                               expectedlen=expectedlen,
                                               **kwargs)
            else:
                out.append(bcolumns)
    return out


def concatenate(tup, axis=0, blen=None, storage=None, create='array', **kwargs):
    """Concatenate arrays."""

    # setup
    storage = _util.get_storage(storage)
    if not isinstance(tup, (tuple, list)):
        raise ValueError('expected tuple or list, found %r' % tup)
    if len(tup) < 2:
        raise ValueError('expected two or more arrays')

    if axis == 0:

        # build output
        expectedlen = sum(len(a) for a in tup)
        out = None
        for a in tup:
            ablen = _util.get_blen_array(a, blen)
            for i in range(0, len(a), ablen):
                j = min(i+ablen, len(a))
                block = a[i:j]
                if out is None:
                    out = getattr(storage, create)(block, expectedlen=expectedlen, **kwargs)
                else:
                    out.append(block)

    else:

        def f(*blocks):
            return np.concatenate(blocks, axis=axis)

        out = apply(tup, f, blen=blen, storage=storage, create=create, **kwargs)

    return out


def binary_op(data, op, other, blen=None, storage=None, create='array',
              **kwargs):
    """Compute a binary operation block-wise over `data`."""

    # normalise scalars
    if hasattr(other, 'shape') and len(other.shape) == 0:
        other = other[()]

    if np.isscalar(other):
        def f(block):
            return op(block, other)
        return apply(data, f, blen=blen, storage=storage, create=create,
                     **kwargs)

    elif len(data) == len(other):
        def f(a, b):
            return op(a, b)
        return apply((data, other), f, blen=blen, storage=storage,
                     create=create, **kwargs)

    else:
        raise NotImplementedError('argument type not supported')


# based on bcolz.chunked_eval
def _get_expression_variables(expression, vm):
    cexpr = compile(expression, '<string>', 'eval')
    if vm == 'numexpr':
        # Check that var is not a numexpr function here.  This is useful for
        # detecting unbound variables in expressions.  This is not necessary
        # for the 'python' engine.
        from numexpr.expressions import functions as numexpr_functions
        return [var for var in cexpr.co_names
                if var not in ['None', 'False', 'True'] and
                var not in numexpr_functions]
    else:
        return [var for var in cexpr.co_names
                if var not in ['None', 'False', 'True']]


# based on bcolz.chunked_eval
def eval_table(tbl, expression, vm='python', blen=None, storage=None,
               create='array', vm_kwargs=None, **kwargs):
    """Evaluate `expression` against columns of a table."""

    # setup
    storage = _util.get_storage(storage)
    names, columns = _util.check_table_like(tbl)
    length = len(columns[0])
    if vm_kwargs is None:
        vm_kwargs = dict()

    # setup vm
    if vm == 'numexpr':
        import numexpr
        evaluate = numexpr.evaluate
    elif vm == 'python':
        # noinspection PyUnusedLocal
        def evaluate(expr, local_dict=None, **kw):
            # takes no keyword arguments
            return eval(expr, dict(), local_dict)
    else:
        raise ValueError('expected vm either "numexpr" or "python"')

    # compile expression and get required columns
    variables = _get_expression_variables(expression, vm)
    required_columns = {v: columns[names.index(v)] for v in variables}

    # determine block size for evaluation
    blen = _util.get_blen_table(required_columns, blen=blen)

    # build output
    out = None
    for i in range(0, length, blen):
        j = min(i+blen, length)
        blocals = {v: c[i:j] for v, c in required_columns.items()}
        res = evaluate(expression, local_dict=blocals, **vm_kwargs)
        if out is None:
            out = getattr(storage, create)(res, expectedlen=length, **kwargs)
        else:
            out.append(res)

    return out


class ChunkedArrayWrapper(ArrayWrapper):
    """Wrapper class for chunked array-like data.

    Parameters
    ----------
    data : array_like
        Data to be wrapped. May be a bcolz carray, h5py dataset, or
        anything providing a similar interface.

    """

    def __init__(self, data):
        data = _util.ensure_array_like(data)
        super(ChunkedArrayWrapper, self).__init__(data)

    @property
    def caption(self):
        r = '<%s' % type(self).__name__
        r += ' shape=%s' % str(self.shape)
        r += ' dtype=%s' % str(self.dtype)
        if self.chunks is not None:
            r += ' chunks=%s' % str(self.chunks)
        if self.nbytes:
            r += '\n   nbytes=%s' % _util.human_readable_size(self.nbytes)
            if self.cbytes:
                r += ' cbytes=%s' % _util.human_readable_size(self.cbytes)
            if self.cratio:
                r += ' cratio=%.1f' % self.cratio
        if self.compression:
            r += '\n   compression=%s' % self.compression
            if self.compression_opts is not None:
                r += ' compression_opts=%s' % self.compression_opts
        values_cls = type(self.values)
        r += '\n   values=%s.%s' % (values_cls.__module__, values_cls.__name__)
        r += '>'
        return r

    def __repr__(self):
        return self.caption

    @property
    def nbytes(self):
        return _util.get_nbytes(self.values)

    @property
    def cbytes(self):
        return _util.get_cbytes(self.values)

    @property
    def compression(self):
        return _util.get_compression(self.values)

    @property
    def compression_opts(self):
        return _util.get_compression_opts(self.values)

    @property
    def shuffle(self):
        return _util.get_shuffle(self.values)

    @property
    def chunks(self):
        return _util.get_chunks(self.values)

    @property
    def cratio(self):
        nbytes = self.nbytes
        cbytes = self.cbytes
        if nbytes and cbytes:
            return nbytes / cbytes
        return None

    # outputs from these methods are not wrapped
    store = store
    reduce_axis = reduce_axis
    max = amax
    min = amin
    sum = asum
    count_nonzero = count_nonzero

    def apply(self, f, blen=None, storage=None, create='array', **kwargs):
        out = apply(self, f, blen=blen, storage=storage, create=create, **kwargs)
        # don't wrap, leave this up to user
        return out

    def apply_method(self, method_name, kwargs=None, **storage_kwargs):
        if kwargs is None:
            kwargs = dict()

        def f(block):
            method = getattr(block, method_name)
            return method(**kwargs)

        out = self.apply(f, **storage_kwargs)
        # don't wrap, leave this up to user
        return out

    def copy(self, start=0, stop=None, blen=None, storage=None, create='array',
             **kwargs):
        out = copy(self, start=start, stop=stop, blen=blen, storage=storage,
                   create=create, **kwargs)
        # can always wrap this
        return type(self)(out)

    def binary_op(self, op, other, blen=None, storage=None, create='array',
                  **kwargs):
        out = binary_op(self, op, other, blen=blen, storage=storage,
                        create=create, **kwargs)
        return out

    def __eq__(self, other, **kwargs):
        return self.binary_op(operator.eq, other, **kwargs)

    def __ne__(self, other, **kwargs):
        return self.binary_op(operator.ne, other, **kwargs)

    def __lt__(self, other, **kwargs):
        return self.binary_op(operator.lt, other, **kwargs)

    def __gt__(self, other, **kwargs):
        return self.binary_op(operator.gt, other, **kwargs)

    def __le__(self, other, **kwargs):
        return self.binary_op(operator.le, other, **kwargs)

    def __ge__(self, other, **kwargs):
        return self.binary_op(operator.ge, other, **kwargs)

    def __add__(self, other, **kwargs):
        return self.binary_op(operator.add, other, **kwargs)

    def __floordiv__(self, other, **kwargs):
        return self.binary_op(operator.floordiv, other, **kwargs)

    def __mod__(self, other, **kwargs):
        return self.binary_op(operator.mod, other, **kwargs)

    def __mul__(self, other, **kwargs):
        return self.binary_op(operator.mul, other, **kwargs)

    def __pow__(self, other, **kwargs):
        return self.binary_op(operator.pow, other, **kwargs)

    def __sub__(self, other, **kwargs):
        return self.binary_op(operator.sub, other, **kwargs)

    def __truediv__(self, other, **kwargs):
        return self.binary_op(operator.truediv, other, **kwargs)

    def __and__(self, other, **kwargs):
        return self.binary_op(operator.and_, other, **kwargs)

    def __lshift__(self, other, **kwargs):
        return self.binary_op(operator.lshift, other, **kwargs)

    def __or__(self, other, **kwargs):
        return self.binary_op(operator.or_, other, **kwargs)

    def __rshift__(self, other, **kwargs):
        return self.binary_op(operator.rshift, other, **kwargs)

    def __xor__(self, other, **kwargs):
        return self.binary_op(operator.xor, other, **kwargs)


class ChunkedTableWrapper(DisplayAsTable):
    """Wrapper class for chunked table-like data.

    Parameters
    ----------
    data: table_like
        Data to be wrapped. May be a tuple or list of columns (array-like),
        a dict mapping names to columns, a bcolz ctable, h5py group,
        numpy recarray, or anything providing a similar interface.
    names : sequence of strings
        Column names.

    """

    array_cls = NumpyRecArrayWrapper

    # noinspection PyMissingConstructor
    def __init__(self, data, names=None):
        names, columns = _util.check_table_like(data, names=names)
        # skip super-class constructor because we are more flexible about type of values here
        self._values = data
        self._names = names
        self._columns = columns
        self.rowcls = namedtuple('row', names)

    @property
    def names(self):
        return self._names

    @property
    def columns(self):
        return self._columns

    def __getitem__(self, item):

        if isinstance(item, string_types):
            # item is column name, return column
            idx = self._names.index(item)
            return ChunkedArrayWrapper(self._columns[idx])

        elif isinstance(item, integer_types):
            # item is row index, return row
            return self.rowcls(*(col[item] for col in self._columns))

        elif isinstance(item, slice):
            # item is row slice, return numpy recarray
            start = 0 if item.start is None else item.start
            if start < 0:
                raise ValueError('negative indices not supported')
            stop = len(self) if item.stop is None else item.stop
            stop = min(stop, len(self))
            step = 1 if item.step is None else item.step
            outshape = (stop - start) // step
            out = np.empty(outshape, dtype=self.dtype)
            for n, c in zip(self._names, self._columns):
                out[n] = c[start:stop:step]
            out = out.view(np.recarray)
            if self.array_cls is not None:
                out = self.array_cls(out)
            return out

        elif isinstance(item, (list, tuple)) and \
                all(isinstance(i, string_types) for i in item):
            # item is sequence of column names, return table
            columns = [self._columns[self._names.index(n)] for n in item]
            return type(self)(columns, names=item)

        else:
            raise IndexError('item not supported for indexing: %s' % repr(item))

    def __array__(self, *args):
        a = np.asarray(self[:])
        if args:
            a = a.astype(args[0])
        return a

    def __getattr__(self, item):
        if item in self._names:
            idx = self._names.index(item)
            return ChunkedArrayWrapper(self._columns[idx])
        else:
            return super(ChunkedTableWrapper, self).__getattr__(item)

    @property
    def caption(self):
        r = '<%s' % type(self).__name__
        r += ' shape=%s' % str(self.shape)
        r += ' dtype=%s' % str(self.dtype)
        if self.nbytes:
            r += '\n   nbytes=%s' % _util.human_readable_size(self.nbytes)
            if self.cbytes:
                r += ' cbytes=%s' % _util.human_readable_size(self.cbytes)
            if self.cratio:
                r += ' cratio=%.1f' % self.cratio
        values_cls = type(self.values)
        r += '\n   values=%s.%s' % (values_cls.__module__, values_cls.__name__)
        r += '>'
        return r

    def __repr__(self):
        return self.caption

    def __len__(self):
        return len(self._columns[0])

    @property
    def shape(self):
        return len(self),

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def dtype(self):
        l = []
        for n, c in zip(self._names, self._columns):
            # need to account for multidimensional columns
            t = (n, c.dtype) if len(c.shape) == 1 else \
                (n, c.dtype, c.shape[1:])
            l.append(t)
        return np.dtype(l)

    @property
    def nbytes(self):
        cols_nbytes = [_util.get_nbytes(c) for c in self._columns]
        if all(cols_nbytes):
            return sum(cols_nbytes)
        return None

    @property
    def cbytes(self):
        cols_cbytes = [_util.get_cbytes(c) for c in self._columns]
        if all(cols_cbytes):
            return sum(cols_cbytes)
        return None

    @property
    def cratio(self):
        nbytes = self.nbytes
        cbytes = self.cbytes
        if nbytes and cbytes:
            return nbytes / cbytes
        return None

    def copy(self, start=0, stop=None, blen=None, storage=None,
             create='table', **kwargs):
        out = copy_table(self, start=start, stop=stop, blen=blen,
                         storage=storage, create=create, **kwargs)
        # can always wrap this
        return type(self)(out, names=self._names)

    def eval(self, expression, **kwargs):
        out = eval_table(self, expression, **kwargs)
        return ChunkedArrayWrapper(out)

    def query(self, expression, vm='python', blen=None, storage=None, create='table',
              vm_kwargs=None, **kwargs):
        condition = self.eval(expression, vm=vm, blen=blen, storage=storage, create='array',
                              vm_kwargs=vm_kwargs)
        out = self.compress(condition, blen=blen, storage=storage, create=create, **kwargs)
        # should already be wrapped
        return out

    def compress(self, condition, blen=None, storage=None, create='table', **kwargs):
        out = compress_table(condition, self, blen=blen, storage=storage, create=create,
                             **kwargs)
        return type(self)(out)

    def take(self, indices, blen=None, storage=None, create='table', **kwargs):
        out = take_table(self, indices, blen=blen, storage=storage, create=create,
                         **kwargs)
        return type(self)(out)
