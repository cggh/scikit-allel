# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
from contextlib import contextmanager
from functools import update_wrapper
import atexit
import os


import numpy as np


from allel.compat import string_types


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
    """Ensure numpy array.

    Parameters
    ----------
    a : array_like
    *ndims : int, optional
        Allowed values for number of dimensions.
    **kwargs
        Passed through to :func:`numpy.array`.

    Returns
    -------
    a : numpy.ndarray

    """
    allow_none = kwargs.pop('allow_none', False)
    kwargs.setdefault('copy', False)
    if a is None and allow_none:
        return None
    a = np.array(a, **kwargs)
    if a.ndim not in ndims:
        if len(ndims) > 1:
            expect_str = 'one of %s' % str(ndims)
        else:
            # noinspection PyUnresolvedReferences
            expect_str = '%s' % ndims[0]
        raise TypeError('bad number of dimensions: expected %s; found %s' %
                        (expect_str, a.ndim))
    return a


def check_ndim(a, ndim):
    if a.ndim != ndim:
        raise TypeError('bad number of dimensions: expected %s; found %s' % (ndim, a.ndim))


def check_shape(a, shape):
    if a.shape != shape:
        raise TypeError('bad shape: expected %s; found %s' % (shape, a.shape))


def check_dtype(a, *dtypes):
    dtypes = [np.dtype(t) for t in dtypes]
    if a.dtype not in dtypes:
        raise TypeError('bad dtype: expected on of %s; found %s' % (dtypes, a.dtype))


def check_dtype_kind(a, *kinds):
    if a.dtype.kind not in kinds:
        raise TypeError('bad dtype kind: expected on of %s; found %s' % (kinds, a.dtype.kind))


def check_integer_dtype(a):
    check_dtype_kind(a, 'u', 'i')


def check_dim0_aligned(*arrays):
    check_dim_aligned(0, *arrays)


def check_dim_aligned(dim, *arrays):
    a = arrays[0]
    for b in arrays[1:]:
        if b.shape[dim] != a.shape[dim]:
            raise ValueError(
                'arrays do not have matching length for dimension %s' % dim
            )


def check_same_ndim(*arrays):
    a = arrays[0]
    for b in arrays[1:]:
        if len(b.shape) != len(a.shape):
            raise ValueError(
                'arrays do not have same number of dimensions'
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
    from scipy.spatial.distance import squareform
    dist = asarray_ndim(dist, 1, 2)
    if dist.ndim == 1:
        dist = squareform(dist)
    else:
        if dist.shape[0] != dist.shape[1]:
            raise ValueError('distance matrix is not square')
    return dist


class _HashedSeq(list):

    __slots__ = 'hashvalue'

    # noinspection PyShadowingBuiltins,PyMissingConstructor
    def __init__(self, tup, hash=hash):
        self[:] = tup
        self.hashvalue = hash(tup)

    def __hash__(self):
        return self.hashvalue


# noinspection PyShadowingBuiltins
def _make_key(args, kwds, typed,
              kwd_mark=('__kwargs__',),
              fasttypes={int, str, frozenset, type(None)},
              sorted=sorted, tuple=tuple, type=type, len=len):
    key = args
    kwd_items = sorted(kwds.items())
    if kwds:
        key += kwd_mark
        for item in kwd_items:
            key += item
    if typed:
        key += tuple(type(v) for v in args)
        if kwds:
            key += tuple(type(v) for _, v in kwd_items)
    else:
        key = args
    if len(key) == 1 and type(key[0]) in fasttypes:
        return key[0]
    return _HashedSeq(key)


def _hdf5_cache_act(filepath, parent, container, key, names, no_cache,
                    user_function, args, kwargs, h5dcreate_kwargs):
    import h5py

    # open the HDF5 file
    with h5py.File(filepath, mode='a') as h5f:

        # find parent group
        if parent is None:
            # use root group
            h5g_parent = h5f
        else:
            h5g_parent = h5f.require_group(parent)

        # find cache container group
        h5g_container = h5g_parent.require_group(container)

        # find cache group
        h5g = h5g_container.require_group(key)

        # call user function and (re)build cache
        if no_cache or '__success__' not in h5g.attrs:

            # reset success mark if present
            if '__success__' in h5g.attrs:
                del h5g.attrs['__success__']

            # compute result
            result = user_function(*args, **kwargs)

            # handle tuple of return values
            if isinstance(result, tuple):

                # determine dataset names
                if names is None:
                    names = ['f%02d' % i for i in range(len(result))]
                elif len(names) < len(result):
                    names = list(names) + ['f%02d' % i
                                           for i in range(len(names),
                                                          len(result))]

                # save data
                for n, r in zip(names, result):
                    if n in h5g:
                        del h5g[n]
                    if np.isscalar(r):
                        h5g.create_dataset(n, data=r)
                    else:
                        h5g.create_dataset(n, data=r, **h5dcreate_kwargs)

            # handle single return value
            else:

                # determine dataset name
                if names is None:
                    n = 'data'
                elif isinstance(names, string_types):
                    n = names
                elif len(names) > 0:
                    n = names[0]
                else:
                    n = 'data'

                # save data
                if n in h5g:
                    del h5g[n]
                if np.isscalar(result):
                    h5g.create_dataset(n, data=result)
                else:
                    h5g.create_dataset(n, data=result,
                                       **h5dcreate_kwargs)

            # mark success
            h5g.attrs['__success__'] = True

        # load from cache
        else:

            # determine dataset names
            if names is None:
                names = sorted(h5g.keys())
            elif isinstance(names, string_types):
                names = (names,)

            # load result from cache
            if len(names) == 1:
                result = h5g[names[0]]
                result = result[:] if len(result.shape) > 0 else result[()]
            else:
                result = tuple(h5g[n] for n in names)
                result = tuple(r[:] if len(r.shape) > 0 else r[()]
                               for r in result)

        return result


def hdf5_cache(filepath=None, parent=None, group=None, names=None, typed=False,
               hashed_key=False, **h5dcreate_kwargs):
    """HDF5 cache decorator.

    Parameters
    ----------
    filepath : string, optional
        Path to HDF5 file. If None a temporary file name will be used.
    parent : string, optional
        Path to group within HDF5 file to use as parent. If None the root
        group will be used.
    group : string, optional
        Path to group within HDF5 file, relative to parent, to use as
        container for cached data. If None the name of the wrapped function
        will be used.
    names : sequence of strings, optional
        Name(s) of dataset(s). If None, default names will be 'f00', 'f01',
        etc.
    typed : bool, optional
        If True, arguments of different types will be cached separately.
        For example, f(3.0) and f(3) will be treated as distinct calls with
        distinct results.
    hashed_key : bool, optional
        If False (default) the key will not be hashed, which makes for
        readable cache group names. If True the key will be hashed, however
        note that on Python >= 3.3 the hash value will not be the same between
        sessions unless the environment variable PYTHONHASHSEED has been set
        to the same value.

    Returns
    -------
    decorator : function

    Examples
    --------

    Without any arguments, will cache using a temporary HDF5 file::

        >>> import allel
        >>> @allel.util.hdf5_cache()
        ... def foo(n):
        ...     print('executing foo')
        ...     return np.arange(n)
        ...
        >>> foo(3)
        executing foo
        array([0, 1, 2])
        >>> foo(3)
        array([0, 1, 2])
        >>> foo.cache_filepath # doctest: +SKIP
        '/tmp/tmp_jwtwgjz'

    Supports multiple return values, including scalars, e.g.::

        >>> @allel.util.hdf5_cache()
        ... def bar(n):
        ...     print('executing bar')
        ...     a = np.arange(n)
        ...     return a, a**2, n**2
        ...
        >>> bar(3)
        executing bar
        (array([0, 1, 2]), array([0, 1, 4]), 9)
        >>> bar(3)
        (array([0, 1, 2]), array([0, 1, 4]), 9)

    Names can also be specified for the datasets, e.g.::

        >>> @allel.util.hdf5_cache(names=['z', 'x', 'y'])
        ... def baz(n):
        ...     print('executing baz')
        ...     a = np.arange(n)
        ...     return a, a**2, n**2
        ...
        >>> baz(3)
        executing baz
        (array([0, 1, 2]), array([0, 1, 4]), 9)
        >>> baz(3)
        (array([0, 1, 2]), array([0, 1, 4]), 9)

    """

    # initialise HDF5 file path
    if filepath is None:
        import tempfile
        filepath = tempfile.mktemp(prefix='scikit_allel_', suffix='.h5')
        atexit.register(os.remove, filepath)

    # initialise defaults for dataset creation
    h5dcreate_kwargs.setdefault('chunks', True)

    def decorator(user_function):

        # setup the name for the cache container group
        if group is None:
            container = user_function.__name__
        else:
            container = group

        def wrapper(*args, **kwargs):

            # load from cache or not
            no_cache = kwargs.pop('no_cache', False)

            # compute a key from the function arguments
            key = _make_key(args, kwargs, typed)
            if hashed_key:
                key = str(hash(key))
            else:
                key = str(key).replace('/', '__slash__')

            return _hdf5_cache_act(filepath, parent, container, key, names,
                                   no_cache, user_function, args, kwargs,
                                   h5dcreate_kwargs)

        wrapper.cache_filepath = filepath
        return update_wrapper(wrapper, user_function)

    return decorator


def contains_newaxis(item):
    if item is None:
        return True
    elif item is np.newaxis:
        return True
    elif isinstance(item, tuple):
        return any((i is None or i is np.newaxis) for i in item)
    return False


def check_ploidy(actual, expect):
    if expect != actual:
        raise ValueError(
            'expected ploidy %s, found %s' % (expect, actual)
        )


def check_min_samples(actual, expect):
    if actual < expect:
        raise ValueError(
            'expected at least %s samples, found %s' % (expect, actual)
        )


def check_type(obj, expected):
    if not isinstance(obj, expected):
        raise TypeError('bad argument type, expected %s, found %s' % (expected, type(obj)))
