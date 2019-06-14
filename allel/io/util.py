# -*- coding: utf-8 -*-
import numpy as np


from allel.util import asarray_ndim


def array_to_hdf5(a, parent, name, **kwargs):
    """Write a Numpy array to an HDF5 dataset.

    Parameters
    ----------
    a : ndarray
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
        kwargs.setdefault('dtype', a.dtype)
        kwargs.setdefault('compression', 'gzip')
        h5d = parent.require_dataset(name, shape=a.shape, **kwargs)
        h5d[...] = a
        return h5d

    finally:
        if h5f is not None:
            h5f.close()


# noinspection PyIncorrectDocstring
def recarray_from_hdf5_group(*args, **kwargs):
    """Load a recarray from columns stored as separate datasets with an
    HDF5 group.

    Either provide an h5py group as a single positional argument,
    or provide two positional arguments giving the HDF5 file path and the
    group node path within the file.

    The following optional parameters may be given.

    Parameters
    ----------
    start : int, optional
        Index to start loading from.
    stop : int, optional
        Index to finish loading at.
    condition : array_like, bool, optional
        A 1-dimensional boolean array of the same length as the columns of the
        table to load, indicating a selection of rows to load.

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
        except Exception as e:
            h5f.close()
            raise e

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
        names = [str(n) for n in names]  # needed for PY2
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

        # check condition
        condition = kwargs.pop('condition', None)  # type: np.ndarray
        condition = asarray_ndim(condition, 1, allow_none=True)
        if condition is not None and condition.size != length:
            raise ValueError('length of condition does not match length '
                             'of datasets')

        # setup output data
        dtype = [(n, d.dtype, d.shape[1:]) for n, d in zip(names, datasets)]
        ra = np.empty(length, dtype=dtype)

        for n, d in zip(names, datasets):
            a = d[start:stop]
            if condition is not None:
                a = np.compress(condition[start:stop], a, axis=0)
            ra[n] = a

        return ra

    finally:
        if h5f is not None:
            h5f.close()


def recarray_to_hdf5_group(ra, parent, name, **kwargs):
    """Write each column in a recarray to a dataset in an HDF5 group.

    Parameters
    ----------
    ra : recarray
        Numpy recarray to store.
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
        for n in ra.dtype.names:
            array_to_hdf5(ra[n], h5g, n, **kwargs)

        return h5g

    finally:
        if h5f is not None:
            h5f.close()
