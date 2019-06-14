# -*- coding: utf-8 -*-
def copy_method_doc(m, n):
    """Copy docstring from `n` to `m`."""
    m.__doc__ = n.__doc__


def memoryview_safe(x):
    """Make array safe to run in a Cython memoryview-based kernel. These
    kernels typically break down with the error ``ValueError: buffer source
    array is read-only`` when running in dask distributed.

    See Also
    --------
    https://github.com/dask/distributed/issues/1978
    https://github.com/cggh/scikit-allel/issues/206

    """
    if not x.flags.writeable:
        if not x.flags.owndata:
            x = x.copy(order='A')
        x.setflags(write=True)
    return x
