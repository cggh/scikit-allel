# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np


def jackknife(values, statistic):
    """Estimate standard error for `statistic` computed over `values` using
    the jackknife.

    Parameters
    ----------
    values : array_like or tuple of array_like
        Input array, or tuple of input arrays.
    statistic : function
        The statistic to compute.

    Returns
    -------
    m : float
        Mean of jackknife values.
    se : float
        Estimate of standard error.
    vj : ndarray
        Statistic values computed for each jackknife iteration.

    """

    if isinstance(values, tuple):
        # multiple input arrays
        n = len(values[0])
        masked_values = [np.ma.asarray(v) for v in values]
        for m in masked_values:
            assert m.ndim == 1, 'only 1D arrays supported'
            assert m.shape[0] == n, 'input arrays not of equal length'
            m.mask = np.zeros(m.shape, dtype=bool)

    else:
        n = len(values)
        masked_values = np.ma.asarray(values)
        assert masked_values.ndim == 1, 'only 1D arrays supported'
        masked_values.mask = np.zeros(masked_values.shape, dtype=bool)

    # values of the statistic calculated in each jackknife iteration
    vj = list()

    for i in range(n):

        if isinstance(values, tuple):
            # multiple input arrays
            for m in masked_values:
                m.mask[i] = True
            x = statistic(*masked_values)
            for m in masked_values:
                m.mask[i] = False

        else:
            masked_values.mask[i] = True
            x = statistic(masked_values)
            masked_values.mask[i] = False

        vj.append(x)

    # convert to array for convenience
    vj = np.array(vj)

    # compute mean of jackknife values
    m = vj.mean()

    # compute standard error
    sv = ((n - 1) / n) * np.sum((vj - m) ** 2)
    se = np.sqrt(sv)

    return m, se, vj
