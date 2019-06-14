# -*- coding: utf-8 -*-
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal


def assert_array_items_equal(expect, actual):

    assert expect.shape == actual.shape
    assert expect.dtype == actual.dtype

    # numpy asserts don't compare object arrays
    # properly; assert that we have the same nans
    # and values
    actual = actual.ravel().tolist()
    expect = expect.ravel().tolist()
    for a, r in zip(actual, expect):
        if isinstance(a, np.ndarray):
            assert_array_equal(a, r)
        elif a != a:
            assert r != r
        else:
            assert a == r


def compare_arrays(expected, actual):
    if expected.dtype.kind == 'f':
        assert_array_almost_equal(expected, actual)
    elif expected.dtype.kind == 'O':
        assert_array_items_equal(expected, actual)
    else:
        assert_array_equal(expected, actual)
