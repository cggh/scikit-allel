# -*- coding: utf-8 -*-
import warnings


import pytest
import numpy as np
from numpy.testing import assert_array_equal


import allel


# needed for PY2/PY3 consistent behaviour
warnings.resetwarnings()
warnings.simplefilter('always')


def test_sfs():
    dac = [0, 1, 2, 1]
    expect = [1, 2, 1]
    actual = allel.sfs(dac)
    assert_array_equal(expect, actual)
    for dtype in 'u2', 'i2', 'u8', 'i8':
        daca = np.asarray(dac, dtype=dtype)
        actual = allel.sfs(daca)
        assert_array_equal(expect, actual)
    # explicitly provide number of chromosomes
    expect = [1, 2, 1, 0]
    actual = allel.sfs(dac, n=3)
    assert_array_equal(expect, actual)
    with pytest.raises(ValueError):
        allel.sfs(dac, n=1)


def test_sfs_folded():
    ac = [[0, 3], [1, 2], [2, 1]]
    expect = [1, 2]
    actual = allel.sfs_folded(ac)
    assert_array_equal(expect, actual)
    for dtype in 'u2', 'i2', 'u8', 'i8':
        aca = np.asarray(ac, dtype=dtype)
        actual = allel.sfs_folded(aca)
        assert_array_equal(expect, actual)


def test_sfs_scaled():
    dac = [0, 1, 2, 1]
    expect = [0, 2, 2]
    actual = allel.sfs_scaled(dac)
    assert_array_equal(expect, actual)
    for dtype in 'u2', 'i2', 'u8', 'i8':
        daca = np.asarray(dac, dtype=dtype)
        actual = allel.sfs_scaled(daca)
        assert_array_equal(expect, actual)
    # explicitly provide number of chromosomes
    expect = [0, 2, 2, 0]
    actual = allel.sfs_scaled(dac, n=3)
    assert_array_equal(expect, actual)
    with pytest.raises(ValueError):
        allel.sfs_scaled(dac, n=1)


def test_joint_sfs():
    # https://github.com/cggh/scikit-allel/issues/144

    warnings.resetwarnings()
    warnings.simplefilter('error')

    dac1 = np.array([0, 1, 2, 3, 4])
    dac2 = np.array([1, 2, 1, 2, 3], dtype='u8')
    s = allel.joint_sfs(dac1, dac2)
    e = [[0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]]
    assert_array_equal(e, s)

    warnings.resetwarnings()
    warnings.simplefilter('always')


def test_joint_sfs_folded():
    # https://github.com/cggh/scikit-allel/issues/144

    warnings.resetwarnings()
    warnings.simplefilter('error')

    ac1 = np.array([[0, 8], [1, 7], [2, 6], [3, 4], [4, 4]])
    ac2 = np.array([[1, 5], [2, 4], [1, 5], [2, 3], [3, 3]], dtype='u8')
    s = allel.joint_sfs_folded(ac1, ac2)
    e = [[0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]]
    assert_array_equal(e, s)

    warnings.resetwarnings()
    warnings.simplefilter('always')
