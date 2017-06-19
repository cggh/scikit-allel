# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import warnings


from allel import joint_sfs, joint_sfs_folded
import numpy as np
from numpy.testing import assert_array_equal


# needed for PY2/PY3 consistent behaviour
warnings.resetwarnings()
warnings.simplefilter('always')


def test_joint_sfs():
    # https://github.com/cggh/scikit-allel/issues/144

    warnings.resetwarnings()
    warnings.simplefilter('error')

    dac1 = np.array([0, 1, 2, 3, 4])
    dac2 = np.array([1, 2, 1, 2, 3], dtype='u8')
    s = joint_sfs(dac1, dac2)
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
    s = joint_sfs_folded(ac1, ac2)
    e = [[0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]]
    assert_array_equal(e, s)

    warnings.resetwarnings()
    warnings.simplefilter('always')
