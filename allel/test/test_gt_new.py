# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


from nose.tools import assert_raises, assert_equal as eq
from allel.test.tools import assert_array_equal as aeq
import numpy as np


from allel.gt_new import GenotypeArray


haploid_data = [[0, 1, -1],
                [1, 1, -1],
                [2, -1, -1],
                [-1, -1, -1]]

diploid_data = [[[0, 0], [0, 1], [-1, -1]],
                [[0, 2], [1, 1], [-1, -1]],
                [[1, 0], [2, 1], [-1, -1]],
                [[2, 2], [-1, -1], [-1, -1]],
                [[-1, -1], [-1, -1], [-1, -1]]]

triploid_data = [[[0, 0, 0], [0, 0, 1], [-1, -1, -1]],
                 [[0, 1, 1], [1, 1, 1], [-1, -1, -1]],
                 [[0, 1, 2], [-1, -1, -1], [-1, -1, -1]],
                 [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]]


def test_constructor():

    # need to instantiate with data arg
    with assert_raises(TypeError):
        GenotypeArray()

    # data has wrong type
    data = 'foo bar'
    with assert_raises(TypeError):
        GenotypeArray(data)

    # data has wrong type
    data = [4., 5., 3.7]
    with assert_raises(TypeError):
        GenotypeArray(data)

    # data has wrong dimensions
    data = [1, 2, 3]
    with assert_raises(ValueError):
        GenotypeArray(data)

    # haploid data
    g = GenotypeArray(haploid_data)
    aeq(haploid_data, g)
    eq(np.int, g.dtype)
    eq(2, g.ndim)
    eq(1, g.ploidy)

    # haploid data (typed)
    g = GenotypeArray(np.array(haploid_data, dtype='i1'))
    aeq(haploid_data, g)
    eq(np.int8, g.dtype)

    # diploid data
    g = GenotypeArray(diploid_data)
    aeq(diploid_data, g)
    eq(np.int, g.dtype)
    eq(3, g.ndim)
    eq(2, g.ploidy)

    # diploid data (typed)
    g = GenotypeArray(np.array(diploid_data, dtype='i1'))
    aeq(diploid_data, g)
    eq(np.int8, g.dtype)

    # triploid data
    g = GenotypeArray(triploid_data)
    aeq(triploid_data, g)
    eq(np.int, g.dtype)
    eq(3, g.ndim)
    eq(3, g.ploidy)

    # triploid data (typed)
    g = GenotypeArray(np.array(triploid_data, dtype='i1'))
    aeq(triploid_data, g)
    eq(np.int8, g.dtype)


def test_slice():

    g = GenotypeArray(haploid_data)
    eq(1, g.ploidy)
    gs = g[:2]
    aeq(haploid_data[:2], gs)
    eq(1, gs.ploidy)

    g = GenotypeArray(diploid_data)
    eq(2, g.ploidy)
    gs = g[:2]
    aeq(diploid_data[:2], gs)
    eq(2, gs.ploidy)

    g = GenotypeArray(triploid_data)
    eq(3, g.ploidy)
    gs = g[:2]
    aeq(triploid_data[:2], gs)
    eq(3, gs.ploidy)


def test_view():

    g = np.array(haploid_data).view(GenotypeArray)
    eq(1, g.ploidy)
    aeq(haploid_data, g)

    # data has wrong type
    data = 'foo bar'
    with assert_raises(TypeError):
        np.array(data).view(GenotypeArray)

    # data has wrong type
    data = [4., 5., 3.7]
    with assert_raises(TypeError):
        np.array(data).view(GenotypeArray)

    # data has wrong dimensions
    data = [1, 2, 3]
    with assert_raises(ValueError):
        np.array(data).view(GenotypeArray)
