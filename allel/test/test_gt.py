# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np
import allel.gt


g_haploid = np.array([[0, 1],
                      [1, 1],
                      [2, -1]], dtype='i1')

g_diploid = np.array([[[0, 0], [0, 1]],
                      [[0, 2], [1, 1]],
                      [[1, 0], [2, 1]],
                      [[2, 2], [-1, -1]]], dtype='i1')

g_triploid = np.array([[[0, 0, 0], [0, 0, 1]],
                       [[0, 1, 1], [1, 1, 1]],
                       [[0, 1, 2], [-1, -1, -1]]], dtype='i1')


def test_is_called():

    # haploid
    expect = np.array([[1, 1],
                       [1, 1],
                       [1, 0]], dtype='b1')
    actual = allel.gt.is_called(g_haploid)
    assert np.array_equal(expect, actual), (expect, actual)

    # diploid
    expect = np.array([[1, 1],
                       [1, 1],
                       [1, 1],
                       [1, 0]], dtype='b1')
    actual = allel.gt.is_called(g_diploid)
    assert np.array_equal(expect, actual), (expect, actual)

    # polyploid
    expect = np.array([[1, 1],
                       [1, 1],
                       [1, 0]], dtype='b1')
    actual = allel.gt.is_called(g_triploid)
    assert np.array_equal(expect, actual), (expect, actual)


def test_is_missing():

    # haploid
    expect = np.array([[0, 0],
                       [0, 0],
                       [0, 1]], dtype='b1')
    actual = allel.gt.is_missing(g_haploid)
    assert np.array_equal(expect, actual), (expect, actual)

    # diploid
    expect = np.array([[0, 0],
                       [0, 0],
                       [0, 0],
                       [0, 1]], dtype='b1')
    actual = allel.gt.is_missing(g_diploid)
    assert np.array_equal(expect, actual), (expect, actual)

    # polyploid
    expect = np.array([[0, 0],
                       [0, 0],
                       [0, 1]], dtype='b1')
    actual = allel.gt.is_missing(g_triploid)
    assert np.array_equal(expect, actual), (expect, actual)


def test_is_hom():

    # haploid - trivially true
    expect = np.array([[1, 1],
                       [1, 1],
                       [1, 1]], dtype='b1')
    actual = allel.gt.is_hom(g_haploid)
    assert np.array_equal(expect, actual), (expect, actual)

    # diploid
    expect = np.array([[1, 0],
                       [0, 1],
                       [0, 0],
                       [1, 0]], dtype='b1')
    actual = allel.gt.is_hom(g_diploid)
    assert np.array_equal(expect, actual), (expect, actual)

    # polyploid
    expect = np.array([[1, 0],
                       [0, 1],
                       [0, 0]], dtype='b1')
    actual = allel.gt.is_hom(g_triploid)
    assert np.array_equal(expect, actual), (expect, actual)


def test_is_het():

    # haploid - trivially false
    expect = np.array([[0, 0],
                       [0, 0],
                       [0, 0]], dtype='b1')
    actual = allel.gt.is_het(g_haploid)
    assert np.array_equal(expect, actual), (expect, actual)

    # diploid
    expect = np.array([[0, 1],
                       [1, 0],
                       [1, 1],
                       [0, 0]], dtype='b1')
    actual = allel.gt.is_het(g_diploid)
    assert np.array_equal(expect, actual), (expect, actual)

    # polyploid
    expect = np.array([[0, 1],
                       [1, 0],
                       [1, 0]], dtype='b1')
    actual = allel.gt.is_het(g_triploid)
    assert np.array_equal(expect, actual), (expect, actual)


def test_is_hom_ref():

    # haploid
    expect = np.array([[1, 0],
                       [0, 0],
                       [0, 0]], dtype='b1')
    actual = allel.gt.is_hom_ref(g_haploid)
    assert np.array_equal(expect, actual), (expect, actual)

    # diploid
    expect = np.array([[1, 0],
                       [0, 0],
                       [0, 0],
                       [0, 0]], dtype='b1')
    actual = allel.gt.is_hom_ref(g_diploid)
    assert np.array_equal(expect, actual), (expect, actual)

    # polyploid
    expect = np.array([[1, 0],
                       [0, 0],
                       [0, 0]], dtype='b1')
    actual = allel.gt.is_hom_ref(g_triploid)
    assert np.array_equal(expect, actual), (expect, actual)


def test_is_hom_alt():

    # haploid
    expect = np.array([[0, 1],
                       [1, 1],
                       [1, 0]], dtype='b1')
    actual = allel.gt.is_hom_alt(g_haploid)
    assert np.array_equal(expect, actual), (expect, actual)

    # diploid
    expect = np.array([[0, 0],
                       [0, 1],
                       [0, 0],
                       [1, 0]], dtype='b1')
    actual = allel.gt.is_hom_alt(g_diploid)
    assert np.array_equal(expect, actual), (expect, actual)

    # polyploid
    expect = np.array([[0, 0],
                       [0, 1],
                       [0, 0]], dtype='b1')
    actual = allel.gt.is_hom_alt(g_triploid)
    assert np.array_equal(expect, actual), (expect, actual)
