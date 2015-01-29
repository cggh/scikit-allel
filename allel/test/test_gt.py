# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


from allel.test.tools import aeq, eq
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
    f = allel.gt.is_called

    # haploid
    expect = np.array([[1, 1],
                       [1, 1],
                       [1, 0]], dtype='b1')
    actual = f(g_haploid)
    aeq(expect, actual)

    # diploid
    expect = np.array([[1, 1],
                       [1, 1],
                       [1, 1],
                       [1, 0]], dtype='b1')
    actual = f(g_diploid)
    aeq(expect, actual)

    # polyploid
    expect = np.array([[1, 1],
                       [1, 1],
                       [1, 0]], dtype='b1')
    actual = f(g_triploid)
    aeq(expect, actual)


def test_is_missing():
    f = allel.gt.is_missing

    # haploid
    expect = np.array([[0, 0],
                       [0, 0],
                       [0, 1]], dtype='b1')
    actual = f(g_haploid)
    aeq(expect, actual)

    # diploid
    expect = np.array([[0, 0],
                       [0, 0],
                       [0, 0],
                       [0, 1]], dtype='b1')
    actual = f(g_diploid)
    aeq(expect, actual)

    # polyploid
    expect = np.array([[0, 0],
                       [0, 0],
                       [0, 1]], dtype='b1')
    actual = f(g_triploid)
    aeq(expect, actual)


def test_is_hom():
    f = allel.gt.is_hom

    # haploid - trivially true
    expect = np.array([[1, 1],
                       [1, 1],
                       [1, 1]], dtype='b1')
    actual = f(g_haploid)
    aeq(expect, actual)

    # diploid
    expect = np.array([[1, 0],
                       [0, 1],
                       [0, 0],
                       [1, 0]], dtype='b1')
    actual = f(g_diploid)
    aeq(expect, actual)

    # polyploid
    expect = np.array([[1, 0],
                       [0, 1],
                       [0, 0]], dtype='b1')
    actual = f(g_triploid)
    aeq(expect, actual)


def test_is_hom_ref():

    # haploid
    expect = np.array([[1, 0],
                       [0, 0],
                       [0, 0]], dtype='b1')
    actual = allel.gt.is_hom(g_haploid, allele=0)
    aeq(expect, actual)
    actual = allel.gt.is_hom_ref(g_haploid)
    aeq(expect, actual)

    # diploid
    expect = np.array([[1, 0],
                       [0, 0],
                       [0, 0],
                       [0, 0]], dtype='b1')
    actual = allel.gt.is_hom(g_diploid, allele=0)
    aeq(expect, actual)
    actual = allel.gt.is_hom_ref(g_diploid)
    aeq(expect, actual)

    # polyploid
    expect = np.array([[1, 0],
                       [0, 0],
                       [0, 0]], dtype='b1')
    actual = allel.gt.is_hom(g_triploid, allele=0)
    aeq(expect, actual)
    actual = allel.gt.is_hom_ref(g_triploid)
    aeq(expect, actual)


def test_is_hom_alt():
    f = allel.gt.is_hom_alt

    # haploid
    expect = np.array([[0, 1],
                       [1, 1],
                       [1, 0]], dtype='b1')
    actual = f(g_haploid)
    aeq(expect, actual)

    # diploid
    expect = np.array([[0, 0],
                       [0, 1],
                       [0, 0],
                       [1, 0]], dtype='b1')
    actual = f(g_diploid)
    aeq(expect, actual)

    # polyploid
    expect = np.array([[0, 0],
                       [0, 1],
                       [0, 0]], dtype='b1')
    actual = f(g_triploid)
    aeq(expect, actual)


def test_is_hom_1():
    f = allel.gt.is_hom

    # haploid
    expect = np.array([[0, 1],
                       [1, 1],
                       [0, 0]], dtype='b1')
    actual = f(g_haploid, allele=1)
    aeq(expect, actual)

    # diploid
    expect = np.array([[0, 0],
                       [0, 1],
                       [0, 0],
                       [0, 0]], dtype='b1')
    actual = f(g_diploid, allele=1)
    aeq(expect, actual)

    # polyploid
    expect = np.array([[0, 0],
                       [0, 1],
                       [0, 0]], dtype='b1')
    actual = f(g_triploid, allele=1)
    aeq(expect, actual)


def test_is_het():
    f = allel.gt.is_het

    # haploid - trivially false
    expect = np.array([[0, 0],
                       [0, 0],
                       [0, 0]], dtype='b1')
    actual = f(g_haploid)
    aeq(expect, actual)

    # diploid
    expect = np.array([[0, 1],
                       [1, 0],
                       [1, 1],
                       [0, 0]], dtype='b1')
    actual = f(g_diploid)
    aeq(expect, actual)

    # polyploid
    expect = np.array([[0, 1],
                       [1, 0],
                       [1, 0]], dtype='b1')
    actual = f(g_triploid)
    aeq(expect, actual)


def test_is_call():
    f = allel.gt.is_call

    # haploid
    expect = np.array([[0, 0],
                       [0, 0],
                       [1, 0]], dtype='b1')
    actual = f(g_haploid, 2)
    aeq(expect, actual)

    # diploid
    expect = np.array([[0, 0],
                       [1, 0],
                       [0, 0],
                       [0, 0]], dtype='b1')
    actual = f(g_diploid, (0, 2))
    aeq(expect, actual)

    # polyploid
    expect = np.array([[0, 0],
                       [0, 0],
                       [1, 0]], dtype='b1')
    actual = f(g_triploid, (0, 1, 2))
    aeq(expect, actual)


def test_count_missing():
    f = allel.gt.count_missing

    expect = 1
    actual = f(g_diploid)
    assert expect == actual, (expect, actual)

    expect = np.array([0, 1])
    actual = f(g_diploid, axis=0)
    aeq(expect, actual)
    actual = f(g_diploid, axis='variants')
    aeq(expect, actual)

    expect = np.array([0, 0, 0, 1])
    actual = f(g_diploid, axis=1)
    aeq(expect, actual)
    actual = f(g_diploid, axis='samples')
    aeq(expect, actual)


def test_count_called():
    f = allel.gt.count_called

    expect = 7
    actual = f(g_diploid)
    eq(expect, actual)

    expect = np.array([4, 3])
    actual = f(g_diploid, axis=0)
    aeq(expect, actual)
    actual = f(g_diploid, axis='variants')
    aeq(expect, actual)

    expect = np.array([2, 2, 2, 1])
    actual = f(g_diploid, axis=1)
    aeq(expect, actual)
    actual = f(g_diploid, axis='samples')
    aeq(expect, actual)


def test_count_hom():
    f = allel.gt.count_hom

    expect = 3
    actual = f(g_diploid)
    eq(expect, actual)

    expect = np.array([2, 1])
    actual = f(g_diploid, axis=0)
    aeq(expect, actual)
    actual = f(g_diploid, axis='variants')
    aeq(expect, actual)

    expect = np.array([1, 1, 0, 1])
    actual = f(g_diploid, axis=1)
    aeq(expect, actual)
    actual = f(g_diploid, axis='samples')
    aeq(expect, actual)


def test_count_hom_ref():

    expect = 1
    actual = allel.gt.count_hom_ref(g_diploid)
    eq(expect, actual)
    actual = allel.gt.count_hom(g_diploid, allele=0)
    eq(expect, actual)

    expect = np.array([1, 0])
    actual = allel.gt.count_hom_ref(g_diploid, axis=0)
    aeq(expect, actual)
    actual = allel.gt.count_hom_ref(g_diploid, axis='variants')
    aeq(expect, actual)
    actual = allel.gt.count_hom(g_diploid, axis=0, allele=0)
    aeq(expect, actual)

    expect = np.array([1, 0, 0, 0])
    actual = allel.gt.count_hom_ref(g_diploid, axis=1)
    aeq(expect, actual)
    actual = allel.gt.count_hom_ref(g_diploid, axis='samples')
    aeq(expect, actual)
    actual = allel.gt.count_hom(g_diploid, axis=1, allele=0)
    aeq(expect, actual)


def test_count_hom_alt():
    f = allel.gt.count_hom_alt

    expect = 2
    actual = f(g_diploid)
    eq(expect, actual)

    expect = np.array([1, 1])
    actual = f(g_diploid, axis=0)
    aeq(expect, actual)
    actual = f(g_diploid, axis='variants')
    aeq(expect, actual)

    expect = np.array([0, 1, 0, 1])
    actual = f(g_diploid, axis=1)
    aeq(expect, actual)
    actual = f(g_diploid, axis='samples')
    aeq(expect, actual)


def test_count_het():
    f = allel.gt.count_het

    expect = 4
    actual = f(g_diploid)
    eq(expect, actual)

    expect = np.array([2, 2])
    actual = f(g_diploid, axis=0)
    aeq(expect, actual)
    actual = f(g_diploid, axis='variants')
    aeq(expect, actual)

    expect = np.array([1, 1, 2, 0])
    actual = f(g_diploid, axis=1)
    aeq(expect, actual)
    actual = f(g_diploid, axis='samples')
    aeq(expect, actual)


def test_count_call():
    f = allel.gt.count_call

    expect = 1
    actual = f(g_diploid, call=(2, 1))
    eq(expect, actual)

    expect = np.array([0, 1])
    actual = f(g_diploid, call=(2, 1), axis=0)
    aeq(expect, actual)
    actual = f(g_diploid, call=(2, 1), axis='variants')
    aeq(expect, actual)

    expect = np.array([0, 0, 1, 0])
    actual = f(g_diploid, call=(2, 1), axis=1)
    aeq(expect, actual)
    actual = f(g_diploid, call=(2, 1), axis='samples')
    aeq(expect, actual)


################################
# Genotype array transformations
################################


# TODO def test_as_haplotypes():
# TODO def test_as_n_alt():
# TODO def test_as_012():
# TODO def test_as_allele_counts():
# TODO def test_pack_diploid():
# TODO def test_unpack_diploid


###############################
# Allele frequency calculations
###############################


# TODO def test_max_allele()
# TODO def test_allelism()
# TODO def test_allele_number()
# TODO def test_allele_count()
# TODO def test_allele_frequency()
# TODO def test_allele_counts()
# TODO def test_allele_frequencies()
# TODO def test_is_variant()
# TODO def test_is_non_variant()
# TODO def test_is_segregating()
# TODO def test_is_non_segregating()
# TODO def test_is_singleton()
# TODO def test_is_doubleton()
# TODO def test_count_variant()
# TODO def test_count_non_variant()
# TODO def test_count_segregating()
# TODO def test_count_non_segregating()
# TODO def test_count_singleton()
# TODO def test_count_doubleton()
