# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


from allel.test.tools import aeq, eq
import numpy as np
import allel.gt


g_haploid = np.array([[0, 1, -1],
                      [1, 1, -1],
                      [2, -1, -1],
                      [-1, -1, -1]], dtype='i1')

g_diploid = np.array([[[0, 0], [0, 1], [-1, -1]],
                      [[0, 2], [1, 1], [-1, -1]],
                      [[1, 0], [2, 1], [-1, -1]],
                      [[2, 2], [-1, -1], [-1, -1]],
                      [[-1, -1], [-1, -1], [-1, -1]]], dtype='i1')

g_triploid = np.array([[[0, 0, 0], [0, 0, 1], [-1, -1, -1]],
                       [[0, 1, 1], [1, 1, 1], [-1, -1, -1]],
                       [[0, 1, 2], [-1, -1, -1], [-1, -1, -1]],
                       [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]], dtype='i1')


def test_is_called():
    f = allel.gt.is_called

    # haploid
    expect = np.array([[1, 1, 0],
                       [1, 1, 0],
                       [1, 0, 0],
                       [0, 0, 0]], dtype='b1')
    actual = f(g_haploid)
    aeq(expect, actual)

    # diploid
    expect = np.array([[1, 1, 0],
                       [1, 1, 0],
                       [1, 1, 0],
                       [1, 0, 0],
                       [0, 0, 0]], dtype='b1')
    actual = f(g_diploid)
    aeq(expect, actual)

    # polyploid
    expect = np.array([[1, 1, 0],
                       [1, 1, 0],
                       [1, 0, 0],
                       [0, 0, 0]], dtype='b1')
    actual = f(g_triploid)
    aeq(expect, actual)


def test_is_missing():
    f = allel.gt.is_missing

    # haploid
    expect = np.array([[0, 0, 1],
                       [0, 0, 1],
                       [0, 1, 1],
                       [1, 1, 1]], dtype='b1')
    actual = f(g_haploid)
    aeq(expect, actual)

    # diploid
    expect = np.array([[0, 0, 1],
                       [0, 0, 1],
                       [0, 0, 1],
                       [0, 1, 1],
                       [1, 1, 1]], dtype='b1')
    actual = f(g_diploid)
    aeq(expect, actual)

    # polyploid
    expect = np.array([[0, 0, 1],
                       [0, 0, 1],
                       [0, 1, 1],
                       [1, 1, 1]], dtype='b1')
    actual = f(g_triploid)
    aeq(expect, actual)


def test_is_hom():
    f = allel.gt.is_hom

    # haploid - trivially true if non-missing
    expect = np.array([[1, 1, 0],
                       [1, 1, 0],
                       [1, 0, 0],
                       [0, 0, 0]], dtype='b1')
    actual = f(g_haploid)
    aeq(expect, actual)

    # diploid
    expect = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 0],
                       [1, 0, 0],
                       [0, 0, 0]], dtype='b1')
    actual = f(g_diploid)
    aeq(expect, actual)

    # polyploid
    expect = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 0],
                       [0, 0, 0]], dtype='b1')
    actual = f(g_triploid)
    aeq(expect, actual)


def test_is_hom_ref():

    # haploid
    expect = np.array([[1, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]], dtype='b1')
    actual = allel.gt.is_hom(g_haploid, allele=0)
    aeq(expect, actual)
    actual = allel.gt.is_hom_ref(g_haploid)
    aeq(expect, actual)

    # diploid
    expect = np.array([[1, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]], dtype='b1')
    actual = allel.gt.is_hom(g_diploid, allele=0)
    aeq(expect, actual)
    actual = allel.gt.is_hom_ref(g_diploid)
    aeq(expect, actual)

    # polyploid
    expect = np.array([[1, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]], dtype='b1')
    actual = allel.gt.is_hom(g_triploid, allele=0)
    aeq(expect, actual)
    actual = allel.gt.is_hom_ref(g_triploid)
    aeq(expect, actual)


def test_is_hom_alt():
    f = allel.gt.is_hom_alt

    # haploid
    expect = np.array([[0, 1, 0],
                       [1, 1, 0],
                       [1, 0, 0],
                       [0, 0, 0]], dtype='b1')
    actual = f(g_haploid)
    aeq(expect, actual)

    # diploid
    expect = np.array([[0, 0, 0],
                       [0, 1, 0],
                       [0, 0, 0],
                       [1, 0, 0],
                       [0, 0, 0]], dtype='b1')
    actual = f(g_diploid)
    aeq(expect, actual)

    # polyploid
    expect = np.array([[0, 0, 0],
                       [0, 1, 0],
                       [0, 0, 0],
                       [0, 0, 0]], dtype='b1')
    actual = f(g_triploid)
    aeq(expect, actual)


def test_is_hom_1():
    f = allel.gt.is_hom

    # haploid
    expect = np.array([[0, 1, 0],
                       [1, 1, 0],
                       [0, 0, 0],
                       [0, 0, 0]], dtype='b1')
    actual = f(g_haploid, allele=1)
    aeq(expect, actual)

    # diploid
    expect = np.array([[0, 0, 0],
                       [0, 1, 0],
                       [0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]], dtype='b1')
    actual = f(g_diploid, allele=1)
    aeq(expect, actual)

    # polyploid
    expect = np.array([[0, 0, 0],
                       [0, 1, 0],
                       [0, 0, 0],
                       [0, 0, 0]], dtype='b1')
    actual = f(g_triploid, allele=1)
    aeq(expect, actual)


def test_is_het():
    f = allel.gt.is_het

    # haploid - trivially false
    expect = np.array([[0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]], dtype='b1')
    actual = f(g_haploid)
    aeq(expect, actual)

    # diploid
    expect = np.array([[0, 1, 0],
                       [1, 0, 0],
                       [1, 1, 0],
                       [0, 0, 0],
                       [0, 0, 0]], dtype='b1')
    actual = f(g_diploid)
    aeq(expect, actual)

    # polyploid
    expect = np.array([[0, 1, 0],
                       [1, 0, 0],
                       [1, 0, 0],
                       [0, 0, 0]], dtype='b1')
    actual = f(g_triploid)
    aeq(expect, actual)


def test_is_call():
    f = allel.gt.is_call

    # haploid
    expect = np.array([[0, 0, 0],
                       [0, 0, 0],
                       [1, 0, 0],
                       [0, 0, 0]], dtype='b1')
    actual = f(g_haploid, 2)
    aeq(expect, actual)

    # diploid
    expect = np.array([[0, 0, 0],
                       [1, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]], dtype='b1')
    actual = f(g_diploid, (0, 2))
    aeq(expect, actual)

    # polyploid
    expect = np.array([[0, 0, 0],
                       [0, 0, 0],
                       [1, 0, 0],
                       [0, 0, 0]], dtype='b1')
    actual = f(g_triploid, (0, 1, 2))
    aeq(expect, actual)


def test_count_missing():
    f = lambda g, axis=None: \
        allel.gt.count(allel.gt.is_missing(g), axis=axis)

    expect = 8
    actual = f(g_diploid)
    assert expect == actual, (expect, actual)

    expect = np.array([1, 2, 5])
    actual = f(g_diploid, axis=0)
    aeq(expect, actual)
    actual = f(g_diploid, axis='variants')
    aeq(expect, actual)

    expect = np.array([1, 1, 1, 2, 3])
    actual = f(g_diploid, axis=1)
    aeq(expect, actual)
    actual = f(g_diploid, axis='samples')
    aeq(expect, actual)


def test_count_called():
    f = lambda g, axis=None: \
        allel.gt.count(allel.gt.is_called(g), axis=axis)

    expect = 7
    actual = f(g_diploid)
    eq(expect, actual)

    expect = np.array([4, 3, 0])
    actual = f(g_diploid, axis=0)
    aeq(expect, actual)
    actual = f(g_diploid, axis='variants')
    aeq(expect, actual)

    expect = np.array([2, 2, 2, 1, 0])
    actual = f(g_diploid, axis=1)
    aeq(expect, actual)
    actual = f(g_diploid, axis='samples')
    aeq(expect, actual)


def test_count_hom():
    f = lambda g, axis=None: \
        allel.gt.count(allel.gt.is_hom(g), axis=axis)

    expect = 3
    actual = f(g_diploid)
    eq(expect, actual)

    expect = np.array([2, 1, 0])
    actual = f(g_diploid, axis=0)
    aeq(expect, actual)
    actual = f(g_diploid, axis='variants')
    aeq(expect, actual)

    expect = np.array([1, 1, 0, 1, 0])
    actual = f(g_diploid, axis=1)
    aeq(expect, actual)
    actual = f(g_diploid, axis='samples')
    aeq(expect, actual)


def test_count_hom_ref():
    f = lambda g, axis=None: \
        allel.gt.count(allel.gt.is_hom_ref(g), axis=axis)

    expect = 1
    actual = f(g_diploid)
    eq(expect, actual)

    expect = np.array([1, 0, 0])
    actual = f(g_diploid, axis=0)
    aeq(expect, actual)
    actual = f(g_diploid, axis='variants')
    aeq(expect, actual)

    expect = np.array([1, 0, 0, 0, 0])
    actual = f(g_diploid, axis=1)
    aeq(expect, actual)
    actual = f(g_diploid, axis='samples')
    aeq(expect, actual)


def test_count_hom_alt():
    f = lambda g, axis=None: \
        allel.gt.count(allel.gt.is_hom_alt(g), axis=axis)

    expect = 2
    actual = f(g_diploid)
    eq(expect, actual)

    expect = np.array([1, 1, 0])
    actual = f(g_diploid, axis=0)
    aeq(expect, actual)
    actual = f(g_diploid, axis='variants')
    aeq(expect, actual)

    expect = np.array([0, 1, 0, 1, 0])
    actual = f(g_diploid, axis=1)
    aeq(expect, actual)
    actual = f(g_diploid, axis='samples')
    aeq(expect, actual)


def test_count_het():
    f = lambda g, axis=None: \
        allel.gt.count(allel.gt.is_het(g), axis=axis)

    expect = 4
    actual = f(g_diploid)
    eq(expect, actual)

    expect = np.array([2, 2, 0])
    actual = f(g_diploid, axis=0)
    aeq(expect, actual)
    actual = f(g_diploid, axis='variants')
    aeq(expect, actual)

    expect = np.array([1, 1, 2, 0, 0])
    actual = f(g_diploid, axis=1)
    aeq(expect, actual)
    actual = f(g_diploid, axis='samples')
    aeq(expect, actual)


def test_count_call():
    f = lambda g, call, axis=None: \
        allel.gt.count(allel.gt.is_call(g, call), axis=axis)

    expect = 1
    actual = f(g_diploid, call=(2, 1))
    eq(expect, actual)

    expect = np.array([0, 1, 0])
    actual = f(g_diploid, call=(2, 1), axis=0)
    aeq(expect, actual)
    actual = f(g_diploid, call=(2, 1), axis='variants')
    aeq(expect, actual)

    expect = np.array([0, 0, 1, 0, 0])
    actual = f(g_diploid, call=(2, 1), axis=1)
    aeq(expect, actual)
    actual = f(g_diploid, call=(2, 1), axis='samples')
    aeq(expect, actual)


################################
# Genotype array transformations
################################


def test_to_haplotypes():
    f = allel.gt.to_haplotypes

    # haploid
    expect = g_haploid
    actual = f(g_haploid)
    aeq(expect, actual)

    # diploid
    expect = np.array([[0, 0, 0, 1, -1, -1],
                       [0, 2, 1, 1, -1, -1],
                       [1, 0, 2, 1, -1, -1],
                       [2, 2, -1, -1, -1, -1],
                       [-1, -1, -1, -1, -1, -1]], dtype='i1')
    actual = f(g_diploid)
    aeq(expect, actual)

    # polyploidy
    expect = np.array([[0, 0, 0, 0, 0, 1, -1, -1, -1],
                       [0, 1, 1, 1, 1, 1, -1, -1, -1],
                       [0, 1, 2, -1, -1, -1, -1, -1, -1],
                       [-1, -1, -1, -1, -1, -1, -1, -1, -1]], dtype='i1')
    actual = f(g_triploid)
    aeq(expect, actual)


def test_from_haplotypes():
    f = allel.gt.from_haplotypes

    # haploid
    expect = g_haploid
    actual = f(g_haploid, ploidy=1)
    aeq(expect, actual)

    # diploid
    h_diploid = np.array([[0, 0, 0, 1, -1, -1],
                          [0, 2, 1, 1, -1, -1],
                          [1, 0, 2, 1, -1, -1],
                          [2, 2, -1, -1, -1, -1],
                          [-1, -1, -1, -1, -1, -1]], dtype='i1')
    expect = g_diploid
    actual = f(h_diploid, ploidy=2)
    aeq(expect, actual)

    # polyploidy
    h_triploid = np.array([[0, 0, 0, 0, 0, 1, -1, -1, -1],
                           [0, 1, 1, 1, 1, 1, -1, -1, -1],
                           [0, 1, 2, -1, -1, -1, -1, -1, -1],
                           [-1, -1, -1, -1, -1, -1, -1, -1, -1]], dtype='i1')
    expect = g_triploid
    actual = f(h_triploid, ploidy=3)
    aeq(expect, actual)


def test_to_n_alt():
    f = allel.gt.to_n_alt

    # haploid
    expect = np.array([[0, 1, 0],
                       [1, 1, 0],
                       [1, 0, 0],
                       [0, 0, 0]], dtype='i1')
    actual = f(g_haploid)
    aeq(expect, actual)

    # diploid
    expect = np.array([[0, 1, 0],
                       [1, 2, 0],
                       [1, 2, 0],
                       [2, 0, 0],
                       [0, 0, 0]], dtype='i1')
    actual = f(g_diploid)
    aeq(expect, actual)

    # polyploid
    expect = np.array([[0, 1, 0],
                       [2, 3, 0],
                       [2, 0, 0],
                       [0, 0, 0]], dtype='i1')
    actual = f(g_triploid)
    aeq(expect, actual)


def test_to_n_alt_fill():
    f = allel.gt.to_n_alt

    # haploid
    expect = np.array([[0, 1, -1],
                       [1, 1, -1],
                       [1, -1, -1],
                       [-1, -1, -1]], dtype='i1')
    actual = f(g_haploid, fill=-1)
    aeq(expect, actual)

    # diploid
    expect = np.array([[0, 1, -1],
                       [1, 2, -1],
                       [1, 2, -1],
                       [2, -1, -1],
                       [-1, -1, -1]], dtype='i1')
    actual = f(g_diploid, fill=-1)
    aeq(expect, actual)

    # polyploid
    expect = np.array([[0, 1, -1],
                       [2, 3, -1],
                       [2, -1, -1],
                       [-1, -1, -1]], dtype='i1')
    actual = f(g_triploid, fill=-1)
    aeq(expect, actual)


def test_to_allele_counts():
    f = allel.gt.to_allele_counts

    # haploid
    expect = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 0]],
                       [[0, 1, 0], [0, 1, 0], [0, 0, 0]],
                       [[0, 0, 1], [0, 0, 0], [0, 0, 0]],
                       [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype='i1')
    actual = f(g_haploid)
    aeq(expect, actual)

    # diploid
    expect = np.array([[[2, 0, 0], [1, 1, 0], [0, 0, 0]],
                       [[1, 0, 1], [0, 2, 0], [0, 0, 0]],
                       [[1, 1, 0], [0, 1, 1], [0, 0, 0]],
                       [[0, 0, 2], [0, 0, 0], [0, 0, 0]],
                       [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype='i1')
    actual = f(g_diploid)
    aeq(expect, actual)

    # triploid
    expect = np.array([[[3, 0, 0], [2, 1, 0], [0, 0, 0]],
                       [[1, 2, 0], [0, 3, 0], [0, 0, 0]],
                       [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
                       [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype='i1')
    actual = f(g_triploid)
    aeq(expect, actual)


def test_to_packed():

    expect = np.array([[0, 1, 239],
                       [2, 17, 239],
                       [16, 33, 239],
                       [34, 239, 239],
                       [239, 239, 239]], dtype='u1')
    actual = allel.gt.to_packed(g_diploid)
    aeq(expect, actual)


def test_from_packed():

    g_diploid_packed = np.array([[0, 1, 239],
                                 [2, 17, 239],
                                 [16, 33, 239],
                                 [34, 239, 239],
                                 [239, 239, 239]], dtype='u1')
    expect = g_diploid
    actual = allel.gt.from_packed(g_diploid_packed)
    aeq(expect, actual)


###############################
# Allele frequency calculations
###############################


def test_max_allele():
    f = allel.gt.max_allele

    # haploid
    expect = 2
    actual = f(g_haploid)
    eq(expect, actual)
    expect = np.array([2, 1, -1])
    actual = f(g_haploid, axis=0)
    aeq(expect, actual)
    expect = np.array([1, 1, 2, -1])
    actual = f(g_haploid, axis=1)
    aeq(expect, actual)

    # diploid
    expect = 2
    actual = f(g_diploid)
    eq(expect, actual)
    expect = np.array([2, 2, -1])
    actual = f(g_diploid, axis=(0, 2))
    aeq(expect, actual)
    expect = np.array([1, 2, 2, 2, -1])
    actual = f(g_diploid, axis=(1, 2))
    aeq(expect, actual)


def test_allelism():
    f = allel.gt.allelism

    # haploid
    expect = np.array([2, 1, 1, 0])
    actual = f(g_haploid)
    aeq(expect, actual)

    # diploid
    expect = np.array([2, 3, 3, 1, 0])
    actual = f(g_diploid)
    aeq(expect, actual)

    # triploid
    expect = np.array([2, 2, 3, 0])
    actual = f(g_triploid)
    aeq(expect, actual)


def test_allele_number():
    f = allel.gt.allele_number

    # haploid
    expect = np.array([2, 2, 1, 0])
    actual = f(g_haploid)
    aeq(expect, actual)

    # diploid
    expect = np.array([4, 4, 4, 2, 0])
    actual = f(g_diploid)
    aeq(expect, actual)

    # triploid
    expect = np.array([6, 6, 3, 0])
    actual = f(g_triploid)
    aeq(expect, actual)


def test_allele_count():
    f = allel.gt.allele_count

    # haploid
    expect = np.array([1, 2, 0, 0])
    actual = f(g_haploid, allele=1)
    aeq(expect, actual)
    expect = np.array([0, 0, 1, 0])
    actual = f(g_haploid, allele=2)
    aeq(expect, actual)

    # diploid
    expect = np.array([1, 2, 2, 0, 0])
    actual = f(g_diploid, allele=1)
    aeq(expect, actual)
    expect = np.array([0, 1, 1, 2, 0])
    actual = f(g_diploid, allele=2)
    aeq(expect, actual)

    # triploid
    expect = np.array([1, 5, 1, 0])
    actual = f(g_triploid, allele=1)
    aeq(expect, actual)
    expect = np.array([0, 0, 1, 0])
    actual = f(g_triploid, allele=2)
    aeq(expect, actual)


def test_allele_frequency():
    f = allel.gt.allele_frequency

    # haploid
    expect = np.array([1/2, 2/2, 0/1, 0])
    actual, _, _ = f(g_haploid, allele=1)
    aeq(expect, actual)
    expect = np.array([0/2, 0/2, 1/1, 0])
    actual, _, _ = f(g_haploid, allele=2)
    aeq(expect, actual)

    # diploid
    expect = np.array([1/4, 2/4, 2/4, 0/2, 0])
    actual, _, _ = f(g_diploid, allele=1)
    aeq(expect, actual)
    expect = np.array([0/4, 1/4, 1/4, 2/2, 0])
    actual, _, _ = f(g_diploid, allele=2)
    aeq(expect, actual)

    # triploid
    expect = np.array([1/6, 5/6, 1/3, 0])
    actual, _, _ = f(g_triploid, allele=1)
    aeq(expect, actual)
    expect = np.array([0/6, 0/6, 1/3, 0])
    actual, _, _ = f(g_triploid, allele=2)
    aeq(expect, actual)


def test_allele_counts():
    f = allel.gt.allele_counts

    # haploid
    expect = np.array([[1, 1, 0],
                       [0, 2, 0],
                       [0, 0, 1],
                       [0, 0, 0]])
    actual = f(g_haploid)
    aeq(expect, actual)

    # diploid
    expect = np.array([[3, 1, 0],
                       [1, 2, 1],
                       [1, 2, 1],
                       [0, 0, 2],
                       [0, 0, 0]])
    actual = f(g_diploid)
    aeq(expect, actual)

    # triploid
    expect = np.array([[5, 1, 0],
                       [1, 5, 0],
                       [1, 1, 1],
                       [0, 0, 0]])
    actual = f(g_triploid)
    aeq(expect, actual)


def test_allele_frequencies():
    f = allel.gt.allele_frequencies

    # haploid
    expect = np.array([[1/2, 1/2, 0/2],
                       [0/2, 2/2, 0/2],
                       [0/1, 0/1, 1/1],
                       [0, 0, 0]])
    actual, _, _ = f(g_haploid)
    aeq(expect, actual)

    # diploid
    expect = np.array([[3/4, 1/4, 0/4],
                       [1/4, 2/4, 1/4],
                       [1/4, 2/4, 1/4],
                       [0/2, 0/2, 2/2],
                       [0, 0, 0]])
    actual, _, _ = f(g_diploid)
    aeq(expect, actual)

    # triploid
    expect = np.array([[5/6, 1/6, 0/6],
                       [1/6, 5/6, 0/6],
                       [1/3, 1/3, 1/3],
                       [0, 0, 0]])
    actual, _, _ = f(g_triploid)
    aeq(expect, actual)


def test_is_count_variant():
    f = allel.gt.is_variant
    c = lambda g: allel.gt.count(f(g))

    # haploid
    expect = np.array([1, 1, 1, 0], dtype='b1')
    actual = f(g_haploid)
    aeq(expect, actual)
    eq(np.sum(expect), c(g_haploid))

    # diploid
    expect = np.array([1, 1, 1, 1, 0], dtype='b1')
    actual = f(g_diploid)
    aeq(expect, actual)
    eq(np.sum(expect), c(g_diploid))

    # triploid
    expect = np.array([1, 1, 1, 0], dtype='b1')
    actual = f(g_triploid)
    aeq(expect, actual)
    eq(np.sum(expect), c(g_triploid))


def test_is_count_non_variant():
    f = allel.gt.is_non_variant
    c = lambda g: allel.gt.count(f(g))

    # haploid
    expect = np.array([0, 0, 0, 1], dtype='b1')
    actual = f(g_haploid)
    aeq(expect, actual)
    eq(np.sum(expect), c(g_haploid))

    # diploid
    expect = np.array([0, 0, 0, 0, 1], dtype='b1')
    actual = f(g_diploid)
    aeq(expect, actual)
    eq(np.sum(expect), c(g_diploid))

    # triploid
    expect = np.array([0, 0, 0, 1], dtype='b1')
    actual = f(g_triploid)
    aeq(expect, actual)
    eq(np.sum(expect), c(g_triploid))


def test_is_count_segregating():
    f = allel.gt.is_segregating
    c = lambda g: allel.gt.count(f(g))

    # haploid
    expect = np.array([1, 0, 0, 0], dtype='b1')
    actual = f(g_haploid)
    aeq(expect, actual)
    eq(np.sum(expect), c(g_haploid))

    # diploid
    expect = np.array([1, 1, 1, 0, 0], dtype='b1')
    actual = f(g_diploid)
    aeq(expect, actual)
    eq(np.sum(expect), c(g_diploid))

    # triploid
    expect = np.array([1, 1, 1, 0], dtype='b1')
    actual = f(g_triploid)
    aeq(expect, actual)
    eq(np.sum(expect), c(g_triploid))


def test_is_count_non_segregating():
    f = allel.gt.is_non_segregating
    c = lambda g, allele=None: allel.gt.count(f(g, allele=allele))

    # haploid
    expect = np.array([0, 1, 1, 1], dtype='b1')
    actual = f(g_haploid)
    aeq(expect, actual)
    eq(np.sum(expect), c(g_haploid))
    expect = np.array([0, 0, 1, 1], dtype='b1')
    actual = f(g_haploid, allele=2)
    aeq(expect, actual)
    eq(np.sum(expect), c(g_haploid, allele=2))

    # diploid
    expect = np.array([0, 0, 0, 1, 1], dtype='b1')
    actual = f(g_diploid)
    aeq(expect, actual)
    eq(np.sum(expect), c(g_diploid))
    expect = np.array([0, 0, 0, 1, 1], dtype='b1')
    actual = f(g_diploid, allele=2)
    aeq(expect, actual)
    eq(np.sum(expect), c(g_diploid, allele=2))

    # triploid
    expect = np.array([0, 0, 0, 1], dtype='b1')
    actual = f(g_triploid)
    aeq(expect, actual)
    eq(np.sum(expect), c(g_triploid))
    expect = np.array([0, 0, 0, 1], dtype='b1')
    actual = f(g_triploid, allele=2)
    aeq(expect, actual)
    eq(np.sum(expect), c(g_triploid, allele=2))


def test_is_count_singleton():
    f = allel.gt.is_singleton
    c = lambda g, allele: allel.gt.count(f(g, allele=allele))

    # haploid
    expect = np.array([1, 0, 0, 0], dtype='b1')
    actual = f(g_haploid, allele=1)
    aeq(expect, actual)
    eq(np.sum(expect), c(g_haploid, allele=1))
    expect = np.array([0, 0, 1, 0], dtype='b1')
    actual = f(g_haploid, allele=2)
    aeq(expect, actual)
    eq(np.sum(expect), c(g_haploid, allele=2))

    # diploid
    expect = np.array([1, 0, 0, 0, 0], dtype='b1')
    actual = f(g_diploid, allele=1)
    aeq(expect, actual)
    eq(np.sum(expect), c(g_diploid, allele=1))
    expect = np.array([0, 1, 1, 0, 0], dtype='b1')
    actual = f(g_diploid, allele=2)
    aeq(expect, actual)
    eq(np.sum(expect), c(g_diploid, allele=2))

    # triploid
    expect = np.array([1, 0, 1, 0], dtype='b1')
    actual = f(g_triploid, allele=1)
    aeq(expect, actual)
    eq(np.sum(expect), c(g_triploid, allele=1))
    expect = np.array([0, 0, 1, 0], dtype='b1')
    actual = f(g_triploid, allele=2)
    aeq(expect, actual)
    eq(np.sum(expect), c(g_triploid, allele=2))


def test_is_count_doubleton():
    f = allel.gt.is_doubleton
    c = lambda g, allele: allel.gt.count(f(g, allele=allele))

    # haploid
    expect = np.array([0, 1, 0, 0], dtype='b1')
    actual = f(g_haploid, allele=1)
    aeq(expect, actual)
    eq(np.sum(expect), c(g_haploid, allele=1))
    expect = np.array([0, 0, 0, 0], dtype='b1')
    actual = f(g_haploid, allele=2)
    aeq(expect, actual)
    eq(np.sum(expect), c(g_haploid, allele=2))

    # diploid
    expect = np.array([0, 1, 1, 0, 0], dtype='b1')
    actual = f(g_diploid, allele=1)
    aeq(expect, actual)
    eq(np.sum(expect), c(g_diploid, allele=1))
    expect = np.array([0, 0, 0, 1, 0], dtype='b1')
    actual = f(g_diploid, allele=2)
    aeq(expect, actual)
    eq(np.sum(expect), c(g_diploid, allele=2))

    # triploid
    expect = np.array([0, 0, 0, 0], dtype='b1')
    actual = f(g_triploid, allele=1)
    aeq(expect, actual)
    eq(np.sum(expect), c(g_triploid, allele=1))
    expect = np.array([0, 0, 0, 0], dtype='b1')
    actual = f(g_triploid, allele=2)
    aeq(expect, actual)
    eq(np.sum(expect), c(g_triploid, allele=2))


def test_windowed_count():
    f = allel.gt.windowed_count
    pos = [1, 12, 15, 27]

    # boolean array, all true
    b = [True, True, True, True]
    expected_counts = [1, 2, 1]
    expected_bin_edges = [1, 11, 21, 31]
    actual_counts, actual_bin_edges = \
        f(pos, b, window=10)
    aeq(expected_counts, actual_counts)
    aeq(expected_bin_edges, actual_bin_edges)

    # boolean array, not all true
    b = [False, True, False, True]
    expected_counts = [0, 1, 1]
    expected_bin_edges = [1, 11, 21, 31]
    actual_counts, actual_bin_edges = \
        f(pos, b, window=10)
    aeq(expected_bin_edges, actual_bin_edges)
    aeq(expected_counts, actual_counts)

    # explicit start and stop
    b = [False, True, False, True]
    expected_counts = [1, 0, 1]
    expected_bin_edges = [5, 15, 25, 27]
    actual_counts, actual_bin_edges = \
        f(pos, b, window=10, start=5, stop=27)
    aeq(expected_bin_edges, actual_bin_edges)
    aeq(expected_counts, actual_counts)

    # boolean array, bad length
    b = [False, True, False]
    try:
        f(pos, b, window=10)
    except allel.errors.ArgumentError:
        pass
    else:
        assert False, 'exception not raised'

    # 2D, 4 variants, 2 samples
    b = [[True, False],
         [True, True],
         [True, False],
         [True, True]]
    expected_counts = [[1, 0],
                       [2, 1],
                       [1, 1]]
    expected_bin_edges = [1, 11, 21, 31]
    actual_counts, actual_bin_edges = \
        f(pos, b, window=10)
    aeq(expected_counts, actual_counts)
    aeq(expected_bin_edges, actual_bin_edges)


def test_windowed_density():
    f = allel.gt.windowed_density
    pos = [1, 12, 15, 27]

    # boolean array, all true
    b = [True, True, True, True]
    # N.B., final bin includes right edge
    expected_densities = [1/10, 2/10, 1/11]
    expected_bin_edges = [1, 11, 21, 31]
    actual_densities, _, actual_bin_edges = \
        f(pos, b, window=10)
    aeq(expected_densities, actual_densities)
    aeq(expected_bin_edges, actual_bin_edges)

    # boolean array, not all true
    b = [False, True, False, True]
    expected_densities = [0/10, 1/10, 1/11]
    expected_bin_edges = [1, 11, 21, 31]
    actual_densities, _, actual_bin_edges = \
        f(pos, b, window=10)
    aeq(expected_bin_edges, actual_bin_edges)
    aeq(expected_densities, actual_densities)

    # explicit start and stop
    b = [False, True, False, True]
    expected_densities = [1/10, 0/10, 1/3]
    expected_bin_edges = [5, 15, 25, 27]
    actual_densities, _, actual_bin_edges = \
        f(pos, b, window=10, start=5, stop=27)
    aeq(expected_bin_edges, actual_bin_edges)
    aeq(expected_densities, actual_densities)

    # boolean array, bad length
    b = [False, True, False]
    try:
        f(pos, b, window=10)
    except allel.errors.ArgumentError:
        pass
    else:
        assert False, 'exception not raised'

    # 2D, 4 variants, 2 samples
    b = [[True, False],
         [True, True],
         [True, False],
         [True, True]]
    expected_densities = [[1/10, 0/10],
                          [2/10, 1/10],
                          [1/11, 1/11]]
    expected_bin_edges = [1, 11, 21, 31]
    actual_densities, _, actual_bin_edges = \
        f(pos, b, window=10)
    aeq(expected_densities, actual_densities)
    aeq(expected_bin_edges, actual_bin_edges)

    # include is_accessible array option
    is_accessible = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              1, 1, 1, 1, 0, 0, 1, 1, 0, 0,
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=bool)
    b = [False, True, False, True]
    expected_densities = [0, 1/6, 1/11]
    expected_bin_edges = [1, 11, 21, 31]
    actual_densities, _, actual_bin_edges = \
        f(pos, b, window=10, is_accessible=is_accessible)
    aeq(expected_bin_edges, actual_bin_edges)
    aeq(expected_densities, actual_densities)
