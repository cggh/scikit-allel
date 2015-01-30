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
    f = allel.gt.count_missing

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
    f = allel.gt.count_called

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
    f = allel.gt.count_hom

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

    expect = 1
    actual = allel.gt.count_hom_ref(g_diploid)
    eq(expect, actual)
    actual = allel.gt.count_hom(g_diploid, allele=0)
    eq(expect, actual)

    expect = np.array([1, 0, 0])
    actual = allel.gt.count_hom_ref(g_diploid, axis=0)
    aeq(expect, actual)
    actual = allel.gt.count_hom_ref(g_diploid, axis='variants')
    aeq(expect, actual)
    actual = allel.gt.count_hom(g_diploid, axis=0, allele=0)
    aeq(expect, actual)

    expect = np.array([1, 0, 0, 0, 0])
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
    f = allel.gt.count_het

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
    f = allel.gt.count_call

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


def test_as_haplotypes():
    f = allel.gt.as_haplotypes

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


def test_as_n_alt():
    f = allel.gt.as_n_alt

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


def test_as_n_alt_fill():
    f = allel.gt.as_n_alt

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


def test_as_allele_counts():
    f = allel.gt.as_allele_counts

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


def test_pack_diploid():

    expect = np.array([[0, 1, 239],
                       [2, 17, 239],
                       [16, 33, 239],
                       [34, 239, 239],
                       [239, 239, 239]], dtype='u1')
    actual = allel.gt.pack_diploid(g_diploid)
    aeq(expect, actual)


def test_unpack_diploid():

    g_diploid_packed = np.array([[0, 1, 239],
                                 [2, 17, 239],
                                 [16, 33, 239],
                                 [34, 239, 239],
                                 [239, 239, 239]], dtype='u1')
    expect = g_diploid
    actual = allel.gt.unpack_diploid(g_diploid_packed)
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
    # TODO

    # diploid
    # TODO

    # triploid
    # TODO


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
