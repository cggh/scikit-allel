# -*- coding: utf-8 -*-
import unittest


import numpy as np
from numpy.testing import assert_array_equal


from allel.stats.mendel import mendel_errors, paint_transmission, \
    INHERIT_PARENT1, INHERIT_PARENT2, INHERIT_NONPARENTAL, \
    INHERIT_NONSEG_REF, INHERIT_NONSEG_ALT, INHERIT_MISSING, \
    INHERIT_PARENT_MISSING, phase_progeny_by_transmission


class TestMendelErrors(unittest.TestCase):

    @staticmethod
    def _test(genotypes, expect):
        parent_genotypes = genotypes[:, 0:2]
        progeny_genotypes = genotypes[:, 2:]

        # run test
        actual = mendel_errors(parent_genotypes, progeny_genotypes)
        assert_array_equal(expect, actual)

        # swap parents, should have no affect
        actual = mendel_errors(parent_genotypes, progeny_genotypes)
        assert_array_equal(expect, actual)

        # swap alleles, should have no effect
        parent_genotypes = parent_genotypes[:, :, ::-1]
        progeny_genotypes = progeny_genotypes[:, :, ::-1]
        actual = mendel_errors(parent_genotypes, progeny_genotypes)
        assert_array_equal(expect, actual)

    def test_consistent(self):
        genotypes = np.array([
            # aa x aa -> aa
            [[0, 0], [0, 0], [0, 0], [-1, -1], [-1, -1], [-1, -1]],
            [[1, 1], [1, 1], [1, 1], [-1, -1], [-1, -1], [-1, -1]],
            [[2, 2], [2, 2], [2, 2], [-1, -1], [-1, -1], [-1, -1]],
            # aa x ab -> aa or ab
            [[0, 0], [0, 1], [0, 0], [0, 1], [-1, -1], [-1, -1]],
            [[0, 0], [0, 2], [0, 0], [0, 2], [-1, -1], [-1, -1]],
            [[1, 1], [0, 1], [1, 1], [0, 1], [-1, -1], [-1, -1]],
            # aa x bb -> ab
            [[0, 0], [1, 1], [0, 1], [-1, -1], [-1, -1], [-1, -1]],
            [[0, 0], [2, 2], [0, 2], [-1, -1], [-1, -1], [-1, -1]],
            [[1, 1], [2, 2], [1, 2], [-1, -1], [-1, -1], [-1, -1]],
            # aa x bc -> ab or ac
            [[0, 0], [1, 2], [0, 1], [0, 2], [-1, -1], [-1, -1]],
            [[1, 1], [0, 2], [0, 1], [1, 2], [-1, -1], [-1, -1]],
            # ab x ab -> aa or ab or bb
            [[0, 1], [0, 1], [0, 0], [0, 1], [1, 1], [-1, -1]],
            [[1, 2], [1, 2], [1, 1], [1, 2], [2, 2], [-1, -1]],
            [[0, 2], [0, 2], [0, 0], [0, 2], [2, 2], [-1, -1]],
            # ab x bc -> ab or ac or bb or bc
            [[0, 1], [1, 2], [0, 1], [0, 2], [1, 1], [1, 2]],
            [[0, 1], [0, 2], [0, 0], [0, 1], [0, 1], [1, 2]],
            # ab x cd -> ac or ad or bc or bd
            [[0, 1], [2, 3], [0, 2], [0, 3], [1, 2], [1, 3]],
        ])
        expect = np.zeros((17, 4))
        self._test(genotypes, expect)

    def test_error_nonparental(self):
        genotypes = np.array([
            # aa x aa -> ab or ac or bb or cc
            [[0, 0], [0, 0], [0, 1], [0, 2], [1, 1], [2, 2]],
            [[1, 1], [1, 1], [0, 1], [1, 2], [0, 0], [2, 2]],
            [[2, 2], [2, 2], [0, 2], [1, 2], [0, 0], [1, 1]],
            # aa x ab -> ac or bc or cc
            [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2], [2, 2]],
            [[0, 0], [0, 2], [0, 1], [1, 2], [1, 1], [1, 1]],
            [[1, 1], [0, 1], [1, 2], [0, 2], [2, 2], [2, 2]],
            # aa x bb -> ac or bc or cc
            [[0, 0], [1, 1], [0, 2], [1, 2], [2, 2], [2, 2]],
            [[0, 0], [2, 2], [0, 1], [1, 2], [1, 1], [1, 1]],
            [[1, 1], [2, 2], [0, 1], [0, 2], [0, 0], [0, 0]],
            # ab x ab -> ac or bc or cc
            [[0, 1], [0, 1], [0, 2], [1, 2], [2, 2], [2, 2]],
            [[0, 2], [0, 2], [0, 1], [1, 2], [1, 1], [1, 1]],
            [[1, 2], [1, 2], [0, 1], [0, 2], [0, 0], [0, 0]],
            # ab x bc -> ad or bd or cd or dd
            [[0, 1], [1, 2], [0, 3], [1, 3], [2, 3], [3, 3]],
            [[0, 1], [0, 2], [0, 3], [1, 3], [2, 3], [3, 3]],
            [[0, 2], [1, 2], [0, 3], [1, 3], [2, 3], [3, 3]],
            # ab x cd -> ae or be or ce or de
            [[0, 1], [2, 3], [0, 4], [1, 4], [2, 4], [3, 4]],
        ])
        expect = np.array([
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [1, 1, 1, 2],
            [1, 1, 1, 2],
            [1, 1, 1, 2],
            [1, 1, 1, 1],
        ])
        self._test(genotypes, expect)

    def test_error_hemiparental(self):
        genotypes = np.array([
            # aa x ab -> bb
            [[0, 0], [0, 1], [1, 1], [-1, -1]],
            [[0, 0], [0, 2], [2, 2], [-1, -1]],
            [[1, 1], [0, 1], [0, 0], [-1, -1]],
            # ab x bc -> aa or cc
            [[0, 1], [1, 2], [0, 0], [2, 2]],
            [[0, 1], [0, 2], [1, 1], [2, 2]],
            [[0, 2], [1, 2], [0, 0], [1, 1]],
            # ab x cd -> aa or bb or cc or dd
            [[0, 1], [2, 3], [0, 0], [1, 1]],
            [[0, 1], [2, 3], [2, 2], [3, 3]],
        ])
        expect = np.array([
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
        ])
        self._test(genotypes, expect)

    def test_error_uniparental(self):
        genotypes = np.array([
            # aa x bb -> aa or bb
            [[0, 0], [1, 1], [0, 0], [1, 1]],
            [[0, 0], [2, 2], [0, 0], [2, 2]],
            [[1, 1], [2, 2], [1, 1], [2, 2]],
            # aa x bc -> aa or bc
            [[0, 0], [1, 2], [0, 0], [1, 2]],
            [[1, 1], [0, 2], [1, 1], [0, 2]],
            # ab x cd -> ab or cd
            [[0, 1], [2, 3], [0, 1], [2, 3]],
        ])
        expect = np.array([
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
        ])
        self._test(genotypes, expect)

    def test_parent_missing(self):
        genotypes = np.array([
            [[-1, -1], [0, 0], [0, 0], [1, 1]],
            [[0, 0], [-1, -1], [0, 0], [2, 2]],
            [[-1, -1], [-1, -1], [1, 1], [2, 2]],
        ])
        expect = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
        ])
        self._test(genotypes, expect)


def test_paint_transmission():

    haplotypes = []
    expect = []

    haplotypes.append([0, 0, 0, 1, 2, -1])
    expect.append([
        INHERIT_NONSEG_REF,
        INHERIT_NONPARENTAL,
        INHERIT_NONPARENTAL,
        INHERIT_MISSING,
    ])

    haplotypes.append([0, 1, 0, 1, 2, -1])
    expect.append([
        INHERIT_PARENT1,
        INHERIT_PARENT2,
        INHERIT_NONPARENTAL,
        INHERIT_MISSING,
    ])

    haplotypes.append([1, 0, 0, 1, 2, -1])
    expect.append([
        INHERIT_PARENT2,
        INHERIT_PARENT1,
        INHERIT_NONPARENTAL,
        INHERIT_MISSING,
    ])

    haplotypes.append([1, 1, 0, 1, 2, -1])
    expect.append([
        INHERIT_NONPARENTAL,
        INHERIT_NONSEG_ALT,
        INHERIT_NONPARENTAL,
        INHERIT_MISSING,
    ])

    haplotypes.append([0, 2, 0, 1, 2, -1])
    expect.append([
        INHERIT_PARENT1,
        INHERIT_NONPARENTAL,
        INHERIT_PARENT2,
        INHERIT_MISSING,
    ])

    haplotypes.append([0, -1, 0, 1, 2, -1])
    expect.append([
        INHERIT_PARENT_MISSING,
        INHERIT_PARENT_MISSING,
        INHERIT_PARENT_MISSING,
        INHERIT_MISSING,
    ])

    haplotypes.append([-1, 1, 0, 1, 2, -1])
    expect.append([
        INHERIT_PARENT_MISSING,
        INHERIT_PARENT_MISSING,
        INHERIT_PARENT_MISSING,
        INHERIT_MISSING,
    ])

    haplotypes.append([-1, -1, 0, 1, 2, -1])
    expect.append([
        INHERIT_PARENT_MISSING,
        INHERIT_PARENT_MISSING,
        INHERIT_PARENT_MISSING,
        INHERIT_MISSING,
    ])

    haplotypes = np.array(haplotypes)
    actual = paint_transmission(haplotypes[:, :2], haplotypes[:, 2:])
    assert_array_equal(expect, actual)


def test_phase_progeny_by_transmission():

    gu = []  # unphased genotypes
    gp = []  # expected genotypes after phasing
    expect_is_phased = []

    # N.B., always mother, father, children...

    # biallelic cases

    gu.append([[0, 0], [0, 0], [0, 0], [0, 1], [1, 1], [-1, -1]])
    gp.append([[0, 0], [0, 0], [0, 0], [0, 1], [1, 1], [-1, -1]])
    expect_is_phased.append([False, False, True, False, False, False])

    gu.append([[1, 1], [1, 1], [1, 1], [0, 1], [0, 0], [-1, -1]])
    gp.append([[1, 1], [1, 1], [1, 1], [0, 1], [0, 0], [-1, -1]])
    expect_is_phased.append([False, False, True, False, False, False])

    gu.append([[0, 0], [0, 1], [0, 0], [0, 1], [1, 1], [-1, -1]])
    gp.append([[0, 0], [0, 1], [0, 0], [0, 1], [1, 1], [-1, -1]])
    expect_is_phased.append([False, False, True, True, False, False])

    gu.append([[0, 1], [0, 0], [0, 0], [0, 1], [1, 1], [-1, -1]])
    gp.append([[0, 1], [0, 0], [0, 0], [1, 0], [1, 1], [-1, -1]])
    expect_is_phased.append([False, False, True, True, False, False])

    gu.append([[0, 0], [1, 1], [0, 0], [0, 1], [1, 1], [-1, -1]])
    gp.append([[0, 0], [1, 1], [0, 0], [0, 1], [1, 1], [-1, -1]])
    expect_is_phased.append([False, False, False, True, False, False])

    gu.append([[1, 1], [0, 0], [0, 0], [0, 1], [1, 1], [-1, -1]])
    gp.append([[1, 1], [0, 0], [0, 0], [1, 0], [1, 1], [-1, -1]])
    expect_is_phased.append([False, False, False, True, False, False])

    gu.append([[0, 1], [0, 1], [0, 0], [0, 1], [1, 1], [-1, -1]])
    gp.append([[0, 1], [0, 1], [0, 0], [0, 1], [1, 1], [-1, -1]])
    expect_is_phased.append([False, False, True, False, True, False])

    # some multi-allelic cases

    gu.append([[0, 0], [1, 2], [0, 0], [0, 1], [0, 2], [1, 2]])
    gp.append([[0, 0], [1, 2], [0, 0], [0, 1], [0, 2], [1, 2]])
    expect_is_phased.append([False, False, False, True, True, False])

    gu.append([[1, 2], [0, 0], [0, 0], [0, 1], [0, 2], [1, 2]])
    gp.append([[1, 2], [0, 0], [0, 0], [1, 0], [2, 0], [1, 2]])
    expect_is_phased.append([False, False, False, True, True, False])

    gu.append([[0, 1], [0, 2], [0, 0], [0, 1], [0, 2], [1, 2]])
    gp.append([[0, 1], [0, 2], [0, 0], [1, 0], [0, 2], [1, 2]])
    expect_is_phased.append([False, False, True, True, True, True])

    gu.append([[0, 2], [0, 1], [0, 0], [0, 1], [0, 2], [1, 2]])
    gp.append([[0, 2], [0, 1], [0, 0], [0, 1], [2, 0], [2, 1]])
    expect_is_phased.append([False, False, True, True, True, True])

    gu.append([[0, 1], [2, 3], [0, 2], [0, 3], [1, 2], [1, 3]])
    gp.append([[0, 1], [2, 3], [0, 2], [0, 3], [1, 2], [1, 3]])
    expect_is_phased.append([False, False, True, True, True, True])

    gu.append([[2, 3], [0, 1], [0, 2], [0, 3], [1, 2], [1, 3]])
    gp.append([[2, 3], [0, 1], [2, 0], [3, 0], [2, 1], [3, 1]])
    expect_is_phased.append([False, False, True, True, True, True])

    # run checks
    g = np.array(gu, dtype='i1')
    ga = phase_progeny_by_transmission(g)
    assert_array_equal(gp, ga)
    assert_array_equal(expect_is_phased, ga.is_phased)
