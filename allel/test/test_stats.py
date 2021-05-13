# -*- coding: utf-8 -*-
import unittest


import numpy as np
import pytest
from pytest import approx


import allel
from allel.test.tools import assert_array_equal as aeq, assert_array_almost_equal
from allel.util import ignore_invalid, mask_inaccessible
from allel import GenotypeArray, HaplotypeArray, SortedIndex, AlleleCountsArray


class TestAccessibilityMasking(unittest.TestCase):

    def test_mask_inaccessible(self):
        np.random.seed(2837)
        for n_vars in [5, 50, 500]:
            pos = np.arange(1, n_vars+1)
            ac = np.random.randint(1, 40, n_vars*2).reshape((n_vars, 2))
            mask = np.random.randint(2, size=n_vars).astype(bool)

            mpos, mac = mask_inaccessible(mask, pos, ac)
            aeq(mac, ac[mask])
            aeq(mpos, pos[mask])

    def test_incomplete_is_accessible(self):
        # is_accessible mask has to cover all positions
        pos = np.array([1, 2, 10])
        ac = np.array([[5, 5], [2, 4]])
        mask = np.array([True, True, False])
        self.assertRaises(ValueError, mask_inaccessible, mask, pos, ac)

    def test_compatible_dims(self):
        # is_accessible mask has to cover all positions
        pos = np.array([1, 2, 10])
        mask = np.array([True, True, False])
        self.assertRaises(ValueError, mask_inaccessible, mask, pos)

    def test_masking_warning(self):
        # assert user is being warning of masking
        pos = np.array([1, 2, 3])
        mask = np.array([True, True, False])
        self.assertWarns(UserWarning, mask_inaccessible, mask, pos)

    def test_fully_masked_windowed_diversty(self):
        ac = allel.AlleleCountsArray(np.array(
            [
                [5, 5],
                [5, 5],
                [1, 9],
                [1, 9]
            ]))
        pos = np.array([1, 2, 3, 4])
        mask = np.array([False, False, True, True])
        pi, _, _, _ = allel.windowed_diversity(pos, ac, size=2, start=1,
                                               stop=5, is_accessible=mask)
        self.assertTrue(np.isnan(pi[0]))

    def test_masked_windowed_diversity(self):
        # four haplotypes, 6 pairwise comparison
        h = allel.HaplotypeArray([[0, 0, 0, 0],
                                  [0, 0, 0, 1],
                                  [0, 0, 1, 1],
                                  [0, 1, 1, 1],
                                  [1, 1, 1, 1],
                                  [0, 0, 1, 2],
                                  [0, 1, 1, 2],
                                  [0, 1, -1, -1],
                                  [-1, -1, -1, -1]])
        ac = h.count_alleles()
        # mean pairwise diversity
        # expect = [0, 3/6, 4/6, 3/6, 0, 5/6, 5/6, 1, -1]
        pos = SortedIndex([2, 4, 7, 14, 15, 18, 19, 25, 27])
        mask = np.tile(np.repeat(np.array([True, False]), 5), 3)
        # expected is every other window with size 5
        expect, _, _, _ = allel.windowed_diversity(pos, ac, size=5, start=1,
                                                   stop=31)
        # only getting every other element
        expect = expect[::2]
        # actual is window of size 10 with the last half masked out
        actual, _, _, _ = allel.windowed_diversity(pos, ac, size=10, start=1,
                                                   stop=31, is_accessible=mask)
        assert_array_almost_equal(expect, actual)

    def test_masked_windowed_divergence(self):
        h = HaplotypeArray([[0, 0, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 1, 1],
                            [0, 1, 1, 1],
                            [1, 1, 1, 1],
                            [0, 0, 1, 2],
                            [0, 1, 1, 2],
                            [0, 1, -1, -1],
                            [-1, -1, -1, -1]])
        h1 = h.take([0, 1], axis=1)
        h2 = h.take([2, 3], axis=1)
        ac1 = h1.count_alleles()
        ac2 = h2.count_alleles()
        pos = SortedIndex([2, 4, 7, 14, 15, 18, 19, 25, 27])
        mask = np.tile(np.repeat(np.array([True, False]), 5), 3)
        expect, _, _, _ = allel.windowed_divergence(pos, ac1, ac2, size=5,
                                                    start=1, stop=31)
        expect = expect[::2]
        actual, _, _, _ = allel.windowed_divergence(pos, ac1, ac2, size=10,
                                                    start=1, stop=31,
                                                    is_accessible=mask)
        assert_array_almost_equal(expect, actual)


class TestWindowUtilities(unittest.TestCase):

    def test_moving_statistic(self):
        f = allel.moving_statistic

        values = [2, 5, 8, 16]
        expect = [7, 24]
        actual = f(values, statistic=np.sum, size=2)
        aeq(expect, actual)

        values = [2, 5, 8, 16]
        expect = [7, 13, 24]
        actual = f(values, statistic=np.sum, size=2, step=1)
        aeq(expect, actual)

    def test_windowed_statistic(self):
        f = allel.windowed_statistic
        pos = [1, 12, 15, 27]

        # boolean array, all true
        b = [True, True, True, True]
        expected_nnz = [1, 2, 1]
        expected_windows = [[1, 10], [11, 20], [21, 27]]
        expected_counts = [1, 2, 1]
        actual_nnz, actual_windows, actual_counts = \
            f(pos, b, np.count_nonzero, 10)
        aeq(expected_nnz, actual_nnz)
        aeq(expected_windows, actual_windows)
        aeq(expected_counts, actual_counts)

        # boolean array, not all true
        b = [False, True, False, True]
        expected_nnz = [0, 1, 1]
        expected_windows = [[1, 10], [11, 20], [21, 27]]
        expected_counts = [1, 2, 1]
        actual_nnz, actual_windows, actual_counts = \
            f(pos, b, np.count_nonzero, 10)
        aeq(expected_windows, actual_windows)
        aeq(expected_nnz, actual_nnz)
        aeq(expected_counts, actual_counts)

        # explicit start and stop
        b = [False, True, False, True]
        expected_nnz = [1, 0, 1]
        expected_windows = [[5, 14], [15, 24], [25, 29]]
        expected_counts = [1, 1, 1]
        actual_nnz, actual_windows, actual_counts = \
            f(pos, b, np.count_nonzero, 10, start=5, stop=29)
        aeq(expected_windows, actual_windows)
        aeq(expected_nnz, actual_nnz)
        aeq(expected_counts, actual_counts)

        # boolean array, bad length
        b = [False, True, False]
        with pytest.raises(ValueError):
            f(pos, b, np.count_nonzero, 10)

        # 2D, 4 variants, 2 samples
        b = [[True, False],
             [True, True],
             [True, False],
             [True, True]]
        expected_nnz = [[1, 0],
                        [2, 1],
                        [1, 1]]
        expected_windows = [[1, 10], [11, 20], [21, 27]]
        expected_counts = [1, 2, 1]
        actual_nnz, actual_windows, actual_counts = \
            f(pos, b, statistic=lambda x: np.sum(x, axis=0), size=10)
        aeq(expected_nnz, actual_nnz)
        aeq(expected_windows, actual_windows)
        aeq(expected_counts, actual_counts)

    def test_per_base(self):
        pos = [1, 12, 15, 27]

        # boolean array, all true
        b = [True, True, True, True]
        # N.B., final bin includes right edge
        expected_nnz = [1, 2, 1]
        expected_windows = [[1, 10], [11, 20], [21, 27]]
        expected_counts = [1, 2, 1]
        expected_densities = [1/10, 2/10, 1/7]
        expected_n_bases = [10, 10, 7]
        nnz, windows, counts = allel.windowed_statistic(
            pos, b, statistic=np.count_nonzero, size=10, start=1
        )
        densities, n_bases = allel.per_base(nnz, windows)
        aeq(expected_nnz, nnz)
        aeq(expected_windows, windows)
        aeq(expected_counts, counts)
        aeq(expected_densities, densities)
        aeq(expected_n_bases, n_bases)

        # boolean array, not all true
        b = [False, True, False, True]
        expected_densities = [0/10, 1/10, 1/7]
        expected_n_bases = [10, 10, 7]
        nnz, windows, counts = allel.windowed_statistic(
            pos, b, statistic=np.count_nonzero, size=10, start=1
        )
        densities, n_bases = allel.per_base(nnz, windows)
        aeq(expected_densities, densities)
        aeq(expected_n_bases, n_bases)

        # 2D, 4 variants, 2 samples
        b = [[True, False],
             [True, True],
             [True, False],
             [True, True]]
        expected_densities = [[1/10, 0/10],
                              [2/10, 1/10],
                              [1/7, 1/7]]
        expected_n_bases = [10, 10, 7]
        nnz, windows, counts = allel.windowed_statistic(
            pos, b, statistic=lambda x: np.sum(x, axis=0), size=10, start=1
        )
        densities, n_bases = allel.per_base(nnz, windows)
        aeq(expected_densities, densities)
        aeq(expected_n_bases, n_bases)

        # include is_accessible array option
        is_accessible = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  1, 1, 1, 1, 0, 0, 1, 1, 0, 0,
                                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=bool)
        b = [False, True, False, True]
        expected_densities = [-1, 1/6, 1/7]
        expected_n_bases = [0, 6, 7]
        nnz, windows, counts = allel.windowed_statistic(
            pos, b, statistic=np.count_nonzero, size=10, start=1
        )
        densities, n_bases = allel.per_base(nnz, windows, is_accessible=is_accessible, fill=-1)
        aeq(expected_densities, densities)
        aeq(expected_n_bases, n_bases)

    def test_equally_accessible_windows(self):
        is_accessible = np.array([1, 0, 0, 1, 1, 0, 1, 0, 1])

        # default options
        actual = allel.equally_accessible_windows(is_accessible, size=2)
        expect = np.array([[1, 4], [5, 7]])
        aeq(expect, actual)

        # with step
        actual = allel.equally_accessible_windows(is_accessible, size=2, step=1)
        expect = np.array([[1, 4], [4, 5], [5, 7], [7, 9]])
        aeq(expect, actual)

        # with start and stop
        actual = allel.equally_accessible_windows(is_accessible, size=2, start=4, stop=5)
        expect = np.array([[4, 5]])
        aeq(expect, actual)


class TestDiversityDivergence(unittest.TestCase):

    def test_mean_pairwise_diversity(self):

        # start with simplest case, two haplotypes, one pairwise comparison
        h = HaplotypeArray([[0, 0],
                            [1, 1],
                            [0, 1],
                            [1, 2],
                            [0, -1],
                            [-1, -1]])
        ac = h.count_alleles()
        expect = [0, 0, 1, 1, -1, -1]
        actual = allel.mean_pairwise_difference(ac, fill=-1)
        aeq(expect, actual)

        # four haplotypes, 6 pairwise comparison
        h = HaplotypeArray([[0, 0, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 1, 1],
                            [0, 1, 1, 1],
                            [1, 1, 1, 1],
                            [0, 0, 1, 2],
                            [0, 1, 1, 2],
                            [0, 1, -1, -1],
                            [-1, -1, -1, -1]])
        ac = h.count_alleles()
        expect = [0, 3/6, 4/6, 3/6, 0, 5/6, 5/6, 1, -1]
        actual = allel.mean_pairwise_difference(ac, fill=-1)
        assert_array_almost_equal(expect, actual)

    def test_sequence_divergence(self):
        from allel import sequence_divergence
        pos = [2, 4, 8]
        ac1 = AlleleCountsArray([[2, 0],
                                 [2, 0],
                                 [2, 0]])
        ac2 = AlleleCountsArray([[0, 2],
                                 [0, 2],
                                 [0, 2]])

        # all variants
        e = 3 / 7
        a = sequence_divergence(pos, ac1, ac2)
        assert e == a

        # start/stop
        e = 2 / 6
        a = sequence_divergence(pos, ac1, ac2, start=0, stop=5)
        assert e == a

        # start/stop, an provided
        an1 = ac1.sum(axis=1)
        an2 = ac2.sum(axis=1)
        e = 2 / 6
        a = sequence_divergence(pos, ac1, ac2, start=0, stop=5, an1=an1,
                                an2=an2)
        assert e == a

    def test_windowed_diversity(self):

        # four haplotypes, 6 pairwise comparison
        h = HaplotypeArray([[0, 0, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 1, 1],
                            [0, 1, 1, 1],
                            [1, 1, 1, 1],
                            [0, 0, 1, 2],
                            [0, 1, 1, 2],
                            [0, 1, -1, -1],
                            [-1, -1, -1, -1]])
        ac = h.count_alleles()
        # mean pairwise diversity
        # expect = [0, 3/6, 4/6, 3/6, 0, 5/6, 5/6, 1, -1]
        pos = SortedIndex([2, 4, 7, 14, 15, 18, 19, 25, 27])
        expect = [(7/6)/10, (13/6)/10, 1/11]
        actual, _, _, _ = allel.windowed_diversity(pos, ac, size=10, start=1, stop=31)
        assert_array_almost_equal(expect, actual)

    def test_mean_pairwise_divergence(self):

        # simplest case, two haplotypes in each population
        h = HaplotypeArray([[0, 0, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 1, 1],
                            [0, 1, 1, 1],
                            [1, 1, 1, 1],
                            [0, 0, 1, 2],
                            [0, 1, 1, 2],
                            [0, 1, -1, -1],
                            [-1, -1, -1, -1]])
        h1 = h.take([0, 1], axis=1)
        h2 = h.take([2, 3], axis=1)
        ac1 = h1.count_alleles()
        ac2 = h2.count_alleles()

        expect = [0/4, 2/4, 4/4, 2/4, 0/4, 4/4, 3/4, -1, -1]
        actual = allel.mean_pairwise_difference_between(ac1, ac2, fill=-1)
        aeq(expect, actual)

    def test_windowed_divergence(self):

        # simplest case, two haplotypes in each population
        h = HaplotypeArray([[0, 0, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 1, 1],
                            [0, 1, 1, 1],
                            [1, 1, 1, 1],
                            [0, 0, 1, 2],
                            [0, 1, 1, 2],
                            [0, 1, -1, -1],
                            [-1, -1, -1, -1]])
        h1 = h.take([0, 1], axis=1)
        h2 = h.take([2, 3], axis=1)
        ac1 = h1.count_alleles()
        ac2 = h2.count_alleles()
        # mean pairwise divergence
        # expect = [0/4, 2/4, 4/4, 2/4, 0/4, 4/4, 3/4, -1, -1]
        pos = SortedIndex([2, 4, 7, 14, 15, 18, 19, 25, 27])
        expect = [(6/4)/10, (9/4)/10, 0/11]
        actual, _, _, _ = allel.windowed_divergence(
            pos, ac1, ac2, size=10, start=1, stop=31
        )
        assert_array_almost_equal(expect, actual)

    def test_tajima_d(self):
        from allel import tajima_d

        # example with calculable value
        ac = AlleleCountsArray([[1, 3],
                                [2, 2],
                                [3, 1]])
        expect = approx(0.168, 0.01)
        actual = tajima_d(ac)
        assert expect == actual

        # too few sites
        ac = AlleleCountsArray([[2, 2],
                                [3, 1]])
        assert np.nan is tajima_d(ac)

        # too few segregating sites
        ac = AlleleCountsArray([[4, 0],
                                [2, 2],
                                [3, 1]])
        assert np.nan is tajima_d(ac)
        # allow people to override if they really want to
        assert approx(0.592, 0.01) == tajima_d(ac, min_sites=2)

    def test_moving_tajima_d(self):
        from allel import moving_tajima_d

        # example with calculable value
        ac = AlleleCountsArray([[1, 3],
                                [2, 2],
                                [3, 1],
                                [1, 3],
                                [2, 2]])
        expect = np.array([0.168] * 3)
        actual = moving_tajima_d(ac, size=3, step=1)
        assert_array_almost_equal(expect, actual, decimal=3)

        # too few sites
        actual = moving_tajima_d(ac, size=2, step=1)
        assert 4 == len(actual)
        assert np.all(np.isnan(actual))

        # too few segregating sites
        ac = AlleleCountsArray([[4, 0],
                                [2, 2],
                                [3, 1],
                                [4, 0],
                                [2, 2]])
        actual = moving_tajima_d(ac, size=3, step=1)
        assert 3 == len(actual)
        assert np.all(np.isnan(actual))
        # allow people to override if they really want to
        expect = np.array([0.592] * 3)
        actual = moving_tajima_d(ac, size=3, step=1, min_sites=2)
        assert_array_almost_equal(expect, actual, decimal=3)

    def test_windowed_tajima_d(self):
        from allel import windowed_tajima_d

        pos = np.array([1, 11, 21, 31, 41])

        # example with calculable value
        ac = AlleleCountsArray([[1, 3],
                                [2, 2],
                                [3, 1],
                                [1, 3],
                                [2, 2]])
        expect = np.array([0.168] * 3)
        actual, _, _ = windowed_tajima_d(pos, ac, size=25, step=10)
        assert_array_almost_equal(expect, actual, decimal=3)

        # too few sites
        actual, _, _ = windowed_tajima_d(pos, ac, size=15, step=10)
        assert 4 == len(actual)
        assert np.all(np.isnan(actual))

        # too few segregating sites
        ac = AlleleCountsArray([[4, 0],
                                [2, 2],
                                [3, 1],
                                [4, 0],
                                [2, 2]])
        actual, _, _ = windowed_tajima_d(pos, ac, size=25, step=10)
        assert 3 == len(actual)
        assert np.all(np.isnan(actual))
        # allow people to override if they really want to
        expect = np.array([0.592] * 3)
        actual, _, _ = windowed_tajima_d(pos, ac, size=25, step=10, min_sites=2)
        assert_array_almost_equal(expect, actual, decimal=3)


class TestHardyWeinberg(unittest.TestCase):

    def test_heterozygosity_observed(self):

        # diploid
        g = GenotypeArray([[[0, 0], [0, 0]],
                           [[1, 1], [1, 1]],
                           [[1, 1], [2, 2]],
                           [[0, 0], [0, 1]],
                           [[0, 0], [0, 2]],
                           [[1, 1], [1, 2]],
                           [[0, 1], [0, 1]],
                           [[0, 1], [1, 2]],
                           [[0, 0], [-1, -1]],
                           [[0, 1], [-1, -1]],
                           [[-1, -1], [-1, -1]]], dtype='i1')
        expect = [0, 0, 0, .5, .5, .5, 1, 1, 0, 1, -1]
        actual = allel.heterozygosity_observed(g, fill=-1)
        aeq(expect, actual)

        # polyploid
        g = GenotypeArray([[[0, 0, 0], [0, 0, 0]],
                           [[1, 1, 1], [1, 1, 1]],
                           [[1, 1, 1], [2, 2, 2]],
                           [[0, 0, 0], [0, 0, 1]],
                           [[0, 0, 0], [0, 0, 2]],
                           [[1, 1, 1], [0, 1, 2]],
                           [[0, 0, 1], [0, 1, 1]],
                           [[0, 1, 1], [0, 1, 2]],
                           [[0, 0, 0], [-1, -1, -1]],
                           [[0, 0, 1], [-1, -1, -1]],
                           [[-1, -1, -1], [-1, -1, -1]]], dtype='i1')
        expect = [0, 0, 0, .5, .5, .5, 1, 1, 0, 1, -1]
        actual = allel.heterozygosity_observed(g, fill=-1)
        aeq(expect, actual)

    def test_heterozygosity_expected(self):

        def refimpl(f, ploidy, fill=0):
            """Limited reference implementation for testing purposes."""

            # check allele frequencies sum to 1
            af_sum = np.sum(f, axis=1)

            # assume three alleles
            p = f[:, 0]
            q = f[:, 1]
            r = f[:, 2]

            out = 1 - p**ploidy - q**ploidy - r**ploidy
            with ignore_invalid():
                out[(af_sum < 1) | np.isnan(af_sum)] = fill

            return out

        # diploid
        g = GenotypeArray([[[0, 0], [0, 0]],
                           [[1, 1], [1, 1]],
                           [[1, 1], [2, 2]],
                           [[0, 0], [0, 1]],
                           [[0, 0], [0, 2]],
                           [[1, 1], [1, 2]],
                           [[0, 1], [0, 1]],
                           [[0, 1], [1, 2]],
                           [[0, 0], [-1, -1]],
                           [[0, 1], [-1, -1]],
                           [[-1, -1], [-1, -1]]], dtype='i1')
        expect1 = [0, 0, 0.5, .375, .375, .375, .5, .625, 0, .5, -1]
        af = g.count_alleles().to_frequencies()
        expect2 = refimpl(af, ploidy=g.ploidy, fill=-1)
        actual = allel.heterozygosity_expected(af, ploidy=g.ploidy, fill=-1)
        assert_array_almost_equal(expect1, actual)
        assert_array_almost_equal(expect2, actual)
        expect3 = [0, 0, 0.5, .375, .375, .375, .5, .625, 0, .5, 0]
        actual = allel.heterozygosity_expected(af, ploidy=g.ploidy, fill=0)
        assert_array_almost_equal(expect3, actual)

        # polyploid
        g = GenotypeArray([[[0, 0, 0], [0, 0, 0]],
                           [[1, 1, 1], [1, 1, 1]],
                           [[1, 1, 1], [2, 2, 2]],
                           [[0, 0, 0], [0, 0, 1]],
                           [[0, 0, 0], [0, 0, 2]],
                           [[1, 1, 1], [0, 1, 2]],
                           [[0, 0, 1], [0, 1, 1]],
                           [[0, 1, 1], [0, 1, 2]],
                           [[0, 0, 0], [-1, -1, -1]],
                           [[0, 0, 1], [-1, -1, -1]],
                           [[-1, -1, -1], [-1, -1, -1]]], dtype='i1')
        af = g.count_alleles().to_frequencies()
        expect = refimpl(af, ploidy=g.ploidy, fill=-1)
        actual = allel.heterozygosity_expected(af, ploidy=g.ploidy, fill=-1)
        assert_array_almost_equal(expect, actual)

    def test_inbreeding_coefficient(self):

        # diploid
        g = GenotypeArray([[[0, 0], [0, 0]],
                           [[1, 1], [1, 1]],
                           [[1, 1], [2, 2]],
                           [[0, 0], [0, 1]],
                           [[0, 0], [0, 2]],
                           [[1, 1], [1, 2]],
                           [[0, 1], [0, 1]],
                           [[0, 1], [1, 2]],
                           [[0, 0], [-1, -1]],
                           [[0, 1], [-1, -1]],
                           [[-1, -1], [-1, -1]]], dtype='i1')
        # ho = np.array([0, 0, 0, .5, .5, .5, 1, 1, 0, 1, -1])
        # he = np.array([0, 0, 0.5, .375, .375, .375, .5, .625, 0, .5, -1])
        # expect = 1 - (ho/he)
        expect = [-1, -1, 1-0, 1-(.5/.375), 1-(.5/.375), 1-(.5/.375),
                  1-(1/.5), 1-(1/.625), -1, 1-(1/.5), -1]
        actual = allel.inbreeding_coefficient(g, fill=-1)
        assert_array_almost_equal(expect, actual)


class TestDistance(unittest.TestCase):

    def test_pdist(self):
        from allel.stats.distance import pdist
        h = HaplotypeArray([[0, 0, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 1, 1],
                            [0, 1, 1, 1],
                            [1, 1, 1, 1],
                            [0, 0, 1, 2],
                            [0, 1, 1, 2],
                            [0, 1, -1, -1],
                            [-1, -1, -1, -1]])
        import scipy.spatial
        d1 = scipy.spatial.distance.pdist(h.T, 'hamming')
        d2 = pdist(h, 'hamming')
        aeq(d1, d2)

    def test_pairwise_distance_multidim(self):
        g = GenotypeArray([[[0, 0], [0, 0]],
                           [[1, 1], [1, 1]],
                           [[1, 1], [2, 2]],
                           [[0, 0], [0, 1]],
                           [[0, 0], [0, 2]],
                           [[1, 1], [1, 2]],
                           [[0, 1], [0, 1]],
                           [[0, 1], [1, 2]],
                           [[0, 0], [-1, -1]],
                           [[0, 1], [-1, -1]],
                           [[-1, -1], [-1, -1]]], dtype='i1')
        gac = g.to_allele_counts()

        def metric(ac1, ac2):
            mpd = allel.mean_pairwise_difference_between(ac1, ac2, fill=0)
            return mpd.sum()

        expect = [allel.mean_pairwise_difference_between(gac[:, 0], gac[:, 1], fill=0).sum()]
        actual = allel.pairwise_distance(gac, metric)
        aeq(expect, actual)

    def test_condensed_coords(self):
        from allel import condensed_coords
        assert 0 == condensed_coords(0, 1, 2)
        assert 0 == condensed_coords(1, 0, 2)
        assert 0 == condensed_coords(0, 1, 3)
        assert 0 == condensed_coords(1, 0, 3)
        assert 1 == condensed_coords(0, 2, 3)
        assert 1 == condensed_coords(2, 0, 3)
        assert 2 == condensed_coords(1, 2, 3)
        assert 2 == condensed_coords(2, 1, 3)

        with pytest.raises(ValueError):
            condensed_coords(0, 0, 1)
            condensed_coords(0, 1, 1)
            condensed_coords(1, 0, 1)
            condensed_coords(0, 0, 2)
            condensed_coords(0, 2, 2)
            condensed_coords(2, 0, 2)
            condensed_coords(1, 1, 2)
            condensed_coords(0, 0, 3)
            condensed_coords(1, 1, 3)
            condensed_coords(2, 2, 3)

    def test_condensed_coords_within(self):
        from allel import condensed_coords_within

        pop = [0, 1]
        n = 3
        expect = [0]
        actual = condensed_coords_within(pop, n)
        assert expect == actual

        pop = [0, 2]
        n = 3
        expect = [1]
        actual = condensed_coords_within(pop, n)
        assert expect == actual

        pop = [1, 2]
        n = 3
        expect = [2]
        actual = condensed_coords_within(pop, n)
        assert expect == actual

        pop = [0, 1, 3]
        n = 4
        expect = [0, 2, 4]
        actual = condensed_coords_within(pop, n)
        assert expect == actual

        pop = [0, 0]
        with pytest.raises(ValueError):
            condensed_coords_within(pop, n)

    def test_condensed_coords_between(self):
        from allel import condensed_coords_between

        pop1 = [0, 1]
        pop2 = [2, 3]
        n = 4
        expect = [1, 2, 3, 4]
        actual = condensed_coords_between(pop1, pop2, n)
        assert expect == actual

        pop1 = [0, 2]
        pop2 = [1, 3]
        n = 4
        expect = [0, 2, 3, 5]
        actual = condensed_coords_between(pop1, pop2, n)
        assert expect == actual

        with pytest.raises(ValueError):
            condensed_coords_between(pop1, pop1, n)


class TestLinkageDisequilibrium(unittest.TestCase):

    def test_rogers_huff_r(self):

        gn = [[0, 1, 2],
              [0, 1, 2]]
        expect = 1.
        actual = allel.rogers_huff_r(gn)
        assert expect == actual

        gn = [[0, 1, 2],
              [2, 1, 0]]
        expect = -1.
        actual = allel.rogers_huff_r(gn)
        assert expect == actual

        gn = [[0, 0, 0],
              [0, 0, 0]]
        actual = allel.rogers_huff_r(gn)
        assert np.isnan(actual)

        gn = [[0, 0, 0],
              [1, 1, 1]]
        actual = allel.rogers_huff_r(gn)
        assert np.isnan(actual)

        gn = [[1, 1, 1],
              [1, 1, 1]]
        actual = allel.rogers_huff_r(gn)
        assert np.isnan(actual)

        gn = [[0, -1, 0],
              [-1, 1, -1]]
        actual = allel.rogers_huff_r(gn)
        assert np.isnan(actual)

        gn = [[0, 1, 0],
              [-1, -1, -1]]
        actual = allel.rogers_huff_r(gn)
        assert np.isnan(actual)

        gn = [[0, 1, 0, 1],
              [0, 1, 1, 0]]
        expect = 0
        actual = allel.rogers_huff_r(gn)
        assert expect == actual

        gn = [[0, 1, 2, -1],
              [0, 1, 2, 2]]
        expect = 1.
        actual = allel.rogers_huff_r(gn)
        assert expect == actual

        gn = [[0, 1, 2, 2],
              [0, 1, 2, -1]]
        expect = 1.
        actual = allel.rogers_huff_r(gn)
        assert expect == actual

        gn = [[0, 1, 2],
              [0, 1, -1]]
        expect = 1.
        actual = allel.rogers_huff_r(gn)
        assert expect == actual

        gn = [[0, 2],
              [2, 0],
              [0, 1]]
        expect = [-1, 1, -1]
        actual = allel.rogers_huff_r(gn)
        assert_array_almost_equal(expect, actual)

        gn = [[0, 2, 0],
              [0, 2, 0],
              [2, 0, 2],
              [0, 2, -1]]
        expect = [1, -1, 1, -1, 1, -1]
        actual = allel.rogers_huff_r(gn)
        assert_array_almost_equal(expect, actual)

    def test_rogers_huff_r_between(self):

        gna = [[0, 1, 2]]
        gnb = [[0, 1, 2]]
        expect = 1.
        actual = allel.rogers_huff_r_between(gna, gnb)
        assert expect == actual

        gna = [[0, 1, 2]]
        gnb = [[2, 1, 0]]
        expect = -1.
        actual = allel.rogers_huff_r_between(gna, gnb)
        assert expect == actual

        gna = [[0, 0, 0]]
        gnb = [[1, 1, 1]]
        actual = allel.rogers_huff_r_between(gna, gnb)
        assert np.isnan(actual)

    def test_locate_unlinked(self):

        gn = [[0, 1, 2],
              [0, 1, 2]]
        expect = [True, False]
        actual = allel.locate_unlinked(gn, size=2, step=2, threshold=.5)
        aeq(expect, actual)

        gn = [[0, 1, 1, 2],
              [0, 1, 1, 2],
              [1, 1, 0, 2],
              [1, 1, 0, 2]]
        actual = allel.locate_unlinked(gn, size=2, step=1, threshold=.5)
        expect = [True, False, True, False]
        aeq(expect, actual)

        gn = [[0, 1, 1, 2],
              [0, 1, 1, 2],
              [0, 1, 1, 2],
              [1, 1, 0, 2],
              [1, 1, 0, 2]]
        actual = allel.locate_unlinked(gn, size=2, step=1, threshold=.5)
        expect = [True, False, True, True, False]
        aeq(expect, actual)
        actual = allel.locate_unlinked(gn, size=3, step=1, threshold=.5)
        expect = [True, False, False, True, False]
        aeq(expect, actual)


class TestAdmixture(unittest.TestCase):

    def test_patterson_f2(self):
        aca = [[0, 2],
               [2, 0],
               [1, 1],
               [0, 0]]
        acb = [[0, 2],
               [0, 2],
               [0, 2],
               [0, 2]]
        expect = [0., 1., 0., np.nan]
        actual = allel.patterson_f2(aca, acb)
        assert_array_almost_equal(expect, actual)

    def test_patterson_f3(self):
        aca = [[0, 2],
               [2, 0],
               [0, 2],
               [0, 2],
               [0, 0]]
        acb = [[2, 0],
               [0, 2],
               [0, 2],
               [0, 2],
               [0, 2]]
        acc = [[1, 1],
               [1, 1],
               [0, 2],
               [2, 0],
               [1, 1]]
        expect_f3 = [-.5, -.5, 0., 1., np.nan]
        actual_f3, actual_hzc = allel.patterson_f3(acc, aca, acb)
        assert_array_almost_equal(expect_f3, actual_f3)
        expect_hzc = [1., 1., 0., 0., 1.]
        assert_array_almost_equal(expect_hzc, actual_hzc)

    def test_patterson_d(self):
        aca = [[0, 2],
               [2, 0],
               [2, 0],
               [1, 1],
               [0, 0]]
        acb = [[0, 2],
               [0, 2],
               [0, 2],
               [1, 1],
               [0, 2]]
        acc = [[2, 0],
               [2, 0],
               [0, 2],
               [1, 1],
               [0, 2]]
        acd = [[2, 0],
               [0, 2],
               [2, 0],
               [1, 1],
               [0, 2]]
        num, den = allel.patterson_d(aca, acb, acc, acd)
        expect_num = [0., 1., -1., 0., np.nan]
        expect_den = [0., 1., 1., 0.25, np.nan]
        assert_array_almost_equal(expect_num, num)
        assert_array_almost_equal(expect_den, den)


class TestRunsOfHomozygosity(unittest.TestCase):

    def test_roh_mhmm_100pct(self):

        # values correspond to start/stop/length/is_marginal
        roh_expected = np.array([[1, 100, 100, True]], dtype=object)
        fraction_expected = 1.0
        gv = np.zeros((4, 2), dtype=np.int16)
        pos = [1, 10, 50, 100]
        roh, fraction = allel.roh_mhmm(gv, pos, contig_size=100)
        aeq(roh.values, roh_expected)
        assert fraction == fraction_expected

    def test_roh_mhmm_0pct(self):

        fraction_expected = 0.0

        gv = np.zeros((4, 2), dtype=np.int16)
        gv[2, 0] = 1

        pos = [1, 10, 50, 100]
        roh, fraction = allel.roh_mhmm(gv, pos, contig_size=100)
        assert roh.shape[0] == 0
        assert fraction == fraction_expected
