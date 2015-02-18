# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import unittest


import numpy as np
from allel.test.tools import assert_array_equal as aeq, assert_array_close


from allel.model import HaplotypeArray, SortedIndex
from allel.stats import moving_statistic, windowed_statistic, \
    mean_pairwise_diversity, mean_pairwise_divergence, windowed_diversity


class TestWindowUtilities(unittest.TestCase):

    def test_moving_statistic(self):
        f = moving_statistic

        values = [2, 5, 8, 16]
        expect = [7, 24]
        actual = f(values, statistic=np.sum, size=2)
        aeq(expect, actual)

        values = [2, 5, 8, 16]
        expect = [7, 13, 24]
        actual = f(values, statistic=np.sum, size=2, step=1)
        aeq(expect, actual)

    def test_windowed_statistic(self):
        f = windowed_statistic
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
        with self.assertRaises(ValueError):
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


class TestDiversity(unittest.TestCase):

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
        actual = mean_pairwise_diversity(ac, fill=-1)
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
        actual = mean_pairwise_diversity(ac, fill=-1)
        assert_array_close(expect, actual)

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
        actual, _, _ = windowed_diversity(pos, ac, size=10, start=1, stop=31)
        assert_array_close(expect, actual)

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
        h1 = h.subset(haplotypes=[0, 1])
        h2 = h.subset(haplotypes=[2, 3])
        ac1 = h1.count_alleles()
        ac2 = h2.count_alleles()

        expect = [0/4, 2/4, 4/4, 2/4, 0/4, 4/4, 3/4, -1, -1]
        actual = mean_pairwise_divergence(ac1, ac2, fill=-1)
        aeq(expect, actual)

    # TODO test windowed_divergence
