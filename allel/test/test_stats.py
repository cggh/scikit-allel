# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import unittest
from allel.test.tools import assert_array_equal as aeq, assert_array_close
import numpy as np


from allel.model import GenotypeArray, PositionIndex
from allel.stats import windowed_nnz, windowed_nnz_per_base, \
    windowed_nucleotide_diversity


class TestWindowedCounts(unittest.TestCase):

    def test_windowed_nnz(self):
        f = windowed_nnz
        pos = [1, 12, 15, 27]

        # boolean array, all true
        b = [True, True, True, True]
        expected_counts = [1, 2, 1]
        expected_bin_edges = [1, 11, 21, 31]
        actual_counts, actual_bin_edges = f(pos, b, window=10)
        aeq(expected_counts, actual_counts)
        aeq(expected_bin_edges, actual_bin_edges)

        # boolean array, not all true
        b = [False, True, False, True]
        expected_counts = [0, 1, 1]
        expected_bin_edges = [1, 11, 21, 31]
        actual_counts, actual_bin_edges = f(pos, b, window=10)
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
        with self.assertRaises(ValueError):
            f(pos, b, window=10)

        # 2D, 4 variants, 2 samples
        b = [[True, False],
             [True, True],
             [True, False],
             [True, True]]
        expected_counts = [[1, 0],
                           [2, 1],
                           [1, 1]]
        expected_bin_edges = [1, 11, 21, 31]
        actual_counts, actual_bin_edges = f(pos, b, window=10)
        aeq(expected_counts, actual_counts)
        aeq(expected_bin_edges, actual_bin_edges)

    def test_windowed_nnz_per_base(self):
        f = windowed_nnz_per_base
        pos = [1, 12, 15, 27]

        # boolean array, all true
        b = [True, True, True, True]
        # N.B., final bin includes right edge
        expected_densities = [1/10, 2/10, 1/11]
        expected_bin_edges = [1, 11, 21, 31]
        actual_densities, actual_bin_edges, _, _ = f(pos, b, window=10)
        aeq(expected_densities, actual_densities)
        aeq(expected_bin_edges, actual_bin_edges)

        # boolean array, not all true
        b = [False, True, False, True]
        expected_densities = [0/10, 1/10, 1/11]
        expected_bin_edges = [1, 11, 21, 31]
        actual_densities, actual_bin_edges, _, _ = f(pos, b, window=10)
        aeq(expected_bin_edges, actual_bin_edges)
        aeq(expected_densities, actual_densities)

        # explicit start and stop
        b = [False, True, False, True]
        expected_densities = [1/10, 0/10, 1/3]
        expected_bin_edges = [5, 15, 25, 27]
        actual_densities, actual_bin_edges, _, _ = \
            f(pos, b, window=10, start=5, stop=27)
        aeq(expected_bin_edges, actual_bin_edges)
        aeq(expected_densities, actual_densities)

        # boolean array, bad length
        b = [False, True, False]
        with self.assertRaises(ValueError):
            f(pos, b, window=10)

        # 2D, 4 variants, 2 samples
        b = [[True, False],
             [True, True],
             [True, False],
             [True, True]]
        expected_densities = [[1/10, 0/10],
                              [2/10, 1/10],
                              [1/11, 1/11]]
        expected_bin_edges = [1, 11, 21, 31]
        actual_densities, actual_bin_edges, _, _ = f(pos, b, window=10)
        aeq(expected_densities, actual_densities)
        aeq(expected_bin_edges, actual_bin_edges)

        # include is_accessible array option
        is_accessible = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  1, 1, 1, 1, 0, 0, 1, 1, 0, 0,
                                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=bool)
        b = [False, True, False, True]
        expected_densities = [-1, 1/6, 1/11]
        expected_bin_edges = [1, 11, 21, 31]
        actual_densities, actual_bin_edges, _, _ = \
            f(pos, b, window=10, is_accessible=is_accessible, fill=-1)
        aeq(expected_bin_edges, actual_bin_edges)
        aeq(expected_densities, actual_densities)

    def test_windowed_nucleotide_diversity(self):

        g = GenotypeArray([[[0, 0], [0, 0]],
                           [[0, 0], [0, 1]],
                           [[0, 0], [1, 1]],
                           [[0, 1], [1, 1]],
                           [[1, 1], [1, 1]],
                           [[0, 0], [1, 2]],
                           [[0, 1], [1, 2]],
                           [[0, 1], [-1, -1]],
                           [[-1, -1], [-1, -1]]])
        pos = PositionIndex([2, 4, 7, 14, 15, 18, 19, 25, 27])
        # mean pairwise differences
        # expect = [0, 3/6, 4/6, 3/6, 0, 5/6, 5/6, 1, -1]
        expect = [(7/6)/10, (13/6)/10, 1/11]
        actual, _, _ = windowed_nucleotide_diversity(g, pos, window=10)
        assert_array_close(expect, actual)
