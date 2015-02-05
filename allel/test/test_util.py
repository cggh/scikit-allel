# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import unittest
from allel.test.tools import assert_array_equal as aeq
import numpy as np
from allel.util import windowed_count, windowed_density


class TestWindowedCounts(unittest.TestCase):

    def test_windowed_count(self):
        f = windowed_count
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

    def test_windowed_density(self):
        f = windowed_density
        pos = [1, 12, 15, 27]

        # boolean array, all true
        b = [True, True, True, True]
        # N.B., final bin includes right edge
        expected_densities = [1/10, 2/10, 1/11]
        expected_bin_edges = [1, 11, 21, 31]
        actual_densities, _, _, actual_bin_edges = f(pos, b, window=10)
        aeq(expected_densities, actual_densities)
        aeq(expected_bin_edges, actual_bin_edges)

        # boolean array, not all true
        b = [False, True, False, True]
        expected_densities = [0/10, 1/10, 1/11]
        expected_bin_edges = [1, 11, 21, 31]
        actual_densities, _, _, actual_bin_edges = f(pos, b, window=10)
        aeq(expected_bin_edges, actual_bin_edges)
        aeq(expected_densities, actual_densities)

        # explicit start and stop
        b = [False, True, False, True]
        expected_densities = [1/10, 0/10, 1/3]
        expected_bin_edges = [5, 15, 25, 27]
        actual_densities, _, _, actual_bin_edges = \
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
        actual_densities, _, _, actual_bin_edges = f(pos, b, window=10)
        aeq(expected_densities, actual_densities)
        aeq(expected_bin_edges, actual_bin_edges)

        # include is_accessible array option
        is_accessible = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  1, 1, 1, 1, 0, 0, 1, 1, 0, 0,
                                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=bool)
        b = [False, True, False, True]
        expected_densities = [0, 1/6, 1/11]
        expected_bin_edges = [1, 11, 21, 31]
        actual_densities, _, _, actual_bin_edges = \
            f(pos, b, window=10, is_accessible=is_accessible)
        aeq(expected_bin_edges, actual_bin_edges)
        aeq(expected_densities, actual_densities)
