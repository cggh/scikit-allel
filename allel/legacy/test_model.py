# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import unittest
import numpy as np
from allel.test.tools import assert_array_equal as aeq, assert_array_close


from allel.model import GenotypeArray, HaplotypeArray, SortedIndex, \
    UniqueIndex


haplotype_data = [[0, 1, -1],
                  [1, 1, -1],
                  [2, -1, -1],
                  [-1, -1, -1]]

diploid_genotype_data = [[[0, 0], [0, 1], [-1, -1]],
                         [[0, 2], [1, 1], [-1, -1]],
                         [[1, 0], [2, 1], [-1, -1]],
                         [[2, 2], [-1, -1], [-1, -1]],
                         [[-1, -1], [-1, -1], [-1, -1]]]

triploid_genotype_data = [[[0, 0, 0], [0, 0, 1], [-1, -1, -1]],
                          [[0, 1, 1], [1, 1, 1], [-1, -1, -1]],
                          [[0, 1, 2], [-1, -1, -1], [-1, -1, -1]],
                          [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]]


class TestPositionIndex(unittest.TestCase):

    def test_constructor(self):
        eq = self.assertEqual

        # missing data arg
        with self.assertRaises(TypeError):
            # noinspection PyArgumentList
            SortedIndex()

        # data has wrong dtype
        data = 'foo bar'
        with self.assertRaises(TypeError):
            SortedIndex(data)

        # data has wrong dtype
        data = [4., 5., 3.7]
        with self.assertRaises(TypeError):
            SortedIndex(data)

        # data has wrong dimensions
        data = [[1, 2], [3, 4]]
        with self.assertRaises(TypeError):
            SortedIndex(data)

        # positions are not sorted
        data = [2, 1, 3, 5]
        with self.assertRaises(ValueError):
            SortedIndex(data)

        # valid data (unique)
        data = [1, 4, 5, 7, 12]
        pos = SortedIndex(data)
        aeq(data, pos)
        eq(np.int, pos.dtype)
        eq(1, pos.ndim)
        eq(5, len(pos))
        assert pos.is_unique

        # valid data (non-unique)
        data = [1, 4, 5, 5, 7, 12]
        pos = SortedIndex(data)
        aeq(data, pos)
        eq(np.int, pos.dtype)
        eq(1, pos.ndim)
        eq(6, len(pos))
        assert not pos.is_unique

        # valid data (typed)
        data = [1, 4, 5, 5, 7, 12]
        pos = SortedIndex(data, dtype='u4')
        aeq(data, pos)
        eq(np.uint32, pos.dtype)

    def test_slice(self):
        eq = self.assertEqual

        data = [1, 4, 5, 5, 7, 12]
        pos = SortedIndex(data, dtype='u4')

        # row slice
        s = pos[1:]
        self.assertIsInstance(s, SortedIndex)
        aeq(data[1:], s)
        eq(5, len(s))
        assert not s.is_unique

        # row slice
        s = pos[3:]
        self.assertIsInstance(s, SortedIndex)
        aeq(data[3:], s)
        eq(3, len(s))
        assert s.is_unique

        # index
        s = pos[0]
        self.assertIsInstance(s, np.uint32)
        self.assertNotIsInstance(s, SortedIndex)
        eq(data[0], s)

    def test_view(self):
        eq = self.assertEqual

        # data has wrong dtype
        data = 'foo bar'
        with self.assertRaises(TypeError):
            np.asarray(data).view(SortedIndex)

        # data has wrong dtype
        data = [4., 5., 3.7]
        with self.assertRaises(TypeError):
            np.asarray(data).view(SortedIndex)

        # data has wrong dimensions
        data = [[1, 2], [3, 4]]
        with self.assertRaises(TypeError):
            np.asarray(data).view(SortedIndex)

        # positions are not sorted
        data = [2, 1, 3, 5]
        with self.assertRaises(ValueError):
            np.asarray(data).view(SortedIndex)

        # valid data (unique)
        data = [1, 4, 5, 7, 12]
        pos = np.asarray(data).view(SortedIndex)
        aeq(data, pos)
        eq(np.int, pos.dtype)
        eq(1, pos.ndim)
        eq(5, len(pos))
        assert pos.is_unique

        # valid data (non-unique)
        data = [1, 4, 5, 5, 7, 12]
        pos = np.asarray(data).view(SortedIndex)
        aeq(data, pos)
        eq(np.int, pos.dtype)
        eq(1, pos.ndim)
        eq(6, len(pos))
        assert not pos.is_unique

        # valid data (typed)
        data = np.array([1, 4, 5, 5, 7, 12], dtype='u4')
        pos = np.asarray(data).view(SortedIndex)
        aeq(data, pos)
        eq(np.uint32, pos.dtype)

    def test_locate_key(self):
        eq = self.assertEqual
        pos = SortedIndex([3, 6, 6, 11])
        f = pos.locate_key
        eq(0, f(3))
        eq(3, f(11))
        eq(slice(1, 3), f(6))
        with self.assertRaises(KeyError):
            f(2)

    def test_locate_keys(self):
        pos = SortedIndex([3, 6, 6, 11, 20, 35])
        f = pos.locate_keys

        # all found
        expect = [False, True, True, False, True, False]
        actual = f([6, 20])
        self.assertNotIsInstance(actual, SortedIndex)
        aeq(expect, actual)

        # not all found, lax
        expect = [False, True, True, False, True, False]
        actual = f([2, 6, 17, 20, 37], strict=False)
        self.assertNotIsInstance(actual, SortedIndex)
        aeq(expect, actual)

        # not all found, strict
        with self.assertRaises(KeyError):
            f([2, 6, 17, 20, 37])

    def test_locate_intersection(self):
        pos1 = SortedIndex([3, 6, 11, 20, 35])
        pos2 = SortedIndex([4, 6, 20, 39])
        expect_loc1 = np.array([False, True, False, True, False])
        expect_loc2 = np.array([False, True, True, False])
        loc1, loc2 = pos1.locate_intersection(pos2)
        self.assertNotIsInstance(loc1, SortedIndex)
        self.assertNotIsInstance(loc2, SortedIndex)
        aeq(expect_loc1, loc1)
        aeq(expect_loc2, loc2)

    def test_intersect(self):
        pos1 = SortedIndex([3, 6, 11, 20, 35])
        pos2 = SortedIndex([4, 6, 20, 39])
        expect = SortedIndex([6, 20])
        actual = pos1.intersect(pos2)
        self.assertIsInstance(actual, SortedIndex)
        aeq(expect, actual)

    def test_locate_range(self):
        eq = self.assertEqual
        pos = SortedIndex([3, 6, 11, 20, 35])
        f = pos.locate_range
        eq(slice(0, 5), f(2, 37))
        eq(slice(0, 5), f(3, 35))
        eq(slice(1, 5), f(4, 37))
        eq(slice(1, 5), f(start=4))
        eq(slice(0, 4), f(2, 32))
        eq(slice(0, 4), f(stop=32))
        eq(slice(1, 4), f(4, 32))
        eq(slice(1, 3), f(4, 19))
        eq(slice(2, 4), f(7, 32))
        eq(slice(2, 3), f(7, 19))
        with self.assertRaises(KeyError):
            f(17, 19)
        with self.assertRaises(KeyError):
            print(f(0, 2))
        with self.assertRaises(KeyError):
            f(36, 2000)

    def test_intersect_range(self):
        pos = SortedIndex([3, 6, 11, 20, 35])
        f = pos.intersect_range
        aeq(pos[:], f(2, 37))
        aeq(pos[:], f(3, 35))
        aeq(pos[1:], f(4, 37))
        aeq(pos[1:], f(start=4))
        aeq(pos[:4], f(2, 32))
        aeq(pos[:4], f(stop=32))
        aeq(pos[1:4], f(4, 32))
        aeq(pos[1:3], f(4, 19))
        aeq(pos[2:4], f(7, 32))
        aeq(pos[2:3], f(7, 19))
        aeq([], f(17, 19))
        aeq([], f(0, 2))
        aeq([], f(36, 2000))

    def test_locate_ranges(self):
        pos = SortedIndex([3, 6, 11, 20, 35])

        # all found
        ranges = np.array([[6, 12], [31, 35]])
        expect = np.array([False, True, True, False, True])
        actual = pos.locate_ranges(ranges[:, 0], ranges[:, 1])
        self.assertNotIsInstance(actual, SortedIndex)
        aeq(expect, actual)

        # not all found, lax
        ranges = np.array([[0, 2], [6, 12], [14, 19], [31, 35], [100, 120]])
        actual = pos.locate_ranges(ranges[:, 0], ranges[:, 1], strict=False)
        self.assertNotIsInstance(actual, SortedIndex)
        aeq(expect, actual)

        # not all found, strict
        with self.assertRaises(KeyError):
            pos.locate_ranges(ranges[:, 0], ranges[:, 1])

    def test_locate_intersection_ranges(self):
        pos = SortedIndex([3, 6, 11, 20, 35])
        f = pos.locate_intersection_ranges

        # all found
        ranges = np.array([[6, 12], [31, 35]])
        expect_loc1 = np.array([False, True, True, False, True])
        expect_loc2 = np.array([True, True])
        actual_loc1, actual_loc2 = f(ranges[:, 0], ranges[:, 1])
        self.assertNotIsInstance(actual_loc1, SortedIndex)
        self.assertNotIsInstance(actual_loc2, SortedIndex)
        aeq(expect_loc1, actual_loc1)
        aeq(expect_loc2, actual_loc2)

        # not all found
        ranges = np.array([[0, 2], [6, 12], [14, 19], [31, 35], [100, 120]])
        expect_loc1 = np.array([False, True, True, False, True])
        expect_loc2 = np.array([False, True, False, True, False])
        actual_loc1, actual_loc2 = f(ranges[:, 0], ranges[:, 1])
        self.assertNotIsInstance(actual_loc1, SortedIndex)
        self.assertNotIsInstance(actual_loc2, SortedIndex)
        aeq(expect_loc1, actual_loc1)
        aeq(expect_loc2, actual_loc2)

    def test_intersect_ranges(self):
        pos = SortedIndex([3, 6, 11, 20, 35])
        f = pos.intersect_ranges

        # all found
        ranges = np.array([[6, 12], [31, 35]])
        expect = [6, 11, 35]
        actual = f(ranges[:, 0], ranges[:, 1])
        self.assertIsInstance(actual, SortedIndex)
        aeq(expect, actual)

        # not all found
        ranges = np.array([[0, 2], [6, 12], [14, 19], [31, 35], [100, 120]])
        expect = [6, 11, 35]
        actual = f(ranges[:, 0], ranges[:, 1])
        self.assertIsInstance(actual, SortedIndex)
        aeq(expect, actual)


class TestLabelIndex(unittest.TestCase):

    def test_constructor(self):
        eq = self.assertEqual

        # missing data arg
        with self.assertRaises(TypeError):
            # noinspection PyArgumentList
            UniqueIndex()

        # data has wrong dimensions
        data = [['A', 'C'], ['B', 'F']]
        with self.assertRaises(TypeError):
            UniqueIndex(data)

        # labels are not unique
        data = ['A', 'B', 'D', 'B']
        with self.assertRaises(ValueError):
            UniqueIndex(data)

        # valid data
        data = ['A', 'C', 'B', 'F']
        lbl = UniqueIndex(data)
        aeq(data, lbl)
        eq(1, lbl.ndim)
        eq(4, len(lbl))

        # valid data (typed)
        data = np.array(['A', 'C', 'B', 'F'], dtype='S1')
        lbl = UniqueIndex(data, dtype='S1')
        aeq(data, lbl)

    def test_slice(self):
        eq = self.assertEqual

        data = ['A', 'C', 'B', 'F']
        lbl = UniqueIndex(data)

        # row slice
        s = lbl[1:]
        self.assertIsInstance(s, UniqueIndex)
        aeq(data[1:], s)
        eq(3, len(s))

        # index
        s = lbl[0]
        self.assertIsInstance(s, str)
        self.assertNotIsInstance(s, UniqueIndex)
        eq(data[0], s)

    def test_view(self):
        eq = self.assertEqual

        # data has wrong dimensions
        data = [['A', 'C'], ['B', 'F']]
        with self.assertRaises(TypeError):
            np.array(data).view(UniqueIndex)

        # labels are not unique
        data = ['A', 'B', 'D', 'B']
        with self.assertRaises(ValueError):
            np.array(data).view(UniqueIndex)

        # valid data
        data = ['A', 'C', 'B', 'F']
        lbl = np.array(data).view(UniqueIndex)
        aeq(data, lbl)
        eq(1, lbl.ndim)
        eq(4, len(lbl))

        # valid data (typed)
        data = np.array(['A', 'C', 'B', 'F'], dtype='a1')
        lbl = data.view(UniqueIndex)
        aeq(data, lbl)

    def test_locate_key(self):
        eq = self.assertEqual
        lbl = UniqueIndex(['A', 'C', 'B', 'F'])
        f = lbl.locate_key
        eq(0, f('A'))
        eq(2, f('B'))
        with self.assertRaises(KeyError):
            f('D')

    def test_locate_keys(self):
        lbl = UniqueIndex(['A', 'C', 'B', 'F'])
        f = lbl.locate_keys

        # all found
        expect = [False, True, False, True]
        actual = f(['F', 'C'])
        self.assertNotIsInstance(actual, UniqueIndex)
        aeq(expect, actual)

        # not all found, lax
        expect = [False, True, False, True]
        actual = f(['X', 'F', 'G', 'C', 'Z'], strict=False)
        self.assertNotIsInstance(actual, UniqueIndex)
        aeq(expect, actual)

        # not all found, strict
        with self.assertRaises(KeyError):
            f(['X', 'F', 'G', 'C', 'Z'])

    def test_locate_intersection(self):
        lbl1 = UniqueIndex(['A', 'C', 'B', 'F'])
        lbl2 = UniqueIndex(['X', 'F', 'G', 'C', 'Z'])
        expect_loc1 = np.array([False, True, False, True])
        expect_loc2 = np.array([False, True, False, True, False])
        loc1, loc2 = lbl1.locate_intersection(lbl2)
        self.assertNotIsInstance(loc1, UniqueIndex)
        self.assertNotIsInstance(loc2, UniqueIndex)
        aeq(expect_loc1, loc1)
        aeq(expect_loc2, loc2)

    def test_intersect(self):
        lbl1 = UniqueIndex(['A', 'C', 'B', 'F'])
        lbl2 = UniqueIndex(['X', 'F', 'G', 'C', 'Z'])

        expect = UniqueIndex(['C', 'F'])
        actual = lbl1.intersect(lbl2)
        self.assertIsInstance(actual, UniqueIndex)
        aeq(expect, actual)

        expect = UniqueIndex(['F', 'C'])
        actual = lbl2.intersect(lbl1)
        self.assertIsInstance(actual, UniqueIndex)
        aeq(expect, actual)
