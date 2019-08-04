# -*- coding: utf-8 -*-
# third-party imports
import numpy as np
import unittest
import pytest


# internal imports
from allel.test.tools import assert_array_equal as aeq
from allel import SortedIndex, UniqueIndex, SortedMultiIndex, ChromPosIndex
from allel.test.model.test_api import SortedIndexInterface, UniqueIndexInterface, \
    SortedMultiIndexInterface, ChromPosIndexInterface


# noinspection PyMethodMayBeStatic
class SortedIndexTests(SortedIndexInterface, unittest.TestCase):

    _class = SortedIndex

    def setup_instance(self, data):
        return SortedIndex(data)

    def test_constructor(self):

        # missing data arg
        with pytest.raises(TypeError):
            # noinspection PyArgumentList
            SortedIndex()

        # data has wrong dtype
        data = 'foo bar'
        with pytest.raises(TypeError):
            SortedIndex(data)

        # data has wrong dimensions
        data = [[1, 2], [3, 4]]
        with pytest.raises(TypeError):
            SortedIndex(data)

        # values are not sorted
        data = [2, 1, 3, 5]
        with pytest.raises(ValueError):
            SortedIndex(data)

        # values are not sorted
        data = [4., 5., 3.7]
        with pytest.raises(ValueError):
            SortedIndex(data)

        # valid data (unique)
        data = [1, 4, 5, 7, 12]
        idx = SortedIndex(data)
        aeq(data, idx)
        assert np.int == idx.dtype
        assert 1 == idx.ndim
        assert 5 == len(idx)
        assert idx.is_unique

        # valid data (non-unique)
        data = [1, 4, 5, 5, 7, 12]
        idx = SortedIndex(data)
        aeq(data, idx)
        assert np.int == idx.dtype
        assert 1 == idx.ndim
        assert 6 == len(idx)
        assert not idx.is_unique

        # valid data (typed)
        data = [1, 4, 5, 5, 7, 12]
        idx = SortedIndex(data, dtype='u4')
        aeq(data, idx)
        assert np.uint32 == idx.dtype

        # valid data (non-numeric)
        data = np.array(['1', '12', '4', '5', '5', '7'], dtype=object)
        idx = SortedIndex(data)
        aeq(data, idx)

    def test_slice(self):

        data = [1, 4, 5, 5, 7, 12]
        idx = SortedIndex(data, dtype='u4')

        # row slice
        s = idx[1:]
        assert isinstance(s, SortedIndex)

        # index
        s = idx[0]
        assert isinstance(s, np.uint32)
        assert not isinstance(s, SortedIndex)
        assert data[0] == s


# noinspection PyMethodMayBeStatic
class UniqueIndexTests(UniqueIndexInterface, unittest.TestCase):

    def setup_instance(self, data):
        return UniqueIndex(data)

    _class = UniqueIndex

    def test_constructor(self):

        # missing data arg
        with pytest.raises(TypeError):
            # noinspection PyArgumentList
            UniqueIndex()

        # data has wrong dimensions
        data = [['A', 'C'], ['B', 'F']]
        with pytest.raises(TypeError):
            UniqueIndex(data)

        # labels are not unique
        data = ['A', 'B', 'D', 'B']
        with pytest.raises(ValueError):
            UniqueIndex(data)

        # valid data
        data = ['A', 'C', 'B', 'F']
        lbl = UniqueIndex(data)
        aeq(data, lbl)
        assert 1 == lbl.ndim
        assert 4 == len(lbl)

        # valid data (typed)
        data = np.array(['A', 'C', 'B', 'F'], dtype='S1')
        lbl = UniqueIndex(data, dtype='S1')
        aeq(data, lbl)

    def test_slice(self):

        data = ['A', 'C', 'B', 'F']
        lbl = UniqueIndex(data)

        # row slice
        s = lbl[1:]
        assert isinstance(s, UniqueIndex)

        # index
        s = lbl[0]
        assert isinstance(s, str)
        assert not isinstance(s, UniqueIndex)


class SortedMultiIndexTests(SortedMultiIndexInterface, unittest.TestCase):

    def setup_instance(self, chrom, pos):
        return SortedMultiIndex(chrom, pos)

    _class = SortedMultiIndex


class ChromPosIndexTests(ChromPosIndexInterface, unittest.TestCase):

    def setup_instance(self, chrom, pos):
        return ChromPosIndex(chrom, pos)

    _class = ChromPosIndex
