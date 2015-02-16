# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np
import unittest
from nose.tools import eq_ as eq, assert_raises
from allel.test.tools import assert_array_equal as aeq


from allel.model import GenotypeArray, HaplotypeArray, SortedIndex, \
    UniqueIndex, GenomeIndex
from allel.test.test_model_api import GenotypeArrayInterface, \
    HaplotypeArrayInterface, SortedIndexInterface, UniqueIndexInterface, \
    GenomeIndexInterface, diploid_genotype_data, triploid_genotype_data, \
    haplotype_data


class GenotypeArrayTests(GenotypeArrayInterface, unittest.TestCase):

    _class = GenotypeArray

    def setup_instance(self, data):
        return GenotypeArray(data)

    def test_constructor(self):

        # missing data arg
        with assert_raises(TypeError):
            # noinspection PyArgumentList
            GenotypeArray()

        # data has wrong dtype
        data = 'foo bar'
        with assert_raises(TypeError):
            GenotypeArray(data)

        # data has wrong dtype
        data = [4., 5., 3.7]
        with assert_raises(TypeError):
            GenotypeArray(data)

        # data has wrong dimensions
        data = [1, 2, 3]
        with assert_raises(TypeError):
            GenotypeArray(data)

        # data has wrong dimensions
        data = [[1, 2], [3, 4]]  # use HaplotypeArray instead
        with assert_raises(TypeError):
            GenotypeArray(data)

        # diploid data (typed)
        g = GenotypeArray(diploid_genotype_data, dtype='i1')
        aeq(diploid_genotype_data, g)
        eq(np.int8, g.dtype)

        # polyploid data (typed)
        g = GenotypeArray(triploid_genotype_data, dtype='i1')
        aeq(triploid_genotype_data, g)
        eq(np.int8, g.dtype)

    def test_slice_types(self):

        g = GenotypeArray(diploid_genotype_data, dtype='i1')

        # row slice
        s = g[1:]
        self.assertIsInstance(s, GenotypeArray)

        # col slice
        s = g[:, 1:]
        self.assertIsInstance(s, GenotypeArray)

        # row index
        s = g[0]
        self.assertIsInstance(s, np.ndarray)
        self.assertNotIsInstance(s, GenotypeArray)

        # col index
        s = g[:, 0]
        self.assertIsInstance(s, np.ndarray)
        self.assertNotIsInstance(s, GenotypeArray)

        # ploidy index
        s = g[:, :, 0]
        self.assertIsInstance(s, np.ndarray)
        self.assertNotIsInstance(s, GenotypeArray)

        # item
        s = g[0, 0, 0]
        self.assertIsInstance(s, np.int8)
        self.assertNotIsInstance(s, GenotypeArray)

    def test_view(self):

        # data has wrong dtype
        data = 'foo bar'
        with assert_raises(TypeError):
            np.array(data).view(GenotypeArray)

        # data has wrong dtype
        data = [4., 5., 3.7]
        with assert_raises(TypeError):
            np.array(data).view(GenotypeArray)

        # data has wrong dimensions
        data = [1, 2, 3]
        with assert_raises(TypeError):
            np.array(data).view(GenotypeArray)

        # data has wrong dimensions
        data = [[1, 2], [3, 4]]  # use HaplotypeArray instead
        with assert_raises(TypeError):
            np.array(data).view(GenotypeArray)

        # diploid data
        g = np.array(diploid_genotype_data).view(GenotypeArray)
        aeq(diploid_genotype_data, g)
        eq(np.int, g.dtype)
        eq(3, g.ndim)
        eq(5, g.n_variants)
        eq(3, g.n_samples)
        eq(2, g.ploidy)

        # polyploid data
        g = np.array(triploid_genotype_data).view(GenotypeArray)
        aeq(triploid_genotype_data, g)
        eq(np.int, g.dtype)
        eq(3, g.ndim)
        eq(4, g.n_variants)
        eq(3, g.n_samples)
        eq(3, g.ploidy)

    def test_haploidify_samples(self):

        # diploid
        g = GenotypeArray([[[0, 1], [2, 3]],
                           [[4, 5], [6, 7]],
                           [[8, 9], [10, 11]]], dtype='i1')
        h = g.haploidify_samples()
        eq(2, h.ndim)
        eq(3, h.n_variants)
        eq(2, h.n_haplotypes)
        eq(np.int8, h.dtype)
        for i in range(g.n_variants):
            for j in range(g.n_samples):
                self.assertIn(h[i, j], set(g[i, j]))
        print(repr(h))

        # triploid
        g = GenotypeArray([[[0, 1, 2], [3, 4, 5]],
                           [[6, 7, 8], [9, 10, 11]],
                           [[12, 13, 14], [15, 16, 17]]], dtype='i1')
        h = g.haploidify_samples()
        eq(2, h.ndim)
        eq(3, h.n_variants)
        eq(2, h.n_haplotypes)
        eq(np.int8, h.dtype)
        for i in range(g.n_variants):
            for j in range(g.n_samples):
                self.assertIn(h[i, j], set(g[i, j]))


class HaplotypeArrayTests(HaplotypeArrayInterface, unittest.TestCase):

    _class = HaplotypeArray

    def setup_instance(self, data):
        return HaplotypeArray(data)

    def test_constructor(self):

        # missing data arg
        with self.assertRaises(TypeError):
            # noinspection PyArgumentList
            HaplotypeArray()

        # data has wrong dtype
        data = 'foo bar'
        with self.assertRaises(TypeError):
            HaplotypeArray(data)

        # data has wrong dtype
        data = [4., 5., 3.7]
        with self.assertRaises(TypeError):
            HaplotypeArray(data)

        # data has wrong dimensions
        data = [1, 2, 3]
        with self.assertRaises(TypeError):
            HaplotypeArray(data)

        # data has wrong dimensions
        data = diploid_genotype_data  # use GenotypeArray instead
        with self.assertRaises(TypeError):
            HaplotypeArray(data)

        # haploid data (typed)
        h = HaplotypeArray(haplotype_data, dtype='i1')
        aeq(haplotype_data, h)
        eq(np.int8, h.dtype)

    def test_slice_types(self):

        h = HaplotypeArray(haplotype_data, dtype='i1')

        # row slice
        s = h[1:]
        self.assertIsInstance(s, HaplotypeArray)

        # col slice
        s = h[:, 1:]
        self.assertIsInstance(s, HaplotypeArray)

        # row index
        s = h[0]
        self.assertIsInstance(s, np.ndarray)
        self.assertNotIsInstance(s, HaplotypeArray)

        # col index
        s = h[:, 0]
        self.assertIsInstance(s, np.ndarray)
        self.assertNotIsInstance(s, HaplotypeArray)

        # item
        s = h[0, 0]
        self.assertIsInstance(s, np.int8)
        self.assertNotIsInstance(s, HaplotypeArray)

    def test_view(self):

        # data has wrong dtype
        data = 'foo bar'
        with self.assertRaises(TypeError):
            np.array(data).view(HaplotypeArray)

        # data has wrong dtype
        data = [4., 5., 3.7]
        with self.assertRaises(TypeError):
            np.array(data).view(HaplotypeArray)

        # data has wrong dimensions
        data = [1, 2, 3]
        with self.assertRaises(TypeError):
            np.array(data).view(HaplotypeArray)

        # data has wrong dimensions
        data = diploid_genotype_data  # use GenotypeArray instead
        with self.assertRaises(TypeError):
            np.array(data).view(HaplotypeArray)

        # haploid data
        h = np.array(haplotype_data).view(HaplotypeArray)
        aeq(haplotype_data, h)
        eq(np.int, h.dtype)
        eq(2, h.ndim)
        eq(4, h.n_variants)
        eq(3, h.n_haplotypes)


class SortedIndexTests(SortedIndexInterface, unittest.TestCase):

    _class = SortedIndex

    def setup_instance(self, data):
        return SortedIndex(data)

    def test_constructor(self):

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

        data = [1, 4, 5, 5, 7, 12]
        pos = SortedIndex(data, dtype='u4')

        # row slice
        s = pos[1:]
        self.assertIsInstance(s, SortedIndex)

        # index
        s = pos[0]
        self.assertIsInstance(s, np.uint32)
        self.assertNotIsInstance(s, SortedIndex)
        eq(data[0], s)

    def test_view(self):

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


class UniqueIndexTests(UniqueIndexInterface, unittest.TestCase):

    def setup_instance(self, data):
        return UniqueIndex(data)

    _class = UniqueIndex

    def test_constructor(self):

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

        data = ['A', 'C', 'B', 'F']
        lbl = UniqueIndex(data)

        # row slice
        s = lbl[1:]
        self.assertIsInstance(s, UniqueIndex)

        # index
        s = lbl[0]
        self.assertIsInstance(s, str)
        self.assertNotIsInstance(s, UniqueIndex)

    def test_view(self):

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


class GenomeIndexTests(GenomeIndexInterface, unittest.TestCase):

    def setup_instance(self, chrom, pos):
        return GenomeIndex(chrom, pos)

    _class = GenomeIndex
