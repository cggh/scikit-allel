# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np
import unittest
from nose.tools import eq_ as eq, assert_raises, assert_is_instance, \
    assert_not_is_instance
from allel.test.tools import assert_array_equal as aeq


from allel.model.ndarray import GenotypeArray, HaplotypeArray, SortedIndex, \
    UniqueIndex, SortedMultiIndex, AlleleCountsArray, VariantTable, \
    FeatureTable
from allel.test.test_model_api import GenotypeArrayInterface, \
    HaplotypeArrayInterface, SortedIndexInterface, UniqueIndexInterface, \
    SortedMultiIndexInterface, diploid_genotype_data, triploid_genotype_data, \
    haplotype_data, AlleleCountsArrayInterface, VariantTableInterface, \
    allele_counts_data, variant_table_data, variant_table_names, \
    variant_table_dtype, FeatureTableInterface, feature_table_data, \
    feature_table_dtype, feature_table_names


class GenotypeArrayTests(GenotypeArrayInterface, unittest.TestCase):

    _class = GenotypeArray

    def setup_instance(self, data, dtype=None):
        return GenotypeArray(data, dtype=dtype)

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
        assert_is_instance(s, GenotypeArray)

        # col slice
        s = g[:, 1:]
        assert_is_instance(s, GenotypeArray)

        # row index
        s = g[0]
        assert_is_instance(s, np.ndarray)
        assert_not_is_instance(s, GenotypeArray)

        # col index
        s = g[:, 0]
        assert_is_instance(s, np.ndarray)
        assert_not_is_instance(s, GenotypeArray)

        # ploidy index
        s = g[:, :, 0]
        assert_is_instance(s, np.ndarray)
        assert_not_is_instance(s, GenotypeArray)

        # item
        s = g[0, 0, 0]
        assert_is_instance(s, np.int8)
        assert_not_is_instance(s, GenotypeArray)

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

    def test_take(self):
        g = self.setup_instance(diploid_genotype_data)
        # take variants not in original order
        indices = [2, 0]
        t = g.take(indices, axis=0)
        eq(2, t.n_variants)
        eq(g.n_samples, t.n_samples)
        eq(g.ploidy, t.ploidy)
        expect = np.array(diploid_genotype_data).take(indices, axis=0)
        aeq(expect, t)

    def test_no_subclass(self):
        g = self.setup_instance(diploid_genotype_data)
        self.assertNotIsInstance(g.reshape((g.shape[0], -1)), GenotypeArray)
        self.assertNotIsInstance(g.flatten(), GenotypeArray)
        self.assertNotIsInstance(g.ravel(), GenotypeArray)
        self.assertNotIsInstance(g.transpose(), GenotypeArray)
        self.assertNotIsInstance(g.T, GenotypeArray)
        self.assertNotIsInstance(g.astype('f4'), GenotypeArray)


class HaplotypeArrayTests(HaplotypeArrayInterface, unittest.TestCase):

    _class = HaplotypeArray

    def setup_instance(self, data, dtype=None):
        return HaplotypeArray(data, dtype=dtype)

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
        assert_is_instance(s, HaplotypeArray)

        # col slice
        s = h[:, 1:]
        assert_is_instance(s, HaplotypeArray)

        # row index
        s = h[0]
        assert_is_instance(s, np.ndarray)
        assert_not_is_instance(s, HaplotypeArray)

        # col index
        s = h[:, 0]
        assert_is_instance(s, np.ndarray)
        assert_not_is_instance(s, HaplotypeArray)

        # item
        s = h[0, 0]
        assert_is_instance(s, np.int8)
        assert_not_is_instance(s, HaplotypeArray)

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


class AlleleCountsArrayTests(AlleleCountsArrayInterface, unittest.TestCase):

    _class = AlleleCountsArray

    def setup_instance(self, data):
        return AlleleCountsArray(data)

    def test_constructor(self):

        # missing data arg
        with self.assertRaises(TypeError):
            # noinspection PyArgumentList
            AlleleCountsArray()

        # data has wrong dtype
        data = 'foo bar'
        with self.assertRaises(TypeError):
            AlleleCountsArray(data)

        # data has wrong dtype
        data = [4., 5., 3.7]
        with self.assertRaises(TypeError):
            AlleleCountsArray(data)

        # data has wrong dimensions
        data = [1, 2, 3]
        with self.assertRaises(TypeError):
            AlleleCountsArray(data)

        # data has wrong dimensions
        data = diploid_genotype_data
        with self.assertRaises(TypeError):
            AlleleCountsArray(data)

        # valid data (typed)
        ac = AlleleCountsArray(allele_counts_data, dtype='u1')
        aeq(allele_counts_data, ac)
        eq(np.uint8, ac.dtype)

    def test_slice_types(self):

        ac = AlleleCountsArray(allele_counts_data, dtype='u1')

        # row slice
        s = ac[1:]
        assert_is_instance(s, AlleleCountsArray)

        # col slice
        s = ac[:, 1:]
        assert_is_instance(s, np.ndarray)
        assert_not_is_instance(s, AlleleCountsArray)

        # row index
        s = ac[0]
        assert_is_instance(s, np.ndarray)
        assert_not_is_instance(s, AlleleCountsArray)

        # col index
        s = ac[:, 0]
        assert_is_instance(s, np.ndarray)
        assert_not_is_instance(s, AlleleCountsArray)

        # item
        s = ac[0, 0]
        assert_is_instance(s, np.uint8)
        assert_not_is_instance(s, AlleleCountsArray)

    def test_view(self):

        # data has wrong dtype
        data = 'foo bar'
        with self.assertRaises(TypeError):
            np.array(data).view(AlleleCountsArray)

        # data has wrong dtype
        data = [4., 5., 3.7]
        with self.assertRaises(TypeError):
            np.array(data).view(AlleleCountsArray)

        # data has wrong dimensions
        data = [1, 2, 3]
        with self.assertRaises(TypeError):
            np.array(data).view(AlleleCountsArray)

        # data has wrong dimensions
        data = diploid_genotype_data
        with self.assertRaises(TypeError):
            np.array(data).view(AlleleCountsArray)

        # valid data
        ac = np.array(allele_counts_data).view(AlleleCountsArray)
        aeq(allele_counts_data, ac)
        eq(np.int, ac.dtype)
        eq(2, ac.ndim)
        eq(6, ac.n_variants)
        eq(3, ac.n_alleles)


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

        # data has wrong dimensions
        data = [[1, 2], [3, 4]]
        with self.assertRaises(TypeError):
            SortedIndex(data)

        # values are not sorted
        data = [2, 1, 3, 5]
        with self.assertRaises(ValueError):
            SortedIndex(data)

        # values are not sorted
        data = [4., 5., 3.7]
        with self.assertRaises(ValueError):
            SortedIndex(data)

        # valid data (unique)
        data = [1, 4, 5, 7, 12]
        idx = SortedIndex(data)
        aeq(data, idx)
        eq(np.int, idx.dtype)
        eq(1, idx.ndim)
        eq(5, len(idx))
        assert idx.is_unique

        # valid data (non-unique)
        data = [1, 4, 5, 5, 7, 12]
        idx = SortedIndex(data)
        aeq(data, idx)
        eq(np.int, idx.dtype)
        eq(1, idx.ndim)
        eq(6, len(idx))
        assert not idx.is_unique

        # valid data (typed)
        data = [1, 4, 5, 5, 7, 12]
        idx = SortedIndex(data, dtype='u4')
        aeq(data, idx)
        eq(np.uint32, idx.dtype)

        # valid data (non-numeric)
        data = ['1', '12', '4', '5', '5', '7']
        idx = SortedIndex(data)
        aeq(data, idx)

    def test_slice(self):

        data = [1, 4, 5, 5, 7, 12]
        idx = SortedIndex(data, dtype='u4')

        # row slice
        s = idx[1:]
        assert_is_instance(s, SortedIndex)

        # index
        s = idx[0]
        assert_is_instance(s, np.uint32)
        assert_not_is_instance(s, SortedIndex)
        eq(data[0], s)

    def test_view(self):

        # data has wrong dtype
        data = 'foo bar'
        with self.assertRaises(TypeError):
            np.asarray(data).view(SortedIndex)

        # data has wrong dimensions
        data = [[1, 2], [3, 4]]
        with self.assertRaises(TypeError):
            np.asarray(data).view(SortedIndex)

        # values are not sorted
        data = [2, 1, 3, 5]
        with self.assertRaises(ValueError):
            np.asarray(data).view(SortedIndex)

        # values are not sorted
        data = [4., 5., 3.7]
        with self.assertRaises(ValueError):
            np.asarray(data).view(SortedIndex)

        # valid data (unique)
        data = [1, 4, 5, 7, 12]
        idx = np.asarray(data).view(SortedIndex)
        aeq(data, idx)
        eq(np.int, idx.dtype)
        eq(1, idx.ndim)
        eq(5, len(idx))
        assert idx.is_unique

        # valid data (non-unique)
        data = [1, 4, 5, 5, 7, 12]
        idx = np.asarray(data).view(SortedIndex)
        aeq(data, idx)
        eq(np.int, idx.dtype)
        eq(1, idx.ndim)
        eq(6, len(idx))
        assert not idx.is_unique

        # valid data (typed)
        data = np.array([1, 4, 5, 5, 7, 12], dtype='u4')
        idx = np.asarray(data).view(SortedIndex)
        aeq(data, idx)
        eq(np.uint32, idx.dtype)

        # valid data (non-numeric)
        data = ['1', '12', '4', '5', '5', '7']
        idx = np.asarray(data).view(SortedIndex)
        aeq(data, idx)


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
        assert_is_instance(s, UniqueIndex)

        # index
        s = lbl[0]
        assert_is_instance(s, str)
        assert_not_is_instance(s, UniqueIndex)

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


class SortedMultiIndexTests(SortedMultiIndexInterface, unittest.TestCase):

    def setup_instance(self, chrom, pos):
        return SortedMultiIndex(chrom, pos)

    _class = SortedMultiIndex


class VariantTableTests(VariantTableInterface, unittest.TestCase):

    _class = VariantTable

    def setup_instance(self, data, index=None, **kwargs):
        return VariantTable(data, index=index, **kwargs)

    def test_constructor(self):

        # missing data arg
        with self.assertRaises(TypeError):
            # noinspection PyArgumentList
            VariantTable()

    def test_get_item_types(self):
        vt = VariantTable(variant_table_data, dtype=variant_table_dtype)

        # row slice
        s = vt[1:]
        assert_is_instance(s, VariantTable)

        # row index
        s = vt[0]
        assert_is_instance(s, np.record)
        assert_not_is_instance(s, VariantTable)

        # col access
        s = vt['CHROM']
        assert_is_instance(s, np.ndarray)
        assert_not_is_instance(s, VariantTable)
        s = vt[['CHROM', 'POS']]
        assert_is_instance(s, VariantTable)

    def test_view(self):
        a = np.rec.array(variant_table_data,
                         dtype=variant_table_dtype)
        vt = a.view(VariantTable)
        aeq(a, vt)
        eq(1, vt.ndim)
        eq(5, vt.n_variants)
        eq(variant_table_names, vt.names)

    def test_take(self):
        a = np.rec.array(variant_table_data,
                         dtype=variant_table_dtype)
        vt = VariantTable(a)
        # take variants not in original order
        indices = [2, 0]
        t = vt.take(indices)
        eq(2, t.n_variants)
        expect = a.take(indices)
        aeq(expect, t)


class FeatureTableTests(FeatureTableInterface, unittest.TestCase):

    _class = FeatureTable

    def setup_instance(self, data, index=None, **kwargs):
        return FeatureTable(data, index=index, **kwargs)

    def test_constructor(self):

        # missing data arg
        with self.assertRaises(TypeError):
            # noinspection PyArgumentList
            FeatureTable()

    def test_get_item_types(self):
        ft = FeatureTable(feature_table_data, dtype=feature_table_dtype)

        # row slice
        s = ft[1:]
        assert_is_instance(s, FeatureTable)

        # row index
        s = ft[0]
        assert_is_instance(s, np.record)
        assert_not_is_instance(s, FeatureTable)

        # col access
        s = ft['seqid']
        assert_is_instance(s, np.ndarray)
        assert_not_is_instance(s, FeatureTable)
        s = ft[['seqid', 'start', 'end']]
        assert_is_instance(s, FeatureTable)

    def test_view(self):
        a = np.rec.array(feature_table_data,
                         dtype=feature_table_dtype)
        ft = a.view(FeatureTable)
        aeq(a, ft)
        eq(1, ft.ndim)
        eq(6, ft.n_features)
        eq(feature_table_names, ft.names)
