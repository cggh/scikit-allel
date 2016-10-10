# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import unittest
import tempfile


import numpy as np
import bcolz
import h5py
from nose.tools import eq_ as eq, assert_raises


from allel.model.ndarray import GenotypeArray, HaplotypeArray, \
    AlleleCountsArray, VariantTable, FeatureTable
from allel.test.tools import assert_array_equal as aeq
from allel.test.test_model_api import GenotypeArrayInterface, \
    HaplotypeArrayInterface, AlleleCountsArrayInterface, \
    diploid_genotype_data, triploid_genotype_data, haplotype_data, \
    allele_counts_data, VariantTableInterface, variant_table_data, \
    variant_table_dtype, FeatureTableInterface, feature_table_data, \
    feature_table_dtype
from allel.model.bcolz import GenotypeCArray, HaplotypeCArray, \
    AlleleCountsCArray, VariantCTable, FeatureCTable


class GenotypeCArrayTests(GenotypeArrayInterface, unittest.TestCase):

    _class = GenotypeCArray

    def setup_instance(self, data, dtype=None):
        return GenotypeCArray(data, dtype=dtype)

    def test_constructor(self):

        # missing data arg
        with assert_raises(ValueError):
            # noinspection PyArgumentList
            GenotypeCArray()

        # data has wrong dtype
        data = 'foo bar'
        with assert_raises(TypeError):
            GenotypeCArray(data)

        # data has wrong dtype
        data = [4., 5., 3.7]
        with assert_raises(TypeError):
            GenotypeCArray(data)

        # data has wrong dimensions
        data = [1, 2, 3]
        with assert_raises(TypeError):
            GenotypeCArray(data)

        # data has wrong dimensions
        data = [[1, 2], [3, 4]]  # use HaplotypeCArray instead
        with assert_raises(TypeError):
            GenotypeCArray(data)

        # diploid data (typed)
        g = GenotypeCArray(diploid_genotype_data, dtype='i1')
        aeq(diploid_genotype_data, g)
        eq(np.int8, g.dtype)

        # polyploid data (typed)
        g = GenotypeCArray(triploid_genotype_data, dtype='i1')
        aeq(triploid_genotype_data, g)
        eq(np.int8, g.dtype)

        # cparams
        g = GenotypeCArray(diploid_genotype_data,
                           cparams=bcolz.cparams(clevel=10))
        aeq(diploid_genotype_data, g)
        eq(10, g.cparams.clevel)

    def test_slice_types(self):

        g = GenotypeCArray(diploid_genotype_data, dtype='i1')

        # row slice
        s = g[1:]
        self.assertNotIsInstance(s, GenotypeCArray)
        self.assertIsInstance(s, GenotypeArray)

        # col slice
        s = g[:, 1:]
        self.assertNotIsInstance(s, GenotypeCArray)
        self.assertIsInstance(s, GenotypeArray)

        # row index
        s = g[0]
        self.assertNotIsInstance(s, GenotypeCArray)
        self.assertNotIsInstance(s, GenotypeArray)
        self.assertIsInstance(s, np.ndarray)

        # col index
        s = g[:, 0]
        self.assertNotIsInstance(s, GenotypeCArray)
        self.assertNotIsInstance(s, GenotypeArray)
        self.assertIsInstance(s, np.ndarray)

        # ploidy index
        s = g[:, :, 0]
        self.assertNotIsInstance(s, GenotypeCArray)
        self.assertNotIsInstance(s, GenotypeArray)
        self.assertIsInstance(s, np.ndarray)

        # item
        s = g[0, 0, 0]
        self.assertNotIsInstance(s, GenotypeCArray)
        self.assertNotIsInstance(s, GenotypeArray)
        self.assertIsInstance(s, np.int8)

    def test_from_hdf5(self):

        # setup HDF5 file
        node_path = 'test'
        tf = tempfile.NamedTemporaryFile(delete=False)
        file_path = tf.name
        tf.close()
        with h5py.File(file_path, mode='w') as h5f:
            h5f.create_dataset(node_path,
                               data=diploid_genotype_data,
                               chunks=(2, 3, 2))

        # file and node path
        g = GenotypeCArray.from_hdf5(file_path, node_path)
        aeq(diploid_genotype_data, g)

        # dataset
        with h5py.File(file_path, mode='r') as h5f:
            dataset = h5f[node_path]
            g = GenotypeCArray.from_hdf5(dataset)
            aeq(diploid_genotype_data, g)

    def test_from_hdf5_condition(self):

        # setup HDF5 file
        node_path = 'test'
        tf = tempfile.NamedTemporaryFile(delete=False)
        file_path = tf.name
        tf.close()
        with h5py.File(file_path, mode='w') as h5f:
            h5f.create_dataset(node_path,
                               data=diploid_genotype_data,
                               chunks=(2, 3, 2))

        # selection
        condition = [False, True, False, True, False]

        # file and node path
        g = GenotypeCArray.from_hdf5(file_path, node_path, condition=condition)
        expect = GenotypeArray(diploid_genotype_data).compress(condition,
                                                               axis=0)
        aeq(expect, g)

        # dataset
        with h5py.File(file_path, mode='r') as h5f:
            dataset = h5f[node_path]
            g = GenotypeCArray.from_hdf5(dataset, condition=condition)
            aeq(expect, g)

    def test_to_hdf5(self):

        # setup HDF5 file
        tf = tempfile.NamedTemporaryFile(delete=False)
        file_path = tf.name
        tf.close()

        # setup genotype array
        node_path = 'test'
        g = GenotypeCArray(diploid_genotype_data, dtype='i1')

        # write using file path and node path
        g.to_hdf5(file_path, node_path)

        # test outcome
        with h5py.File(file_path, mode='r') as h5f:
            h5d = h5f[node_path]
            aeq(g[:], h5d[:])

        # write using group
        with h5py.File(file_path, mode='w') as h5f:
            g.to_hdf5(h5f, node_path)

        # test outcome
        with h5py.File(file_path, mode='r') as h5f:
            h5d = h5f[node_path]
            aeq(g[:], h5d[:])

    def test_take(self):
        g = self.setup_instance(diploid_genotype_data)
        # take variants not in original order
        # not supported for carrays
        indices = [2, 0]
        with assert_raises(ValueError):
            g.take(indices, axis=0)

    def test_to_n_ref_array_like(self):
        # see also https://github.com/cggh/scikit-allel/issues/66

        gn = self.setup_instance(diploid_genotype_data).to_n_ref(fill=-1)
        t = gn > 0
        eq(4, np.count_nonzero(t))
        expect = np.array([[1, 1, 0],
                           [1, 0, 0],
                           [1, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]], dtype='b1')
        aeq(expect, t)

        # numpy reductions trigger the issue

        expect = np.array([2, 1, 1, 0, 0])
        actual = np.sum(t, axis=1)
        aeq(expect, actual)

        expect = np.array([0, 0, 0, 0, 0])
        actual = np.min(t, axis=1)
        aeq(expect, actual)

        expect = np.array([1, 1, 1, 0, 0])
        actual = np.max(t, axis=1)
        aeq(expect, actual)


class HaplotypeCArrayTests(HaplotypeArrayInterface, unittest.TestCase):

    _class = HaplotypeCArray

    def setup_instance(self, data, dtype=None):
        return HaplotypeCArray(data, dtype=dtype)

    def test_constructor(self):

        # missing data arg
        with assert_raises(ValueError):
            # noinspection PyArgumentList
            HaplotypeCArray()

        # data has wrong dtype
        data = 'foo bar'
        with assert_raises(TypeError):
            HaplotypeCArray(data)

        # data has wrong dtype
        data = [4., 5., 3.7]
        with assert_raises(TypeError):
            HaplotypeCArray(data)

        # data has wrong dimensions
        data = [1, 2, 3]
        with assert_raises(TypeError):
            HaplotypeCArray(data)

        # data has wrong dimensions
        data = [[[1, 2], [3, 4]]]  # use GenotypeCArray instead
        with assert_raises(TypeError):
            HaplotypeCArray(data)

        # typed data (typed)
        h = HaplotypeCArray(haplotype_data, dtype='i1')
        aeq(haplotype_data, h)
        eq(np.int8, h.dtype)

        # cparams
        h = HaplotypeCArray(haplotype_data,
                            cparams=bcolz.cparams(clevel=10))
        aeq(haplotype_data, h)
        eq(10, h.cparams.clevel)

    def test_slice_types(self):

        h = HaplotypeCArray(haplotype_data, dtype='i1')

        # row slice
        s = h[1:]
        self.assertNotIsInstance(s, HaplotypeCArray)
        self.assertIsInstance(s, HaplotypeArray)

        # col slice
        s = h[:, 1:]
        self.assertNotIsInstance(s, HaplotypeCArray)
        self.assertIsInstance(s, HaplotypeArray)

        # row index
        s = h[0]
        self.assertNotIsInstance(s, HaplotypeCArray)
        self.assertNotIsInstance(s, HaplotypeArray)
        self.assertIsInstance(s, np.ndarray)

        # col index
        s = h[:, 0]
        self.assertNotIsInstance(s, HaplotypeCArray)
        self.assertNotIsInstance(s, HaplotypeArray)
        self.assertIsInstance(s, np.ndarray)

        # item
        s = h[0, 0]
        self.assertNotIsInstance(s, HaplotypeCArray)
        self.assertNotIsInstance(s, HaplotypeArray)
        self.assertIsInstance(s, np.int8)

    def test_from_hdf5(self):

        # setup HDF5 file
        node_path = 'test'
        tf = tempfile.NamedTemporaryFile(delete=False)
        file_path = tf.name
        tf.close()
        with h5py.File(file_path, mode='w') as h5f:
            h5f.create_dataset(node_path,
                               data=haplotype_data,
                               chunks=(2, 3))

        # file and node path
        h = HaplotypeCArray.from_hdf5(file_path, node_path)
        aeq(haplotype_data, h)

        # dataset
        with h5py.File(file_path, mode='r') as h5f:
            dataset = h5f[node_path]
            h = HaplotypeCArray.from_hdf5(dataset)
            aeq(haplotype_data, h)


class AlleleCountsCArrayTests(AlleleCountsArrayInterface, unittest.TestCase):

    _class = AlleleCountsCArray

    def setup_instance(self, data):
        return AlleleCountsCArray(data)

    def test_constructor(self):

        # missing data arg
        with assert_raises(ValueError):
            # noinspection PyArgumentList
            AlleleCountsCArray()

        # data has wrong dtype
        data = 'foo bar'
        with assert_raises(TypeError):
            AlleleCountsCArray(data)

        # data has wrong dtype
        data = [4., 5., 3.7]
        with assert_raises(TypeError):
            AlleleCountsCArray(data)

        # data has wrong dimensions
        data = [1, 2, 3]
        with assert_raises(TypeError):
            AlleleCountsCArray(data)

        # data has wrong dimensions
        data = [[[1, 2], [3, 4]]]
        with assert_raises(TypeError):
            AlleleCountsCArray(data)

        # typed data (typed)
        ac = AlleleCountsCArray(allele_counts_data, dtype='u1')
        aeq(allele_counts_data, ac)
        eq(np.uint8, ac.dtype)

        # cparams
        ac = AlleleCountsCArray(allele_counts_data,
                                cparams=bcolz.cparams(clevel=10))
        aeq(allele_counts_data, ac)
        eq(10, ac.cparams.clevel)

    def test_slice_types(self):

        h = AlleleCountsCArray(allele_counts_data, dtype='u1')

        # row slice
        s = h[1:]
        self.assertNotIsInstance(s, AlleleCountsCArray)
        self.assertIsInstance(s, AlleleCountsArray)

        # col slice
        s = h[:, 1:]
        self.assertNotIsInstance(s, AlleleCountsCArray)
        self.assertNotIsInstance(s, AlleleCountsArray)
        self.assertIsInstance(s, np.ndarray)

        # row index
        s = h[0]
        self.assertNotIsInstance(s, AlleleCountsCArray)
        self.assertNotIsInstance(s, AlleleCountsArray)
        self.assertIsInstance(s, np.ndarray)

        # col index
        s = h[:, 0]
        self.assertNotIsInstance(s, AlleleCountsCArray)
        self.assertNotIsInstance(s, AlleleCountsArray)
        self.assertIsInstance(s, np.ndarray)

        # item
        s = h[0, 0]
        self.assertNotIsInstance(s, AlleleCountsCArray)
        self.assertNotIsInstance(s, AlleleCountsArray)
        self.assertIsInstance(s, np.uint8)


class VariantCTableTests(VariantTableInterface, unittest.TestCase):

    _class = VariantCTable

    def setup_instance(self, data, **kwargs):
        return VariantCTable(data, **kwargs)

    def test_constructor(self):

        # missing data arg
        with self.assertRaises(ValueError):
            VariantCTable()

    def test_slice_types(self):
        a = np.rec.array(variant_table_data, dtype=variant_table_dtype)
        vt = VariantCTable(a)

        # row slice
        s = vt[1:]
        self.assertNotIsInstance(s, VariantCTable)
        self.assertIsInstance(s, VariantTable)

        # row index
        s = vt[0]
        self.assertNotIsInstance(s, VariantCTable)
        self.assertNotIsInstance(s, VariantTable)
        self.assertIsInstance(s, (np.record, np.void))

        # col access
        s = vt['CHROM']
        self.assertNotIsInstance(s, VariantCTable)
        self.assertNotIsInstance(s, VariantTable)
        self.assertIsInstance(s, bcolz.carray)

    def test_take(self):
        a = np.rec.array(variant_table_data, dtype=variant_table_dtype)
        vt = VariantCTable(a)
        # take variants not in original order
        # not supported for carrays
        indices = [2, 0]
        with assert_raises(ValueError):
            vt.take(indices)

    def test_from_hdf5_group(self):

        # setup HDF5 file
        node_path = 'test'
        tf = tempfile.NamedTemporaryFile(delete=False)
        file_path = tf.name
        tf.close()
        a = np.rec.array(variant_table_data, dtype=variant_table_dtype)
        # reorder columns because will come back out in sorted order
        a = a[sorted(a.dtype.names)]
        with h5py.File(file_path, mode='w') as h5f:
            h5g = h5f.create_group(node_path)
            for n in a.dtype.names:
                h5g.create_dataset(n, data=a[n], chunks=True,
                                   compression='gzip')

        # file and node path
        vt = self._class.from_hdf5_group(file_path, node_path)
        self.assertIsInstance(vt, self._class)
        aeq(a, vt[:])

        # dataset
        with h5py.File(file_path, mode='r') as h5f:
            h5g = h5f[node_path]
            vt = self._class.from_hdf5_group(h5g)
            self.assertIsInstance(vt, self._class)
            aeq(a, vt[:])

    def test_to_hdf5_group(self):

        # setup HDF5 file
        node_path = 'test'
        tf = tempfile.NamedTemporaryFile(delete=False)
        file_path = tf.name
        tf.close()
        a = np.rec.array(variant_table_data, dtype=variant_table_dtype)
        # reorder columns because will come back out in sorted order
        a = a[sorted(a.dtype.names)]
        vt = self.setup_instance(a)

        # write using file path and node path
        vt.to_hdf5_group(file_path, node_path)

        with h5py.File(file_path, mode='r') as h5f:
            h5g = h5f[node_path]
            eq(sorted(a.dtype.names), sorted(h5g.keys()))
            for n in a.dtype.names:
                aeq(a[n], h5g[n][:])

        # write using group and node path
        with h5py.File(file_path, mode='w') as h5f:
            vt.to_hdf5_group(h5f, node_path)

        with h5py.File(file_path, mode='r') as h5f:
            h5g = h5f[node_path]
            eq(sorted(a.dtype.names), sorted(h5g.keys()))
            for n in a.dtype.names:
                aeq(a[n], h5g[n][:])


class FeatureCTableTests(FeatureTableInterface, unittest.TestCase):

    _class = FeatureCTable

    def setup_instance(self, data, **kwargs):
        return FeatureCTable(data, **kwargs)

    def test_constructor(self):

        # missing data arg
        with self.assertRaises(ValueError):
            FeatureCTable()

    def test_slice_types(self):
        a = np.rec.array(feature_table_data, dtype=feature_table_dtype)
        vt = FeatureCTable(a)

        # row slice
        s = vt[1:]
        self.assertNotIsInstance(s, FeatureCTable)
        self.assertIsInstance(s, FeatureTable)

        # row index
        s = vt[0]
        self.assertNotIsInstance(s, FeatureCTable)
        self.assertNotIsInstance(s, FeatureTable)
        self.assertIsInstance(s, (np.record, np.void))

        # col access
        s = vt['seqid']
        self.assertNotIsInstance(s, FeatureCTable)
        self.assertNotIsInstance(s, FeatureTable)
        self.assertIsInstance(s, bcolz.carray)
