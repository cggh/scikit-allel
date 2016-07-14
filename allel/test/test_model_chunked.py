# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import unittest
import numpy as np
import bcolz
import h5py
import zarr
from nose.tools import assert_raises, eq_ as eq


from allel.model.ndarray import GenotypeArray, HaplotypeArray, \
    AlleleCountsArray, VariantTable, FeatureTable
from allel.test.tools import assert_array_equal as aeq
from allel.test.test_model_api import GenotypeArrayInterface, \
    diploid_genotype_data, triploid_genotype_data, HaplotypeArrayInterface, \
    haplotype_data, allele_counts_data, AlleleCountsArrayInterface, \
    VariantTableInterface, variant_table_data, variant_table_dtype, \
    variant_table_names, feature_table_data, feature_table_dtype, \
    feature_table_names, FeatureTableInterface
from allel import chunked
from allel.model.chunked import GenotypeChunkedArray, HaplotypeChunkedArray,\
    AlleleCountsChunkedArray, VariantChunkedTable, FeatureChunkedTable


class GenotypeChunkedArrayTests(GenotypeArrayInterface, unittest.TestCase):

    _class = GenotypeChunkedArray

    def setUp(self):
        chunked.storage_registry['default'] = chunked.bcolzmem_storage

    def setup_instance(self, data, dtype=None):
        data = chunked.storage_registry['default'].array(data, dtype=dtype,
                                                         chunklen=2)
        return GenotypeChunkedArray(data)

    def test_constructor(self):

        # missing data arg
        with assert_raises(TypeError):
            # noinspection PyArgumentList
            GenotypeChunkedArray()

        # data has wrong dtype
        data = 'foo bar'
        with assert_raises(ValueError):
            GenotypeChunkedArray(data)

        # data has wrong dtype
        data = np.array([4., 5., 3.7])
        with assert_raises(ValueError):
            GenotypeChunkedArray(data)

        # data has wrong dimensions
        data = np.array([1, 2, 3])
        with assert_raises(ValueError):
            GenotypeChunkedArray(data)

        # data has wrong dimensions
        data = np.array([[1, 2], [3, 4]])  # use HaplotypeChunkedArray instead
        with assert_raises(ValueError):
            GenotypeChunkedArray(data)

        # diploid data (typed)
        g = self.setup_instance(np.array(diploid_genotype_data, dtype='i1'))
        aeq(diploid_genotype_data, g)
        eq(np.int8, g.dtype)

        # polyploid data (typed)
        g = self.setup_instance(np.array(triploid_genotype_data, dtype='i1'))
        aeq(triploid_genotype_data, g)
        eq(np.int8, g.dtype)

    def test_storage(self):
        g = self.setup_instance(np.array(diploid_genotype_data))
        # default is bcolz mem
        assert isinstance(g.data, bcolz.carray)
        assert g.data.rootdir is None, g.data.rootdir

    def test_slice_types(self):

        g = self.setup_instance(np.array(diploid_genotype_data, dtype='i1'))

        # row slice
        s = g[1:]
        self.assertNotIsInstance(s, GenotypeChunkedArray)
        self.assertIsInstance(s, GenotypeArray)

        # col slice
        s = g[:, 1:]
        self.assertNotIsInstance(s, GenotypeChunkedArray)
        self.assertIsInstance(s, GenotypeArray)

        # row index
        s = g[0]
        self.assertNotIsInstance(s, GenotypeChunkedArray)
        self.assertNotIsInstance(s, GenotypeArray)
        self.assertIsInstance(s, np.ndarray)

        # col index
        s = g[:, 0]
        self.assertNotIsInstance(s, GenotypeChunkedArray)
        self.assertNotIsInstance(s, GenotypeArray)
        self.assertIsInstance(s, np.ndarray)

        # ploidy index
        s = g[:, :, 0]
        self.assertNotIsInstance(s, GenotypeChunkedArray)
        self.assertNotIsInstance(s, GenotypeArray)
        self.assertIsInstance(s, np.ndarray)

        # item
        s = g[0, 0, 0]
        self.assertNotIsInstance(s, GenotypeChunkedArray)
        self.assertNotIsInstance(s, GenotypeArray)
        self.assertIsInstance(s, np.int8)

    def test_take(self):
        g = self.setup_instance(diploid_genotype_data)
        # take variants not in original order
        # not supported for carrays
        indices = [2, 0]
        with assert_raises(NotImplementedError):
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


class GenotypeChunkedArrayTestsBColzTmpStorage(GenotypeChunkedArrayTests):

    def setUp(self):
        chunked.storage_registry['default'] = chunked.bcolztmp_storage

    def setup_instance(self, data, dtype=None):
        data = chunked.bcolztmp_storage.array(data, dtype=dtype, chunklen=2)
        return GenotypeChunkedArray(data)

    def test_storage(self):
        g = self.setup_instance(np.array(diploid_genotype_data))
        assert isinstance(g.data, bcolz.carray)
        assert g.data.rootdir is not None


class GenotypeChunkedArrayTestsBColzCustomStorage(GenotypeChunkedArrayTests):

    def setUp(self):
        chunked.storage_registry['default'] = chunked.BcolzMemStorage(
            cparams=bcolz.cparams(cname='zlib', clevel=1)
        )

    def setup_instance(self, data, dtype=None):
        data = chunked.storage_registry['default'].array(data, dtype=dtype,
                                                         chunklen=2)
        return GenotypeChunkedArray(data)

    def test_storage(self):
        g = self.setup_instance(np.array(diploid_genotype_data))
        assert isinstance(g.data, bcolz.carray)
        eq('zlib', g.data.cparams.cname)
        eq(1, g.data.cparams.clevel)


class GenotypeChunkedArrayTestsHDF5MemStorage(GenotypeChunkedArrayTests):

    def setUp(self):
        chunked.storage_registry['default'] = chunked.hdf5mem_storage

    def setup_instance(self, data, dtype=None):
        data = chunked.hdf5mem_storage.array(data, dtype=dtype)
        return GenotypeChunkedArray(data)

    def test_storage(self):
        g = self.setup_instance(np.array(diploid_genotype_data))
        assert isinstance(g.data, h5py.Dataset)


class GenotypeChunkedArrayTestsZarrMemStorage(GenotypeChunkedArrayTests):

    def setUp(self):
        chunked.storage_registry['default'] = chunked.zarrmem_storage

    def setup_instance(self, data, dtype=None):
        data = chunked.zarrmem_storage.array(data, dtype=dtype)
        return GenotypeChunkedArray(data)

    def test_storage(self):
        g = self.setup_instance(np.array(diploid_genotype_data))
        assert isinstance(g.data, zarr.core.Array)


class GenotypeChunkedArrayTestsZarrTmpStorage(GenotypeChunkedArrayTests):

    def setUp(self):
        chunked.storage_registry['default'] = chunked.zarrtmp_storage

    def setup_instance(self, data, dtype=None):
        data = chunked.zarrtmp_storage.array(data, dtype=dtype)
        return GenotypeChunkedArray(data)

    def test_storage(self):
        g = self.setup_instance(np.array(diploid_genotype_data))
        assert isinstance(g.data, zarr.Array)
        assert isinstance(g.data.store, zarr.DirectoryStore)


class GenotypeChunkedArrayTestsHDF5TmpStorage(GenotypeChunkedArrayTests):

    def setUp(self):
        chunked.storage_registry['default'] = chunked.hdf5tmp_storage

    def setup_instance(self, data, dtype=None):
        data = chunked.hdf5tmp_storage.array(data, dtype=dtype)
        return GenotypeChunkedArray(data)

    def test_storage(self):
        g = self.setup_instance(np.array(diploid_genotype_data))
        assert isinstance(g.data, h5py.Dataset)


class GenotypeChunkedArrayTestsHDF5TmpLZFStorage(GenotypeChunkedArrayTests):

    def setUp(self):
        chunked.storage_registry['default'] = chunked.hdf5tmp_lzf_storage

    def setup_instance(self, data, dtype=None):
        data = chunked.hdf5tmp_lzf_storage.array(data, dtype=dtype)
        return GenotypeChunkedArray(data)

    def test_storage(self):
        g = self.setup_instance(np.array(diploid_genotype_data))
        assert isinstance(g.data, h5py.Dataset)
        assert g.data.compression == 'lzf'


class HaplotypeChunkedArrayTests(HaplotypeArrayInterface, unittest.TestCase):

    _class = HaplotypeChunkedArray

    def setUp(self):
        chunked.storage_registry['default'] = chunked.bcolzmem_storage

    def setup_instance(self, data, dtype=None):
        data = chunked.storage_registry['default'].array(data, dtype=dtype,
                                                         chunklen=2)
        return HaplotypeChunkedArray(data)

    def test_constructor(self):

        # missing data arg
        with assert_raises(TypeError):
            # noinspection PyArgumentList
            HaplotypeChunkedArray()

        # data has wrong dtype
        data = 'foo bar'
        with assert_raises(ValueError):
            HaplotypeChunkedArray(data)

        # data has wrong dtype
        data = np.array([4., 5., 3.7])
        with assert_raises(ValueError):
            HaplotypeChunkedArray(data)

        # data has wrong dimensions
        data = np.array([1, 2, 3])
        with assert_raises(ValueError):
            HaplotypeChunkedArray(data)

        # data has wrong dimensions
        data = np.array([[[1, 2], [3, 4]]])  # use GenotypeCArray instead
        with assert_raises(ValueError):
            HaplotypeChunkedArray(data)

        # typed data (typed)
        h = HaplotypeChunkedArray(np.array(haplotype_data, dtype='i1'))
        aeq(haplotype_data, h)
        eq(np.int8, h.dtype)

    def test_slice_types(self):

        h = self.setup_instance(np.array(haplotype_data, dtype='i1'))

        # row slice
        s = h[1:]
        self.assertNotIsInstance(s, HaplotypeChunkedArray)
        self.assertIsInstance(s, HaplotypeArray)

        # col slice
        s = h[:, 1:]
        self.assertNotIsInstance(s, HaplotypeChunkedArray)
        self.assertIsInstance(s, HaplotypeArray)

        # row index
        s = h[0]
        self.assertNotIsInstance(s, HaplotypeChunkedArray)
        self.assertNotIsInstance(s, HaplotypeArray)
        self.assertIsInstance(s, np.ndarray)

        # col index
        s = h[:, 0]
        self.assertNotIsInstance(s, HaplotypeChunkedArray)
        self.assertNotIsInstance(s, HaplotypeArray)
        self.assertIsInstance(s, np.ndarray)

        # item
        s = h[0, 0]
        self.assertNotIsInstance(s, HaplotypeChunkedArray)
        self.assertNotIsInstance(s, HaplotypeArray)
        self.assertIsInstance(s, np.int8)


class AlleleCountsChunkedArrayTests(AlleleCountsArrayInterface,
                                    unittest.TestCase):

    _class = AlleleCountsChunkedArray

    def setUp(self):
        chunked.storage_registry['default'] = chunked.bcolzmem_storage

    def setup_instance(self, data):
        data = chunked.storage_registry['default'].array(data, chunklen=2)
        return AlleleCountsChunkedArray(data)

    def test_constructor(self):

        # missing data arg
        with assert_raises(TypeError):
            # noinspection PyArgumentList
            AlleleCountsChunkedArray()

        # data has wrong dtype
        data = 'foo bar'
        with assert_raises(ValueError):
            AlleleCountsChunkedArray(data)

        # data has wrong dtype
        data = np.array([4., 5., 3.7])
        with assert_raises(ValueError):
            AlleleCountsChunkedArray(data)

        # data has wrong dimensions
        data = np.array([1, 2, 3])
        with assert_raises(ValueError):
            AlleleCountsChunkedArray(data)

        # data has wrong dimensions
        data = np.array([[[1, 2], [3, 4]]])
        with assert_raises(ValueError):
            AlleleCountsChunkedArray(data)

        # typed data (typed)
        ac = AlleleCountsChunkedArray(np.array(allele_counts_data, dtype='u1'))
        aeq(allele_counts_data, ac)
        eq(np.uint8, ac.dtype)

    def test_slice_types(self):

        ac = self.setup_instance(np.array(allele_counts_data, dtype='u2'))

        # row slice
        s = ac[1:]
        self.assertNotIsInstance(s, AlleleCountsChunkedArray)
        self.assertIsInstance(s, AlleleCountsArray)

        # col slice
        s = ac[:, 1:]
        self.assertNotIsInstance(s, AlleleCountsChunkedArray)
        self.assertNotIsInstance(s, AlleleCountsArray)
        self.assertIsInstance(s, np.ndarray)

        # row index
        s = ac[0]
        self.assertNotIsInstance(s, AlleleCountsChunkedArray)
        self.assertNotIsInstance(s, AlleleCountsArray)
        self.assertIsInstance(s, np.ndarray)

        # col index
        s = ac[:, 0]
        self.assertNotIsInstance(s, AlleleCountsChunkedArray)
        self.assertNotIsInstance(s, AlleleCountsArray)
        self.assertIsInstance(s, np.ndarray)

        # item
        s = ac[0, 0]
        self.assertNotIsInstance(s, AlleleCountsChunkedArray)
        self.assertNotIsInstance(s, AlleleCountsArray)
        self.assertIsInstance(s, np.uint16)


class AlleleCountsChunkedArrayTestsHDF5Mem(AlleleCountsChunkedArrayTests):

    def setUp(self):
        chunked.storage_registry['default'] = chunked.hdf5mem_storage

    def setup_instance(self, data):
        data = chunked.storage_registry['default'].array(data)
        return AlleleCountsChunkedArray(data)


class VariantChunkedTableTests(VariantTableInterface, unittest.TestCase):

    _class = VariantChunkedTable

    def setUp(self):
        chunked.storage_registry['default'] = chunked.bcolzmem_storage

    def setup_instance(self, data, **kwargs):
        data = chunked.storage_registry['default'].table(data, chunklen=2)
        return VariantChunkedTable(data, **kwargs)

    def test_storage(self):
        a = np.rec.array(variant_table_data, dtype=variant_table_dtype)
        vt = self.setup_instance(a)
        assert isinstance(vt.data, bcolz.ctable)

    def test_constructor(self):

        # missing data arg
        with self.assertRaises(TypeError):
            # noinspection PyArgumentList
            VariantChunkedTable()

        # recarray
        ra = np.rec.array(variant_table_data, dtype=variant_table_dtype)
        vt = VariantChunkedTable(ra)
        eq(5, len(vt))
        aeq(ra, vt)

        # dict
        d = {n: ra[n] for n in variant_table_names}
        vt = VariantChunkedTable(d, names=variant_table_names)
        eq(5, len(vt))
        aeq(ra, vt)

    def test_slice_types(self):
        ra = np.rec.array(variant_table_data, dtype=variant_table_dtype)
        vt = VariantChunkedTable(ra)

        # row slice
        s = vt[1:]
        self.assertNotIsInstance(s, VariantChunkedTable)
        self.assertIsInstance(s, VariantTable)

        # row index
        s = vt[0]
        self.assertNotIsInstance(s, VariantChunkedTable)
        self.assertNotIsInstance(s, VariantTable)
        self.assertIsInstance(s, (np.record, np.void, tuple))

        # col access
        s = vt['CHROM']
        self.assertNotIsInstance(s, VariantChunkedTable)
        self.assertNotIsInstance(s, VariantTable)
        self.assertIsInstance(s, chunked.ChunkedArray)

    def test_take(self):
        a = np.rec.array(variant_table_data, dtype=variant_table_dtype)
        vt = VariantChunkedTable(a)
        # take variants not in original order
        # not supported for carrays
        indices = [2, 0]
        with assert_raises(NotImplementedError):
            vt.take(indices)

    def test_eval_vm(self):
        a = np.rec.array(variant_table_data, dtype=variant_table_dtype)
        vt = self.setup_instance(a)

        expr = '(DP > 30) & (QD < 4)'
        r = vt.eval(expr, vm='numexpr')
        aeq([False, False, True, False, True], r)
        r = vt.eval(expr, vm='python')
        aeq([False, False, True, False, True], r)


class VariantChunkedTableTestsHDF5Storage(VariantChunkedTableTests):

    def setUp(self):
        chunked.storage_registry['default'] = chunked.hdf5mem_storage

    def setup_instance(self, data, **kwargs):
        data = chunked.storage_registry['default'].table(data)
        return VariantChunkedTable(data, **kwargs)

    def test_storage(self):
        a = np.rec.array(variant_table_data, dtype=variant_table_dtype)
        vt = self.setup_instance(a)
        assert isinstance(vt.data, h5py.Group)


class FeatureChunkedTableTests(FeatureTableInterface, unittest.TestCase):

    _class = FeatureChunkedTable

    def setUp(self):
        chunked.storage_registry['default'] = chunked.bcolzmem_storage

    def setup_instance(self, data, **kwargs):
        print('before', data)
        data = chunked.storage_registry['default'].table(data, chunklen=2)
        print('after', data)
        return FeatureChunkedTable(data, **kwargs)

    def test_storage(self):
        a = np.rec.array(variant_table_data, dtype=variant_table_dtype)
        vt = self.setup_instance(a)
        assert isinstance(vt.data, bcolz.ctable)

    def test_constructor(self):

        # missing data arg
        with self.assertRaises(TypeError):
            # noinspection PyArgumentList
            FeatureChunkedTable()

        # recarray
        ra = np.rec.array(feature_table_data, dtype=feature_table_dtype)
        ft = FeatureChunkedTable(ra)
        eq(6, len(ft))
        aeq(ra, ft)

        # dict
        d = {n: ra[n] for n in feature_table_names}
        ft = FeatureChunkedTable(d, names=feature_table_names)
        eq(6, len(ft))
        aeq(ra, ft)

    def test_slice_types(self):
        ra = np.rec.array(feature_table_data, dtype=feature_table_dtype)
        ft = FeatureChunkedTable(ra)

        # row slice
        s = ft[1:]
        self.assertNotIsInstance(s, FeatureChunkedTable)
        self.assertIsInstance(s, FeatureTable)

        # row index
        s = ft[0]
        self.assertNotIsInstance(s, FeatureChunkedTable)
        self.assertNotIsInstance(s, FeatureTable)
        self.assertIsInstance(s, (np.record, np.void, tuple))

        # col access
        s = ft['seqid']
        self.assertNotIsInstance(s, FeatureChunkedTable)
        self.assertNotIsInstance(s, FeatureTable)
        self.assertIsInstance(s, chunked.ChunkedArray)
