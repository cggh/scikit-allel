# -*- coding: utf-8 -*-
import unittest
import numpy as np
import h5py
import zarr
import pytest


from allel import GenotypeArray, HaplotypeArray, AlleleCountsArray, VariantTable, \
    GenotypeVector, GenotypeAlleleCountsArray, GenotypeAlleleCountsVector
from allel import GenotypeChunkedArray, HaplotypeChunkedArray, AlleleCountsChunkedArray, \
    VariantChunkedTable, GenotypeAlleleCountsChunkedArray
from allel.test.tools import assert_array_equal as aeq
from allel.test.model.test_api import GenotypeArrayInterface, \
    diploid_genotype_data, triploid_genotype_data, HaplotypeArrayInterface, \
    haplotype_data, allele_counts_data, AlleleCountsArrayInterface, \
    VariantTableInterface, variant_table_data, variant_table_dtype, \
    variant_table_names, GenotypeAlleleCountsArrayInterface, \
    diploid_genotype_ac_data, triploid_genotype_ac_data
from allel import chunked
from allel.chunked.storage_zarr import ZarrTable


class GenotypeChunkedArrayTests(GenotypeArrayInterface, unittest.TestCase):

    _class = GenotypeChunkedArray

    def setUp(self):
        chunked.storage_registry['default'] = chunked.zarrmem_storage

    def setup_instance(self, data, **kwargs):
        data = chunked.storage_registry['default'].array(data, chunks=2, **kwargs)
        return GenotypeChunkedArray(data)

    def test_constructor(self):

        # missing data arg
        with pytest.raises(TypeError):
            # noinspection PyArgumentList
            GenotypeChunkedArray()

        # data has wrong dtype
        data = 'foo bar'
        with pytest.raises(TypeError):
            GenotypeChunkedArray(data)

        # data has wrong dtype
        data = np.array([4., 5., 3.7])
        with pytest.raises(TypeError):
            GenotypeChunkedArray(data)

        # data has wrong dimensions
        data = np.array([1, 2, 3])
        with pytest.raises(TypeError):
            GenotypeChunkedArray(data)

        # data has wrong dimensions
        data = np.array([[1, 2], [3, 4]])  # use HaplotypeChunkedArray instead
        with pytest.raises(TypeError):
            GenotypeChunkedArray(data)

        # diploid data (typed)
        g = self.setup_instance(np.array(diploid_genotype_data, dtype='i1'))
        aeq(diploid_genotype_data, g)
        assert np.int8 == g.dtype

        # polyploid data (typed)
        g = self.setup_instance(np.array(triploid_genotype_data, dtype='i1'))
        aeq(triploid_genotype_data, g)
        assert np.int8 == g.dtype

    def test_storage(self):
        g = self.setup_instance(np.array(diploid_genotype_data))
        # default is zarr mem
        assert isinstance(g.values, zarr.core.Array)

    def test_slice_types(self):

        g = self.setup_instance(np.array(diploid_genotype_data, dtype='i1'))

        # total slice
        s = g[:]
        self.assertIsInstance(s, GenotypeArray)

        # row slice
        s = g[1:]
        self.assertIsInstance(s, GenotypeArray)

        # col slice
        s = g[:, 1:]
        self.assertIsInstance(s, GenotypeArray)

        # row index
        s = g[0]
        self.assertIsInstance(s, GenotypeVector)

        # col index
        s = g[:, 0]
        self.assertIsInstance(s, GenotypeVector)

        # ploidy index
        s = g[:, :, 0]
        self.assertIsInstance(s, np.ndarray)

        # item
        s = g[0, 0, 0]
        self.assertIsInstance(s, np.int8)

    def test_take(self):
        g = self.setup_instance(diploid_genotype_data)
        # take variants not in original order
        # not supported for carrays
        indices = [2, 0]
        with pytest.raises(NotImplementedError):
            g.take(indices, axis=0)

    def test_to_n_ref_array_like(self):
        # see also https://github.com/cggh/scikit-allel/issues/66

        gn = self.setup_instance(diploid_genotype_data).to_n_ref(fill=-1)
        t = np.array(gn) > 0
        assert 4 == np.count_nonzero(t)
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


class GenotypeChunkedArrayTestsHDF5MemStorage(GenotypeChunkedArrayTests):

    def setUp(self):
        chunked.storage_registry['default'] = chunked.hdf5mem_storage

    def setup_instance(self, data, **kwargs):
        data = chunked.hdf5mem_storage.array(data, **kwargs)
        return GenotypeChunkedArray(data)

    def test_storage(self):
        g = self.setup_instance(np.array(diploid_genotype_data))
        assert isinstance(g.values, h5py.Dataset)


class GenotypeChunkedArrayTestsZarrMemStorage(GenotypeChunkedArrayTests):

    def setUp(self):
        chunked.storage_registry['default'] = chunked.zarrmem_storage

    def setup_instance(self, data, **kwargs):
        data = chunked.zarrmem_storage.array(data, **kwargs)
        return GenotypeChunkedArray(data)

    def test_storage(self):
        g = self.setup_instance(np.array(diploid_genotype_data))
        assert isinstance(g.values, zarr.core.Array)


class GenotypeChunkedArrayTestsZarrTmpStorage(GenotypeChunkedArrayTests):

    def setUp(self):
        chunked.storage_registry['default'] = chunked.zarrtmp_storage

    def setup_instance(self, data, **kwargs):
        data = chunked.zarrtmp_storage.array(data, **kwargs)
        return GenotypeChunkedArray(data)

    def test_storage(self):
        g = self.setup_instance(np.array(diploid_genotype_data))
        assert isinstance(g.values, zarr.Array)
        assert isinstance(g.values.store, zarr.DirectoryStore)


class GenotypeChunkedArrayTestsHDF5TmpStorage(GenotypeChunkedArrayTests):

    def setUp(self):
        chunked.storage_registry['default'] = chunked.hdf5tmp_storage

    def setup_instance(self, data, dtype=None):
        data = chunked.hdf5tmp_storage.array(data, dtype=dtype)
        return GenotypeChunkedArray(data)

    def test_storage(self):
        g = self.setup_instance(np.array(diploid_genotype_data))
        assert isinstance(g.values, h5py.Dataset)


class GenotypeChunkedArrayTestsHDF5TmpLZFStorage(GenotypeChunkedArrayTests):

    def setUp(self):
        chunked.storage_registry['default'] = chunked.hdf5tmp_lzf_storage

    def setup_instance(self, data, dtype=None):
        data = chunked.hdf5tmp_lzf_storage.array(data, dtype=dtype)
        return GenotypeChunkedArray(data)

    def test_storage(self):
        g = self.setup_instance(np.array(diploid_genotype_data))
        assert isinstance(g.values, h5py.Dataset)
        assert g.values.compression == 'lzf'


# noinspection PyMethodMayBeStatic
class HaplotypeChunkedArrayTests(HaplotypeArrayInterface, unittest.TestCase):

    _class = HaplotypeChunkedArray

    def setUp(self):
        chunked.storage_registry['default'] = chunked.zarrmem_storage

    def setup_instance(self, data, dtype=None):
        data = chunked.storage_registry['default'].array(data, dtype=dtype,
                                                         chunks=2)
        return HaplotypeChunkedArray(data)

    def test_constructor(self):

        # missing data arg
        with pytest.raises(TypeError):
            # noinspection PyArgumentList
            HaplotypeChunkedArray()

        # data has wrong dtype
        data = 'foo bar'
        with pytest.raises(TypeError):
            HaplotypeChunkedArray(data)

        # data has wrong dtype
        data = np.array([4., 5., 3.7])
        with pytest.raises(TypeError):
            HaplotypeChunkedArray(data)

        # data has wrong dimensions
        data = np.array([1, 2, 3])
        with pytest.raises(TypeError):
            HaplotypeChunkedArray(data)

        # data has wrong dimensions
        data = np.array([[[1, 2], [3, 4]]])  # use GenotypeCArray instead
        with pytest.raises(TypeError):
            HaplotypeChunkedArray(data)

        # typed data (typed)
        h = HaplotypeChunkedArray(np.array(haplotype_data, dtype='i1'))
        aeq(haplotype_data, h)
        assert np.int8 == h.dtype

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


# noinspection PyMethodMayBeStatic
class AlleleCountsChunkedArrayTests(AlleleCountsArrayInterface,
                                    unittest.TestCase):

    _class = AlleleCountsChunkedArray

    def setUp(self):
        chunked.storage_registry['default'] = chunked.zarrmem_storage

    def setup_instance(self, data):
        data = chunked.storage_registry['default'].array(data, chunks=2)
        return AlleleCountsChunkedArray(data)

    def test_constructor(self):

        # missing data arg
        with pytest.raises(TypeError):
            # noinspection PyArgumentList
            AlleleCountsChunkedArray()

        # data has wrong dtype
        data = 'foo bar'
        with pytest.raises(TypeError):
            AlleleCountsChunkedArray(data)

        # data has wrong dtype
        data = np.array([4., 5., 3.7])
        with pytest.raises(TypeError):
            AlleleCountsChunkedArray(data)

        # data has wrong dimensions
        data = np.array([1, 2, 3])
        with pytest.raises(TypeError):
            AlleleCountsChunkedArray(data)

        # data has wrong dimensions
        data = np.array([[[1, 2], [3, 4]]])
        with pytest.raises(TypeError):
            AlleleCountsChunkedArray(data)

        # typed data (typed)
        ac = AlleleCountsChunkedArray(np.array(allele_counts_data, dtype='u1'))
        aeq(allele_counts_data, ac)
        assert np.uint8 == ac.dtype

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


class GenotypeAlleleCountsChunkedArrayTests(GenotypeAlleleCountsArrayInterface,
                                            unittest.TestCase):

    _class = GenotypeAlleleCountsChunkedArray

    def setUp(self):
        chunked.storage_registry['default'] = chunked.zarrmem_storage

    def setup_instance(self, data, **kwargs):
        data = chunked.storage_registry['default'].array(data, chunks=2, **kwargs)
        return GenotypeAlleleCountsChunkedArray(data)

    def test_constructor(self):

        # missing data arg
        with pytest.raises(TypeError):
            # noinspection PyArgumentList
            GenotypeAlleleCountsChunkedArray()

        # data has wrong dtype
        data = 'foo bar'
        with pytest.raises(TypeError):
            GenotypeAlleleCountsChunkedArray(data)

        # data has wrong dtype
        data = np.array([4., 5., 3.7])
        with pytest.raises(TypeError):
            GenotypeAlleleCountsChunkedArray(data)

        # data has wrong dimensions
        data = np.array([1, 2, 3])
        with pytest.raises(TypeError):
            GenotypeAlleleCountsChunkedArray(data)

        # data has wrong dimensions
        data = np.array([[1, 2], [3, 4]])  # use HaplotypeChunkedArray instead
        with pytest.raises(TypeError):
            GenotypeAlleleCountsChunkedArray(data)

        # diploid data (typed)
        g = self.setup_instance(np.array(diploid_genotype_ac_data, dtype='i1'))
        aeq(diploid_genotype_ac_data, g)
        assert np.int8 == g.dtype

        # polyploid data (typed)
        g = self.setup_instance(np.array(triploid_genotype_ac_data, dtype='i1'))
        aeq(triploid_genotype_ac_data, g)
        assert np.int8 == g.dtype

    def test_storage(self):
        g = self.setup_instance(np.array(diploid_genotype_ac_data))
        # default is zarr mem
        assert isinstance(g.values, zarr.core.Array)

    def test_slice_types(self):

        g = self.setup_instance(np.array(diploid_genotype_ac_data, dtype='i1'))

        # total slice
        s = g[:]
        self.assertIsInstance(s, GenotypeAlleleCountsArray)

        # row slice
        s = g[1:]
        self.assertIsInstance(s, GenotypeAlleleCountsArray)

        # col slice
        s = g[:, 1:]
        self.assertIsInstance(s, GenotypeAlleleCountsArray)

        # row index
        s = g[0]
        self.assertIsInstance(s, GenotypeAlleleCountsVector)

        # col index
        s = g[:, 0]
        self.assertIsInstance(s, GenotypeAlleleCountsVector)

        # ploidy index
        s = g[:, :, 0]
        self.assertIsInstance(s, np.ndarray)

        # item
        s = g[0, 0, 0]
        self.assertIsInstance(s, np.int8)


class VariantChunkedTableTests(VariantTableInterface, unittest.TestCase):

    _class = VariantChunkedTable

    def setUp(self):
        chunked.storage_registry['default'] = chunked.zarrmem_storage

    def setup_instance(self, data, **kwargs):
        data = chunked.storage_registry['default'].table(data, chunks=2)
        return VariantChunkedTable(data, **kwargs)

    def test_storage(self):
        a = np.rec.array(variant_table_data, dtype=variant_table_dtype)
        vt = self.setup_instance(a)
        assert isinstance(vt.values, ZarrTable)

    def test_constructor(self):

        # missing data arg
        with self.assertRaises(TypeError):
            # noinspection PyArgumentList
            VariantChunkedTable()

        # recarray
        ra = np.rec.array(variant_table_data, dtype=variant_table_dtype)
        vt = VariantChunkedTable(ra)
        assert 5 == len(vt)
        aeq(ra, vt)

        # dict
        d = {n: ra[n] for n in variant_table_names}
        vt = VariantChunkedTable(d, names=variant_table_names)
        assert 5 == len(vt)
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
        self.assertIsInstance(s, chunked.ChunkedArrayWrapper)

        # bad access
        with pytest.raises(IndexError):
            # noinspection PyStatementEffect
            vt[:, 0]

    def test_take(self):
        a = np.rec.array(variant_table_data, dtype=variant_table_dtype)
        vt = VariantChunkedTable(a)
        # take variants not in original order
        # not supported for carrays
        indices = [2, 0]
        with pytest.raises(NotImplementedError):
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
        assert isinstance(vt.values, h5py.Group)


class VariantChunkedTableTestsZarrStorage(VariantChunkedTableTests):

    def setUp(self):
        chunked.storage_registry['default'] = chunked.zarrmem_storage

    def setup_instance(self, data, **kwargs):
        data = chunked.storage_registry['default'].table(data)
        return VariantChunkedTable(data, **kwargs)

    def test_storage(self):
        a = np.rec.array(variant_table_data, dtype=variant_table_dtype)
        vt = self.setup_instance(a)
        assert isinstance(vt.values, ZarrTable)

    # noinspection PyMethodMayBeStatic
    def test_zarr_group(self):
        z = zarr.group()
        z.create_dataset('chrom', data=['1', '2', '3'])
        z.create_dataset('pos', data=[2, 4, 6])
        vt = VariantChunkedTable(z)
        assert isinstance(vt.values, zarr.Group)

    # noinspection PyMethodMayBeStatic
    def test_hdf5_to_zarr(self):
        # https://github.com/cggh/scikit-allel/issues/353
        chrom = [b'chr1', b'chr1', b'chr2', b'chr2', b'chr3']
        pos = [2, 7, 3, 9, 6]
        dp = [35, 12, 78, 22, 99]
        qd = [4.5, 6.7, 1.2, 4.4, 2.8]
        ac = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]
        with h5py.File('callset.h5', mode='w') as h5f:
            h5g = h5f.create_group('/3L/variants')
            h5g.create_dataset('CHROM', data=chrom, chunks=True)
            h5g.create_dataset('POS', data=pos, chunks=True)
            h5g.create_dataset('DP', data=dp, chunks=True)
            h5g.create_dataset('QD', data=qd, chunks=True)
            h5g.create_dataset('AC', data=ac, chunks=True)

        callset = h5py.File('callset.h5', mode='r')
        vt = VariantChunkedTable(callset['/3L/variants'], names=['CHROM', 'POS', 'AC', 'QD', 'DP'])
        vs = vt.eval('DP > 15')[:]
        v = vt.compress(vs, axis=0)
        assert isinstance(v, VariantChunkedTable)
        assert isinstance(v.values, ZarrTable)


class AlleleCountsChunkedTableTests(unittest.TestCase):

    def setUp(self):
        chunked.storage_registry['default'] = chunked.zarrmem_storage

    # noinspection PyMethodMayBeStatic
    def test_count_alleles_subpops(self):

        data = chunked.storage_registry['default'].array(diploid_genotype_data, chunks=2)
        g = GenotypeChunkedArray(data)
        subpops = {'foo': [0, 2], 'bar': [1]}
        ac_subpops = g.count_alleles_subpops(subpops)
        for p in subpops.keys():
            ac = g.take(subpops[p], axis=1).count_alleles()
            aeq(ac, ac_subpops[p])

        loc = np.array([True, False, True, False, True])
        t = ac_subpops.compress(loc)
        assert 3 == len(t)
