# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import unittest
import numpy as np
import dask
import dask.array as da
import dask.async
from nose.tools import assert_raises, eq_ as eq


from allel.model.ndarray import GenotypeArray, HaplotypeArray, \
    AlleleCountsArray
from allel.test.tools import assert_array_equal as aeq
from allel.test.test_model_api import GenotypeArrayInterface, \
    diploid_genotype_data, triploid_genotype_data, HaplotypeArrayInterface, \
    haplotype_data, allele_counts_data, AlleleCountsArrayInterface
from allel.model.dask import GenotypeDaskArray, HaplotypeDaskArray, \
    AlleleCountsDaskArray


# use synchronous scheduler because getting random hangs with default
da.set_options(get=dask.async.get_sync)


class GenotypeDaskArrayTests(GenotypeArrayInterface, unittest.TestCase):

    _class = GenotypeDaskArray

    def setup_instance(self, data, dtype=None):
        return GenotypeDaskArray.from_array(np.asarray(data, dtype=dtype),
                                            chunks=(2, 2, None))

    def test_constructor(self):

        # missing data arg
        with assert_raises(TypeError):
            # noinspection PyArgumentList
            GenotypeDaskArray.from_array()

        # data has wrong dtype
        data = 'foo bar'
        with assert_raises(ValueError):
            GenotypeDaskArray.from_array(data)

        # data has wrong dtype
        data = np.array([4., 5., 3.7])
        with assert_raises(ValueError):
            GenotypeDaskArray.from_array(data)

        # data has wrong dimensions
        data = np.array([1, 2, 3])
        with assert_raises(ValueError):
            GenotypeDaskArray.from_array(data)

        # data has wrong dimensions
        data = np.array([[1, 2], [3, 4]])  # use HaplotypeDaskArray instead
        with assert_raises(ValueError):
            GenotypeDaskArray.from_array(data)

        # diploid data (typed)
        gd = self.setup_instance(np.array(diploid_genotype_data, dtype='i1'))
        aeq(diploid_genotype_data, gd)
        eq(np.int8, gd.dtype)

        # polyploid data (typed)
        gd = self.setup_instance(np.array(triploid_genotype_data, dtype='i1'))
        aeq(triploid_genotype_data, gd)
        eq(np.int8, gd.dtype)

    def test_slice_types(self):

        gd = self.setup_instance(np.array(diploid_genotype_data, dtype='i1'))
        self.assertIsInstance(gd, GenotypeDaskArray)
        self.assertIsInstance(gd.compute(), GenotypeArray)

        # N.B., all slices return dask arrays, computation is always
        # deferred until an explicit call to compute()

        # row slice
        s = gd[1:]
        self.assertIsInstance(s, GenotypeDaskArray)
        self.assertIsInstance(s.compute(), GenotypeArray)

        # col slice
        s = gd[:, 1:]
        self.assertIsInstance(s, GenotypeDaskArray)
        self.assertIsInstance(s.compute(), GenotypeArray)

        # row index
        s = gd[0]
        self.assertNotIsInstance(s, GenotypeDaskArray)
        self.assertIsInstance(s, da.Array)
        self.assertNotIsInstance(s.compute(), GenotypeArray)
        self.assertIsInstance(s.compute(), np.ndarray)

        # col index
        s = gd[:, 0]
        self.assertNotIsInstance(s, GenotypeDaskArray)
        self.assertIsInstance(s, da.Array)
        self.assertNotIsInstance(s.compute(), GenotypeArray)
        self.assertIsInstance(s.compute(), np.ndarray)

        # ploidy index
        s = gd[:, :, 0]
        self.assertNotIsInstance(s, GenotypeDaskArray)
        self.assertIsInstance(s, da.Array)
        self.assertNotIsInstance(s.compute(), GenotypeArray)
        self.assertIsInstance(s.compute(), np.ndarray)

        # item
        s = gd[0, 0, 0]
        self.assertNotIsInstance(s, GenotypeDaskArray)
        self.assertIsInstance(s, da.Array)
        self.assertNotIsInstance(s.compute(), GenotypeArray)
        self.assertIsInstance(s.compute()[()], np.int8)

    def test_take(self):
        g = np.array(diploid_genotype_data)
        gd = self.setup_instance(g)
        # take variants not in original order
        indices = [2, 0]
        expect = g.take(indices, axis=0)
        actual = gd.take(indices, axis=0)
        aeq(expect, actual)


class HaplotypeDaskArrayTests(HaplotypeArrayInterface, unittest.TestCase):

    _class = HaplotypeDaskArray

    def setup_instance(self, data, dtype=None):
        return HaplotypeDaskArray.from_array(np.asarray(data, dtype=dtype),
                                             chunks=(2, 2))

    def test_constructor(self):

        # missing data arg
        with assert_raises(TypeError):
            # noinspection PyArgumentList
            HaplotypeDaskArray.from_array()

        # data has wrong dtype
        data = 'foo bar'
        with assert_raises(ValueError):
            HaplotypeDaskArray.from_array(data)

        # data has wrong dtype
        data = np.array([4., 5., 3.7])
        with assert_raises(ValueError):
            HaplotypeDaskArray.from_array(data)

        # data has wrong dimensions
        data = np.array([1, 2, 3])
        with assert_raises(ValueError):
            HaplotypeDaskArray.from_array(data)

        # data has wrong dimensions
        data = np.array([[[1, 2], [3, 4]]])  # use GenotypeDaskArray instead
        with assert_raises(ValueError):
            HaplotypeDaskArray.from_array(data)

        # valid data (typed)
        hd = self.setup_instance(np.array(haplotype_data, dtype='i1'))
        aeq(haplotype_data, hd)
        eq(np.int8, hd.dtype)

    def test_slice_types(self):

        hd = self.setup_instance(np.array(haplotype_data, dtype='i1'))
        self.assertIsInstance(hd, HaplotypeDaskArray)
        self.assertIsInstance(hd.compute(), HaplotypeArray)

        # N.B., all slices return dask arrays, computation is always
        # deferred until an explicit call to compute()

        # row slice
        s = hd[1:]
        self.assertIsInstance(s, HaplotypeDaskArray)
        self.assertIsInstance(s.compute(), HaplotypeArray)

        # col slice
        s = hd[:, 1:]
        self.assertIsInstance(s, HaplotypeDaskArray)
        self.assertIsInstance(s.compute(), HaplotypeArray)

        # row index
        s = hd[0]
        self.assertNotIsInstance(s, HaplotypeDaskArray)
        self.assertIsInstance(s, da.Array)
        self.assertNotIsInstance(s.compute(), HaplotypeArray)
        self.assertIsInstance(s.compute(), np.ndarray)

        # col index
        s = hd[:, 0]
        self.assertNotIsInstance(s, HaplotypeDaskArray)
        self.assertIsInstance(s, da.Array)
        self.assertNotIsInstance(s.compute(), HaplotypeArray)
        self.assertIsInstance(s.compute(), np.ndarray)

        # item
        s = hd[0, 0]
        self.assertNotIsInstance(s, HaplotypeDaskArray)
        self.assertIsInstance(s, da.Array)
        self.assertNotIsInstance(s.compute(), HaplotypeArray)
        self.assertIsInstance(s.compute()[()], np.int8)


class AlleleCountsDaskArrayTests(AlleleCountsArrayInterface,
                                 unittest.TestCase):

    _class = AlleleCountsDaskArray

    def setup_instance(self, data):
        return AlleleCountsDaskArray.from_array(data, chunks=(2, None))

    def test_constructor(self):

        # missing data arg
        with assert_raises(TypeError):
            # noinspection PyArgumentList
            AlleleCountsDaskArray.from_array()

        # data has wrong dtype
        data = 'foo bar'
        with assert_raises(ValueError):
            AlleleCountsDaskArray.from_array(data)

        # data has wrong dtype
        data = np.array([4., 5., 3.7])
        with assert_raises(ValueError):
            AlleleCountsDaskArray.from_array(data)

        # data has wrong dimensions
        data = np.array([1, 2, 3])
        with assert_raises(ValueError):
            AlleleCountsDaskArray.from_array(data)

        # data has wrong dimensions
        data = np.array([[[1, 2], [3, 4]]])
        with assert_raises(ValueError):
            AlleleCountsDaskArray.from_array(data)

        # valid data (typed)
        hd = self.setup_instance(np.array(allele_counts_data, dtype='u2'))
        aeq(allele_counts_data, hd)
        eq(np.uint16, hd.dtype)

    def test_slice_types(self):

        acd = self.setup_instance(np.array(allele_counts_data, dtype='u2'))
        self.assertIsInstance(acd, AlleleCountsDaskArray)
        self.assertIsInstance(acd.compute(), AlleleCountsArray)

        # N.B., all slices return dask arrays, computation is always
        # deferred until an explicit call to compute()

        # row slice
        s = acd[1:]
        self.assertIsInstance(s, AlleleCountsDaskArray)
        self.assertIsInstance(s.compute(), AlleleCountsArray)

        # col slice
        s = acd[:, 1:]
        self.assertNotIsInstance(s, AlleleCountsDaskArray)
        self.assertIsInstance(s, da.Array)
        self.assertNotIsInstance(s.compute(), AlleleCountsArray)
        self.assertIsInstance(s.compute(), np.ndarray)

        # row index
        s = acd[0]
        self.assertNotIsInstance(s, AlleleCountsDaskArray)
        self.assertIsInstance(s, da.Array)
        self.assertNotIsInstance(s.compute(), AlleleCountsArray)
        self.assertIsInstance(s.compute(), np.ndarray)

        # col index
        s = acd[:, 0]
        self.assertNotIsInstance(s, AlleleCountsDaskArray)
        self.assertIsInstance(s, da.Array)
        self.assertNotIsInstance(s.compute(), AlleleCountsArray)
        self.assertIsInstance(s.compute(), np.ndarray)

        # item
        s = acd[0, 0]
        self.assertNotIsInstance(s, AlleleCountsDaskArray)
        self.assertIsInstance(s, da.Array)
        self.assertNotIsInstance(s.compute(), AlleleCountsArray)
        self.assertIsInstance(s.compute()[()], np.uint16)
