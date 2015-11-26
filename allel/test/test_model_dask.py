# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import unittest
import numpy as np
import bcolz
import h5py
import dask
import dask.array as da
import dask.async
# use synchronous scheduler because getting random hangs with default
da.set_options(get=dask.async.get_sync)
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
from allel.model.dask import GenotypeDaskArray


class GenotypeDaskArrayTests(GenotypeArrayInterface, unittest.TestCase):

    _class = GenotypeDaskArray

    def setup_instance(self, data):
        data = np.asarray(data)
        return GenotypeDaskArray(data, chunks=(2, 2, data.shape[2]))

    def test_constructor(self):

        # missing data arg
        with assert_raises(TypeError):
            # noinspection PyArgumentList
            GenotypeDaskArray()

        # data has wrong dtype
        data = 'foo bar'
        with assert_raises(ValueError):
            GenotypeDaskArray(data)

        # data has wrong dtype
        data = np.array([4., 5., 3.7])
        with assert_raises(ValueError):
            GenotypeDaskArray(data)

        # data has wrong dimensions
        data = np.array([1, 2, 3])
        with assert_raises(ValueError):
            GenotypeDaskArray(data)

        # data has wrong dimensions
        data = np.array([[1, 2], [3, 4]])  # use HaplotypeDaskArray instead
        with assert_raises(ValueError):
            GenotypeDaskArray(data)

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

        # col index
        s = gd[:, 0]
        self.assertNotIsInstance(s, GenotypeDaskArray)
        self.assertIsInstance(s, da.Array)
        self.assertNotIsInstance(s.compute(), GenotypeArray)

        # ploidy index
        s = gd[:, :, 0]
        self.assertNotIsInstance(s, GenotypeDaskArray)
        self.assertIsInstance(s, da.Array)
        self.assertNotIsInstance(s.compute(), GenotypeArray)

        # item
        s = gd[0, 0, 0]
        self.assertNotIsInstance(s, GenotypeDaskArray)
        self.assertIsInstance(s, da.Array)
        self.assertNotIsInstance(s.compute(), GenotypeArray)

    def test_take(self):
        g = np.array(diploid_genotype_data)
        gd = self.setup_instance(g)
        # take variants not in original order
        indices = [2, 0]
        expect = g.take(indices, axis=0)
        actual = gd.take(indices, axis=0)
        aeq(expect, actual)
