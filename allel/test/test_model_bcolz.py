# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import unittest


import numpy as np
import bcolz
from nose.tools import eq_ as eq, assert_raises


from allel.model import GenotypeArray, HaplotypeArray
from allel.test.tools import assert_array_equal as aeq
from allel.test.test_model_api import GenotypeArrayInterface, \
    HaplotypeArrayInterface, diploid_genotype_data, triploid_genotype_data, \
    haplotype_data
from allel.bcolz import GenotypeCArray


class GenotypeCArrayTests(GenotypeArrayInterface, unittest.TestCase):

    _class = GenotypeCArray

    def setup_instance(self, data):
        return GenotypeCArray(data)

    def test_constructor(self):

        # missing data arg
        with assert_raises(TypeError):
            # noinspection PyArgumentList
            GenotypeCArray()

        # data has wrong dtype
        data = 'foo bar'
        with assert_raises(NotImplementedError):
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

