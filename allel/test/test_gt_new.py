# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import unittest


import numpy as np
from allel.test.tools import assert_array_equal as aeq


from allel.gt_new import GenotypeArray


haploid_data = [[0, 1, -1],
                [1, 1, -1],
                [2, -1, -1],
                [-1, -1, -1]]

diploid_data = [[[0, 0], [0, 1], [-1, -1]],
                [[0, 2], [1, 1], [-1, -1]],
                [[1, 0], [2, 1], [-1, -1]],
                [[2, 2], [-1, -1], [-1, -1]],
                [[-1, -1], [-1, -1], [-1, -1]]]

triploid_data = [[[0, 0, 0], [0, 0, 1], [-1, -1, -1]],
                 [[0, 1, 1], [1, 1, 1], [-1, -1, -1]],
                 [[0, 1, 2], [-1, -1, -1], [-1, -1, -1]],
                 [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]]


class TestGenotypeArray(unittest.TestCase):

    def test_constructor(self):
        eq = self.assertEqual

        # missing data arg
        with self.assertRaises(TypeError):
            GenotypeArray()

        # data has wrong dtype
        data = 'foo bar'
        with self.assertRaises(TypeError):
            GenotypeArray(data)

        # data has wrong dtype
        data = [4., 5., 3.7]
        with self.assertRaises(TypeError):
            GenotypeArray(data)

        # data has wrong dimensions
        data = [1, 2, 3]
        with self.assertRaises(TypeError):
            GenotypeArray(data)

        # data has wrong dimensions
        data = haploid_data  # use HaplotypeArray instead
        with self.assertRaises(TypeError):
            GenotypeArray(data)

        # diploid data
        g = GenotypeArray(diploid_data)
        aeq(diploid_data, g)
        eq(np.int, g.dtype)
        eq(3, g.ndim)
        eq(5, g.n_variants)
        eq(3, g.n_samples)
        eq(2, g.ploidy)

        # diploid data (typed)
        g = GenotypeArray(np.array(diploid_data, dtype='i1'))
        aeq(diploid_data, g)
        eq(np.int8, g.dtype)

        # triploid data
        g = GenotypeArray(triploid_data)
        aeq(triploid_data, g)
        eq(np.int, g.dtype)
        eq(3, g.ndim)
        eq(4, g.n_variants)
        eq(3, g.n_samples)
        eq(3, g.ploidy)

        # triploid data (typed)
        g = GenotypeArray(np.array(triploid_data, dtype='i1'))
        aeq(triploid_data, g)
        eq(np.int8, g.dtype)

    def test_slice(self):
        eq = self.assertEqual

        g = GenotypeArray(diploid_data)
        eq(2, g.ploidy)

        # row slice
        s = g[1:]
        self.assertIsInstance(s, GenotypeArray)
        aeq(diploid_data[1:], s)
        eq(4, s.n_variants)
        eq(3, s.n_samples)
        eq(2, s.ploidy)

        # col slice
        s = g[:, 1:]
        self.assertIsInstance(s, GenotypeArray)
        aeq(np.array(diploid_data)[:, 1:], s)
        eq(5, s.n_variants)
        eq(2, s.n_samples)
        eq(2, s.ploidy)

        # row index
        s = g[0]
        self.assertIsInstance(s, np.ndarray)
        self.assertNotIsInstance(s, GenotypeArray)
        aeq(diploid_data[0], s)

        # col index
        s = g[:, 0]
        self.assertIsInstance(s, np.ndarray)
        self.assertNotIsInstance(s, GenotypeArray)
        aeq(np.array(diploid_data)[:, 0], s)

        # ploidy index
        s = g[:, :, 0]
        self.assertIsInstance(s, np.ndarray)
        self.assertNotIsInstance(s, GenotypeArray)
        aeq(np.array(diploid_data)[:, :, 0], s)

    def test_view(self):
        eq = self.assertEqual

        # data has wrong dtype
        data = 'foo bar'
        with self.assertRaises(TypeError):
            np.array(data).view(GenotypeArray)

        # data has wrong dtype
        data = [4., 5., 3.7]
        with self.assertRaises(TypeError):
            np.array(data).view(GenotypeArray)

        # data has wrong dimensions
        data = [1, 2, 3]
        with self.assertRaises(TypeError):
            np.array(data).view(GenotypeArray)

        # data has wrong dimensions
        data = haploid_data  # use HaplotypeArray instead
        with self.assertRaises(TypeError):
            np.array(data).view(GenotypeArray)

        # diploid data
        g = np.array(diploid_data).view(GenotypeArray)
        aeq(diploid_data, g)
        eq(np.int, g.dtype)
        eq(3, g.ndim)
        eq(5, g.n_variants)
        eq(3, g.n_samples)
        eq(2, g.ploidy)

        # triploid data
        g = np.array(triploid_data).view(GenotypeArray)
        aeq(triploid_data, g)
        eq(np.int, g.dtype)
        eq(3, g.ndim)
        eq(4, g.n_variants)
        eq(3, g.n_samples)
        eq(3, g.ploidy)

    def test_is_called(self):

        # diploid
        expect = np.array([[1, 1, 0],
                           [1, 1, 0],
                           [1, 1, 0],
                           [1, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = GenotypeArray(diploid_data).is_called()
        aeq(expect, actual)

        # polyploid
        expect = np.array([[1, 1, 0],
                           [1, 1, 0],
                           [1, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = GenotypeArray(triploid_data).is_called()
        aeq(expect, actual)

    def test_is_missing(self):

        # diploid
        expect = np.array([[0, 0, 1],
                           [0, 0, 1],
                           [0, 0, 1],
                           [0, 1, 1],
                           [1, 1, 1]], dtype='b1')
        actual = GenotypeArray(diploid_data).is_missing()
        aeq(expect, actual)

        # polyploid
        expect = np.array([[0, 0, 1],
                           [0, 0, 1],
                           [0, 1, 1],
                           [1, 1, 1]], dtype='b1')
        actual = GenotypeArray(triploid_data).is_missing()
        aeq(expect, actual)

    def test_is_hom(self):

        # diploid
        expect = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0],
                           [1, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = GenotypeArray(diploid_data).is_hom()
        aeq(expect, actual)

        # polyploid
        expect = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = GenotypeArray(triploid_data).is_hom()
        aeq(expect, actual)

    def test_is_hom_ref(self):

        # diploid
        expect = np.array([[1, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = GenotypeArray(diploid_data).is_hom(allele=0)
        aeq(expect, actual)
        actual = GenotypeArray(diploid_data).is_hom_ref()
        aeq(expect, actual)

        # polyploid
        expect = np.array([[1, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = GenotypeArray(triploid_data).is_hom(allele=0)
        aeq(expect, actual)
        actual = GenotypeArray(triploid_data).is_hom_ref()
        aeq(expect, actual)

    def test_is_hom_alt(self):

        # diploid
        expect = np.array([[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0],
                           [1, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = GenotypeArray(diploid_data).is_hom_alt()
        aeq(expect, actual)

        # polyploid
        expect = np.array([[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = GenotypeArray(triploid_data).is_hom_alt()
        aeq(expect, actual)

    def test_is_hom_1(self):

        # diploid
        expect = np.array([[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = GenotypeArray(diploid_data).is_hom(allele=1)
        aeq(expect, actual)

        # polyploid
        expect = np.array([[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = GenotypeArray(triploid_data).is_hom(allele=1)
        aeq(expect, actual)

    def test_is_het(self):

        # diploid
        expect = np.array([[0, 1, 0],
                           [1, 0, 0],
                           [1, 1, 0],
                           [0, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = GenotypeArray(diploid_data).is_het()
        aeq(expect, actual)

        # polyploid
        expect = np.array([[0, 1, 0],
                           [1, 0, 0],
                           [1, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = GenotypeArray(triploid_data).is_het()
        aeq(expect, actual)

    def test_is_call(self):

        # diploid
        expect = np.array([[0, 0, 0],
                           [1, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = GenotypeArray(diploid_data).is_call(call=(0, 2))
        aeq(expect, actual)

        # polyploid
        expect = np.array([[0, 0, 0],
                           [0, 0, 0],
                           [1, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = GenotypeArray(triploid_data).is_call(call=(0, 1, 2))
        aeq(expect, actual)
