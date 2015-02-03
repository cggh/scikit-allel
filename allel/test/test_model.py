# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import unittest


import numpy as np
from allel.test.tools import assert_array_equal as aeq


from allel.model import GenotypeArray, HaplotypeArray, PosArray


haplotype_data = [[0, 1, -1],
                  [1, 1, -1],
                  [2, -1, -1],
                  [-1, -1, -1]]

diploid_genotype_data = [[[0, 0], [0, 1], [-1, -1]],
                         [[0, 2], [1, 1], [-1, -1]],
                         [[1, 0], [2, 1], [-1, -1]],
                         [[2, 2], [-1, -1], [-1, -1]],
                         [[-1, -1], [-1, -1], [-1, -1]]]

triploid_genotype_data = [[[0, 0, 0], [0, 0, 1], [-1, -1, -1]],
                          [[0, 1, 1], [1, 1, 1], [-1, -1, -1]],
                          [[0, 1, 2], [-1, -1, -1], [-1, -1, -1]],
                          [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]]


class TestGenotypeArray(unittest.TestCase):

    def test_constructor(self):
        eq = self.assertEqual

        # missing data arg
        with self.assertRaises(TypeError):
            # noinspection PyArgumentList
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
        data = haplotype_data  # use HaplotypeArray instead
        with self.assertRaises(TypeError):
            GenotypeArray(data)

        # diploid data
        g = GenotypeArray(diploid_genotype_data)
        aeq(diploid_genotype_data, g)
        eq(np.int, g.dtype)
        eq(3, g.ndim)
        eq(5, g.n_variants)
        eq(3, g.n_samples)
        eq(2, g.ploidy)

        # diploid data (typed)
        g = GenotypeArray(np.array(diploid_genotype_data, dtype='i1'))
        aeq(diploid_genotype_data, g)
        eq(np.int8, g.dtype)

        # polyploid data
        g = GenotypeArray(triploid_genotype_data)
        aeq(triploid_genotype_data, g)
        eq(np.int, g.dtype)
        eq(3, g.ndim)
        eq(4, g.n_variants)
        eq(3, g.n_samples)
        eq(3, g.ploidy)

        # polyploid data (typed)
        g = GenotypeArray(np.array(triploid_genotype_data, dtype='i1'))
        aeq(triploid_genotype_data, g)
        eq(np.int8, g.dtype)

    def test_slice(self):
        eq = self.assertEqual

        g = GenotypeArray(np.array(diploid_genotype_data, dtype='i1'))
        eq(2, g.ploidy)

        # row slice
        s = g[1:]
        self.assertIsInstance(s, GenotypeArray)
        aeq(diploid_genotype_data[1:], s)
        eq(4, s.n_variants)
        eq(3, s.n_samples)
        eq(2, s.ploidy)

        # col slice
        s = g[:, 1:]
        self.assertIsInstance(s, GenotypeArray)
        aeq(np.array(diploid_genotype_data)[:, 1:], s)
        eq(5, s.n_variants)
        eq(2, s.n_samples)
        eq(2, s.ploidy)

        # row index
        s = g[0]
        self.assertIsInstance(s, np.ndarray)
        self.assertNotIsInstance(s, GenotypeArray)
        aeq(diploid_genotype_data[0], s)

        # col index
        s = g[:, 0]
        self.assertIsInstance(s, np.ndarray)
        self.assertNotIsInstance(s, GenotypeArray)
        aeq(np.array(diploid_genotype_data)[:, 0], s)

        # ploidy index
        s = g[:, :, 0]
        self.assertIsInstance(s, np.ndarray)
        self.assertNotIsInstance(s, GenotypeArray)
        aeq(np.array(diploid_genotype_data)[:, :, 0], s)

        # item
        s = g[0, 0, 0]
        self.assertIsInstance(s, np.int8)
        self.assertNotIsInstance(s, GenotypeArray)
        eq(0, s)

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
        data = haplotype_data  # use HaplotypeArray instead
        with self.assertRaises(TypeError):
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

    def test_is_called(self):

        # diploid
        expect = np.array([[1, 1, 0],
                           [1, 1, 0],
                           [1, 1, 0],
                           [1, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = GenotypeArray(diploid_genotype_data).is_called()
        aeq(expect, actual)

        # polyploid
        expect = np.array([[1, 1, 0],
                           [1, 1, 0],
                           [1, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = GenotypeArray(triploid_genotype_data).is_called()
        aeq(expect, actual)

    def test_is_missing(self):

        # diploid
        expect = np.array([[0, 0, 1],
                           [0, 0, 1],
                           [0, 0, 1],
                           [0, 1, 1],
                           [1, 1, 1]], dtype='b1')
        actual = GenotypeArray(diploid_genotype_data).is_missing()
        aeq(expect, actual)

        # polyploid
        expect = np.array([[0, 0, 1],
                           [0, 0, 1],
                           [0, 1, 1],
                           [1, 1, 1]], dtype='b1')
        actual = GenotypeArray(triploid_genotype_data).is_missing()
        aeq(expect, actual)

    def test_is_hom(self):

        # diploid
        expect = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0],
                           [1, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = GenotypeArray(diploid_genotype_data).is_hom()
        aeq(expect, actual)

        # polyploid
        expect = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = GenotypeArray(triploid_genotype_data).is_hom()
        aeq(expect, actual)

    def test_is_hom_ref(self):

        # diploid
        expect = np.array([[1, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = GenotypeArray(diploid_genotype_data).is_hom(allele=0)
        aeq(expect, actual)
        actual = GenotypeArray(diploid_genotype_data).is_hom_ref()
        aeq(expect, actual)

        # polyploid
        expect = np.array([[1, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = GenotypeArray(triploid_genotype_data).is_hom(allele=0)
        aeq(expect, actual)
        actual = GenotypeArray(triploid_genotype_data).is_hom_ref()
        aeq(expect, actual)

    def test_is_hom_alt(self):

        # diploid
        expect = np.array([[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0],
                           [1, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = GenotypeArray(diploid_genotype_data).is_hom_alt()
        aeq(expect, actual)

        # polyploid
        expect = np.array([[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = GenotypeArray(triploid_genotype_data).is_hom_alt()
        aeq(expect, actual)

    def test_is_hom_1(self):

        # diploid
        expect = np.array([[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = GenotypeArray(diploid_genotype_data).is_hom(allele=1)
        aeq(expect, actual)

        # polyploid
        expect = np.array([[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = GenotypeArray(triploid_genotype_data).is_hom(allele=1)
        aeq(expect, actual)

    def test_is_het(self):

        # diploid
        expect = np.array([[0, 1, 0],
                           [1, 0, 0],
                           [1, 1, 0],
                           [0, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = GenotypeArray(diploid_genotype_data).is_het()
        aeq(expect, actual)

        # polyploid
        expect = np.array([[0, 1, 0],
                           [1, 0, 0],
                           [1, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = GenotypeArray(triploid_genotype_data).is_het()
        aeq(expect, actual)

    def test_is_call(self):

        # diploid
        expect = np.array([[0, 0, 0],
                           [1, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = GenotypeArray(diploid_genotype_data).is_call(call=(0, 2))
        aeq(expect, actual)

        # polyploid
        expect = np.array([[0, 0, 0],
                           [0, 0, 0],
                           [1, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = GenotypeArray(triploid_genotype_data).is_call(call=(0, 1, 2))
        aeq(expect, actual)

    def test_count_called(self):
        eq = self.assertEqual
        g = GenotypeArray(diploid_genotype_data)
        f = g.count_called

        expect = 7
        actual = f()
        eq(expect, actual)

        expect = np.array([4, 3, 0])
        actual = f(axis=0)
        aeq(expect, actual)

        expect = np.array([2, 2, 2, 1, 0])
        actual = f(axis=1)
        aeq(expect, actual)

    def test_count_missing(self):
        eq = self.assertEqual
        g = GenotypeArray(diploid_genotype_data)
        f = g.count_missing

        expect = 8
        actual = f()
        eq(expect, actual)

        expect = np.array([1, 2, 5])
        actual = f(axis=0)
        aeq(expect, actual)

        expect = np.array([1, 1, 1, 2, 3])
        actual = f(axis=1)
        aeq(expect, actual)

    def test_count_hom(self):
        eq = self.assertEqual
        g = GenotypeArray(diploid_genotype_data)
        f = g.count_hom

        expect = 3
        actual = f()
        eq(expect, actual)

        expect = np.array([2, 1, 0])
        actual = f(axis=0)
        aeq(expect, actual)

        expect = np.array([1, 1, 0, 1, 0])
        actual = f(axis=1)
        aeq(expect, actual)

    def test_count_hom_ref(self):
        eq = self.assertEqual
        g = GenotypeArray(diploid_genotype_data)
        f = g.count_hom_ref

        expect = 1
        actual = f()
        eq(expect, actual)

        expect = np.array([1, 0, 0])
        actual = f(axis=0)
        aeq(expect, actual)

        expect = np.array([1, 0, 0, 0, 0])
        actual = f(axis=1)
        aeq(expect, actual)

    def test_count_hom_alt(self):
        eq = self.assertEqual
        g = GenotypeArray(diploid_genotype_data)
        f = g.count_hom_alt

        expect = 2
        actual = f()
        eq(expect, actual)

        expect = np.array([1, 1, 0])
        actual = f(axis=0)
        aeq(expect, actual)

        expect = np.array([0, 1, 0, 1, 0])
        actual = f(axis=1)
        aeq(expect, actual)

    def test_count_het(self):
        eq = self.assertEqual
        g = GenotypeArray(diploid_genotype_data)
        f = g.count_het

        expect = 4
        actual = f()
        eq(expect, actual)

        expect = np.array([2, 2, 0])
        actual = f(axis=0)
        aeq(expect, actual)

        expect = np.array([1, 1, 2, 0, 0])
        actual = f(axis=1)
        aeq(expect, actual)

    def test_count_call(self):
        eq = self.assertEqual
        g = GenotypeArray(diploid_genotype_data)
        f = g.count_call

        expect = 1
        actual = f(call=(2, 1))
        eq(expect, actual)

        expect = np.array([0, 1, 0])
        actual = f(call=(2, 1), axis=0)
        aeq(expect, actual)

        expect = np.array([0, 0, 1, 0, 0])
        actual = f(call=(2, 1), axis=1)
        aeq(expect, actual)

    def test_to_haplotypes(self):

        # diploid
        expect = np.array([[0, 0, 0, 1, -1, -1],
                           [0, 2, 1, 1, -1, -1],
                           [1, 0, 2, 1, -1, -1],
                           [2, 2, -1, -1, -1, -1],
                           [-1, -1, -1, -1, -1, -1]], dtype='i1')
        actual = GenotypeArray(diploid_genotype_data).to_haplotypes()
        aeq(expect, actual)

        # polyploid
        expect = np.array([[0, 0, 0, 0, 0, 1, -1, -1, -1],
                           [0, 1, 1, 1, 1, 1, -1, -1, -1],
                           [0, 1, 2, -1, -1, -1, -1, -1, -1],
                           [-1, -1, -1, -1, -1, -1, -1, -1, -1]], dtype='i1')
        actual = GenotypeArray(triploid_genotype_data).to_haplotypes()
        aeq(expect, actual)

    def test_from_haplotypes(self):
        eq = self.assertEqual

        # diploid
        h_diploid = np.array([[0, 0, 0, 1, -1, -1],
                              [0, 2, 1, 1, -1, -1],
                              [1, 0, 2, 1, -1, -1],
                              [2, 2, -1, -1, -1, -1],
                              [-1, -1, -1, -1, -1, -1]], dtype='i1')
        expect = diploid_genotype_data
        actual = GenotypeArray.from_haplotypes(h_diploid, ploidy=2)
        self.assertIsInstance(actual, GenotypeArray)
        aeq(expect, actual)
        eq(2, actual.ploidy)

        # polyploidy
        h_triploid = np.array([[0, 0, 0, 0, 0, 1, -1, -1, -1],
                               [0, 1, 1, 1, 1, 1, -1, -1, -1],
                               [0, 1, 2, -1, -1, -1, -1, -1, -1],
                               [-1, -1, -1, -1, -1, -1, -1, -1, -1]],
                              dtype='i1')
        expect = triploid_genotype_data
        actual = GenotypeArray.from_haplotypes(h_triploid, ploidy=3)
        self.assertIsInstance(actual, GenotypeArray)
        aeq(expect, actual)
        eq(3, actual.ploidy)

    def test_to_n_alt(self):

        # diploid
        expect = np.array([[0, 1, 0],
                           [1, 2, 0],
                           [1, 2, 0],
                           [2, 0, 0],
                           [0, 0, 0]], dtype='i1')
        actual = GenotypeArray(diploid_genotype_data).to_n_alt()
        aeq(expect, actual)

        # polyploid
        expect = np.array([[0, 1, 0],
                           [2, 3, 0],
                           [2, 0, 0],
                           [0, 0, 0]], dtype='i1')
        actual = GenotypeArray(triploid_genotype_data).to_n_alt()
        aeq(expect, actual)

        # diploid with fill
        expect = np.array([[0, 1, -1],
                           [1, 2, -1],
                           [1, 2, -1],
                           [2, -1, -1],
                           [-1, -1, -1]], dtype='i1')
        actual = GenotypeArray(diploid_genotype_data).to_n_alt(fill=-1)
        aeq(expect, actual)

        # polyploid with fill
        expect = np.array([[0, 1, -1],
                           [2, 3, -1],
                           [2, -1, -1],
                           [-1, -1, -1]], dtype='i1')
        actual = GenotypeArray(triploid_genotype_data).to_n_alt(fill=-1)
        aeq(expect, actual)

    def test_to_allele_counts(self):

        # diploid
        expect = np.array([[[2, 0, 0], [1, 1, 0], [0, 0, 0]],
                           [[1, 0, 1], [0, 2, 0], [0, 0, 0]],
                           [[1, 1, 0], [0, 1, 1], [0, 0, 0]],
                           [[0, 0, 2], [0, 0, 0], [0, 0, 0]],
                           [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype='i1')
        actual = GenotypeArray(diploid_genotype_data).to_allele_counts()
        aeq(expect, actual)

        # polyploid
        expect = np.array([[[3, 0, 0], [2, 1, 0], [0, 0, 0]],
                           [[1, 2, 0], [0, 3, 0], [0, 0, 0]],
                           [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
                           [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype='i1')
        actual = GenotypeArray(triploid_genotype_data).to_allele_counts()
        aeq(expect, actual)

    def test_to_packed(self):

        expect = np.array([[0, 1, 239],
                           [2, 17, 239],
                           [16, 33, 239],
                           [34, 239, 239],
                           [239, 239, 239]], dtype='u1')
        actual = GenotypeArray(diploid_genotype_data).to_packed()
        aeq(expect, actual)

    def test_from_packed(self):
        packed_data = np.array([[0, 1, 239],
                                [2, 17, 239],
                                [16, 33, 239],
                                [34, 239, 239],
                                [239, 239, 239]], dtype='u1')
        expect = diploid_genotype_data
        actual = GenotypeArray.from_packed(packed_data)
        aeq(expect, actual)

    def test_max(self):
        eq = self.assertEqual

        # overall max
        expect = 2
        actual = GenotypeArray(diploid_genotype_data).max()
        eq(expect, actual)

        # max by sample
        expect = np.array([2, 2, -1])
        actual = GenotypeArray(diploid_genotype_data).max(axis=(0, 2))
        aeq(expect, actual)

        # max by variant
        expect = np.array([1, 2, 2, 2, -1])
        actual = GenotypeArray(diploid_genotype_data).max(axis=(1, 2))
        aeq(expect, actual)

    def test_allelism(self):

        # diploid
        expect = np.array([2, 3, 3, 1, 0])
        actual = GenotypeArray(diploid_genotype_data).allelism()
        aeq(expect, actual)

        # polyploid
        expect = np.array([2, 2, 3, 0])
        actual = GenotypeArray(triploid_genotype_data).allelism()
        aeq(expect, actual)

    def test_allele_number(self):

        # diploid
        expect = np.array([4, 4, 4, 2, 0])
        actual = GenotypeArray(diploid_genotype_data).allele_number()
        aeq(expect, actual)

        # polyploid
        expect = np.array([6, 6, 3, 0])
        actual = GenotypeArray(triploid_genotype_data).allele_number()
        aeq(expect, actual)

    def test_allele_count(self):

        # diploid
        g = GenotypeArray(diploid_genotype_data)
        expect = np.array([1, 2, 2, 0, 0])
        actual = g.allele_count(allele=1)
        aeq(expect, actual)
        expect = np.array([0, 1, 1, 2, 0])
        actual = g.allele_count(allele=2)
        aeq(expect, actual)

        # polyploid
        g = GenotypeArray(triploid_genotype_data)
        expect = np.array([1, 5, 1, 0])
        actual = g.allele_count(allele=1)
        aeq(expect, actual)
        expect = np.array([0, 0, 1, 0])
        actual = g.allele_count(allele=2)
        aeq(expect, actual)

    def test_allele_frequency(self):

        # diploid
        g = GenotypeArray(diploid_genotype_data)
        expect = np.array([1/4, 2/4, 2/4, 0/2, 0])
        actual, _, _ = g.allele_frequency(allele=1)
        aeq(expect, actual)
        expect = np.array([0/4, 1/4, 1/4, 2/2, 0])
        actual, _, _ = g.allele_frequency(allele=2)
        aeq(expect, actual)

        # polyploid
        g = GenotypeArray(triploid_genotype_data)
        expect = np.array([1/6, 5/6, 1/3, 0])
        actual, _, _ = g.allele_frequency(allele=1)
        aeq(expect, actual)
        expect = np.array([0/6, 0/6, 1/3, 0])
        actual, _, _ = g.allele_frequency(allele=2)
        aeq(expect, actual)

    def test_allele_counts(self):

        # diploid
        g = GenotypeArray(diploid_genotype_data)
        expect = np.array([[3, 1, 0],
                           [1, 2, 1],
                           [1, 2, 1],
                           [0, 0, 2],
                           [0, 0, 0]])
        actual = g.allele_counts()
        aeq(expect, actual)

        # polyploid
        g = GenotypeArray(triploid_genotype_data)
        expect = np.array([[5, 1, 0],
                           [1, 5, 0],
                           [1, 1, 1],
                           [0, 0, 0]])
        actual = g.allele_counts()
        aeq(expect, actual)

    def test_allele_frequencies(self):

        # diploid
        g = GenotypeArray(diploid_genotype_data)
        expect = np.array([[3/4, 1/4, 0/4],
                           [1/4, 2/4, 1/4],
                           [1/4, 2/4, 1/4],
                           [0/2, 0/2, 2/2],
                           [0, 0, 0]])
        actual, _, _ = g.allele_frequencies()
        aeq(expect, actual)

        # polyploid
        g = GenotypeArray(triploid_genotype_data)
        expect = np.array([[5/6, 1/6, 0/6],
                           [1/6, 5/6, 0/6],
                           [1/3, 1/3, 1/3],
                           [0, 0, 0]])
        actual, _, _ = g.allele_frequencies()
        aeq(expect, actual)

    def test_is_count_variant(self):
        eq = self.assertEqual
        
        # diploid
        g = GenotypeArray(diploid_genotype_data)
        expect = np.array([1, 1, 1, 1, 0], dtype='b1')
        actual = g.is_variant()
        aeq(expect, actual)
        eq(np.sum(expect), g.count_variant())
    
        # polyploid
        g = GenotypeArray(triploid_genotype_data)
        expect = np.array([1, 1, 1, 0], dtype='b1')
        actual = g.is_variant()
        aeq(expect, actual)
        eq(np.sum(expect), g.count_variant())
    
    def test_is_count_non_variant(self):
        eq = self.assertEqual
    
        # diploid
        g = GenotypeArray(diploid_genotype_data)
        expect = np.array([0, 0, 0, 0, 1], dtype='b1')
        actual = g.is_non_variant()
        aeq(expect, actual)
        eq(np.sum(expect), g.count_non_variant())
    
        # polyploid
        g = GenotypeArray(triploid_genotype_data)
        expect = np.array([0, 0, 0, 1], dtype='b1')
        actual = g.is_non_variant()
        aeq(expect, actual)
        eq(np.sum(expect), g.count_non_variant())
        
    def test_is_count_segregating(self):
        eq = self.assertEqual
        
        # diploid
        g = GenotypeArray(diploid_genotype_data)
        expect = np.array([1, 1, 1, 0, 0], dtype='b1')
        actual = g.is_segregating()
        aeq(expect, actual)
        eq(np.sum(expect), g.count_segregating())
    
        # polyploid
        g = GenotypeArray(triploid_genotype_data)
        expect = np.array([1, 1, 1, 0], dtype='b1')
        actual = g.is_segregating()
        aeq(expect, actual)
        eq(np.sum(expect), g.count_segregating())
    
    def test_is_count_non_segregating(self):
        eq = self.assertEqual
        
        # diploid
        g = GenotypeArray(diploid_genotype_data)
        expect = np.array([0, 0, 0, 1, 1], dtype='b1')
        actual = g.is_non_segregating()
        aeq(expect, actual)
        eq(np.sum(expect), g.count_non_segregating())
        expect = np.array([0, 0, 0, 1, 1], dtype='b1')
        actual = g.is_non_segregating(allele=2)
        aeq(expect, actual)
        eq(np.sum(expect), g.count_non_segregating(allele=2))
    
        # polyploid
        g = GenotypeArray(triploid_genotype_data)
        expect = np.array([0, 0, 0, 1], dtype='b1')
        actual = g.is_non_segregating()
        aeq(expect, actual)
        eq(np.sum(expect), g.count_non_segregating())
        expect = np.array([0, 0, 0, 1], dtype='b1')
        actual = g.is_non_segregating(allele=2)
        aeq(expect, actual)
        eq(np.sum(expect), g.count_non_segregating(allele=2))
        
    def test_is_count_singleton(self):
        eq = self.assertEqual
    
        # diploid
        g = GenotypeArray(diploid_genotype_data)
        expect = np.array([1, 0, 0, 0, 0], dtype='b1')
        actual = g.is_singleton(allele=1)
        aeq(expect, actual)
        eq(np.sum(expect), g.count_singleton(allele=1))
        expect = np.array([0, 1, 1, 0, 0], dtype='b1')
        actual = g.is_singleton(allele=2)
        aeq(expect, actual)
        eq(np.sum(expect), g.count_singleton(allele=2))
    
        # polyploid
        g = GenotypeArray(triploid_genotype_data)
        expect = np.array([1, 0, 1, 0], dtype='b1')
        actual = g.is_singleton(allele=1)
        aeq(expect, actual)
        eq(np.sum(expect), g.count_singleton(allele=1))
        expect = np.array([0, 0, 1, 0], dtype='b1')
        actual = g.is_singleton(allele=2)
        aeq(expect, actual)
        eq(np.sum(expect), g.count_singleton(allele=2))
    
    def test_is_count_doubleton(self):
        eq = self.assertEqual
    
        # diploid
        g = GenotypeArray(diploid_genotype_data)
        expect = np.array([0, 1, 1, 0, 0], dtype='b1')
        actual = g.is_doubleton(allele=1)
        aeq(expect, actual)
        eq(np.sum(expect), g.count_doubleton(allele=1))
        expect = np.array([0, 0, 0, 1, 0], dtype='b1')
        actual = g.is_doubleton(allele=2)
        aeq(expect, actual)
        eq(np.sum(expect), g.count_doubleton(allele=2))
    
        # polyploid
        g = GenotypeArray(triploid_genotype_data)
        expect = np.array([0, 0, 0, 0], dtype='b1')
        actual = g.is_doubleton(allele=1)
        aeq(expect, actual)
        eq(np.sum(expect), g.count_doubleton(allele=1))
        expect = np.array([0, 0, 0, 0], dtype='b1')
        actual = g.is_doubleton(allele=2)
        aeq(expect, actual)
        eq(np.sum(expect), g.count_doubleton(allele=2))


class TestHaplotypeArray(unittest.TestCase):

    def test_constructor(self):
        eq = self.assertEqual

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

        # haploid data
        h = HaplotypeArray(haplotype_data)
        aeq(haplotype_data, h)
        eq(np.int, h.dtype)
        eq(2, h.ndim)
        eq(4, h.n_variants)
        eq(3, h.n_haplotypes)

        # haploid data (typed)
        h = HaplotypeArray(np.array(haplotype_data, dtype='i1'))
        aeq(haplotype_data, h)
        eq(np.int8, h.dtype)

    def test_slice(self):
        eq = self.assertEqual

        h = HaplotypeArray(np.array(haplotype_data, dtype='i1'))

        # row slice
        s = h[1:]
        self.assertIsInstance(s, HaplotypeArray)
        aeq(haplotype_data[1:], s)
        eq(3, s.n_variants)
        eq(3, s.n_haplotypes)

        # col slice
        s = h[:, 1:]
        self.assertIsInstance(s, HaplotypeArray)
        aeq(np.array(haplotype_data)[:, 1:], s)
        eq(4, s.n_variants)
        eq(2, s.n_haplotypes)

        # row index
        s = h[0]
        self.assertIsInstance(s, np.ndarray)
        self.assertNotIsInstance(s, HaplotypeArray)
        aeq(haplotype_data[0], s)

        # col index
        s = h[:, 0]
        self.assertIsInstance(s, np.ndarray)
        self.assertNotIsInstance(s, HaplotypeArray)
        aeq(np.array(haplotype_data)[:, 0], s)

        # item
        s = h[0, 0]
        self.assertIsInstance(s, np.int8)
        self.assertNotIsInstance(s, HaplotypeArray)
        eq(0, s)

    def test_view(self):
        eq = self.assertEqual

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

    def test_is_called(self):
        expect = np.array([[1, 1, 0],
                           [1, 1, 0],
                           [1, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = HaplotypeArray(haplotype_data) >= 0
        aeq(expect, actual)

    def test_is_missing(self):
        expect = np.array([[0, 0, 1],
                           [0, 0, 1],
                           [0, 1, 1],
                           [1, 1, 1]], dtype='b1')
        actual = HaplotypeArray(haplotype_data) < 0
        aeq(expect, actual)

    def test_is_ref(self):
        expect = np.array([[1, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = HaplotypeArray(haplotype_data) == 0
        aeq(expect, actual)

    def test_is_alt(self):
        expect = np.array([[0, 1, 0],
                           [1, 1, 0],
                           [1, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = HaplotypeArray(haplotype_data) > 0
        aeq(expect, actual)

    def test_is_call(self):
        expect = np.array([[0, 0, 0],
                           [0, 0, 0],
                           [1, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = HaplotypeArray(haplotype_data) == 2
        aeq(expect, actual)

    # TODO test allele frequency calculations
    # TODO test is_variant, count_variant etc.


class TestPosArray(unittest.TestCase):
    
    def test_constructor(self):
        eq = self.assertEqual

        # missing data arg
        with self.assertRaises(TypeError):
            # noinspection PyArgumentList
            PosArray()

        # data has wrong dtype
        data = 'foo bar'
        with self.assertRaises(TypeError):
            PosArray(data)

        # data has wrong dtype
        data = [4., 5., 3.7]
        with self.assertRaises(TypeError):
            PosArray(data)

        # data has wrong dimensions
        data = [[1, 2], [3, 4]]
        with self.assertRaises(TypeError):
            PosArray(data)

        # positions are not sorted
        data = [2, 1, 3, 5]
        with self.assertRaises(ValueError):
            PosArray(data)
        
        # valid data
        data = [1, 4, 5, 5, 7, 12]
        pos = PosArray(data)
        aeq(data, pos)
        eq(np.int, pos.dtype)
        eq(1, pos.ndim)
        eq(6, pos.n_variants)

        # valid data (typed)
        data = np.array([1, 4, 5, 5, 7, 12], dtype='u4')
        pos = PosArray(data)
        aeq(data, pos)
        eq(np.uint32, pos.dtype)

    def test_slice(self):
        eq = self.assertEqual

        data = np.array([1, 4, 5, 5, 7, 12], dtype=np.int32)
        pos = PosArray(data)

        # row slice
        s = pos[1:]
        self.assertIsInstance(s, PosArray)
        aeq(data[1:], s)
        eq(5, s.n_variants)

        # index
        s = pos[0]
        self.assertIsInstance(s, np.int32)
        self.assertNotIsInstance(s, PosArray)
        eq(data[0], s)

    def test_view(self):
        eq = self.assertEqual

        # data has wrong dtype
        data = 'foo bar'
        with self.assertRaises(TypeError):
            np.asarray(data).view(PosArray)

        # data has wrong dtype
        data = [4., 5., 3.7]
        with self.assertRaises(TypeError):
            np.asarray(data).view(PosArray)

        # data has wrong dimensions
        data = [[1, 2], [3, 4]]
        with self.assertRaises(TypeError):
            np.asarray(data).view(PosArray)

        # positions are not sorted
        data = [2, 1, 3, 5]
        with self.assertRaises(ValueError):
            np.asarray(data).view(PosArray)
        
        # valid data
        data = [1, 4, 5, 5, 7, 12]
        pos = np.asarray(data).view(PosArray)
        aeq(data, pos)
        eq(np.int, pos.dtype)
        eq(1, pos.ndim)
        eq(6, pos.n_variants)

        # valid data (typed)
        data = np.array([1, 4, 5, 5, 7, 12], dtype='u4')
        pos = np.asarray(data).view(PosArray)
        aeq(data, pos)
        eq(np.uint32, pos.dtype)

    def test_locate_position(self):
        eq = self.assertEqual
        pos = PosArray([3, 6, 11])
        f = pos.locate_position
        eq(0, f(3))
        eq(1, f(6))
        eq(2, f(11))
        self.assertIsNone(f(1))
        self.assertIsNone(f(7))
        self.assertIsNone(f(12))

    def test_locate_positions(self):
        pos1 = PosArray([3, 6, 11, 20, 35])
        pos2 = PosArray([4, 6, 20, 39])
        expect_cond1 = np.array([False, True, False, True, False])
        expect_cond2 = np.array([False, True, True, False])
        cond1, cond2 = pos1.locate_positions(pos2)
        aeq(expect_cond1, cond1)
        aeq(expect_cond2, cond2)

    def test_intersect(self):
        pos1 = PosArray([3, 6, 11, 20, 35])
        pos2 = PosArray([4, 6, 20, 39])
        expect = PosArray([6, 20])
        actual = pos1.intersect(pos2)
        aeq(expect, actual)
        
    def test_locate_interval(self):
        eq = self.assertEqual
        pos = PosArray([3, 6, 11, 20, 35])
        eq(slice(0, 5), pos.locate_interval(2, 37))
        eq(slice(1, 5), pos.locate_interval(4, 37))
        eq(slice(0, 4), pos.locate_interval(2, 32))
        eq(slice(1, 4), pos.locate_interval(4, 32))
        eq(slice(1, 3), pos.locate_interval(4, 19))
        eq(slice(2, 4), pos.locate_interval(7, 32))
        eq(slice(2, 3), pos.locate_interval(7, 19))
        eq(slice(3, 3), pos.locate_interval(17, 19))
        eq(slice(0, 0), pos.locate_interval(0, 0))
        eq(slice(5, 5), pos.locate_interval(1000, 2000))

    def test_locate_intervals(self):
        pos = PosArray([3, 6, 11, 20, 35])
        intervals = np.array([[0, 2], [6, 17], [12, 15], [31, 35], [100, 120]])
        expect_cond1 = np.array([False, True, True, False, True])
        expect_cond2 = np.array([False, True, False, True, False])
        cond1, cond2 = pos.locate_intervals(intervals[:, 0], intervals[:, 1])
        aeq(expect_cond1, cond1)
        aeq(expect_cond2, cond2)
        aeq([6, 11, 35], pos[cond1])
        aeq([[6, 17], [31, 35]], intervals[cond2])

    # TODO test windowed counts
