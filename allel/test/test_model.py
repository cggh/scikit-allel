# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import unittest
import numpy as np
from allel.test.tools import assert_array_equal as aeq, assert_array_close


from allel.model import GenotypeArray, HaplotypeArray, PositionIndex, \
    LabelIndex


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
        g = GenotypeArray(diploid_genotype_data, dtype='i1')
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
        g = GenotypeArray(triploid_genotype_data, dtype='i1')
        aeq(triploid_genotype_data, g)
        eq(np.int8, g.dtype)

    def test_slice(self):
        eq = self.assertEqual

        g = GenotypeArray(diploid_genotype_data, dtype='i1')
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
        actual = GenotypeArray(diploid_genotype_data).view_haplotypes()
        aeq(expect, actual)

        # polyploid
        expect = np.array([[0, 0, 0, 0, 0, 1, -1, -1, -1],
                           [0, 1, 1, 1, 1, 1, -1, -1, -1],
                           [0, 1, 2, -1, -1, -1, -1, -1, -1],
                           [-1, -1, -1, -1, -1, -1, -1, -1, -1]], dtype='i1')
        actual = GenotypeArray(triploid_genotype_data).view_haplotypes()
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
        actual = HaplotypeArray(h_diploid).view_genotypes(ploidy=2)
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
        actual = HaplotypeArray(h_triploid).view_genotypes(ploidy=3)
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
        expect = np.array([1/4, 2/4, 2/4, 0/2, -1])
        actual, _, _ = g.allele_frequency(allele=1, fill=-1)
        aeq(expect, actual)
        expect = np.array([0/4, 1/4, 1/4, 2/2, -1])
        actual, _, _ = g.allele_frequency(allele=2, fill=-1)
        aeq(expect, actual)

        # polyploid
        g = GenotypeArray(triploid_genotype_data)
        expect = np.array([1/6, 5/6, 1/3, -1])
        actual, _, _ = g.allele_frequency(allele=1, fill=-1)
        aeq(expect, actual)
        expect = np.array([0/6, 0/6, 1/3, -1])
        actual, _, _ = g.allele_frequency(allele=2, fill=-1)
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
                           [-1, -1, -1]])
        actual, _, _ = g.allele_frequencies(fill=-1)
        aeq(expect, actual)

        # polyploid
        g = GenotypeArray(triploid_genotype_data)
        expect = np.array([[5/6, 1/6, 0/6],
                           [1/6, 5/6, 0/6],
                           [1/3, 1/3, 1/3],
                           [-1, -1, -1]])
        actual, _, _ = g.allele_frequencies(fill=-1)
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

    def test_haploidify_samples(self):
        eq = self.assertEqual

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

    def test_heterozygosity_observed(self):

        # diploid
        g = GenotypeArray([[[0, 0], [0, 0]],
                           [[1, 1], [1, 1]],
                           [[1, 1], [2, 2]],
                           [[0, 0], [0, 1]],
                           [[0, 0], [0, 2]],
                           [[1, 1], [1, 2]],
                           [[0, 1], [0, 1]],
                           [[0, 1], [1, 2]],
                           [[0, 0], [-1, -1]],
                           [[0, 1], [-1, -1]],
                           [[-1, -1], [-1, -1]]], dtype='i1')
        expect = [0, 0, 0, .5, .5, .5, 1, 1, 0, 1, -1]
        actual = g.heterozygosity_observed(fill=-1)
        aeq(expect, actual)

        # polyploid
        g = GenotypeArray([[[0, 0, 0], [0, 0, 0]],
                           [[1, 1, 1], [1, 1, 1]],
                           [[1, 1, 1], [2, 2, 2]],
                           [[0, 0, 0], [0, 0, 1]],
                           [[0, 0, 0], [0, 0, 2]],
                           [[1, 1, 1], [0, 1, 2]],
                           [[0, 0, 1], [0, 1, 1]],
                           [[0, 1, 1], [0, 1, 2]],
                           [[0, 0, 0], [-1, -1, -1]],
                           [[0, 0, 1], [-1, -1, -1]],
                           [[-1, -1, -1], [-1, -1, -1]]], dtype='i1')
        expect = [0, 0, 0, .5, .5, .5, 1, 1, 0, 1, -1]
        actual = g.heterozygosity_observed(fill=-1)
        aeq(expect, actual)

    def test_heterozygosity_expected(self):

        def refimpl(g, fill=0):
            """Limited reference implementation for testing purposes."""

            ploidy = g.ploidy

            # calculate allele frequencies
            af, _, an = g.allele_frequencies()

            # assume three alleles
            p = af[:, 0]
            q = af[:, 1]
            r = af[:, 2]

            out = 1 - p**ploidy - q**ploidy - r**ploidy
            out[an == 0] = fill

            return out

        # diploid
        g = GenotypeArray([[[0, 0], [0, 0]],
                           [[1, 1], [1, 1]],
                           [[1, 1], [2, 2]],
                           [[0, 0], [0, 1]],
                           [[0, 0], [0, 2]],
                           [[1, 1], [1, 2]],
                           [[0, 1], [0, 1]],
                           [[0, 1], [1, 2]],
                           [[0, 0], [-1, -1]],
                           [[0, 1], [-1, -1]],
                           [[-1, -1], [-1, -1]]], dtype='i1')
        expect1 = [0, 0, 0.5, .375, .375, .375, .5, .625, 0, .5, -1]
        expect2 = refimpl(g, fill=-1)
        actual = g.heterozygosity_expected(fill=-1)
        assert_array_close(expect1, actual)
        assert_array_close(expect2, actual)
        expect3 = [0, 0, 0.5, .375, .375, .375, .5, .625, 0, .5, 0]
        actual = g.heterozygosity_expected(fill=0)
        assert_array_close(expect3, actual)

        # polyploid
        g = GenotypeArray([[[0, 0, 0], [0, 0, 0]],
                           [[1, 1, 1], [1, 1, 1]],
                           [[1, 1, 1], [2, 2, 2]],
                           [[0, 0, 0], [0, 0, 1]],
                           [[0, 0, 0], [0, 0, 2]],
                           [[1, 1, 1], [0, 1, 2]],
                           [[0, 0, 1], [0, 1, 1]],
                           [[0, 1, 1], [0, 1, 2]],
                           [[0, 0, 0], [-1, -1, -1]],
                           [[0, 0, 1], [-1, -1, -1]],
                           [[-1, -1, -1], [-1, -1, -1]]], dtype='i1')
        expect = refimpl(g, fill=-1)
        actual = g.heterozygosity_expected(fill=-1)
        assert_array_close(expect, actual)

    def test_inbreeding_coefficient(self):

        # diploid
        g = GenotypeArray([[[0, 0], [0, 0]],
                           [[1, 1], [1, 1]],
                           [[1, 1], [2, 2]],
                           [[0, 0], [0, 1]],
                           [[0, 0], [0, 2]],
                           [[1, 1], [1, 2]],
                           [[0, 1], [0, 1]],
                           [[0, 1], [1, 2]],
                           [[0, 0], [-1, -1]],
                           [[0, 1], [-1, -1]],
                           [[-1, -1], [-1, -1]]], dtype='i1')
        # ho = np.array([0, 0, 0, .5, .5, .5, 1, 1, 0, 1, -1])
        # he = np.array([0, 0, 0.5, .375, .375, .375, .5, .625, 0, .5, -1])
        # expect = 1 - (ho/he)
        # expect[he == 0] = -1
        expect = [-1, -1, 1-0, 1-(.5/.375), 1-(.5/.375), 1-(.5/.375),
                  1-(1/.5), 1-(1/.625), -1, 1-(1/.5), -1]
        actual = g.inbreeding_coefficient(fill=-1)
        assert_array_close(expect, actual)

    def test_mean_pairwise_difference(self):

        # four haplotypes, 6 pairwise comparison
        g = GenotypeArray([[[0, 0], [0, 0]],
                           [[0, 0], [0, 1]],
                           [[0, 0], [1, 1]],
                           [[0, 1], [1, 1]],
                           [[1, 1], [1, 1]],
                           [[0, 0], [1, 2]],
                           [[0, 1], [1, 2]],
                           [[0, 1], [-1, -1]],
                           [[-1, -1], [-1, -1]]])
        expect = [0, 3/6, 4/6, 3/6, 0, 5/6, 5/6, 1, -1]
        actual = g.mean_pairwise_difference(fill=-1)
        assert_array_close(expect, actual)


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
        h = HaplotypeArray(haplotype_data, dtype='i1')
        aeq(haplotype_data, h)
        eq(np.int8, h.dtype)

    def test_slice(self):
        eq = self.assertEqual

        h = HaplotypeArray(haplotype_data, dtype='i1')

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

    def test_allelism(self):
        expect = np.array([2, 1, 1, 0])
        actual = HaplotypeArray(haplotype_data).allelism()
        aeq(expect, actual)

    def test_allele_number(self):
        expect = np.array([2, 2, 1, 0])
        actual = HaplotypeArray(haplotype_data).allele_number()
        aeq(expect, actual)

    def test_allele_count(self):
        expect = np.array([1, 2, 0, 0])
        actual = HaplotypeArray(haplotype_data).allele_count(allele=1)
        aeq(expect, actual)
        expect = np.array([0, 0, 1, 0])
        actual = HaplotypeArray(haplotype_data).allele_count(allele=2)
        aeq(expect, actual)

    def test_allele_frequency(self):
        expect = np.array([1/2, 2/2, 0/1, -1])
        h = HaplotypeArray(haplotype_data)
        actual, _, _ = h.allele_frequency(allele=1, fill=-1)
        aeq(expect, actual)
        expect = np.array([0/2, 0/2, 1/1, -1])
        actual, _, _ = h.allele_frequency(allele=2, fill=-1)
        aeq(expect, actual)

    def test_allele_counts(self):
        expect = np.array([[1, 1, 0],
                           [0, 2, 0],
                           [0, 0, 1],
                           [0, 0, 0]])
        actual = HaplotypeArray(haplotype_data).allele_counts()
        aeq(expect, actual)

    def test_allele_frequencies(self):
        expect = np.array([[1/2, 1/2, 0/2],
                           [0/2, 2/2, 0/2],
                           [0/1, 0/1, 1/1],
                           [-1, -1, -1]])
        actual, _, _ = \
            HaplotypeArray(haplotype_data).allele_frequencies(fill=-1)
        aeq(expect, actual)

    def test_is_count_variant(self):
        expect = np.array([1, 1, 1, 0], dtype='b1')
        h = HaplotypeArray(haplotype_data)
        actual = h.is_variant()
        aeq(expect, actual)
        self.assertEqual(np.sum(expect), h.count_variant())

    def test_is_count_non_variant(self):
        expect = np.array([0, 0, 0, 1], dtype='b1')
        h = HaplotypeArray(haplotype_data)
        actual = h.is_non_variant()
        aeq(expect, actual)
        self.assertEqual(np.sum(expect), h.count_non_variant())

    def test_is_count_segregating(self):
        expect = np.array([1, 0, 0, 0], dtype='b1')
        h = HaplotypeArray(haplotype_data)
        actual = h.is_segregating()
        aeq(expect, actual)
        self.assertEqual(np.sum(expect), h.count_segregating())

    def test_is_count_non_segregating(self):
        expect = np.array([0, 1, 1, 1], dtype='b1')
        h = HaplotypeArray(haplotype_data)
        actual = h.is_non_segregating()
        aeq(expect, actual)
        self.assertEqual(np.sum(expect), h.count_non_segregating())

        expect = np.array([0, 0, 1, 1], dtype='b1')
        h = HaplotypeArray(haplotype_data)
        actual = h.is_non_segregating(allele=2)
        aeq(expect, actual)
        self.assertEqual(np.sum(expect), h.count_non_segregating(allele=2))

    def test_is_count_singleton(self):
        expect = np.array([1, 0, 0, 0], dtype='b1')
        h = HaplotypeArray(haplotype_data)
        actual = h.is_singleton(allele=1)
        aeq(expect, actual)
        self.assertEqual(np.sum(expect), h.count_singleton(allele=1))

        expect = np.array([0, 0, 1, 0], dtype='b1')
        h = HaplotypeArray(haplotype_data)
        actual = h.is_singleton(allele=2)
        aeq(expect, actual)
        self.assertEqual(np.sum(expect), h.count_singleton(allele=2))

    def test_is_count_doubleton(self):
        expect = np.array([0, 1, 0, 0], dtype='b1')
        h = HaplotypeArray(haplotype_data)
        actual = h.is_doubleton(allele=1)
        aeq(expect, actual)
        self.assertEqual(np.sum(expect), h.count_doubleton(allele=1))

        expect = np.array([0, 0, 0, 0], dtype='b1')
        h = HaplotypeArray(haplotype_data)
        actual = h.is_doubleton(allele=2)
        aeq(expect, actual)
        self.assertEqual(np.sum(expect), h.count_doubleton(allele=2))

    def test_mean_pairwise_difference(self):

        # start with simplest case, two haplotypes, one pairwise comparison
        h = HaplotypeArray([[0, 0],
                            [1, 1],
                            [0, 1],
                            [1, 2],
                            [0, -1],
                            [-1, -1]])
        expect = [0, 0, 1, 1, -1, -1]
        actual = h.mean_pairwise_difference(fill=-1)
        aeq(expect, actual)

        # four haplotypes, 6 pairwise comparison
        h = HaplotypeArray([[0, 0, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 1, 1],
                            [0, 1, 1, 1],
                            [1, 1, 1, 1],
                            [0, 0, 1, 2],
                            [0, 1, 1, 2],
                            [0, 1, -1, -1],
                            [-1, -1, -1, -1]])
        expect = [0, 3/6, 4/6, 3/6, 0, 5/6, 5/6, 1, -1]
        actual = h.mean_pairwise_difference(fill=-1)
        assert_array_close(expect, actual)


class TestPositionIndex(unittest.TestCase):

    def test_constructor(self):
        eq = self.assertEqual

        # missing data arg
        with self.assertRaises(TypeError):
            # noinspection PyArgumentList
            PositionIndex()

        # data has wrong dtype
        data = 'foo bar'
        with self.assertRaises(TypeError):
            PositionIndex(data)

        # data has wrong dtype
        data = [4., 5., 3.7]
        with self.assertRaises(TypeError):
            PositionIndex(data)

        # data has wrong dimensions
        data = [[1, 2], [3, 4]]
        with self.assertRaises(TypeError):
            PositionIndex(data)

        # positions are not sorted
        data = [2, 1, 3, 5]
        with self.assertRaises(ValueError):
            PositionIndex(data)

        # valid data (unique)
        data = [1, 4, 5, 7, 12]
        pos = PositionIndex(data)
        aeq(data, pos)
        eq(np.int, pos.dtype)
        eq(1, pos.ndim)
        eq(5, len(pos))
        assert pos.is_unique

        # valid data (non-unique)
        data = [1, 4, 5, 5, 7, 12]
        pos = PositionIndex(data)
        aeq(data, pos)
        eq(np.int, pos.dtype)
        eq(1, pos.ndim)
        eq(6, len(pos))
        assert not pos.is_unique

        # valid data (typed)
        data = [1, 4, 5, 5, 7, 12]
        pos = PositionIndex(data, dtype='u4')
        aeq(data, pos)
        eq(np.uint32, pos.dtype)

    def test_slice(self):
        eq = self.assertEqual

        data = [1, 4, 5, 5, 7, 12]
        pos = PositionIndex(data, dtype='u4')

        # row slice
        s = pos[1:]
        self.assertIsInstance(s, PositionIndex)
        aeq(data[1:], s)
        eq(5, len(s))
        assert not s.is_unique

        # row slice
        s = pos[3:]
        self.assertIsInstance(s, PositionIndex)
        aeq(data[3:], s)
        eq(3, len(s))
        assert s.is_unique

        # index
        s = pos[0]
        self.assertIsInstance(s, np.uint32)
        self.assertNotIsInstance(s, PositionIndex)
        eq(data[0], s)

    def test_view(self):
        eq = self.assertEqual

        # data has wrong dtype
        data = 'foo bar'
        with self.assertRaises(TypeError):
            np.asarray(data).view(PositionIndex)

        # data has wrong dtype
        data = [4., 5., 3.7]
        with self.assertRaises(TypeError):
            np.asarray(data).view(PositionIndex)

        # data has wrong dimensions
        data = [[1, 2], [3, 4]]
        with self.assertRaises(TypeError):
            np.asarray(data).view(PositionIndex)

        # positions are not sorted
        data = [2, 1, 3, 5]
        with self.assertRaises(ValueError):
            np.asarray(data).view(PositionIndex)

        # valid data (unique)
        data = [1, 4, 5, 7, 12]
        pos = np.asarray(data).view(PositionIndex)
        aeq(data, pos)
        eq(np.int, pos.dtype)
        eq(1, pos.ndim)
        eq(5, len(pos))
        assert pos.is_unique

        # valid data (non-unique)
        data = [1, 4, 5, 5, 7, 12]
        pos = np.asarray(data).view(PositionIndex)
        aeq(data, pos)
        eq(np.int, pos.dtype)
        eq(1, pos.ndim)
        eq(6, len(pos))
        assert not pos.is_unique

        # valid data (typed)
        data = np.array([1, 4, 5, 5, 7, 12], dtype='u4')
        pos = np.asarray(data).view(PositionIndex)
        aeq(data, pos)
        eq(np.uint32, pos.dtype)

    def test_locate_key(self):
        eq = self.assertEqual
        pos = PositionIndex([3, 6, 6, 11])
        f = pos.locate_key
        eq(0, f(3))
        eq(3, f(11))
        eq(slice(1, 3), f(6))
        with self.assertRaises(KeyError):
            f(2)

    def test_locate_keys(self):
        pos = PositionIndex([3, 6, 6, 11, 20, 35])
        f = pos.locate_keys

        # all found
        expect = [False, True, True, False, True, False]
        actual = f([6, 20])
        self.assertNotIsInstance(actual, PositionIndex)
        aeq(expect, actual)

        # not all found, lax
        expect = [False, True, True, False, True, False]
        actual = f([2, 6, 17, 20, 37], strict=False)
        self.assertNotIsInstance(actual, PositionIndex)
        aeq(expect, actual)

        # not all found, strict
        with self.assertRaises(KeyError):
            f([2, 6, 17, 20, 37])

    def test_locate_intersection(self):
        pos1 = PositionIndex([3, 6, 11, 20, 35])
        pos2 = PositionIndex([4, 6, 20, 39])
        expect_loc1 = np.array([False, True, False, True, False])
        expect_loc2 = np.array([False, True, True, False])
        loc1, loc2 = pos1.locate_intersection(pos2)
        self.assertNotIsInstance(loc1, PositionIndex)
        self.assertNotIsInstance(loc2, PositionIndex)
        aeq(expect_loc1, loc1)
        aeq(expect_loc2, loc2)

    def test_intersect(self):
        pos1 = PositionIndex([3, 6, 11, 20, 35])
        pos2 = PositionIndex([4, 6, 20, 39])
        expect = PositionIndex([6, 20])
        actual = pos1.intersect(pos2)
        self.assertIsInstance(actual, PositionIndex)
        aeq(expect, actual)

    def test_locate_range(self):
        eq = self.assertEqual
        pos = PositionIndex([3, 6, 11, 20, 35])
        f = pos.locate_range
        eq(slice(0, 5), f(2, 37))
        eq(slice(0, 5), f(3, 35))
        eq(slice(1, 5), f(4, 37))
        eq(slice(1, 5), f(start=4))
        eq(slice(0, 4), f(2, 32))
        eq(slice(0, 4), f(stop=32))
        eq(slice(1, 4), f(4, 32))
        eq(slice(1, 3), f(4, 19))
        eq(slice(2, 4), f(7, 32))
        eq(slice(2, 3), f(7, 19))
        with self.assertRaises(KeyError):
            f(17, 19)
        with self.assertRaises(KeyError):
            print(f(0, 2))
        with self.assertRaises(KeyError):
            f(36, 2000)

    def test_intersect_range(self):
        pos = PositionIndex([3, 6, 11, 20, 35])
        f = pos.intersect_range
        aeq(pos[:], f(2, 37))
        aeq(pos[:], f(3, 35))
        aeq(pos[1:], f(4, 37))
        aeq(pos[1:], f(start=4))
        aeq(pos[:4], f(2, 32))
        aeq(pos[:4], f(stop=32))
        aeq(pos[1:4], f(4, 32))
        aeq(pos[1:3], f(4, 19))
        aeq(pos[2:4], f(7, 32))
        aeq(pos[2:3], f(7, 19))
        aeq([], f(17, 19))
        aeq([], f(0, 2))
        aeq([], f(36, 2000))

    def test_locate_ranges(self):
        pos = PositionIndex([3, 6, 11, 20, 35])

        # all found
        ranges = np.array([[6, 12], [31, 35]])
        expect = np.array([False, True, True, False, True])
        actual = pos.locate_ranges(ranges[:, 0], ranges[:, 1])
        self.assertNotIsInstance(actual, PositionIndex)
        aeq(expect, actual)

        # not all found, lax
        ranges = np.array([[0, 2], [6, 12], [14, 19], [31, 35], [100, 120]])
        actual = pos.locate_ranges(ranges[:, 0], ranges[:, 1], strict=False)
        self.assertNotIsInstance(actual, PositionIndex)
        aeq(expect, actual)

        # not all found, strict
        with self.assertRaises(KeyError):
            pos.locate_ranges(ranges[:, 0], ranges[:, 1])

    def test_locate_intersection_ranges(self):
        pos = PositionIndex([3, 6, 11, 20, 35])
        f = pos.locate_intersection_ranges

        # all found
        ranges = np.array([[6, 12], [31, 35]])
        expect_loc1 = np.array([False, True, True, False, True])
        expect_loc2 = np.array([True, True])
        actual_loc1, actual_loc2 = f(ranges[:, 0], ranges[:, 1])
        self.assertNotIsInstance(actual_loc1, PositionIndex)
        self.assertNotIsInstance(actual_loc2, PositionIndex)
        aeq(expect_loc1, actual_loc1)
        aeq(expect_loc2, actual_loc2)

        # not all found
        ranges = np.array([[0, 2], [6, 12], [14, 19], [31, 35], [100, 120]])
        expect_loc1 = np.array([False, True, True, False, True])
        expect_loc2 = np.array([False, True, False, True, False])
        actual_loc1, actual_loc2 = f(ranges[:, 0], ranges[:, 1])
        self.assertNotIsInstance(actual_loc1, PositionIndex)
        self.assertNotIsInstance(actual_loc2, PositionIndex)
        aeq(expect_loc1, actual_loc1)
        aeq(expect_loc2, actual_loc2)

    def test_intersect_ranges(self):
        pos = PositionIndex([3, 6, 11, 20, 35])
        f = pos.intersect_ranges

        # all found
        ranges = np.array([[6, 12], [31, 35]])
        expect = [6, 11, 35]
        actual = f(ranges[:, 0], ranges[:, 1])
        self.assertIsInstance(actual, PositionIndex)
        aeq(expect, actual)

        # not all found
        ranges = np.array([[0, 2], [6, 12], [14, 19], [31, 35], [100, 120]])
        expect = [6, 11, 35]
        actual = f(ranges[:, 0], ranges[:, 1])
        self.assertIsInstance(actual, PositionIndex)
        aeq(expect, actual)


class TestLabelIndex(unittest.TestCase):

    def test_constructor(self):
        eq = self.assertEqual

        # missing data arg
        with self.assertRaises(TypeError):
            # noinspection PyArgumentList
            LabelIndex()

        # data has wrong dimensions
        data = [['A', 'C'], ['B', 'F']]
        with self.assertRaises(TypeError):
            LabelIndex(data)

        # labels are not unique
        data = ['A', 'B', 'D', 'B']
        with self.assertRaises(ValueError):
            LabelIndex(data)

        # valid data
        data = ['A', 'C', 'B', 'F']
        lbl = LabelIndex(data)
        aeq(data, lbl)
        eq(1, lbl.ndim)
        eq(4, len(lbl))

        # valid data (typed)
        data = np.array(['A', 'C', 'B', 'F'], dtype='S1')
        lbl = LabelIndex(data, dtype='S1')
        aeq(data, lbl)

    def test_slice(self):
        eq = self.assertEqual

        data = ['A', 'C', 'B', 'F']
        lbl = LabelIndex(data)

        # row slice
        s = lbl[1:]
        self.assertIsInstance(s, LabelIndex)
        aeq(data[1:], s)
        eq(3, len(s))

        # index
        s = lbl[0]
        self.assertIsInstance(s, str)
        self.assertNotIsInstance(s, LabelIndex)
        eq(data[0], s)

    def test_view(self):
        eq = self.assertEqual

        # data has wrong dimensions
        data = [['A', 'C'], ['B', 'F']]
        with self.assertRaises(TypeError):
            np.array(data).view(LabelIndex)

        # labels are not unique
        data = ['A', 'B', 'D', 'B']
        with self.assertRaises(ValueError):
            np.array(data).view(LabelIndex)

        # valid data
        data = ['A', 'C', 'B', 'F']
        lbl = np.array(data).view(LabelIndex)
        aeq(data, lbl)
        eq(1, lbl.ndim)
        eq(4, len(lbl))

        # valid data (typed)
        data = np.array(['A', 'C', 'B', 'F'], dtype='a1')
        lbl = data.view(LabelIndex)
        aeq(data, lbl)

    def test_locate_key(self):
        eq = self.assertEqual
        lbl = LabelIndex(['A', 'C', 'B', 'F'])
        f = lbl.locate_key
        eq(0, f('A'))
        eq(2, f('B'))
        with self.assertRaises(KeyError):
            f('D')

    def test_locate_keys(self):
        lbl = LabelIndex(['A', 'C', 'B', 'F'])
        f = lbl.locate_keys

        # all found
        expect = [False, True, False, True]
        actual = f(['F', 'C'])
        self.assertNotIsInstance(actual, LabelIndex)
        aeq(expect, actual)

        # not all found, lax
        expect = [False, True, False, True]
        actual = f(['X', 'F', 'G', 'C', 'Z'], strict=False)
        self.assertNotIsInstance(actual, LabelIndex)
        aeq(expect, actual)

        # not all found, strict
        with self.assertRaises(KeyError):
            f(['X', 'F', 'G', 'C', 'Z'])

    def test_locate_intersection(self):
        lbl1 = LabelIndex(['A', 'C', 'B', 'F'])
        lbl2 = LabelIndex(['X', 'F', 'G', 'C', 'Z'])
        expect_loc1 = np.array([False, True, False, True])
        expect_loc2 = np.array([False, True, False, True, False])
        loc1, loc2 = lbl1.locate_intersection(lbl2)
        self.assertNotIsInstance(loc1, LabelIndex)
        self.assertNotIsInstance(loc2, LabelIndex)
        aeq(expect_loc1, loc1)
        aeq(expect_loc2, loc2)

    def test_intersect(self):
        lbl1 = LabelIndex(['A', 'C', 'B', 'F'])
        lbl2 = LabelIndex(['X', 'F', 'G', 'C', 'Z'])

        expect = LabelIndex(['C', 'F'])
        actual = lbl1.intersect(lbl2)
        self.assertIsInstance(actual, LabelIndex)
        aeq(expect, actual)

        expect = LabelIndex(['F', 'C'])
        actual = lbl2.intersect(lbl1)
        self.assertIsInstance(actual, LabelIndex)
        aeq(expect, actual)
