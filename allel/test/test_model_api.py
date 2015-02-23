# -*- coding: utf-8 -*-
"""This module defines interfaces for the classes in the allel.model module.
These interfaces are defined as test cases, but are abstracted so that the
tests can be re-used for alternative implementations of the same interfaces.

"""
from __future__ import absolute_import, print_function, division


import numpy as np
from nose.tools import eq_ as eq, assert_raises, \
    assert_is_instance, assert_not_is_instance
from allel.test.tools import assert_array_equal as aeq


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

allele_counts_data = [[3, 1, 0],
                      [1, 2, 1],
                      [1, 2, 1],
                      [0, 0, 2],
                      [0, 0, 0]]

variant_table_names = ('CHROM', 'POS', 'DP', 'QD')
variant_table_data = [[b'chr1', 2, 35, 4.5],
                      [b'chr1', 7, 12, 6.7],
                      [b'chr2', 3, 78, 1.2],
                      [b'chr2', 9, 22, 4.4],
                      [b'chr3', 6, 99, 2.8]]


class GenotypeArrayInterface(object):

    def setup_instance(self, data):
        # to be implemented in sub-classes
        pass

    # to be overriden in sub-classes
    _class = None

    # basic properties and data access methods
    ##########################################

    def test_properties(self):
        # Test the instance properties.

        # diploid data
        g = self.setup_instance(diploid_genotype_data)
        eq(3, g.ndim)
        eq((5, 3, 2), g.shape)
        eq(5, g.n_variants)
        eq(3, g.n_samples)
        eq(2, g.ploidy)

        # polyploid data
        g = self.setup_instance(triploid_genotype_data)
        eq(3, g.ndim)
        eq((4, 3, 3), g.shape)
        eq(4, g.n_variants)
        eq(3, g.n_samples)
        eq(3, g.ploidy)

    def test_array_like(self):
        # Test that an instance is array-like, in that it can be used as
        # input argument to np.array(). I.e., there is a standard way to get
        # a vanilla numpy array representation of the data.

        # diploid data
        g = self.setup_instance(diploid_genotype_data)
        a = np.array(g, copy=False)
        aeq(diploid_genotype_data, a)

        # polyploid data
        g = self.setup_instance(triploid_genotype_data)
        a = np.array(g, copy=False)
        aeq(triploid_genotype_data, a)

    def test_slice(self):
        # Test contiguous slicing and item indexing.

        g = self.setup_instance(diploid_genotype_data)

        # row slice
        s = g[1:]
        aeq(diploid_genotype_data[1:], s)
        # slice which preserves dimensionality should return GenotypeArray
        eq(4, s.n_variants)
        eq(3, s.n_samples)
        eq(2, s.ploidy)

        # col slice
        s = g[:, 1:]
        aeq(np.array(diploid_genotype_data)[:, 1:], s)
        # slice which preserves dimensionality should return GenotypeArray
        eq(5, s.n_variants)
        eq(2, s.n_samples)
        eq(2, s.ploidy)

        # row index
        s = g[0]
        aeq(diploid_genotype_data[0], s)
        assert not hasattr(s, 'n_variants')

        # col index
        s = g[:, 0]
        aeq(np.array(diploid_genotype_data)[:, 0], s)
        assert not hasattr(s, 'n_samples')

        # ploidy index
        s = g[:, :, 0]
        aeq(np.array(diploid_genotype_data)[:, :, 0], s)
        assert not hasattr(s, 'ploidy')

        # item
        s = g[0, 0, 0]
        eq(0, s)

    def test_take(self):
        # Test the take() method.

        g = self.setup_instance(diploid_genotype_data)

        # take variants
        indices = [0, 2]
        t = g.take(indices, axis=0)
        eq(2, t.n_variants)
        eq(g.n_samples, t.n_samples)
        eq(g.ploidy, t.ploidy)
        expect = np.array(diploid_genotype_data).take(indices, axis=0)
        aeq(expect, t)

        # take samples
        indices = [0, 2]
        t = g.take(indices, axis=1)
        eq(g.n_variants, t.n_variants)
        eq(2, t.n_samples)
        eq(g.ploidy, t.ploidy)
        expect = np.array(diploid_genotype_data).take(indices, axis=1)
        aeq(expect, t)

    def test_compress(self):
        # Test the compress() method.

        g = self.setup_instance(diploid_genotype_data)

        # compress variants
        condition = [True, False, True, False, False]
        t = g.compress(condition, axis=0)
        eq(2, t.n_variants)
        eq(g.n_samples, t.n_samples)
        eq(g.ploidy, t.ploidy)
        expect = np.array(diploid_genotype_data).compress(condition, axis=0)
        aeq(expect, t)

        # compress samples
        condition = [True, False, True]
        t = g.compress(condition, axis=1)
        eq(g.n_variants, t.n_variants)
        eq(2, t.n_samples)
        eq(g.ploidy, t.ploidy)
        expect = np.array(diploid_genotype_data).compress(condition, axis=1)
        aeq(expect, t)

    def test_subset(self):
        # Test the subset() method.

        g = self.setup_instance(diploid_genotype_data)

        # test with indices
        variants = [0, 2]
        samples = [0, 2]
        s = g.subset(variants=variants, samples=samples)
        expect = np.array(diploid_genotype_data)\
            .take(variants, axis=0)\
            .take(samples, axis=1)
        aeq(expect, s)

        # test with condition
        variants = [True, False, True, False, False]
        samples = [True, False, True]
        s = g.subset(variants=variants, samples=samples)
        expect = np.array(diploid_genotype_data)\
            .compress(variants, axis=0)\
            .compress(samples, axis=1)
        aeq(expect, s)

        # mix and match
        variants = [0, 2]
        samples = [True, False, True]
        s = g.subset(variants=variants, samples=samples)
        expect = np.array(diploid_genotype_data)\
            .take(variants, axis=0)\
            .compress(samples, axis=1)
        aeq(expect, s)

        # mix and match
        variants = [True, False, True, False, False]
        samples = [0, 2]
        s = g.subset(variants=variants, samples=samples)
        expect = np.array(diploid_genotype_data)\
            .compress(variants, axis=0)\
            .take(samples, axis=1)
        aeq(expect, s)

    # genotype counting methods
    ###########################

    def test_is_called(self):

        # diploid
        expect = np.array([[1, 1, 0],
                           [1, 1, 0],
                           [1, 1, 0],
                           [1, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = self.setup_instance(diploid_genotype_data).is_called()
        aeq(expect, actual)

        # polyploid
        expect = np.array([[1, 1, 0],
                           [1, 1, 0],
                           [1, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = self.setup_instance(triploid_genotype_data).is_called()
        aeq(expect, actual)

    def test_is_missing(self):

        # diploid
        expect = np.array([[0, 0, 1],
                           [0, 0, 1],
                           [0, 0, 1],
                           [0, 1, 1],
                           [1, 1, 1]], dtype='b1')
        actual = self.setup_instance(diploid_genotype_data).is_missing()
        aeq(expect, actual)

        # polyploid
        expect = np.array([[0, 0, 1],
                           [0, 0, 1],
                           [0, 1, 1],
                           [1, 1, 1]], dtype='b1')
        actual = self.setup_instance(triploid_genotype_data).is_missing()
        aeq(expect, actual)

    def test_is_hom(self):

        # diploid
        expect = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0],
                           [1, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = self.setup_instance(diploid_genotype_data).is_hom()
        aeq(expect, actual)

        # polyploid
        expect = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = self.setup_instance(triploid_genotype_data).is_hom()
        aeq(expect, actual)

    def test_is_hom_ref(self):

        # diploid
        expect = np.array([[1, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = self.setup_instance(diploid_genotype_data).is_hom(allele=0)
        aeq(expect, actual)
        actual = self.setup_instance(diploid_genotype_data).is_hom_ref()
        aeq(expect, actual)

        # polyploid
        expect = np.array([[1, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = self.setup_instance(triploid_genotype_data).is_hom(allele=0)
        aeq(expect, actual)
        actual = self.setup_instance(triploid_genotype_data).is_hom_ref()
        aeq(expect, actual)

    def test_is_hom_alt(self):

        # diploid
        expect = np.array([[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0],
                           [1, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = self.setup_instance(diploid_genotype_data).is_hom_alt()
        aeq(expect, actual)

        # polyploid
        expect = np.array([[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = self.setup_instance(triploid_genotype_data).is_hom_alt()
        aeq(expect, actual)

    def test_is_hom_1(self):

        # diploid
        expect = np.array([[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = self.setup_instance(diploid_genotype_data).is_hom(allele=1)
        aeq(expect, actual)

        # polyploid
        expect = np.array([[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = self.setup_instance(triploid_genotype_data).is_hom(allele=1)
        aeq(expect, actual)

    def test_is_het(self):

        # diploid
        expect = np.array([[0, 1, 0],
                           [1, 0, 0],
                           [1, 1, 0],
                           [0, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = self.setup_instance(diploid_genotype_data).is_het()
        aeq(expect, actual)

        # polyploid
        expect = np.array([[0, 1, 0],
                           [1, 0, 0],
                           [1, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = self.setup_instance(triploid_genotype_data).is_het()
        aeq(expect, actual)

    def test_is_call(self):

        # diploid
        expect = np.array([[0, 0, 0],
                           [1, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = self.setup_instance(diploid_genotype_data).is_call((0, 2))
        aeq(expect, actual)

        # polyploid
        expect = np.array([[0, 0, 0],
                           [0, 0, 0],
                           [1, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = self.setup_instance(triploid_genotype_data).is_call((0, 1, 2))
        aeq(expect, actual)

    def test_count_called(self):

        g = self.setup_instance(diploid_genotype_data)
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

        g = self.setup_instance(diploid_genotype_data)
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

        g = self.setup_instance(diploid_genotype_data)
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

        g = self.setup_instance(diploid_genotype_data)
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

        g = self.setup_instance(diploid_genotype_data)
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

        g = self.setup_instance(diploid_genotype_data)
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

        g = self.setup_instance(diploid_genotype_data)
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

    # data transformation methods
    #############################

    def test_to_haplotypes(self):

        # diploid
        expect = np.array([[0, 0, 0, 1, -1, -1],
                           [0, 2, 1, 1, -1, -1],
                           [1, 0, 2, 1, -1, -1],
                           [2, 2, -1, -1, -1, -1],
                           [-1, -1, -1, -1, -1, -1]], dtype='i1')
        actual = self.setup_instance(diploid_genotype_data).to_haplotypes()
        aeq(expect, actual)

        # polyploid
        expect = np.array([[0, 0, 0, 0, 0, 1, -1, -1, -1],
                           [0, 1, 1, 1, 1, 1, -1, -1, -1],
                           [0, 1, 2, -1, -1, -1, -1, -1, -1],
                           [-1, -1, -1, -1, -1, -1, -1, -1, -1]], dtype='i1')
        actual = self.setup_instance(triploid_genotype_data).to_haplotypes()
        aeq(expect, actual)

    def test_to_n_alt(self):

        # diploid
        expect = np.array([[0, 1, 0],
                           [1, 2, 0],
                           [1, 2, 0],
                           [2, 0, 0],
                           [0, 0, 0]], dtype='i1')
        actual = self.setup_instance(diploid_genotype_data).to_n_alt()
        aeq(expect, actual)

        # polyploid
        expect = np.array([[0, 1, 0],
                           [2, 3, 0],
                           [2, 0, 0],
                           [0, 0, 0]], dtype='i1')
        actual = self.setup_instance(triploid_genotype_data).to_n_alt()
        aeq(expect, actual)

        # diploid with fill
        expect = np.array([[0, 1, -1],
                           [1, 2, -1],
                           [1, 2, -1],
                           [2, -1, -1],
                           [-1, -1, -1]], dtype='i1')
        actual = self.setup_instance(diploid_genotype_data).to_n_alt(fill=-1)
        aeq(expect, actual)

        # polyploid with fill
        expect = np.array([[0, 1, -1],
                           [2, 3, -1],
                           [2, -1, -1],
                           [-1, -1, -1]], dtype='i1')
        actual = self.setup_instance(triploid_genotype_data).to_n_alt(fill=-1)
        aeq(expect, actual)

    def test_to_allele_counts(self):

        # diploid
        expect = np.array([[[2, 0, 0], [1, 1, 0], [0, 0, 0]],
                           [[1, 0, 1], [0, 2, 0], [0, 0, 0]],
                           [[1, 1, 0], [0, 1, 1], [0, 0, 0]],
                           [[0, 0, 2], [0, 0, 0], [0, 0, 0]],
                           [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype='i1')
        actual = self.setup_instance(diploid_genotype_data).to_allele_counts()
        aeq(expect, actual)

        # polyploid
        expect = np.array([[[3, 0, 0], [2, 1, 0], [0, 0, 0]],
                           [[1, 2, 0], [0, 3, 0], [0, 0, 0]],
                           [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
                           [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype='i1')
        actual = self.setup_instance(triploid_genotype_data).to_allele_counts()
        aeq(expect, actual)

    def test_to_packed(self):

        expect = np.array([[0, 1, 239],
                           [2, 17, 239],
                           [16, 33, 239],
                           [34, 239, 239],
                           [239, 239, 239]], dtype='u1')
        actual = self.setup_instance(diploid_genotype_data).to_packed()
        aeq(expect, actual)

    def test_from_packed(self):
        packed_data = np.array([[0, 1, 239],
                                [2, 17, 239],
                                [16, 33, 239],
                                [34, 239, 239],
                                [239, 239, 239]], dtype='u1')
        expect = diploid_genotype_data
        actual = self._class.from_packed(packed_data)
        aeq(expect, actual)

    def test_max(self):

        # overall
        expect = 2
        actual = self.setup_instance(diploid_genotype_data).max()
        eq(expect, actual)

        # by sample
        expect = np.array([2, 2, -1])
        actual = self.setup_instance(diploid_genotype_data).max(axis=(0, 2))
        aeq(expect, actual)

        # by variant
        expect = np.array([1, 2, 2, 2, -1])
        actual = self.setup_instance(diploid_genotype_data).max(axis=(1, 2))
        aeq(expect, actual)

    def test_min(self):

        # overall
        expect = -1
        actual = self.setup_instance(diploid_genotype_data).min()
        eq(expect, actual)

        # by sample
        expect = np.array([-1, -1, -1])
        actual = self.setup_instance(diploid_genotype_data).min(axis=(0, 2))
        aeq(expect, actual)

        # by variant
        expect = np.array([-1, -1, -1, -1, -1])
        actual = self.setup_instance(diploid_genotype_data).min(axis=(1, 2))
        aeq(expect, actual)

    def test_count_alleles(self):

        # diploid
        g = self.setup_instance(diploid_genotype_data)
        expect = np.array([[3, 1, 0],
                           [1, 2, 1],
                           [1, 2, 1],
                           [0, 0, 2],
                           [0, 0, 0]])
        actual = g.count_alleles()
        aeq(expect, actual)
        eq(5, actual.n_variants)
        eq(3, actual.n_alleles)

        # polyploid
        g = self.setup_instance(triploid_genotype_data)
        expect = np.array([[5, 1, 0],
                           [1, 5, 0],
                           [1, 1, 1],
                           [0, 0, 0]])
        actual = g.count_alleles()
        aeq(expect, actual)
        eq(4, actual.n_variants)
        eq(3, actual.n_alleles)


class HaplotypeArrayInterface(object):

    def setup_instance(self, data):
        # to be implemented in sub-classes
        pass

    # to be overriden in sub-classes
    _class = None

    # basic properties and data access methods
    ##########################################

    def test_properties(self):

        # haploid data
        h = self.setup_instance(haplotype_data)
        eq(2, h.ndim)
        eq((4, 3), h.shape)
        eq(4, h.n_variants)
        eq(3, h.n_haplotypes)

    def test_array_like(self):
        # Test that an instance is array-like, in that it can be used as
        # input argument to np.array(). I.e., there is a standard way to get
        # a vanilla numpy array representation of the data.

        h = self.setup_instance(haplotype_data)
        a = np.array(h, copy=False)
        aeq(haplotype_data, a)

    def test_slice(self):

        h = self.setup_instance(haplotype_data)

        # row slice
        s = h[1:]
        aeq(haplotype_data[1:], s)
        eq(3, s.n_variants)
        eq(3, s.n_haplotypes)

        # col slice
        s = h[:, 1:]
        aeq(np.array(haplotype_data)[:, 1:], s)
        eq(4, s.n_variants)
        eq(2, s.n_haplotypes)

        # row index
        s = h[0]
        assert not hasattr(s, 'n_variants')
        aeq(haplotype_data[0], s)

        # col index
        s = h[:, 0]
        assert not hasattr(s, 'n_samples')
        aeq(np.array(haplotype_data)[:, 0], s)

        # item
        s = h[0, 0]
        eq(0, s)

    def test_take(self):
        # Test the take() method.

        h = self.setup_instance(haplotype_data)

        # take variants
        indices = [0, 2]
        t = h.take(indices, axis=0)
        eq(2, t.n_variants)
        eq(h.n_haplotypes, t.n_haplotypes)
        expect = np.array(haplotype_data).take(indices, axis=0)
        aeq(expect, t)

        # take samples
        indices = [0, 2]
        t = h.take(indices, axis=1)
        eq(h.n_variants, t.n_variants)
        eq(2, t.n_haplotypes)
        expect = np.array(haplotype_data).take(indices, axis=1)
        aeq(expect, t)

    def test_compress(self):
        # Test the compress() method.

        h = self.setup_instance(haplotype_data)

        # compress variants
        condition = [True, False, True, False]
        t = h.compress(condition, axis=0)
        eq(2, t.n_variants)
        eq(h.n_haplotypes, t.n_haplotypes)
        expect = np.array(haplotype_data).compress(condition, axis=0)
        aeq(expect, t)

        # compress samples
        condition = [True, False, True]
        t = h.compress(condition, axis=1)
        eq(h.n_variants, t.n_variants)
        eq(2, t.n_haplotypes)
        expect = np.array(haplotype_data).compress(condition, axis=1)
        aeq(expect, t)

    def test_subset(self):
        # Test the subset() method.

        h = self.setup_instance(haplotype_data)

        # test with indices
        variants = [0, 2]
        haplotypes = [0, 2]
        s = h.subset(variants=variants, haplotypes=haplotypes)
        expect = np.array(haplotype_data)\
            .take(variants, axis=0)\
            .take(haplotypes, axis=1)
        aeq(expect, s)

        # test with condition
        variants = [True, False, True, False]
        haplotypes = [True, False, True]
        s = h.subset(variants=variants, haplotypes=haplotypes)
        expect = np.array(haplotype_data)\
            .compress(variants, axis=0)\
            .compress(haplotypes, axis=1)
        aeq(expect, s)

        # mix and match
        variants = [0, 2]
        haplotypes = [True, False, True]
        s = h.subset(variants=variants, haplotypes=haplotypes)
        expect = np.array(haplotype_data)\
            .take(variants, axis=0)\
            .compress(haplotypes, axis=1)
        aeq(expect, s)

        # mix and match
        variants = [True, False, True, False]
        haplotypes = [0, 2]
        s = h.subset(variants=variants, haplotypes=haplotypes)
        expect = np.array(haplotype_data)\
            .compress(variants, axis=0)\
            .take(haplotypes, axis=1)
        aeq(expect, s)

    def test_is_called(self):
        expect = np.array([[1, 1, 0],
                           [1, 1, 0],
                           [1, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = self.setup_instance(haplotype_data) >= 0
        aeq(expect, actual)
        actual = self.setup_instance(haplotype_data).is_called()
        aeq(expect, actual)

    def test_is_missing(self):
        expect = np.array([[0, 0, 1],
                           [0, 0, 1],
                           [0, 1, 1],
                           [1, 1, 1]], dtype='b1')
        actual = self.setup_instance(haplotype_data) < 0
        aeq(expect, actual)
        actual = self.setup_instance(haplotype_data).is_missing()
        aeq(expect, actual)

    def test_is_ref(self):
        expect = np.array([[1, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = self.setup_instance(haplotype_data) == 0
        aeq(expect, actual)
        actual = self.setup_instance(haplotype_data).is_ref()
        aeq(expect, actual)

    def test_is_alt(self):
        expect = np.array([[0, 1, 0],
                           [1, 1, 0],
                           [1, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = self.setup_instance(haplotype_data) > 0
        aeq(expect, actual)
        actual = self.setup_instance(haplotype_data).is_alt()
        aeq(expect, actual)

    def test_is_call(self):
        expect = np.array([[0, 0, 0],
                           [0, 0, 0],
                           [1, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = self.setup_instance(haplotype_data) == 2
        aeq(expect, actual)
        actual = self.setup_instance(haplotype_data).is_call(2)
        aeq(expect, actual)

    # TODO test to_genotypes()

    def test_max(self):

        # overall
        expect = 2
        actual = self.setup_instance(haplotype_data).max()
        eq(expect, actual)

        # by sample
        expect = np.array([2, 1, -1])
        actual = self.setup_instance(haplotype_data).max(axis=0)
        aeq(expect, actual)

        # by variant
        expect = np.array([1, 1, 2, -1])
        actual = self.setup_instance(haplotype_data).max(axis=1)
        aeq(expect, actual)

    def test_min(self):

        # overall
        expect = -1
        actual = self.setup_instance(haplotype_data).min()
        eq(expect, actual)

        # by sample
        expect = np.array([-1, -1, -1])
        actual = self.setup_instance(haplotype_data).min(axis=0)
        aeq(expect, actual)

        # by variant
        expect = np.array([-1, -1, -1, -1])
        actual = self.setup_instance(haplotype_data).min(axis=1)
        aeq(expect, actual)

    def test_count_alleles(self):
        expect = np.array([[1, 1, 0],
                           [0, 2, 0],
                           [0, 0, 1],
                           [0, 0, 0]])
        actual = self.setup_instance(haplotype_data).count_alleles()
        aeq(expect, actual)
        eq(4, actual.n_variants)
        eq(3, actual.n_alleles)


class AlleleCountsArrayInterface(object):

    def setup_instance(self, data):
        # to be implemented in sub-classes
        pass

    # to be overriden in sub-classes
    _class = None

    # basic properties and data access methods
    ##########################################

    def test_properties(self):
        ac = self.setup_instance(allele_counts_data)
        eq(2, ac.ndim)
        eq((5, 3), ac.shape)
        eq(5, ac.n_variants)
        eq(3, ac.n_alleles)

    def test_array_like(self):
        # Test that an instance is array-like, in that it can be used as
        # input argument to np.array(). I.e., there is a standard way to get
        # a vanilla numpy array representation of the data.

        ac = self.setup_instance(allele_counts_data)
        a = np.array(ac, copy=False)
        aeq(allele_counts_data, a)

    def test_slice(self):
        ac = self.setup_instance(allele_counts_data)

        # row slice
        s = ac[1:]
        aeq(allele_counts_data[1:], s)
        # if length of second dimension is preserved, expect result to be
        # wrapped
        assert hasattr(s, 'n_variants')
        assert hasattr(s, 'n_alleles')

        # col slice
        s = ac[:, 1:]
        aeq(np.array(allele_counts_data)[:, 1:], s)
        assert not hasattr(s, 'n_variants')
        assert not hasattr(s, 'n_alleles')

        # row index
        s = ac[0]
        assert not hasattr(s, 'n_variants')
        assert not hasattr(s, 'n_alleles')
        aeq(allele_counts_data[0], s)

        # col index
        s = ac[:, 0]
        assert not hasattr(s, 'n_variants')
        assert not hasattr(s, 'n_alleles')
        aeq(np.array(allele_counts_data)[:, 0], s)

        # item
        s = ac[0, 0]
        eq(3, s)

    def test_take(self):
        # Test the take() method.

        ac = self.setup_instance(allele_counts_data)

        # take variants
        indices = [0, 2]
        t = ac.take(indices, axis=0)
        eq(2, t.n_variants)
        eq(ac.n_alleles, t.n_alleles)
        expect = np.array(allele_counts_data).take(indices, axis=0)
        aeq(expect, t)

    def test_compress(self):
        # Test the compress() method.

        ac = self.setup_instance(allele_counts_data)

        # compress variants
        condition = [True, False, True, False, True]
        t = ac.compress(condition, axis=0)
        eq(3, t.n_variants)
        eq(ac.n_alleles, t.n_alleles)
        expect = np.array(allele_counts_data).compress(condition, axis=0)
        aeq(expect, t)

    def test_to_frequencies(self):
        ac = self.setup_instance(allele_counts_data)
        expect = np.array([[3/4, 1/4, 0/4],
                           [1/4, 2/4, 1/4],
                           [1/4, 2/4, 1/4],
                           [0/2, 0/2, 2/2],
                           [-1, -1, -1]])
        actual = ac.to_frequencies(fill=-1)
        aeq(expect, actual)

    def test_allelism(self):

        expect = np.array([2, 3, 3, 1, 0])
        actual = self.setup_instance(allele_counts_data).allelism()
        aeq(expect, actual)

    def test_is_count_variant(self):
        ac = self.setup_instance(allele_counts_data)
        expect = np.array([1, 1, 1, 1, 0], dtype='b1')
        actual = ac.is_variant()
        aeq(expect, actual)
        eq(np.sum(expect), ac.count_variant())

    def test_is_count_non_variant(self):
        ac = self.setup_instance(allele_counts_data)
        expect = np.array([0, 0, 0, 0, 1], dtype='b1')
        actual = ac.is_non_variant()
        aeq(expect, actual)
        eq(np.sum(expect), ac.count_non_variant())

    def test_is_count_segregating(self):
        ac = self.setup_instance(allele_counts_data)
        expect = np.array([1, 1, 1, 0, 0], dtype='b1')
        actual = ac.is_segregating()
        aeq(expect, actual)
        eq(np.sum(expect), ac.count_segregating())

    def test_is_count_non_segregating(self):
        ac = self.setup_instance(allele_counts_data)
        expect = np.array([0, 0, 0, 1, 1], dtype='b1')
        actual = ac.is_non_segregating()
        aeq(expect, actual)
        eq(np.sum(expect), ac.count_non_segregating())
        expect = np.array([0, 0, 0, 1, 0], dtype='b1')
        actual = ac.is_non_segregating(allele=2)
        aeq(expect, actual)
        eq(np.sum(expect), ac.count_non_segregating(allele=2))

    def test_is_count_singleton(self):
        ac = self.setup_instance(allele_counts_data)
        expect = np.array([1, 0, 0, 0, 0], dtype='b1')
        actual = ac.is_singleton(allele=1)
        aeq(expect, actual)
        eq(np.sum(expect), ac.count_singleton(allele=1))
        expect = np.array([0, 1, 1, 0, 0], dtype='b1')
        actual = ac.is_singleton(allele=2)
        aeq(expect, actual)
        eq(np.sum(expect), ac.count_singleton(allele=2))

    def test_is_count_doubleton(self):
        ac = self.setup_instance(allele_counts_data)
        expect = np.array([0, 1, 1, 0, 0], dtype='b1')
        actual = ac.is_doubleton(allele=1)
        aeq(expect, actual)
        eq(np.sum(expect), ac.count_doubleton(allele=1))
        expect = np.array([0, 0, 0, 1, 0], dtype='b1')
        actual = ac.is_doubleton(allele=2)
        aeq(expect, actual)
        eq(np.sum(expect), ac.count_doubleton(allele=2))


class SortedIndexInterface(object):

    _class = None

    def setup_instance(self, data):
        pass

    def test_properties(self):

        # valid data (unique)
        data = [1, 4, 5, 7, 12]
        pos = self.setup_instance(data)
        eq(1, pos.ndim)
        eq(5, len(pos))
        assert pos.is_unique

        # valid data (non-unique)
        data = [1, 4, 5, 5, 7, 12]
        pos = self.setup_instance(data)
        eq(1, pos.ndim)
        eq(6, len(pos))
        assert not pos.is_unique

    def test_array_like(self):

        data = [1, 4, 5, 7, 12]
        pos = self.setup_instance(data)
        a = np.array(pos, copy=False)
        aeq(data, a)

    def test_slice(self):

        data = [1, 4, 5, 5, 7, 12]
        pos = self.setup_instance(data)

        # row slice
        s = pos[1:]
        aeq(data[1:], s)
        eq(5, len(s))
        assert not s.is_unique

        # row slice
        s = pos[3:]
        aeq(data[3:], s)
        eq(3, len(s))
        assert s.is_unique

        # index
        s = pos[0]
        eq(data[0], s)

    def test_locate_key(self):
        pos = self.setup_instance([3, 6, 6, 11])
        f = pos.locate_key
        eq(0, f(3))
        eq(3, f(11))
        eq(slice(1, 3), f(6))
        with assert_raises(KeyError):
            f(2)

    def test_locate_keys(self):
        pos = self.setup_instance([3, 6, 6, 11, 20, 35])
        f = pos.locate_keys

        # all found
        expect = [False, True, True, False, True, False]
        actual = f([6, 20])
        assert_not_is_instance(actual, self._class)
        aeq(expect, actual)

        # not all found, lax
        expect = [False, True, True, False, True, False]
        actual = f([2, 6, 17, 20, 37], strict=False)
        assert_not_is_instance(actual, self._class)
        aeq(expect, actual)

        # not all found, strict
        with assert_raises(KeyError):
            f([2, 6, 17, 20, 37])

    def test_locate_intersection(self):
        pos1 = self.setup_instance([3, 6, 11, 20, 35])
        pos2 = self.setup_instance([4, 6, 20, 39])
        expect_loc1 = np.array([False, True, False, True, False])
        expect_loc2 = np.array([False, True, True, False])
        loc1, loc2 = pos1.locate_intersection(pos2)
        assert_not_is_instance(loc1, self._class)
        assert_not_is_instance(loc2, self._class)
        aeq(expect_loc1, loc1)
        aeq(expect_loc2, loc2)

    def test_intersect(self):
        pos1 = self.setup_instance([3, 6, 11, 20, 35])
        pos2 = self.setup_instance([4, 6, 20, 39])
        expect = self.setup_instance([6, 20])
        actual = pos1.intersect(pos2)
        assert_is_instance(actual, self._class)
        aeq(expect, actual)

    def test_locate_range(self):
        pos = self.setup_instance([3, 6, 11, 20, 35])
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
        with assert_raises(KeyError):
            f(17, 19)
        with assert_raises(KeyError):
            f(0, 2)
        with assert_raises(KeyError):
            f(36, 2000)

    def test_intersect_range(self):
        pos = self.setup_instance([3, 6, 11, 20, 35])
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
        pos = self.setup_instance([3, 6, 11, 20, 35])

        # all found
        ranges = np.array([[6, 12], [31, 35]])
        expect = np.array([False, True, True, False, True])
        actual = pos.locate_ranges(ranges[:, 0], ranges[:, 1])
        assert_not_is_instance(actual, self._class)
        aeq(expect, actual)

        # not all found, lax
        ranges = np.array([[0, 2], [6, 12], [14, 19], [31, 35], [100, 120]])
        actual = pos.locate_ranges(ranges[:, 0], ranges[:, 1], strict=False)
        assert_not_is_instance(actual, self._class)
        aeq(expect, actual)

        # not all found, strict
        with assert_raises(KeyError):
            pos.locate_ranges(ranges[:, 0], ranges[:, 1])

    def test_locate_intersection_ranges(self):
        pos = self.setup_instance([3, 6, 11, 20, 35])
        f = pos.locate_intersection_ranges

        # all found
        ranges = np.array([[6, 12], [31, 35]])
        expect_loc1 = np.array([False, True, True, False, True])
        expect_loc2 = np.array([True, True])
        actual_loc1, actual_loc2 = f(ranges[:, 0], ranges[:, 1])
        assert_not_is_instance(actual_loc1, self._class)
        assert_not_is_instance(actual_loc2, self._class)
        aeq(expect_loc1, actual_loc1)
        aeq(expect_loc2, actual_loc2)

        # not all found
        ranges = np.array([[0, 2], [6, 12], [14, 19], [31, 35], [100, 120]])
        expect_loc1 = np.array([False, True, True, False, True])
        expect_loc2 = np.array([False, True, False, True, False])
        actual_loc1, actual_loc2 = f(ranges[:, 0], ranges[:, 1])
        assert_not_is_instance(actual_loc1, self._class)
        assert_not_is_instance(actual_loc2, self._class)
        aeq(expect_loc1, actual_loc1)
        aeq(expect_loc2, actual_loc2)

    def test_intersect_ranges(self):
        pos = self.setup_instance([3, 6, 11, 20, 35])
        f = pos.intersect_ranges

        # all found
        ranges = np.array([[6, 12], [31, 35]])
        expect = [6, 11, 35]
        actual = f(ranges[:, 0], ranges[:, 1])
        assert_is_instance(actual, self._class)
        aeq(expect, actual)

        # not all found
        ranges = np.array([[0, 2], [6, 12], [14, 19], [31, 35], [100, 120]])
        expect = [6, 11, 35]
        actual = f(ranges[:, 0], ranges[:, 1])
        assert_is_instance(actual, self._class)
        aeq(expect, actual)


class UniqueIndexInterface(object):

    _class = None

    def setup_instance(self, data):
        pass

    def test_properties(self):
        data = ['A', 'C', 'B', 'F']
        lbl = self.setup_instance(data)
        eq(1, lbl.ndim)
        eq(4, len(lbl))

    def test_array_like(self):
        data = ['A', 'C', 'B', 'F']
        lbl = self.setup_instance(data)
        a = np.array(lbl, copy=False)
        aeq(data, a)

    def test_slice(self):

        data = ['A', 'C', 'B', 'F']
        lbl = self.setup_instance(data)

        # row slice
        s = lbl[1:]
        aeq(data[1:], s)
        eq(3, len(s))

        # index
        s = lbl[0]
        eq(data[0], s)

    def test_locate_key(self):
        lbl = self.setup_instance(['A', 'C', 'B', 'F'])
        f = lbl.locate_key
        eq(0, f('A'))
        eq(2, f('B'))
        with assert_raises(KeyError):
            f('D')

    def test_locate_keys(self):
        lbl = self.setup_instance(['A', 'C', 'B', 'F'])
        f = lbl.locate_keys

        # all found
        expect = [False, True, False, True]
        actual = f(['F', 'C'])
        assert_not_is_instance(actual, self._class)
        aeq(expect, actual)

        # not all found, lax
        expect = [False, True, False, True]
        actual = f(['X', 'F', 'G', 'C', 'Z'], strict=False)
        assert_not_is_instance(actual, self._class)
        aeq(expect, actual)

        # not all found, strict
        with assert_raises(KeyError):
            f(['X', 'F', 'G', 'C', 'Z'])

    def test_locate_intersection(self):
        lbl1 = self.setup_instance(['A', 'C', 'B', 'F'])
        lbl2 = self.setup_instance(['X', 'F', 'G', 'C', 'Z'])
        expect_loc1 = np.array([False, True, False, True])
        expect_loc2 = np.array([False, True, False, True, False])
        loc1, loc2 = lbl1.locate_intersection(lbl2)
        assert_not_is_instance(loc1, self._class)
        assert_not_is_instance(loc2, self._class)
        aeq(expect_loc1, loc1)
        aeq(expect_loc2, loc2)

    def test_intersect(self):
        lbl1 = self.setup_instance(['A', 'C', 'B', 'F'])
        lbl2 = self.setup_instance(['X', 'F', 'G', 'C', 'Z'])

        expect = self.setup_instance(['C', 'F'])
        actual = lbl1.intersect(lbl2)
        aeq(expect, actual)

        expect = self.setup_instance(['F', 'C'])
        actual = lbl2.intersect(lbl1)
        aeq(expect, actual)


class SortedMultiIndexInterface(object):

    _class = None

    def setup_instance(self, chrom, pos):
        pass

    def test_properties(self):
        chrom = [0, 0, 1, 1, 1, 2]
        pos = [1, 4, 2, 5, 5, 3]
        idx = self.setup_instance(chrom, pos)
        eq(6, len(idx))

    def test_locate_key(self):
        chrom = [0, 0, 1, 1, 1, 2]
        pos = [1, 4, 2, 5, 5, 3]
        idx = self.setup_instance(chrom, pos)
        f = idx.locate_key
        eq(slice(0, 2), f(0))
        eq(0, f(0, 1))
        eq(2, f(1, 2))
        eq(slice(3, 5), f(1, 5))
        with assert_raises(KeyError):
            f(2, 4)
        with assert_raises(KeyError):
            f(3, 4)

    def test_locate_range(self):
        chrom = [0, 0, 1, 1, 1, 2]
        pos = [1, 4, 2, 5, 5, 3]
        idx = self.setup_instance(chrom, pos)
        f = idx.locate_range

        eq(slice(0, 2), f(0))
        eq(slice(2, 5), f(1))
        eq(slice(5, 6), f(2))
        eq(slice(0, 2), f(0, 1, 5))
        eq(slice(0, 1), f(0, 1, 3))
        eq(slice(2, 5), f(1, 1, 5))
        eq(slice(2, 3), f(1, 1, 4))
        eq(slice(5, 6), f(2, 2, 6))
        with assert_raises(KeyError):
            f(0, 17, 19)
        with assert_raises(KeyError):
            f(1, 3, 4)
        with assert_raises(KeyError):
            f(2, 1, 2)
        with assert_raises(KeyError):
            f(3, 2, 4)


class VariantTableInterface(object):

    _class = None

    def setup_instance(self, data, **kwargs):
        pass

    def test_properties(self):
        a = np.rec.array(variant_table_data, names=variant_table_names)
        vt = self.setup_instance(a)
        eq(5, vt.n_variants)
        eq(variant_table_names, vt.names)

    def test_array_like(self):
        # Test that an instance is array-like, in that it can be used as
        # input argument to np.rec.array(). I.e., there is a standard way to
        # get a vanilla numpy array representation of the data.

        a = np.rec.array(variant_table_data, names=variant_table_names)
        vt = self.setup_instance(a)
        b = np.array(vt, copy=False)
        aeq(a, b)

    def test_get_item(self):
        a = np.rec.array(variant_table_data, names=variant_table_names)
        vt = self.setup_instance(a)

        # row slice
        s = vt[1:]
        eq(4, s.n_variants)
        eq(variant_table_names, s.names)
        a = np.rec.array(variant_table_data, names=variant_table_names)
        aeq(a[1:], s)

        # row index
        s = vt[1]
        eq(tuple(variant_table_data[1]), tuple(s))

        # column access
        s = vt['CHROM']
        a = np.rec.array(variant_table_data, names=variant_table_names)
        aeq(a['CHROM'], s)

        # multi-column access
        s = vt[['CHROM', 'POS']]
        a = np.rec.array(variant_table_data, names=variant_table_names)
        eq(5, s.n_variants)
        eq(('CHROM', 'POS'), s.names)
        aeq(a[['CHROM', 'POS']], s)

    def test_take(self):
        a = np.rec.array(variant_table_data, names=variant_table_names)
        vt = self.setup_instance(a)
        indices = [0, 2]
        t = vt.take(indices)
        expect = a.take(indices)
        aeq(expect, t)
        eq(2, t.n_variants)
        eq(variant_table_names, t.names)

    def test_compress(self):
        a = np.rec.array(variant_table_data, names=variant_table_names)
        vt = self.setup_instance(a)
        condition = [True, False, True, False, False]
        t = vt.compress(condition)
        expect = a.compress(condition)
        aeq(expect, t)
        eq(2, t.n_variants)
        eq(variant_table_names, t.names)

    def test_eval(self):
        a = np.rec.array(variant_table_data, names=variant_table_names)
        vt = self.setup_instance(a)

        expr = '(DP > 30) & (QD < 4)'
        r = vt.eval(expr)
        aeq([False, False, True, False, True], r)

    def test_query(self):
        a = np.rec.array(variant_table_data, names=variant_table_names)
        vt = self.setup_instance(a)

        query = '(DP > 30) & (QD < 4)'
        r = vt.query(query)
        aeq(a.take([2, 4]), r)

    def test_index(self):
        a = np.rec.array(variant_table_data, names=variant_table_names)

        # multi chromosome/contig
        vt = self.setup_instance(a, index=('CHROM', 'POS'))
        eq(slice(0, 2), vt.index.locate_key(b'chr1'))
        eq(1, vt.index.locate_key(b'chr1', 7))
        eq(slice(2, 4), vt.index.locate_range(b'chr2', 3, 9))

        # single chromosome/contig index
        vt = self.setup_instance(a[2:4][['POS', 'DP', 'QD']], index='POS')
        eq(0, vt.index.locate_key(3))
        eq(slice(0, 2), vt.index.locate_range(3, 9))
