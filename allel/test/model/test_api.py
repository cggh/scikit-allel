# -*- coding: utf-8 -*-
"""This module defines interfaces for the classes in the allel.model module.
These interfaces are defined as test cases, but are abstracted so that the
tests can be re-used for alternative implementations of the same interfaces.

"""
from __future__ import absolute_import, print_function, division


from datetime import date
import tempfile


import numpy as np
from nose.tools import eq_ as eq, assert_raises, \
    assert_is_instance, assert_not_is_instance, assert_almost_equal, \
    assert_sequence_equal
from allel.test.tools import assert_array_equal as aeq


import allel


haplotype_data = [
    [0, 1, -1],
    [1, 1, -1],
    [2, -1, -1],
    [-1, -1, -1]
]

diploid_genotype_data = [
    [[0, 0], [0, 1], [-1, -1]],
    [[0, 2], [1, 1], [-1, -1]],
    [[1, 0], [2, 1], [-1, -1]],
    [[2, 2], [-1, -1], [-1, -1]],
    [[-1, -1], [-1, -1], [-1, -1]]
]

diploid_genotype_ac_data = [
    [[2, 0, 0], [1, 1, 0], [0, 0, 0]],
    [[1, 0, 1], [0, 2, 0], [0, 0, 0]],
    [[1, 1, 0], [0, 1, 1], [0, 0, 0]],
    [[0, 0, 2], [0, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
]

triploid_genotype_data = [
    [[0, 0, 0], [0, 0, 1], [-1, -1, -1]],
    [[0, 1, 1], [1, 1, 1], [-1, -1, -1]],
    [[0, 1, 2], [-1, -1, -1], [-1, -1, -1]],
    [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]
]

triploid_genotype_ac_data = [
    [[3, 0, 0], [2, 1, 0], [0, 0, 0]],
    [[1, 2, 0], [0, 3, 0], [0, 0, 0]],
    [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
]

allele_counts_data = [
    [3, 1, 0],
    [1, 2, 1],
    [1, 2, 1],
    [0, 0, 2],
    [0, 0, 0],
    [0, 1, 2]
]

variant_table_data = [
    [b'chr1', 2, 35, 4.5, (1, 2)],
    [b'chr1', 7, 12, 6.7, (3, 4)],
    [b'chr2', 3, 78, 1.2, (5, 6)],
    [b'chr2', 9, 22, 4.4, (7, 8)],
    [b'chr3', 6, 99, 2.8, (9, 10)]
]
variant_table_dtype = [
    ('CHROM', 'S4'),
    ('POS', 'u4'),
    ('DP', int),
    ('QD', float),
    ('AC', (int, 2))
]
variant_table_names = tuple(t[0] for t in variant_table_dtype)

feature_table_data = [
    [b'chr1', b'DB', b'gene', 1000, 2000, -1, b'+', -1, b'gene1', b'.'],
    [b'chr1', b'DB', b'mRNA', 1000, 2000, -1, b'+', -1, b'rna1', b'gene1'],
    [b'chr1', b'DB', b'exon', 1100, 1300, -1, b'+', -1, b'exon1', b'rna1'],
    [b'chr1', b'DB', b'exon', 1500, 1800, -1, b'+', -1, b'exon2', b'rna1'],
    [b'chr1', b'DB', b'CDS', 1100, 1400, -1, b'+', 0, b'.', b'rna1'],
    [b'chr1', b'DB', b'CDS', 1431, 1800, -1, b'+', 1, b'.', b'rna1'],
]
feature_table_dtype = [
    ('seqid', 'S4'),
    ('source', 'S2'),
    ('type', 'S15'),
    ('start', int),
    ('end', int),
    ('score', float),
    ('strand', 'S1'),
    ('phase', int),
    ('ID', 'S5'),
    ('Parent', 'S5')
]
feature_table_names = tuple(t[0] for t in feature_table_dtype)


class GenotypeArrayInterface(object):

    def setup_instance(self, data, dtype=None):
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

        # take samples not in original order
        indices = [2, 0]
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
        sel0 = [0, 2]
        sel1 = [0, 2]
        s = g.subset(sel0, sel1)
        expect = np.array(diploid_genotype_data)\
            .take(sel0, axis=0)\
            .take(sel1, axis=1)
        aeq(expect, s)

        # test with condition
        sel0 = [True, False, True, False, False]
        sel1 = [True, False, True]
        s = g.subset(sel0, sel1)
        expect = np.array(diploid_genotype_data)\
            .compress(sel0, axis=0)\
            .compress(sel1, axis=1)
        aeq(expect, s)

        # mix and match
        sel0 = [0, 2]
        sel1 = [True, False, True]
        s = g.subset(sel0, sel1)
        expect = np.array(diploid_genotype_data)\
            .take(sel0, axis=0)\
            .compress(sel1, axis=1)
        aeq(expect, s)

        # mix and match
        sel0 = [True, False, True, False, False]
        sel1 = [0, 2]
        s = g.subset(sel0, sel1)
        expect = np.array(diploid_genotype_data)\
            .compress(sel0, axis=0)\
            .take(sel1, axis=1)
        aeq(expect, s)

        # check argument type inference
        sel0 = list(range(g.shape[0]))
        sel1 = None
        s = g.subset(sel0, sel1)
        expect = np.array(diploid_genotype_data)
        aeq(expect, s)

        # check argument type inference
        sel0 = None
        sel1 = list(range(g.shape[1]))
        s = g.subset(sel0, sel1)
        expect = np.array(diploid_genotype_data)
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

    def test_to_n_ref(self):

        # diploid
        expect = np.array([[2, 1, 0],
                           [1, 0, 0],
                           [1, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]], dtype='i1')
        actual = self.setup_instance(diploid_genotype_data).to_n_ref()
        aeq(expect, actual)

        # polyploid
        expect = np.array([[3, 2, 0],
                           [1, 0, 0],
                           [1, 0, 0],
                           [0, 0, 0]], dtype='i1')
        actual = self.setup_instance(triploid_genotype_data).to_n_ref()
        aeq(expect, actual)

        # diploid with fill
        expect = np.array([[2, 1, -1],
                           [1, 0, -1],
                           [1, 0, -1],
                           [0, -1, -1],
                           [-1, -1, -1]], dtype='i1')
        actual = self.setup_instance(diploid_genotype_data).to_n_ref(fill=-1)
        aeq(expect, actual)

        # polyploid with fill
        expect = np.array([[3, 2, -1],
                           [1, 0, -1],
                           [1, -1, -1],
                           [-1, -1, -1]], dtype='i1')
        actual = self.setup_instance(triploid_genotype_data).to_n_ref(fill=-1)
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
        g = self.setup_instance(diploid_genotype_data)
        expect = np.array(diploid_genotype_ac_data, dtype='i1')
        gac = g.to_allele_counts()
        aeq(expect, gac)

        # polyploid
        g = self.setup_instance(triploid_genotype_data)
        expect = np.array(triploid_genotype_ac_data, dtype='i1')
        gac = g.to_allele_counts()
        aeq(expect, gac)

    def test_to_packed(self):
        expect = np.array([[0, 1, 239],
                           [2, 17, 239],
                           [16, 33, 239],
                           [34, 239, 239],
                           [239, 239, 239]], dtype='u1')
        for dtype in None, 'i1', 'i2', 'i4', 'i8':
            actual = self.setup_instance(diploid_genotype_data, dtype=dtype).to_packed()
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

    def test_to_gt(self):

        # diploid
        expect = [[b'0/0', b'0/1', b'./.'],
                  [b'0/2', b'1/1', b'./.'],
                  [b'1/0', b'2/1', b'./.'],
                  [b'2/2', b'./.', b'./.'],
                  [b'./.', b'./.', b'./.']]
        actual = self.setup_instance(diploid_genotype_data).to_gt()
        aeq(expect, actual)

        # polyploid
        expect = [[b'0/0/0', b'0/0/1', b'././.'],
                  [b'0/1/1', b'1/1/1', b'././.'],
                  [b'0/1/2', b'././.', b'././.'],
                  [b'././.', b'././.', b'././.']]
        actual = self.setup_instance(triploid_genotype_data).to_gt()
        aeq(expect, actual)

        # all zeroes
        data = [[[0, 0]]]
        expect = [[b'0/0']]
        actual = self.setup_instance(data).to_gt()
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
        for dtype in None, 'i1', 'i2', 'i4', 'i8':
            # make sure we test the optimisations too

            # diploid
            g = self.setup_instance(diploid_genotype_data, dtype=dtype)
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
            g = self.setup_instance(triploid_genotype_data, dtype=dtype)
            expect = np.array([[5, 1, 0],
                               [1, 5, 0],
                               [1, 1, 1],
                               [0, 0, 0]])
            actual = g.count_alleles()
            aeq(expect, actual)
            eq(4, actual.n_variants)
            eq(3, actual.n_alleles)

    def test_count_alleles_subpop(self):
        for dtype in None, 'i1', 'i2', 'i4', 'i8':
            # make sure we test the optimisations too
            g = self.setup_instance(diploid_genotype_data, dtype=dtype)
            expect = np.array([[2, 0, 0],
                               [1, 0, 1],
                               [1, 1, 0],
                               [0, 0, 2],
                               [0, 0, 0]])
            actual = g.count_alleles(subpop=[0, 2])
            aeq(expect, actual)
            eq(5, actual.n_variants)
            eq(3, actual.n_alleles)

    def test_count_alleles_subpops(self):
        for dtype in None, 'i1', 'i2', 'i4', 'i8':
            # make sure we test the optimisations too
            g = self.setup_instance(diploid_genotype_data, dtype=dtype)
            subpops = {'sub1': [0, 2], 'sub2': [1, 2]}
            expect_sub1 = np.array([[2, 0, 0],
                                    [1, 0, 1],
                                    [1, 1, 0],
                                    [0, 0, 2],
                                    [0, 0, 0]])
            expect_sub2 = np.array([[1, 1, 0],
                                    [0, 2, 0],
                                    [0, 1, 1],
                                    [0, 0, 0],
                                    [0, 0, 0]])
            actual = g.count_alleles_subpops(subpops=subpops)
            aeq(expect_sub1, actual['sub1'])
            aeq(expect_sub2, actual['sub2'])
            eq(5, actual['sub1'].n_variants)
            eq(3, actual['sub1'].n_alleles)
            eq(5, actual['sub2'].n_variants)
            eq(3, actual['sub2'].n_alleles)

    def test_count_alleles_max_allele(self):

        for dtype in None, 'i1', 'i2', 'i4', 'i8':
            # make sure we test the optimisations too

            # diploid
            g = self.setup_instance(diploid_genotype_data, dtype=dtype)
            expect = np.array([[3, 1, 0],
                               [1, 2, 1],
                               [1, 2, 1],
                               [0, 0, 2],
                               [0, 0, 0]])
            actual = g.count_alleles()
            eq(3, actual.n_alleles)
            aeq(expect, actual)
            actual = g.count_alleles(max_allele=2)
            eq(3, actual.n_alleles)
            aeq(expect, actual)
            actual = g.count_alleles(max_allele=1)
            eq(2, actual.n_alleles)
            aeq(expect[:, :2], actual)
            actual = g.count_alleles(max_allele=0)
            eq(1, actual.n_alleles)
            aeq(expect[:, :1], actual)

            # polyploid
            g = self.setup_instance(triploid_genotype_data, dtype=dtype)
            expect = np.array([[5, 1, 0],
                               [1, 5, 0],
                               [1, 1, 1],
                               [0, 0, 0]])
            actual = g.count_alleles()
            eq(3, actual.n_alleles)
            aeq(expect, actual)
            actual = g.count_alleles(max_allele=2)
            eq(3, actual.n_alleles)
            aeq(expect, actual)
            actual = g.count_alleles(max_allele=1)
            eq(2, actual.n_alleles)
            aeq(expect[:, :2], actual)
            actual = g.count_alleles(max_allele=0)
            eq(1, actual.n_alleles)
            aeq(expect[:, :1], actual)

    def test_map_alleles(self):
        for dtype in None, 'i1', 'i2', 'i4', 'i8':
            a = np.array(diploid_genotype_data, dtype=dtype)
            g = self.setup_instance(a)
            mapping = np.array([[0, 1, 2],
                                [2, 0, 1],
                                [1, 2, 0],
                                [2, 1, 0],
                                [2, 0, 1]], dtype='i1')
            expect = [[[0, 0], [0, 1], [-1, -1]],
                      [[2, 1], [0, 0], [-1, -1]],
                      [[2, 1], [0, 2], [-1, -1]],
                      [[0, 0], [-1, -1], [-1, -1]],
                      [[-1, -1], [-1, -1], [-1, -1]]]
            actual = g.map_alleles(mapping)
            aeq(expect, actual)

    def test_set_mask(self):

        # diploid case
        a = np.array(diploid_genotype_data, dtype=np.int8)
        g = self.setup_instance(a)
        eq(7, g.count_called())
        eq(4, g.count_het())
        eq(3, g.count_hom())
        eq(1, g.count_hom_ref())
        eq(2, g.count_hom_alt())
        eq(8, g.count_missing())
        expect_ac = [[3, 1, 0],
                     [1, 2, 1],
                     [1, 2, 1],
                     [0, 0, 2],
                     [0, 0, 0]]
        aeq(expect_ac, g.count_alleles())
        m = [[True, False, False],
             [False, False, False],
             [False, True, False],
             [False, False, True],
             [True, False, True]]
        g.mask = m
        eq(5, g.count_called())
        eq(3, g.count_het())
        eq(2, g.count_hom())
        eq(0, g.count_hom_ref())
        eq(2, g.count_hom_alt())
        eq(10, g.count_missing())
        expect_ac = [[1, 1, 0],
                     [1, 2, 1],
                     [1, 1, 0],
                     [0, 0, 2],
                     [0, 0, 0]]
        aeq(expect_ac, g.count_alleles())

        # polyploid
        a = np.array(triploid_genotype_data, dtype=np.int8)
        g = self.setup_instance(a)
        eq(5, g.count_called())
        eq(3, g.count_het())
        eq(2, g.count_hom())
        eq(1, g.count_hom_ref())
        eq(1, g.count_hom_alt())
        eq(7, g.count_missing())
        expect_ac = [[5, 1, 0],
                     [1, 5, 0],
                     [1, 1, 1],
                     [0, 0, 0]]
        aeq(expect_ac, g.count_alleles())
        m = [[True, False, False],
             [False, False, False],
             [False, True, False],
             [False, False, True]]
        g.mask = m
        eq(4, g.count_called())
        eq(3, g.count_het())
        eq(1, g.count_hom())
        eq(0, g.count_hom_ref())
        eq(1, g.count_hom_alt())
        eq(8, g.count_missing())
        expect_ac = [[2, 1, 0],
                     [1, 5, 0],
                     [1, 1, 1],
                     [0, 0, 0]]
        aeq(expect_ac, g.count_alleles())

    def test_fill_masked(self):

        # diploid case
        a = np.array(diploid_genotype_data, dtype=np.int8)
        g = self.setup_instance(a)
        m = [[True, False, False],
             [False, False, False],
             [False, True, False],
             [False, False, True],
             [True, False, True]]
        g.mask = m
        gm = g.fill_masked()
        expect = [[[-1, -1], [0, 1], [-1, -1]],
                  [[0, 2], [1, 1], [-1, -1]],
                  [[1, 0], [-1, -1], [-1, -1]],
                  [[2, 2], [-1, -1], [-1, -1]],
                  [[-1, -1], [-1, -1], [-1, -1]]]
        aeq(expect, gm)

        # polyploid
        a = np.array(triploid_genotype_data, dtype=np.int8)
        g = self.setup_instance(a)
        m = [[True, False, False],
             [False, False, False],
             [False, True, False],
             [False, False, True]]
        g.mask = m
        gm = g.fill_masked()
        expect = [[[-1, -1, -1], [0, 0, 1], [-1, -1, -1]],
                  [[0, 1, 1], [1, 1, 1], [-1, -1, -1]],
                  [[0, 1, 2], [-1, -1, -1], [-1, -1, -1]],
                  [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]]
        aeq(expect, gm)

    def test_concatenate(self):
        a = np.array(diploid_genotype_data, dtype=np.int8)
        g1 = self.setup_instance(a)
        g2 = self.setup_instance(a)
        for axis in 0, 1:
            actual = g1.concatenate(g2, axis=axis)
            expect = np.concatenate([a, a], axis=axis)
            aeq(expect, actual)


class HaplotypeArrayInterface(object):

    def setup_instance(self, data, dtype=None):
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
        sel0 = [0, 2]
        sel1 = [0, 2]
        s = h.subset(sel0, sel1)
        expect = np.array(haplotype_data)\
            .take(sel0, axis=0)\
            .take(sel1, axis=1)
        aeq(expect, s)

        # test with condition
        sel0 = [True, False, True, False]
        sel1 = [True, False, True]
        s = h.subset(sel0, sel1)
        expect = np.array(haplotype_data)\
            .compress(sel0, axis=0)\
            .compress(sel1, axis=1)
        aeq(expect, s)

        # mix and match
        sel0 = [0, 2]
        sel1 = [True, False, True]
        s = h.subset(sel0, sel1)
        expect = np.array(haplotype_data)\
            .take(sel0, axis=0)\
            .compress(sel1, axis=1)
        aeq(expect, s)

        # mix and match
        sel0 = [True, False, True, False]
        sel1 = [0, 2]
        s = h.subset(sel0, sel1)
        expect = np.array(haplotype_data)\
            .compress(sel0, axis=0)\
            .take(sel1, axis=1)
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
        for dtype in None, 'i1', 'i2', 'i4', 'i8':
            h = self.setup_instance(haplotype_data, dtype=dtype)
            actual = h.count_alleles()
            aeq(expect, actual)
            eq(4, actual.n_variants)
            eq(3, actual.n_alleles)

    def test_count_alleles_subpop(self):
        expect = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1],
                           [0, 0, 0]])
        for dtype in None, 'i1', 'i2', 'i4', 'i8':
            h = self.setup_instance(haplotype_data, dtype=dtype)
            actual = h.count_alleles(subpop=[0, 2])
            aeq(expect, actual)
            eq(4, actual.n_variants)
            eq(3, actual.n_alleles)

    def test_count_alleles_subpops(self):
        expect_sub1 = np.array([[1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1],
                                [0, 0, 0]])
        expect_sub2 = np.array([[0, 1, 0],
                                [0, 1, 0],
                                [0, 0, 0],
                                [0, 0, 0]])
        for dtype in None, 'i1', 'i2', 'i4', 'i8':
            h = self.setup_instance(haplotype_data, dtype=dtype)
            subpops = {'sub1': [0, 2], 'sub2': [1, 2]}
            actual = h.count_alleles_subpops(subpops=subpops)
            aeq(expect_sub1, actual['sub1'])
            aeq(expect_sub2, actual['sub2'])
            eq(4, actual['sub1'].n_variants)
            eq(3, actual['sub1'].n_alleles)
            eq(4, actual['sub2'].n_variants)
            eq(3, actual['sub2'].n_alleles)

    def test_count_alleles_max_allele(self):
        expect = np.array([[1, 1, 0],
                           [0, 2, 0],
                           [0, 0, 1],
                           [0, 0, 0]])
        for dtype in None, 'i1', 'i2', 'i4', 'i8':
            h = self.setup_instance(haplotype_data, dtype=dtype)
            actual = h.count_alleles()
            eq(3, actual.n_alleles)
            aeq(expect, actual)
            actual = h.count_alleles(max_allele=2)
            eq(3, actual.n_alleles)
            aeq(expect, actual)
            actual = h.count_alleles(max_allele=1)
            eq(2, actual.n_alleles)
            aeq(expect[:, :2], actual)
            actual = h.count_alleles(max_allele=0)
            eq(1, actual.n_alleles)
            aeq(expect[:, :1], actual)

    def test_map_alleles(self):

        a = np.array(haplotype_data)
        h = self.setup_instance(a)
        mapping = np.array([[0, 1, 2],
                            [2, 0, 1],
                            [1, 2, 0],
                            [2, 1, 0]])
        expect = [[0, 1, -1],
                  [0, 0, -1],
                  [0, -1, -1],
                  [-1, -1, -1]]
        actual = h.map_alleles(mapping)
        aeq(expect, actual)

        for dtype in None, 'i1', 'i2', 'i4', 'i8':
            a = np.array(haplotype_data, dtype=dtype)
            h = self.setup_instance(a)
            mapping = np.array(mapping, dtype=dtype)
            actual = h.map_alleles(mapping)
            aeq(expect, actual)

    def test_concatenate(self):
        a = np.array(haplotype_data, dtype=np.int8)
        h1 = self.setup_instance(a)
        h2 = self.setup_instance(a)
        for axis in 0, 1:
            actual = h1.concatenate(h2, axis=axis)
            expect = np.concatenate([a, a], axis=axis)
            aeq(expect, actual)


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
        eq((6, 3), ac.shape)
        eq(6, ac.n_variants)
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
        ac = self.setup_instance(allele_counts_data)
        condition = [True, False, True, False, True, False]
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
                           [-1, -1, -1],
                           [0/3, 1/3, 2/3]])
        actual = ac.to_frequencies(fill=-1)
        aeq(expect, actual)

    def test_allelism(self):
        expect = np.array([2, 3, 3, 1, 0, 2])
        actual = self.setup_instance(allele_counts_data).allelism()
        aeq(expect, actual)

    def test_max_allele(self):
        expect = np.array([1, 2, 2, 2, -1, 2])
        actual = self.setup_instance(allele_counts_data).max_allele()
        aeq(expect, actual)

    def test_is_count_variant(self):
        ac = self.setup_instance(allele_counts_data)
        expect = np.array([1, 1, 1, 1, 0, 1], dtype='b1')
        actual = ac.is_variant()
        aeq(expect, actual)
        eq(np.sum(expect), ac.count_variant())

    def test_is_count_non_variant(self):
        ac = self.setup_instance(allele_counts_data)
        expect = np.array([0, 0, 0, 0, 1, 0], dtype='b1')
        actual = ac.is_non_variant()
        aeq(expect, actual)
        eq(np.sum(expect), ac.count_non_variant())

    def test_is_count_segregating(self):
        ac = self.setup_instance(allele_counts_data)
        expect = np.array([1, 1, 1, 0, 0, 1], dtype='b1')
        actual = ac.is_segregating()
        aeq(expect, actual)
        eq(np.sum(expect), ac.count_segregating())

    def test_is_count_non_segregating(self):
        ac = self.setup_instance(allele_counts_data)
        expect = np.array([0, 0, 0, 1, 1, 0], dtype='b1')
        actual = ac.is_non_segregating()
        aeq(expect, actual)
        eq(np.sum(expect), ac.count_non_segregating())
        expect = np.array([0, 0, 0, 1, 0, 0], dtype='b1')
        actual = ac.is_non_segregating(allele=2)
        aeq(expect, actual)
        eq(np.sum(expect), ac.count_non_segregating(allele=2))

    def test_is_count_singleton(self):
        ac = self.setup_instance(allele_counts_data)
        expect = np.array([1, 0, 0, 0, 0, 1], dtype='b1')
        actual = ac.is_singleton(allele=1)
        aeq(expect, actual)
        eq(np.sum(expect), ac.count_singleton(allele=1))
        expect = np.array([0, 1, 1, 0, 0, 0], dtype='b1')
        actual = ac.is_singleton(allele=2)
        aeq(expect, actual)
        eq(np.sum(expect), ac.count_singleton(allele=2))

    def test_is_count_doubleton(self):
        ac = self.setup_instance(allele_counts_data)
        expect = np.array([0, 1, 1, 0, 0, 0], dtype='b1')
        actual = ac.is_doubleton(allele=1)
        aeq(expect, actual)
        eq(np.sum(expect), ac.count_doubleton(allele=1))
        expect = np.array([0, 0, 0, 1, 0, 1], dtype='b1')
        actual = ac.is_doubleton(allele=2)
        aeq(expect, actual)
        eq(np.sum(expect), ac.count_doubleton(allele=2))

    def test_is_biallelic(self):
        ac = self.setup_instance(allele_counts_data)
        expect = np.array([1, 0, 0, 0, 0, 1], dtype='b1')
        actual = ac.is_biallelic()
        aeq(expect, actual)

    def test_is_biallelic_01(self):
        ac = self.setup_instance(allele_counts_data)
        expect = np.array([1, 0, 0, 0, 0, 0], dtype='b1')
        actual = ac.is_biallelic_01()
        aeq(expect, actual)
        expect = np.array([1, 0, 0, 0, 0, 0], dtype='b1')
        actual = ac.is_biallelic_01(min_mac=1)
        aeq(expect, actual)
        expect = np.array([0, 0, 0, 0, 0, 0], dtype='b1')
        actual = ac.is_biallelic_01(min_mac=2)
        aeq(expect, actual)

    def test_map_alleles(self):
        ac = self.setup_instance(allele_counts_data)
        mapping = np.array([[0, 1, 2],
                            [2, 0, 1],
                            [1, 2, 0],
                            [2, 1, 0],
                            [2, 0, 1],
                            [0, 2, 1]])
        expect = [[3, 1, 0],
                  [2, 1, 1],
                  [1, 1, 2],
                  [2, 0, 0],
                  [0, 0, 0],
                  [0, 2, 1]]
        actual = ac.map_alleles(mapping)
        aeq(expect, actual)

    def test_concatenate(self):
        a = np.array(allele_counts_data, dtype=np.int8)
        ac1 = self.setup_instance(a)
        ac2 = self.setup_instance(a)
        for axis in 0, 1:
            actual = ac1.concatenate(ac2, axis=axis)
            expect = np.concatenate([a, a], axis=axis)
            aeq(expect, actual)


# noinspection PyNoneFunctionAssignment
class GenotypeAlleleCountsArrayInterface(object):

    def setup_instance(self, data, dtype=None):
        # to be implemented in sub-classes
        pass

    # to be overriden in sub-classes
    _class = None

    # basic properties and data access methods
    ##########################################

    def test_properties(self):
        # Test the instance properties.

        # diploid data
        g = self.setup_instance(diploid_genotype_ac_data)
        eq(3, g.ndim)
        eq((5, 3, 3), g.shape)
        eq(5, g.n_variants)
        eq(3, g.n_samples)
        eq(3, g.n_alleles)

        # polyploid data
        g = self.setup_instance(triploid_genotype_ac_data)
        eq(3, g.ndim)
        eq((4, 3, 3), g.shape)
        eq(4, g.n_variants)
        eq(3, g.n_samples)
        eq(3, g.n_alleles)

    def test_array_like(self):
        # Test that an instance is array-like, in that it can be used as
        # input argument to np.array(). I.e., there is a standard way to get
        # a vanilla numpy array representation of the data.

        # diploid data
        g = self.setup_instance(diploid_genotype_ac_data)
        a = np.array(g, copy=False)
        aeq(diploid_genotype_ac_data, a)

        # polyploid data
        g = self.setup_instance(triploid_genotype_ac_data)
        a = np.array(g, copy=False)
        aeq(triploid_genotype_ac_data, a)

    def test_slice(self):
        # Test contiguous slicing and item indexing.

        g = self.setup_instance(diploid_genotype_ac_data)

        # row slice
        s = g[1:]
        aeq(diploid_genotype_ac_data[1:], s)
        # slice which preserves dimensionality should return GenotypeArray
        eq(4, s.n_variants)
        eq(3, s.n_samples)
        eq(3, s.n_alleles)

        # col slice
        s = g[:, 1:]
        aeq(np.array(diploid_genotype_ac_data)[:, 1:], s)
        # slice which preserves dimensionality should return GenotypeArray
        eq(5, s.n_variants)
        eq(2, s.n_samples)
        eq(3, s.n_alleles)

        # row index
        s = g[0]
        aeq(diploid_genotype_ac_data[0], s)
        assert not hasattr(s, 'n_variants')

        # col index
        s = g[:, 0]
        aeq(np.array(diploid_genotype_ac_data)[:, 0], s)
        assert not hasattr(s, 'n_samples')

        # allele index
        s = g[:, :, 0]
        aeq(np.array(diploid_genotype_ac_data)[:, :, 0], s)
        assert not hasattr(s, 'n_alleles')

        # item
        s = g[0, 0, 0]
        eq(2, s)

    def test_take(self):
        # Test the take() method.

        g = self.setup_instance(diploid_genotype_ac_data)

        # take variants
        indices = [0, 2]
        t = g.take(indices, axis=0)
        eq(2, t.n_variants)
        eq(g.n_samples, t.n_samples)
        eq(g.n_alleles, t.n_alleles)
        expect = np.array(diploid_genotype_ac_data).take(indices, axis=0)
        aeq(expect, t)

        # take samples
        indices = [0, 2]
        t = g.take(indices, axis=1)
        eq(g.n_variants, t.n_variants)
        eq(2, t.n_samples)
        eq(g.n_alleles, t.n_alleles)
        expect = np.array(diploid_genotype_ac_data).take(indices, axis=1)
        aeq(expect, t)

        # take samples not in original order
        indices = [2, 0]
        t = g.take(indices, axis=1)
        eq(g.n_variants, t.n_variants)
        eq(2, t.n_samples)
        eq(g.n_alleles, t.n_alleles)
        expect = np.array(diploid_genotype_ac_data).take(indices, axis=1)
        aeq(expect, t)

    def test_compress(self):
        # Test the compress() method.

        g = self.setup_instance(diploid_genotype_ac_data)

        # compress variants
        condition = [True, False, True, False, False]
        t = g.compress(condition, axis=0)
        eq(2, t.n_variants)
        eq(g.n_samples, t.n_samples)
        eq(g.n_alleles, t.n_alleles)
        expect = np.array(diploid_genotype_ac_data).compress(condition, axis=0)
        aeq(expect, t)

        # compress samples
        condition = [True, False, True]
        t = g.compress(condition, axis=1)
        eq(g.n_variants, t.n_variants)
        eq(2, t.n_samples)
        eq(g.n_alleles, t.n_alleles)
        expect = np.array(diploid_genotype_ac_data).compress(condition, axis=1)
        aeq(expect, t)

    def test_subset(self):
        # Test the subset() method.

        g = self.setup_instance(diploid_genotype_ac_data)

        # test with indices
        sel0 = [0, 2]
        sel1 = [0, 2]
        s = g.subset(sel0, sel1)
        expect = np.array(diploid_genotype_ac_data)\
            .take(sel0, axis=0)\
            .take(sel1, axis=1)
        aeq(expect, s)

        # test with condition
        sel0 = [True, False, True, False, False]
        sel1 = [True, False, True]
        s = g.subset(sel0, sel1)
        expect = np.array(diploid_genotype_ac_data)\
            .compress(sel0, axis=0)\
            .compress(sel1, axis=1)
        aeq(expect, s)

        # mix and match
        sel0 = [0, 2]
        sel1 = [True, False, True]
        s = g.subset(sel0, sel1)
        expect = np.array(diploid_genotype_ac_data)\
            .take(sel0, axis=0)\
            .compress(sel1, axis=1)
        aeq(expect, s)

        # mix and match
        sel0 = [True, False, True, False, False]
        sel1 = [0, 2]
        s = g.subset(sel0, sel1)
        expect = np.array(diploid_genotype_ac_data)\
            .compress(sel0, axis=0)\
            .take(sel1, axis=1)
        aeq(expect, s)

        # check argument type inference
        sel0 = list(range(g.shape[0]))
        sel1 = None
        s = g.subset(sel0, sel1)
        expect = np.array(diploid_genotype_ac_data)
        aeq(expect, s)

        # check argument type inference
        sel0 = None
        sel1 = list(range(g.shape[1]))
        s = g.subset(sel0, sel1)
        expect = np.array(diploid_genotype_ac_data)
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
        actual = self.setup_instance(diploid_genotype_ac_data).is_called()
        aeq(expect, actual)

        # polyploid
        expect = np.array([[1, 1, 0],
                           [1, 1, 0],
                           [1, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = self.setup_instance(triploid_genotype_ac_data).is_called()
        aeq(expect, actual)

    def test_is_missing(self):

        # diploid
        expect = np.array([[0, 0, 1],
                           [0, 0, 1],
                           [0, 0, 1],
                           [0, 1, 1],
                           [1, 1, 1]], dtype='b1')
        actual = self.setup_instance(diploid_genotype_ac_data).is_missing()
        aeq(expect, actual)

        # polyploid
        expect = np.array([[0, 0, 1],
                           [0, 0, 1],
                           [0, 1, 1],
                           [1, 1, 1]], dtype='b1')
        actual = self.setup_instance(triploid_genotype_ac_data).is_missing()
        aeq(expect, actual)

    def test_is_hom(self):

        # diploid
        expect = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0],
                           [1, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = self.setup_instance(diploid_genotype_ac_data).is_hom()
        aeq(expect, actual)

        # polyploid
        expect = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = self.setup_instance(triploid_genotype_ac_data).is_hom()
        aeq(expect, actual)

    def test_is_hom_ref(self):

        # diploid
        expect = np.array([[1, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = self.setup_instance(diploid_genotype_ac_data).is_hom(allele=0)
        aeq(expect, actual)
        actual = self.setup_instance(diploid_genotype_ac_data).is_hom_ref()
        aeq(expect, actual)

        # polyploid
        expect = np.array([[1, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = self.setup_instance(triploid_genotype_ac_data).is_hom(allele=0)
        aeq(expect, actual)
        actual = self.setup_instance(triploid_genotype_ac_data).is_hom_ref()
        aeq(expect, actual)

    def test_is_hom_alt(self):

        # diploid
        expect = np.array([[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0],
                           [1, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = self.setup_instance(diploid_genotype_ac_data).is_hom_alt()
        aeq(expect, actual)

        # polyploid
        expect = np.array([[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = self.setup_instance(triploid_genotype_ac_data).is_hom_alt()
        aeq(expect, actual)

    def test_is_hom_1(self):

        # diploid
        expect = np.array([[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = self.setup_instance(diploid_genotype_ac_data).is_hom(allele=1)
        aeq(expect, actual)

        # polyploid
        expect = np.array([[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = self.setup_instance(triploid_genotype_ac_data).is_hom(allele=1)
        aeq(expect, actual)

    def test_is_het(self):

        # diploid
        expect = np.array([[0, 1, 0],
                           [1, 0, 0],
                           [1, 1, 0],
                           [0, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = self.setup_instance(diploid_genotype_ac_data).is_het()
        aeq(expect, actual)

        # polyploid
        expect = np.array([[0, 1, 0],
                           [1, 0, 0],
                           [1, 0, 0],
                           [0, 0, 0]], dtype='b1')
        actual = self.setup_instance(triploid_genotype_ac_data).is_het()
        aeq(expect, actual)

    def test_count_alleles(self):

        # diploid
        g = self.setup_instance(diploid_genotype_ac_data)
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
        g = self.setup_instance(triploid_genotype_ac_data)
        expect = np.array([[5, 1, 0],
                           [1, 5, 0],
                           [1, 1, 1],
                           [0, 0, 0]])
        actual = g.count_alleles()
        aeq(expect, actual)
        eq(4, actual.n_variants)
        eq(3, actual.n_alleles)

    def test_count_alleles_subpop(self):
        g = self.setup_instance(diploid_genotype_ac_data)
        expect = np.array([[2, 0, 0],
                           [1, 0, 1],
                           [1, 1, 0],
                           [0, 0, 2],
                           [0, 0, 0]])
        actual = g.count_alleles(subpop=[0, 2])
        aeq(expect, actual)
        eq(5, actual.n_variants)
        eq(3, actual.n_alleles)


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
        a = np.rec.array(variant_table_data, dtype=variant_table_dtype)
        vt = self.setup_instance(a)
        eq(5, len(vt))
        eq(5, vt.n_variants)
        assert_sequence_equal(variant_table_names, vt.names)

    def test_array_like(self):
        # Test that an instance is array-like, in that it can be used as
        # input argument to np.rec.array(). I.e., there is a standard way to
        # get a vanilla numpy array representation of the data.

        a = np.rec.array(variant_table_data, dtype=variant_table_dtype)
        vt = self.setup_instance(a)
        b = np.asarray(vt)
        aeq(a, b)

    def test_get_item(self):
        a = np.rec.array(variant_table_data, dtype=variant_table_dtype)
        vt = self.setup_instance(a)

        # total slice
        s = vt[:]
        eq(5, s.n_variants)
        eq(variant_table_names, s.names)
        aeq(a, s)

        # row slice
        s = vt[1:]
        eq(4, s.n_variants)
        eq(variant_table_names, s.names)
        aeq(a[1:], s)

        # row index
        s = vt[1]
        # compare item by item
        for x, y in zip(variant_table_data[1], s):
            if np.isscalar(x):
                assert_almost_equal(x, y)
            else:
                eq(tuple(x), tuple(y))

        # column access
        s = vt['CHROM']
        aeq(a['CHROM'], s)

        # multi-column access
        s = vt[['CHROM', 'POS']]
        eq(5, s.n_variants)
        assert_sequence_equal(('CHROM', 'POS'), s.names)
        aeq(a[['CHROM', 'POS']], s)

    def test_take(self):
        a = np.rec.array(variant_table_data, dtype=variant_table_dtype)
        vt = self.setup_instance(a)
        indices = [0, 2]
        t = vt.take(indices)
        expect = a.take(indices)
        aeq(expect, t)
        eq(2, t.n_variants)
        eq(variant_table_names, t.names)

    def test_compress(self):
        a = np.rec.array(variant_table_data, dtype=variant_table_dtype)
        vt = self.setup_instance(a)
        condition = [True, False, True, False, False]
        t = vt.compress(condition)
        expect = a.compress(condition)
        aeq(expect, t)
        eq(2, t.n_variants)
        assert_sequence_equal(variant_table_names, t.names)

    def test_eval(self):
        a = np.rec.array(variant_table_data, dtype=variant_table_dtype)
        vt = self.setup_instance(a)

        expr = '(DP > 30) & (QD < 4)'
        for vm in 'numexpr', 'python':
            r = vt.eval(expr, vm=vm)
            aeq([False, False, True, False, True], r)

    def test_query(self):
        a = np.rec.array(variant_table_data, dtype=variant_table_dtype)
        vt = self.setup_instance(a)

        query = '(DP > 30) & (QD < 4)'
        for vm in 'numexpr', 'python':
            r = vt.query(query, vm=vm)
            aeq(a.take([2, 4]), r)

    def test_index(self):
        a = np.rec.array(variant_table_data, dtype=variant_table_dtype)

        # multi chromosome/contig
        vt = self.setup_instance(a, index=('CHROM', 'POS'))
        eq(slice(0, 2), vt.index.locate_key(b'chr1'))
        eq(1, vt.index.locate_key(b'chr1', 7))
        eq(slice(2, 4), vt.index.locate_range(b'chr2', 3, 9))

        # single chromosome/contig index
        vt = self.setup_instance(a[2:4][['POS', 'DP', 'QD']], index='POS')
        eq(0, vt.index.locate_key(3))
        eq(slice(0, 2), vt.index.locate_range(3, 9))

    def test_to_vcf(self):

        # define columns
        chrom = [b'chr1', b'chr1', b'chr2', b'chr2', b'chr3']
        pos = [2, 6, 3, 8, 1]
        # noinspection PyShadowingBuiltins
        id = ['a', 'b', 'c', 'd', 'e']
        ref = [b'A', b'C', b'T', b'G', b'N']
        alt = [(b'T', b'.'),
               (b'G', b'.'),
               (b'A', b'C'),
               (b'C', b'A'),
               (b'X', b'.')]
        qual = [1.2, 2.3, 3.4, 4.5, 5.6]
        filter_qd = [True, True, True, False, False]
        filter_dp = [True, False, True, False, False]
        dp = [12, 23, 34, 45, 56]
        qd = [12.3, 23.4, 34.5, 45.6, 56.7]
        flg = [True, False, True, False, True]
        ac = [(1, -1), (3, -1), (5, 6), (7, 8), (9, -1)]
        xx = [(1.2, 2.3), (3.4, 4.5), (5.6, 6.7), (7.8, 8.9), (9.0, 9.9)]

        # compile into recarray
        columns = [chrom, pos, id, ref, alt, qual, filter_dp, filter_qd,
                   dp, qd, flg, ac, xx]
        records = list(zip(*columns))
        dtype = [('chrom', 'S4'),
                 ('pos', 'u4'),
                 ('ID', 'S1'),
                 ('ref', 'S1'),
                 ('alt', ('S1', 2)),
                 ('qual', 'f4'),
                 ('filter_dp', bool),
                 ('filter_qd', bool),
                 ('dp', int),
                 ('qd', float),
                 ('flg', bool),
                 ('ac', (int, 2)),
                 ('xx', (float, 2))]
        a = np.array(records, dtype=dtype)

        # wrap
        vt = self.setup_instance(a)

        # check dtype
        eq(a.dtype, vt.dtype)

        # expectation
        expect_vcf = """##fileformat=VCFv4.1
##fileDate={today}
##source=scikit-allel-{version}
##INFO=<ID=DP,Number=1,Type=Integer,Description="">
##INFO=<ID=QD,Number=1,Type=Float,Description="">
##INFO=<ID=ac,Number=A,Type=Integer,Description="Allele counts">
##INFO=<ID=flg,Number=0,Type=Flag,Description="">
##INFO=<ID=xx,Number=2,Type=Float,Description="">
##FILTER=<ID=QD,Description="">
##FILTER=<ID=dp,Description="Low depth">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
chr1\t2\ta\tA\tT\t1.2\tQD;dp\tDP=12;QD=12.3;ac=1;flg;xx=1.2,2.3
chr1\t6\tb\tC\tG\t2.3\tQD\tDP=23;QD=23.4;ac=3;xx=3.4,4.5
chr2\t3\tc\tT\tA,C\t3.4\tQD;dp\tDP=34;QD=34.5;ac=5,6;flg;xx=5.6,6.7
chr2\t8\td\tG\tC,A\t4.5\tPASS\tDP=45;QD=45.6;ac=7,8;xx=7.8,8.9
chr3\t1\te\tN\tX\t5.6\tPASS\tDP=56;QD=56.7;ac=9;flg;xx=9.0,9.9
""".format(today=date.today().strftime('%Y%m%d'), version=allel.__version__)

        # create a named temp file
        f = tempfile.NamedTemporaryFile(delete=False)
        f.close()

        # write the VCF
        rename = {'dp': 'DP', 'qd': 'QD', 'filter_qd': 'QD'}
        fill = {'ALT': b'.', 'ac': -1}
        number = {'ac': 'A'}
        description = {'ac': 'Allele counts', 'filter_dp': 'Low depth'}
        vt.to_vcf(f.name, rename=rename, fill=fill, number=number,
                  description=description)

        # check the result
        actual_vcf = open(f.name).read()
        # compare line-by-line
        for l1, l2 in zip(expect_vcf.split('\n'), actual_vcf.split('\n')):
            print('expect:', l1)
            print('actual:', l2)
            eq(l1, l2)

    def test_to_vcf_no_filters(self):

        # define columns
        chrom = [b'chr1', b'chr1', b'chr2', b'chr2', b'chr3']
        pos = [2, 6, 3, 8, 1]
        # noinspection PyShadowingBuiltins
        id = ['a', 'b', 'c', 'd', 'e']
        ref = [b'A', b'C', b'T', b'G', b'N']
        alt = [(b'T', b'.'),
               (b'G', b'.'),
               (b'A', b'C'),
               (b'C', b'A'),
               (b'X', b'.')]
        qual = [1.2, 2.3, 3.4, 4.5, 5.6]
        dp = [12, 23, 34, 45, 56]
        qd = [12.3, 23.4, 34.5, 45.6, 56.7]
        flg = [True, False, True, False, True]
        ac = [(1, -1), (3, -1), (5, 6), (7, 8), (9, -1)]
        xx = [(1.2, 2.3), (3.4, 4.5), (5.6, 6.7), (7.8, 8.9), (9.0, 9.9)]

        # compile into recarray
        columns = [chrom, pos, id, ref, alt, qual, dp, qd, flg, ac, xx]
        records = list(zip(*columns))
        dtype = [('chrom', 'S4'),
                 ('pos', 'u4'),
                 ('ID', 'S1'),
                 ('ref', 'S1'),
                 ('alt', ('S1', 2)),
                 ('qual', 'f4'),
                 ('dp', int),
                 ('qd', float),
                 ('flg', bool),
                 ('ac', (int, 2)),
                 ('xx', (float, 2))]
        a = np.array(records, dtype=dtype)

        # wrap
        vt = self.setup_instance(a)

        # check dtype
        eq(a.dtype, vt.dtype)

        # expectation
        expect_vcf = """##fileformat=VCFv4.1
##fileDate={today}
##source=scikit-allel-{version}
##INFO=<ID=DP,Number=1,Type=Integer,Description="">
##INFO=<ID=QD,Number=1,Type=Float,Description="">
##INFO=<ID=ac,Number=A,Type=Integer,Description="Allele counts">
##INFO=<ID=flg,Number=0,Type=Flag,Description="">
##INFO=<ID=xx,Number=2,Type=Float,Description="">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
chr1\t2\ta\tA\tT\t1.2\t.\tDP=12;QD=12.3;ac=1;flg;xx=1.2,2.3
chr1\t6\tb\tC\tG\t2.3\t.\tDP=23;QD=23.4;ac=3;xx=3.4,4.5
chr2\t3\tc\tT\tA,C\t3.4\t.\tDP=34;QD=34.5;ac=5,6;flg;xx=5.6,6.7
chr2\t8\td\tG\tC,A\t4.5\t.\tDP=45;QD=45.6;ac=7,8;xx=7.8,8.9
chr3\t1\te\tN\tX\t5.6\t.\tDP=56;QD=56.7;ac=9;flg;xx=9.0,9.9
""".format(today=date.today().strftime('%Y%m%d'), version=allel.__version__)

        # create a named temp file
        f = tempfile.NamedTemporaryFile(delete=False)
        f.close()

        # write the VCF
        rename = {'dp': 'DP', 'qd': 'QD'}
        fill = {'ALT': b'.', 'ac': -1}
        number = {'ac': 'A'}
        description = {'ac': 'Allele counts'}
        vt.to_vcf(f.name, rename=rename, fill=fill, number=number,
                  description=description)

        # check the result
        actual_vcf = open(f.name).read()
        # compare line-by-line
        for l1, l2 in zip(expect_vcf.split('\n'), actual_vcf.split('\n')):
            print('expect:', l1)
            print('actual:', l2)
            eq(l1, l2)

    def test_to_vcf_no_info(self):

        # define columns
        chrom = [b'chr1', b'chr1', b'chr2', b'chr2', b'chr3']
        pos = [2, 6, 3, 8, 1]
        # noinspection PyShadowingBuiltins
        id = ['a', 'b', 'c', 'd', 'e']
        ref = [b'A', b'C', b'T', b'G', b'N']
        alt = [(b'T', b'.'),
               (b'G', b'.'),
               (b'A', b'C'),
               (b'C', b'A'),
               (b'X', b'.')]
        qual = [1.2, 2.3, 3.4, 4.5, 5.6]

        # compile into recarray
        columns = [chrom, pos, id, ref, alt, qual]
        records = list(zip(*columns))
        dtype = [('chrom', 'S4'),
                 ('pos', 'u4'),
                 ('ID', 'S1'),
                 ('ref', 'S1'),
                 ('alt', ('S1', 2)),
                 ('qual', 'f4')]
        a = np.array(records, dtype=dtype)

        # wrap
        vt = self.setup_instance(a)

        # check dtype
        eq(a.dtype, vt.dtype)

        # expectation
        expect_vcf = """##fileformat=VCFv4.1
##fileDate={today}
##source=scikit-allel-{version}
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
chr1\t2\ta\tA\tT\t1.2\t.\t.
chr1\t6\tb\tC\tG\t2.3\t.\t.
chr2\t3\tc\tT\tA,C\t3.4\t.\t.
chr2\t8\td\tG\tC,A\t4.5\t.\t.
chr3\t1\te\tN\tX\t5.6\t.\t.
""".format(today=date.today().strftime('%Y%m%d'), version=allel.__version__)

        # create a named temp file
        f = tempfile.NamedTemporaryFile(delete=False)
        f.close()

        # write the VCF
        fill = {'ALT': b'.'}
        vt.to_vcf(f.name, fill=fill)

        # check the result
        actual_vcf = open(f.name).read()
        # compare line-by-line
        for l1, l2 in zip(expect_vcf.split('\n'), actual_vcf.split('\n')):
            print('expect:', l1)
            print('actual:', l2)
            eq(l1, l2)


class FeatureTableInterface(object):

    _class = None

    def setup_instance(self, data, **kwargs):
        pass

    def test_properties(self):
        a = np.rec.array(feature_table_data, dtype=feature_table_dtype)
        ft = self.setup_instance(a)
        eq(6, len(ft))
        eq(6, ft.n_features)
        assert_sequence_equal(feature_table_names, ft.names)

    def test_get_item(self):
        a = np.rec.array(feature_table_data, dtype=feature_table_dtype)
        ft = self.setup_instance(a)

        # total slice
        s = ft[:]
        eq(6, s.n_features)
        eq(feature_table_names, s.names)
        aeq(a, s)

        # row slice
        s = ft[1:]
        eq(5, s.n_features)
        eq(feature_table_names, s.names)
        aeq(a[1:], s)

        # row index
        s = ft[1]
        # compare item by item
        for x, y in zip(feature_table_data[1], s):
            eq(x, y)

        # column access
        s = ft['seqid']
        aeq(a['seqid'], s)

        # multi-column access
        s = ft[['seqid', 'start', 'end']]
        eq(6, s.n_features)
        assert_sequence_equal(('seqid', 'start', 'end'), s.names)
        aeq(a[['seqid', 'start', 'end']], s)

    def test_take(self):
        a = np.rec.array(feature_table_data, dtype=feature_table_dtype)
        ft = self.setup_instance(a)
        indices = [0, 2]
        t = ft.take(indices)
        expect = a.take(indices)
        aeq(expect, t)
        eq(2, t.n_features)
        assert_sequence_equal(feature_table_names, t.names)

    def test_compress(self):
        a = np.rec.array(feature_table_data, dtype=feature_table_dtype)
        ft = self.setup_instance(a)
        condition = [True, False, True, False, False, False]
        t = ft.compress(condition)
        expect = a.compress(condition)
        aeq(expect, t)
        eq(2, t.n_features)
        assert_sequence_equal(feature_table_names, t.names)

    def test_eval(self):
        a = np.rec.array(feature_table_data, dtype=feature_table_dtype)
        ft = self.setup_instance(a)
        expr = 'type == b"exon"'
        for vm in 'numexpr', 'python':
            r = ft.eval(expr, vm=vm)
            aeq([False, False, True, True, False, False], r)

    def test_query(self):
        a = np.rec.array(feature_table_data, dtype=feature_table_dtype)
        ft = self.setup_instance(a)
        expr = 'type == b"exon"'
        for vm in 'numexpr', 'python':
            r = ft.query(expr, vm=vm)
            aeq(a.take([2, 3]), r)

    def test_from_gff3(self):
        ft = self._class.from_gff3('fixture/sample.gff')
        eq(177, len(ft))

    def test_from_gff3_region(self):
        ft = self._class.from_gff3('fixture/sample.sorted.gff.gz', region='apidb|MAL1')
        eq(44, len(ft))
        ft = self._class.from_gff3('fixture/sample.sorted.gff.gz',
                                   region='apidb|MAL1:42000-50000')
        eq(7, len(ft))
        with assert_raises(ValueError):
            # should be empty
            self._class.from_gff3('fixture/sample.sorted.gff.gz', region='foo')


def test_create_allele_mapping():

    # biallelic case
    ref = [b'A', b'C', b'T', b'G']
    alt = [b'T', b'G', b'C', b'A']
    alleles = [[b'A', b'T'],  # no transformation
               [b'G', b'C'],  # swap
               [b'T', b'A'],  # 1 missing
               [b'A', b'C']]  # 1 missing
    expect = [[0, 1],
              [1, 0],
              [0, -1],
              [-1, 0]]
    actual = allel.create_allele_mapping(ref, alt, alleles)
    aeq(expect, actual)

    # multiallelic case
    ref = [b'A', b'C', b'T']
    alt = [[b'T', b'G'],
           [b'A', b'T'],
           [b'G', b'.']]
    alleles = [[b'A', b'T'],
               [b'C', b'T'],
               [b'G', b'A']]
    expect = [[0, 1, -1],
              [0, -1, 1],
              [-1, 0, -1]]
    actual = allel.create_allele_mapping(ref, alt, alleles)
    aeq(expect, actual)
