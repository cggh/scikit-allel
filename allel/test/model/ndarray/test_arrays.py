# -*- coding: utf-8 -*-
# third-party imports
import numpy as np
import unittest
import pytest


# internal imports
from allel.test.tools import assert_array_equal as aeq
from allel import GenotypeArray, HaplotypeArray, AlleleCountsArray, GenotypeVector, \
    GenotypeAlleleCountsArray, GenotypeAlleleCountsVector
from allel.test.model.test_api import GenotypeArrayInterface, HaplotypeArrayInterface, \
    diploid_genotype_data, triploid_genotype_data, haplotype_data, \
    AlleleCountsArrayInterface, allele_counts_data, GenotypeAlleleCountsArrayInterface, \
    diploid_genotype_ac_data, triploid_genotype_ac_data


# noinspection PyMethodMayBeStatic
class GenotypeArrayTests(GenotypeArrayInterface, unittest.TestCase):

    _class = GenotypeArray

    def setup_instance(self, data, dtype=None):
        return GenotypeArray(data, dtype=dtype)

    def test_constructor(self):

        # missing data arg
        with pytest.raises(TypeError):
            # noinspection PyArgumentList
            GenotypeArray()

        # data has wrong dtype
        data = 'foo bar'
        with pytest.raises(TypeError):
            GenotypeArray(data)

        # data has wrong dtype
        data = [4., 5., 3.7]
        with pytest.raises(TypeError):
            GenotypeArray(data)

        # data has wrong dimensions
        data = [1, 2, 3]
        with pytest.raises(TypeError):
            GenotypeArray(data)

        # data has wrong dimensions
        data = [[1, 2], [3, 4]]  # use HaplotypeArray instead
        with pytest.raises(TypeError):
            GenotypeArray(data)

        # diploid data (typed)
        g = GenotypeArray(diploid_genotype_data, dtype='i1')
        aeq(diploid_genotype_data, g)
        assert np.int8 == g.dtype

        # polyploid data (typed)
        g = GenotypeArray(triploid_genotype_data, dtype='i1')
        aeq(triploid_genotype_data, g)
        assert np.int8 == g.dtype

    def test_slice_types(self):

        g = GenotypeArray(diploid_genotype_data, dtype='i1')

        # row slice
        s = g[1:]
        assert isinstance(s, GenotypeArray)

        # col slice
        s = g[:, 1:]
        assert isinstance(s, GenotypeArray)

        # row index
        s = g[0]
        assert isinstance(s, GenotypeVector)
        assert not isinstance(s, GenotypeArray)

        # col index
        s = g[:, 0]
        assert isinstance(s, GenotypeVector)
        assert not isinstance(s, GenotypeArray)

        # ploidy index
        s = g[:, :, 0]
        assert isinstance(s, np.ndarray)
        assert not isinstance(s, GenotypeArray)

        # item
        s = g[0, 0, 0]
        assert isinstance(s, np.int8)
        assert not isinstance(s, GenotypeArray)

    def test_haploidify_samples(self):

        # diploid
        g = GenotypeArray([[[0, 1], [2, 3]],
                           [[4, 5], [6, 7]],
                           [[8, 9], [10, 11]]], dtype='i1')
        h = g.haploidify_samples()
        assert 2 == h.ndim
        assert 3 == h.n_variants
        assert 2 == h.n_haplotypes
        assert np.int8 == h.dtype
        for i in range(g.n_variants):
            for j in range(g.n_samples):
                self.assertIn(h[i, j], set(g[i, j]))

        # triploid
        g = GenotypeArray([[[0, 1, 2], [3, 4, 5]],
                           [[6, 7, 8], [9, 10, 11]],
                           [[12, 13, 14], [15, 16, 17]]], dtype='i1')
        h = g.haploidify_samples()
        assert 2 == h.ndim
        assert 3 == h.n_variants
        assert 2 == h.n_haplotypes
        assert np.int8 == h.dtype
        for i in range(g.n_variants):
            for j in range(g.n_samples):
                self.assertIn(h[i, j], set(g[i, j]))

    def test_take(self):
        g = self.setup_instance(diploid_genotype_data)
        # take variants not in original order
        indices = [2, 0]
        t = g.take(indices, axis=0)
        assert 2 == t.n_variants
        assert g.n_samples == t.n_samples
        assert g.ploidy == t.ploidy
        expect = np.array(diploid_genotype_data).take(indices, axis=0)
        aeq(expect, t)

    def test_no_subclass(self):
        g = self.setup_instance(diploid_genotype_data)
        self.assertNotIsInstance(g.reshape((g.shape[0], -1)), GenotypeArray)
        self.assertNotIsInstance(g.flatten(), GenotypeArray)
        self.assertNotIsInstance(g.ravel(), GenotypeArray)
        self.assertNotIsInstance(g.transpose(), GenotypeArray)
        self.assertNotIsInstance(g.T, GenotypeArray)
        self.assertNotIsInstance(g.astype('f4'), GenotypeArray)


# noinspection PyMethodMayBeStatic
class HaplotypeArrayTests(HaplotypeArrayInterface, unittest.TestCase):

    _class = HaplotypeArray

    def setup_instance(self, data, dtype=None):
        return HaplotypeArray(data, dtype=dtype)

    def test_constructor(self):

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

        # haploid data (typed)
        h = HaplotypeArray(haplotype_data, dtype='i1')
        aeq(haplotype_data, h)
        assert np.int8 == h.dtype

    def test_slice_types(self):

        h = HaplotypeArray(haplotype_data, dtype='i1')

        # row slice
        s = h[1:]
        assert isinstance(s, HaplotypeArray)

        # col slice
        s = h[:, 1:]
        assert isinstance(s, HaplotypeArray)

        # row index
        s = h[0]
        assert isinstance(s, np.ndarray)
        assert not isinstance(s, HaplotypeArray)

        # col index
        s = h[:, 0]
        assert isinstance(s, np.ndarray)
        assert not isinstance(s, HaplotypeArray)

        # item
        s = h[0, 0]
        assert isinstance(s, np.int8)
        assert not isinstance(s, HaplotypeArray)


# noinspection PyMethodMayBeStatic
class AlleleCountsArrayTests(AlleleCountsArrayInterface, unittest.TestCase):

    _class = AlleleCountsArray

    def setup_instance(self, data):
        return AlleleCountsArray(data)

    def test_constructor(self):

        # missing data arg
        with pytest.raises(TypeError):
            # noinspection PyArgumentList
            AlleleCountsArray()

        # data has wrong dtype
        data = 'foo bar'
        with pytest.raises(TypeError):
            AlleleCountsArray(data)

        # data has wrong dtype
        data = [4., 5., 3.7]
        with pytest.raises(TypeError):
            AlleleCountsArray(data)

        # data has wrong dimensions
        data = [1, 2, 3]
        with pytest.raises(TypeError):
            AlleleCountsArray(data)

        # data has wrong dimensions
        data = diploid_genotype_data
        with pytest.raises(TypeError):
            AlleleCountsArray(data)

        # valid data (typed)
        ac = AlleleCountsArray(allele_counts_data, dtype='u1')
        aeq(allele_counts_data, ac)
        assert np.uint8 == ac.dtype

    def test_slice_types(self):
        ac = AlleleCountsArray(allele_counts_data, dtype='u1')

        # total slice
        s = ac[:]
        assert isinstance(s, AlleleCountsArray)

        # row slice
        s = ac[1:]
        assert isinstance(s, AlleleCountsArray)

        # col slice
        s = ac[:, 1:]
        assert isinstance(s, np.ndarray)

        # row index
        s = ac[0]
        assert isinstance(s, np.ndarray)

        # col index
        s = ac[:, 0]
        assert isinstance(s, np.ndarray)

        # item
        s = ac[0, 0]
        assert isinstance(s, np.uint8)

    def test_reduce_types(self):
        ac = AlleleCountsArray(allele_counts_data, dtype='u1')

        for m in 'sum', 'max', 'argmax':
            x = getattr(ac, m)(axis=1)
            assert isinstance(x, np.ndarray)


# noinspection PyMethodMayBeStatic
class GenotypeVectorTests(unittest.TestCase):

    def test_properties(self):

        # diploid row
        gv = GenotypeVector(diploid_genotype_data[0])
        assert 2 == gv.ndim
        assert (3, 2) == gv.shape
        assert np.dtype(int) == gv.dtype
        assert 3 == gv.n_calls
        assert 2 == gv.ploidy
        assert 6 == gv.n_allele_calls

        # diploid column
        gv = GenotypeVector(np.array(diploid_genotype_data, dtype='i1')[:, 0])
        assert 2 == gv.ndim
        assert (5, 2) == gv.shape
        assert np.dtype('i1') == gv.dtype
        assert 5 == gv.n_calls
        assert 2 == gv.ploidy
        assert 10 == gv.n_allele_calls

        # triploid row
        gv = GenotypeVector(triploid_genotype_data[0])
        assert 2 == gv.ndim
        assert (3, 3) == gv.shape
        assert np.dtype(int) == gv.dtype
        assert 3 == gv.n_calls
        assert 3 == gv.ploidy
        assert 9 == gv.n_allele_calls

        # triploid column
        gv = GenotypeVector(np.array(triploid_genotype_data, dtype='i1')[:, 0])
        assert 2 == gv.ndim
        assert (4, 3) == gv.shape
        assert np.dtype('i1') == gv.dtype
        assert 4 == gv.n_calls
        assert 3 == gv.ploidy
        assert 12 == gv.n_allele_calls

    def test_input_data(self):

        # ndim 3
        with pytest.raises(TypeError):
            GenotypeVector(diploid_genotype_data)
        with pytest.raises(TypeError):
            GenotypeVector(triploid_genotype_data)

        # ndim 1
        with pytest.raises(TypeError):
            GenotypeVector([0, 1, 0, -1])

        # ndim 1
        with pytest.raises(TypeError):
            GenotypeVector([0, 1, 0, -1])

        # wrong dtype
        with pytest.raises(TypeError):
            GenotypeVector(np.array(diploid_genotype_data, dtype='f4')[:, 0])

    def test_getitem(self):
        gv = GenotypeVector(diploid_genotype_data[0])

        # these should return the same type
        gs = gv[:]
        assert isinstance(gs, GenotypeVector)
        gs = gv[:, :]
        assert isinstance(gs, GenotypeVector)
        gs = gv[...]
        assert isinstance(gs, GenotypeVector)
        gs = gv[0:2]
        assert isinstance(gs, GenotypeVector)
        gs = gv[0:2, :]
        assert isinstance(gs, GenotypeVector)
        gs = gv[np.array([True, False, True], dtype=bool)]
        assert isinstance(gs, GenotypeVector)
        gs = gv[[0, 2]]
        assert isinstance(gs, GenotypeVector)

        # these should return plain array
        gs = gv[:, 0]
        assert not isinstance(gs, GenotypeVector)
        gs = gv[np.newaxis, :2, 0]  # change dimension semantics
        assert not isinstance(gs, GenotypeVector)

    # __getitem__
    # to_html_str
    # _repr_html_
    # mask getter
    # mask setter
    # fill_masked
    # is_called
    # is_missing
    # is_hom
    # is_hom_ref
    # is_hom_alt
    # is_het
    # is_call
    # count_called
    # count_missing
    # count_hom
    # count_hom_ref
    # count_hom_alt
    # count_het
    # count_call
    # to_haplotypes
    # to_n_ref
    # to_n_alt
    # to_allele_counts
    # haploidify
    # to_gt
    # map_alleles


# noinspection PyMethodMayBeStatic
class GenotypeAlleleCountsArrayTests(GenotypeAlleleCountsArrayInterface, unittest.TestCase):

    _class = GenotypeAlleleCountsArray

    def setup_instance(self, data, dtype=None):
        return GenotypeAlleleCountsArray(data, dtype=dtype)

    def test_constructor(self):

        # missing data arg
        with pytest.raises(TypeError):
            # noinspection PyArgumentList
            GenotypeAlleleCountsArray()

        # data has wrong dtype
        data = 'foo bar'
        with pytest.raises(TypeError):
            GenotypeAlleleCountsArray(data)

        # data has wrong dtype
        data = [4., 5., 3.7]
        with pytest.raises(TypeError):
            GenotypeAlleleCountsArray(data)

        # data has wrong dimensions
        data = [1, 2, 3]
        with pytest.raises(TypeError):
            GenotypeAlleleCountsArray(data)

        # data has wrong dimensions
        data = [[1, 2], [3, 4]]  # use HaplotypeArray instead
        with pytest.raises(TypeError):
            GenotypeAlleleCountsArray(data)

        # diploid data (typed)
        g = GenotypeAlleleCountsArray(diploid_genotype_ac_data, dtype='i1')
        aeq(diploid_genotype_ac_data, g)
        assert np.int8 == g.dtype

        # polyploid data (typed)
        g = GenotypeAlleleCountsArray(triploid_genotype_ac_data, dtype='i1')
        aeq(triploid_genotype_ac_data, g)
        assert np.int8 == g.dtype

    def test_slice_types(self):

        g = GenotypeAlleleCountsArray(diploid_genotype_ac_data, dtype='i1')

        # row slice
        s = g[1:]
        assert isinstance(s, GenotypeAlleleCountsArray)

        # col slice
        s = g[:, 1:]
        assert isinstance(s, GenotypeAlleleCountsArray)

        # row index
        s = g[0]
        assert isinstance(s, GenotypeAlleleCountsVector)
        assert not isinstance(s, GenotypeAlleleCountsArray)

        # col index
        s = g[:, 0]
        assert isinstance(s, GenotypeAlleleCountsVector)
        assert not isinstance(s, GenotypeAlleleCountsArray)

        # ploidy index
        s = g[:, :, 0]
        assert isinstance(s, np.ndarray)
        assert not isinstance(s, GenotypeAlleleCountsArray)

        # item
        s = g[0, 0, 0]
        assert isinstance(s, np.int8)
        assert not isinstance(s, GenotypeAlleleCountsArray)
