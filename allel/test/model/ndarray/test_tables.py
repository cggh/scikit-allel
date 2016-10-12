# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


# third-party imports
import numpy as np
import unittest
from nose.tools import eq_ as eq, assert_raises, assert_is_instance, \
    assert_not_is_instance
from allel.test.tools import assert_array_equal as aeq


# internal imports
from allel import VariantTable, FeatureTable
from allel.test.model.test_api import VariantTableInterface, variant_table_data, \
    variant_table_dtype, FeatureTableInterface, feature_table_data, feature_table_dtype


# noinspection PyMethodMayBeStatic
class VariantTableTests(VariantTableInterface, unittest.TestCase):

    _class = VariantTable

    def setup_instance(self, data, index=None, **kwargs):
        return VariantTable(data, index=index, **kwargs)

    def test_constructor(self):

        # missing data arg
        with assert_raises(TypeError):
            # noinspection PyArgumentList
            VariantTable()

    def test_get_item_types(self):
        vt = VariantTable(variant_table_data, dtype=variant_table_dtype)

        # row slice
        s = vt[1:]
        assert_is_instance(s, VariantTable)

        # row index
        s = vt[0]
        assert_is_instance(s, np.record)
        assert_not_is_instance(s, VariantTable)

        # col access
        s = vt['CHROM']
        assert_is_instance(s, np.ndarray)
        assert_not_is_instance(s, VariantTable)
        s = vt[['CHROM', 'POS']]
        assert_is_instance(s, VariantTable)

    def test_take(self):
        a = np.rec.array(variant_table_data,
                         dtype=variant_table_dtype)
        vt = VariantTable(a)
        # take variants not in original order
        indices = [2, 0]
        t = vt.take(indices)
        eq(2, t.n_variants)
        expect = a.take(indices)
        aeq(expect, t)


# noinspection PyMethodMayBeStatic
class FeatureTableTests(FeatureTableInterface, unittest.TestCase):

    _class = FeatureTable

    def setup_instance(self, data, **kwargs):
        return FeatureTable(data, **kwargs)

    def test_constructor(self):

        # missing data arg
        with assert_raises(TypeError):
            # noinspection PyArgumentList
            FeatureTable()

    def test_get_item_types(self):
        ft = FeatureTable(feature_table_data, dtype=feature_table_dtype)

        # row slice
        s = ft[1:]
        assert_is_instance(s, FeatureTable)

        # row index
        s = ft[0]
        assert_is_instance(s, np.record)
        assert_not_is_instance(s, FeatureTable)

        # col access
        s = ft['seqid']
        assert_is_instance(s, np.ndarray)
        assert_not_is_instance(s, FeatureTable)
        s = ft[['seqid', 'start', 'end']]
        assert_is_instance(s, FeatureTable)
