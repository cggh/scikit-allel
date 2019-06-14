# -*- coding: utf-8 -*-
# third-party imports
import numpy as np
import unittest
import pytest


# internal imports
from allel.test.tools import assert_array_equal as aeq
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
        with pytest.raises(TypeError):
            # noinspection PyArgumentList
            VariantTable()

    def test_get_item_types(self):
        vt = VariantTable(variant_table_data, dtype=variant_table_dtype)

        # row slice
        s = vt[1:]
        assert isinstance(s, VariantTable)

        # row index
        s = vt[0]
        assert isinstance(s, np.record)
        assert not isinstance(s, VariantTable)

        # col access
        s = vt['CHROM']
        assert isinstance(s, np.ndarray)
        assert not isinstance(s, VariantTable)
        s = vt[['CHROM', 'POS']]
        assert isinstance(s, VariantTable)

    def test_take(self):
        a = np.rec.array(variant_table_data,
                         dtype=variant_table_dtype)
        vt = VariantTable(a)
        # take variants not in original order
        indices = [2, 0]
        t = vt.take(indices)
        assert 2 == t.n_variants
        expect = a.take(indices)
        aeq(expect, t)


# noinspection PyMethodMayBeStatic
class FeatureTableTests(FeatureTableInterface, unittest.TestCase):

    _class = FeatureTable

    def setup_instance(self, data, **kwargs):
        return FeatureTable(data, **kwargs)

    def test_constructor(self):

        # missing data arg
        with pytest.raises(TypeError):
            # noinspection PyArgumentList
            FeatureTable()

    def test_get_item_types(self):
        ft = FeatureTable(feature_table_data, dtype=feature_table_dtype)

        # row slice
        s = ft[1:]
        assert isinstance(s, FeatureTable)

        # row index
        s = ft[0]
        assert isinstance(s, np.record)
        assert not isinstance(s, FeatureTable)

        # col access
        s = ft['seqid']
        assert isinstance(s, np.ndarray)
        assert not isinstance(s, FeatureTable)
        s = ft[['seqid', 'start', 'end']]
        assert isinstance(s, FeatureTable)
