# -*- coding: utf-8 -*-
import os
import sys


import pytest


from allel.io.gff import gff3_to_dataframe


def fixture_path(fn):
    return os.path.join(os.path.dirname(__file__), os.pardir, 'data', fn)


expected_columns = ['seqid', 'source', 'type', 'start', 'end', 'score', 'strand', 'phase']


def test_gff3_to_dataframe():
    fn = fixture_path('sample.gff')
    df = gff3_to_dataframe(fn)
    assert 177 == len(df)
    assert expected_columns == df.columns.tolist()


def test_gff3_to_dataframe_attributes():
    fn = fixture_path('sample.gff')
    attributes = ['ID', 'description', 'foo']
    df = gff3_to_dataframe(fn, attributes=attributes)
    assert 177 == len(df)
    assert expected_columns + attributes == df.columns.tolist()
    # test correct fill value
    actual = df.iloc[0].foo
    assert '.' == actual, actual


def test_gff3_to_dataframe_region():
    if sys.platform != 'linux':
        pytest.skip('skipping tabix tests if not on linux')
    fn = fixture_path('sample.sorted.gff.gz')
    df = gff3_to_dataframe(fn, region='apidb|MAL1')
    assert 44 == len(df)
    assert expected_columns == df.columns.tolist()
    df = gff3_to_dataframe(fn, region='apidb|MAL1:42000-50000')
    assert 7 == len(df)
    assert expected_columns == df.columns.tolist()
    # should be empty region
    df = gff3_to_dataframe(fn, region='foo')
    assert 0 == len(df)
    assert expected_columns == df.columns.tolist()
