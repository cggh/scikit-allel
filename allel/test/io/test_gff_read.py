# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import os


from allel.io.gff import gff3_to_dataframe


def fixture_path(fn):
    return os.path.join(os.path.dirname(__file__), os.pardir, 'data', fn)


expected_columns = ['seqid', 'source', 'type', 'start', 'end', 'score', 'strand', 'phase']


def test_gff3_to_dataframe():
    fn = fixture_path('sample.gff')
    df = gff3_to_dataframe(fn)
    assert 177 == len(df)
    assert expected_columns == df.columns.tolist()


def test_gff3_to_dataframe_region():
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
