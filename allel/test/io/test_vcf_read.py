# -*- coding: utf-8 -*-
import io
import os
import shutil
import itertools
import gzip
import warnings
import tempfile
import atexit


import zarr
import h5py
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pytest
from pytest import approx
from allel.io.vcf_read import (iter_vcf_chunks, read_vcf, vcf_to_zarr, vcf_to_hdf5,
                               vcf_to_npz, ANNTransformer, vcf_to_dataframe, vcf_to_csv,
                               vcf_to_recarray, read_vcf_headers)
from allel.test.tools import compare_arrays


# needed for PY2/PY3 consistent behaviour
warnings.resetwarnings()
warnings.simplefilter('always')


# setup temp dir for testing
tempdir = tempfile.mkdtemp()
atexit.register(shutil.rmtree, tempdir)


def fixture_path(fn):
    return os.path.join(os.path.dirname(__file__), os.pardir, 'data', fn)


def test_read_vcf_chunks():
    vcf_path = fixture_path('sample.vcf')

    fields, samples, headers, it = iter_vcf_chunks(vcf_path, fields='*', chunk_length=4,
                                                   buffer_size=100)

    # check headers
    assert 'q10' in headers.filters
    assert 's50' in headers.filters
    assert 'AA' in headers.infos
    assert 'AC' in headers.infos
    assert 'AF' in headers.infos
    assert 'AN' in headers.infos
    assert 'DB' in headers.infos
    assert 'DP' in headers.infos
    assert 'H2' in headers.infos
    assert 'NS' in headers.infos
    assert 'DP' in headers.formats
    assert 'GQ' in headers.formats
    assert 'GT' in headers.formats
    assert 'HQ' in headers.formats
    assert ['NA00001', 'NA00002', 'NA00003'] == headers.samples
    assert ['NA00001', 'NA00002', 'NA00003'] == samples.tolist()
    assert '1' == headers.infos['AA']['Number']
    assert 'String' == headers.infos['AA']['Type']
    assert 'Ancestral Allele' == headers.infos['AA']['Description']
    assert '2' == headers.formats['HQ']['Number']
    assert 'Integer' == headers.formats['HQ']['Type']
    assert 'Haplotype Quality' == headers.formats['HQ']['Description']

    # check chunk lengths
    chunks = [chunk for chunk, _, _, _ in it]
    assert 3 == len(chunks)
    assert 4 == chunks[0]['variants/POS'].shape[0]
    assert 4 == chunks[1]['variants/POS'].shape[0]
    assert 1 == chunks[2]['variants/POS'].shape[0]

    # check chunk contents
    expected_fields = [
        # fixed fields
        'variants/CHROM',
        'variants/POS',
        'variants/ID',
        'variants/REF',
        'variants/ALT',
        'variants/QUAL',
        'variants/FILTER_PASS',
        'variants/FILTER_q10',
        'variants/FILTER_s50',
        # INFO fields
        'variants/AA',
        'variants/AC',
        'variants/AF',
        'variants/AN',
        'variants/DB',
        'variants/DP',
        'variants/H2',
        'variants/NS',
        # special computed fields
        'variants/altlen',
        'variants/numalt',
        'variants/is_snp',
        # FORMAT fields
        'calldata/GT',
        'calldata/GQ',
        'calldata/HQ',
        'calldata/DP',
    ]
    for chunk in chunks:
        assert sorted(expected_fields) == sorted(chunk.keys())


def test_fields_all():
    vcf_path = fixture_path('sample.vcf')
    callset = read_vcf(vcf_path, fields='*')
    expected_fields = [
        'samples',
        # fixed fields
        'variants/CHROM',
        'variants/POS',
        'variants/ID',
        'variants/REF',
        'variants/ALT',
        'variants/QUAL',
        'variants/FILTER_PASS',
        'variants/FILTER_q10',
        'variants/FILTER_s50',
        # INFO fields
        'variants/AA',
        'variants/AC',
        'variants/AF',
        'variants/AN',
        'variants/DB',
        'variants/DP',
        'variants/H2',
        'variants/NS',
        # special computed fields
        'variants/altlen',
        'variants/numalt',
        'variants/is_snp',
        # FORMAT fields
        'calldata/GT',
        'calldata/GQ',
        'calldata/HQ',
        'calldata/DP',
    ]
    assert sorted(expected_fields) == sorted(callset.keys())


def test_fields_exclude():
    vcf_path = fixture_path('sample.vcf')
    exclude = ['variants/altlen', 'ID', 'calldata/DP']
    callset = read_vcf(vcf_path, fields='*', exclude_fields=exclude)
    expected_fields = [
        'samples',
        # fixed fields
        'variants/CHROM',
        'variants/POS',
        'variants/REF',
        'variants/ALT',
        'variants/QUAL',
        'variants/FILTER_PASS',
        'variants/FILTER_q10',
        'variants/FILTER_s50',
        # INFO fields
        'variants/AA',
        'variants/AC',
        'variants/AF',
        'variants/AN',
        'variants/DB',
        'variants/DP',
        'variants/H2',
        'variants/NS',
        # special computed fields
        'variants/numalt',
        'variants/is_snp',
        # FORMAT fields
        'calldata/GT',
        'calldata/GQ',
        'calldata/HQ',
    ]
    assert sorted(expected_fields) == sorted(callset.keys())


def test_fields_rename():
    vcf_path = fixture_path('sample.vcf')
    rename = {'CHROM': 'variants/chromosome',
              'variants/altlen': 'spam/eggs',
              'calldata/GT': 'foo/bar'}
    callset = read_vcf(vcf_path, fields='*', rename_fields=rename)
    expected_fields = [
        'samples',
        # fixed fields
        'variants/chromosome',
        'variants/POS',
        'variants/ID',
        'variants/REF',
        'variants/ALT',
        'variants/QUAL',
        'variants/FILTER_PASS',
        'variants/FILTER_q10',
        'variants/FILTER_s50',
        # INFO fields
        'variants/AA',
        'variants/AC',
        'variants/AF',
        'variants/AN',
        'variants/DB',
        'variants/DP',
        'variants/H2',
        'variants/NS',
        # special computed fields
        'spam/eggs',
        'variants/numalt',
        'variants/is_snp',
        # FORMAT fields
        'foo/bar',
        'calldata/DP',
        'calldata/GQ',
        'calldata/HQ',
    ]
    assert sorted(expected_fields) == sorted(callset.keys())


def test_fields_rename_clash():
    vcf_path = fixture_path('sample.vcf')

    # rename two fields to the same path
    rename = {'CHROM': 'variants/chromosome',
              'variants/altlen': 'spam/eggs',
              'calldata/GT': 'spam/eggs'}
    with pytest.raises(ValueError):
        read_vcf(vcf_path, fields='*', rename_fields=rename)

    # rename two fields to the same path (case insensitive)
    rename = {'CHROM': 'variants/chromosome',
              'variants/altlen': 'spam/eggs',
              'calldata/GT': 'SPAM/EGGS'}
    with pytest.raises(ValueError):
        read_vcf(vcf_path, fields='*', rename_fields=rename)

    # parent clash
    rename = {'CHROM': 'variants/chromosome',
              'variants/altlen': 'spam/eggs',
              'calldata/GT': 'spam'}
    with pytest.raises(ValueError):
        read_vcf(vcf_path, fields='*', rename_fields=rename)

    # parent clash
    rename = {'CHROM': 'variants/chromosome',
              'variants/altlen': 'spam/eggs',
              'calldata/GT': 'SPAM'}
    with pytest.raises(ValueError):
        read_vcf(vcf_path, fields='*', rename_fields=rename)

    # parent clash
    rename = {'CHROM': 'variants/chromosome',
              'variants/altlen': 'spam',
              'calldata/GT': 'spam/eggs'}
    with pytest.raises(ValueError):
        read_vcf(vcf_path, fields='*', rename_fields=rename)

    # parent clash
    rename = {'CHROM': 'variants/chromosome',
              'variants/altlen': 'spam',
              'calldata/GT': 'SPAM/EGGS'}
    with pytest.raises(ValueError):
        read_vcf(vcf_path, fields='*', rename_fields=rename)


def test_fields_default():
    vcf_path = fixture_path('sample.vcf')
    callset = read_vcf(vcf_path)
    expected_fields = [
        'samples',
        'variants/CHROM',
        'variants/POS',
        'variants/ID',
        'variants/REF',
        'variants/ALT',
        'variants/QUAL',
        'variants/FILTER_PASS',
        'calldata/GT',
    ]
    assert sorted(expected_fields) == sorted(callset.keys())


def test_fields_all_variants():
    vcf_path = fixture_path('sample.vcf')
    callset = read_vcf(vcf_path, fields='variants/*')
    expected_fields = [
        # fixed fields
        'variants/CHROM',
        'variants/POS',
        'variants/ID',
        'variants/REF',
        'variants/ALT',
        'variants/QUAL',
        'variants/FILTER_PASS',
        'variants/FILTER_q10',
        'variants/FILTER_s50',
        # INFO fields
        'variants/AA',
        'variants/AC',
        'variants/AF',
        'variants/AN',
        'variants/DB',
        'variants/DP',
        'variants/H2',
        'variants/NS',
        # special computed fields
        'variants/altlen',
        'variants/numalt',
        'variants/is_snp',
    ]
    assert sorted(expected_fields) == sorted(callset.keys())


def test_fields_info():
    vcf_path = fixture_path('sample.vcf')
    callset = read_vcf(vcf_path, fields='INFO')
    expected_fields = [
        # INFO fields
        'variants/AA',
        'variants/AC',
        'variants/AF',
        'variants/AN',
        'variants/DB',
        'variants/DP',
        'variants/H2',
        'variants/NS',
    ]
    assert sorted(expected_fields) == sorted(callset.keys())


def test_fields_filter():
    vcf_path = fixture_path('sample.vcf')
    callset1 = read_vcf(vcf_path, fields='FILTER')
    expected_fields = [
        'variants/FILTER_PASS',
        'variants/FILTER_q10',
        'variants/FILTER_s50',
    ]
    assert sorted(expected_fields) == sorted(callset1.keys())

    # this has explicit PASS definition in header, shouldn't cause problems
    vcf_path = fixture_path('test16.vcf')
    callset2 = read_vcf(vcf_path, fields='FILTER')
    expected_fields = [
        'variants/FILTER_PASS',
        'variants/FILTER_q10',
        'variants/FILTER_s50',
    ]
    assert sorted(expected_fields) == sorted(callset2.keys())
    for k in callset1.keys():
        assert_array_equal(callset1[k], callset2[k])


def test_fields_all_calldata():
    vcf_path = fixture_path('sample.vcf')
    callset = read_vcf(vcf_path, fields='calldata/*')
    expected_fields = [
        'calldata/GT',
        'calldata/GQ',
        'calldata/HQ',
        'calldata/DP',
    ]
    assert sorted(expected_fields) == sorted(callset.keys())


def test_fields_selected():
    vcf_path = fixture_path('sample.vcf')

    # without samples
    callset = read_vcf(vcf_path,
                       fields=['CHROM', 'variants/POS', 'AC', 'variants/AF', 'GT',
                               'calldata/HQ', 'FILTER_q10', 'variants/numalt'])
    expected_fields = [
        'variants/CHROM',
        'variants/POS',
        'variants/FILTER_q10',
        'variants/AC',
        'variants/AF',
        'variants/numalt',
        # FORMAT fields
        'calldata/GT',
        'calldata/HQ',
    ]
    assert sorted(expected_fields) == sorted(callset.keys())

    # with samples
    callset = read_vcf(vcf_path,
                       fields=['CHROM', 'variants/POS', 'AC', 'variants/AF', 'GT',
                               'calldata/HQ', 'FILTER_q10', 'variants/numalt', 'samples'],
                       chunk_length=4, buffer_size=100)
    expected_fields = [
        'samples',
        'variants/CHROM',
        'variants/POS',
        'variants/FILTER_q10',
        'variants/AC',
        'variants/AF',
        'variants/numalt',
        # FORMAT fields
        'calldata/GT',
        'calldata/HQ',
    ]
    assert sorted(expected_fields) == sorted(callset.keys())


def test_fields_dups():
    vcf_path = fixture_path('sample.vcf')

    # silently collapse dups
    callset = read_vcf(vcf_path,
                       fields=['CHROM', 'variants/CHROM', 'variants/AF', 'variants/AF',
                               'numalt', 'variants/numalt'])
    expected_fields = [
        'variants/CHROM',
        'variants/AF',
        'variants/numalt'
    ]
    assert sorted(expected_fields) == sorted(callset.keys())


def test_fields_dups_case_insensitive():
    vcf_path = fixture_path('altlen.vcf')

    # allow case-insensitive dups here (but not in vcf_to_zarr)
    callset = read_vcf(vcf_path, fields=['ALTLEN', 'altlen'])
    expected_fields = [
        'variants/ALTLEN',
        'variants/altlen',
    ]
    assert sorted(expected_fields) == sorted(callset.keys())


def _test_read_vcf_content(vcf, chunk_length, buffer_size):

    # object dtype for strings

    if isinstance(vcf, str):
        input_file = vcf
        close = False
    else:
        input_file = vcf()
        close = True

    callset = read_vcf(input_file,
                       fields='*',
                       chunk_length=chunk_length,
                       buffer_size=buffer_size,
                       types={'calldata/DP': 'object'})
    if close:
        input_file.close()

    # samples
    assert (3,) == callset['samples'].shape
    assert 'O' == callset['samples'].dtype.kind
    assert ['NA00001', 'NA00002', 'NA00003'] == callset['samples'].tolist()

    # fixed fields
    assert (9,) == callset['variants/CHROM'].shape
    assert np.dtype(object) == callset['variants/CHROM'].dtype
    assert '19' == callset['variants/CHROM'][0]
    assert (9,) == callset['variants/POS'].shape
    assert 111 == callset['variants/POS'][0]
    assert (9,) == callset['variants/ID'].shape
    assert np.dtype(object) == callset['variants/ID'].dtype
    assert 'rs6054257' == callset['variants/ID'][2]
    assert (9,) == callset['variants/REF'].shape
    assert np.dtype(object) == callset['variants/REF'].dtype
    assert 'A' == callset['variants/REF'][0]
    assert (9, 3) == callset['variants/ALT'].shape
    assert np.dtype(object) == callset['variants/ALT'].dtype
    assert 'ATG' == callset['variants/ALT'][8, 1]
    assert (9,) == callset['variants/QUAL'].shape
    assert 10.0 == callset['variants/QUAL'][1]
    assert (9,) == callset['variants/FILTER_PASS'].shape
    assert callset['variants/FILTER_PASS'][2]
    assert not callset['variants/FILTER_PASS'][3]
    assert (9,) == callset['variants/FILTER_q10'].shape
    assert callset['variants/FILTER_q10'][3]

    # INFO fields
    assert 3 == callset['variants/NS'][2]
    assert .5 == callset['variants/AF'][2, 0]
    assert callset['variants/DB'][2]
    assert (3, 1, -1) == tuple(callset['variants/AC'][6])

    # test calldata content
    assert (9, 3, 2) == callset['calldata/GT'].shape
    assert (0, 0) == tuple(callset['calldata/GT'][0, 0])
    assert (-1, -1) == tuple(callset['calldata/GT'][6, 2])
    assert (-1, -1) == tuple(callset['calldata/GT'][7, 2])
    assert (9, 3, 2) == callset['calldata/HQ'].shape
    assert (10, 15) == tuple(callset['calldata/HQ'][0, 0])
    assert (9, 3) == callset['calldata/DP'].shape
    assert np.dtype(object) == callset['calldata/DP'].dtype
    assert ('4', '2', '3') == tuple(callset['calldata/DP'][6])

    # String (S) dtype

    if isinstance(vcf, str):
        input_file = vcf
        close = False
    else:
        input_file = vcf()
        close = True

    types = {'CHROM': 'S12', 'ID': 'S20', 'REF': 'S20', 'ALT': 'S20', 'calldata/DP': 'S3',
             'samples': 'S20'}
    callset = read_vcf(input_file, fields='*', chunk_length=chunk_length,
                       buffer_size=buffer_size, types=types)
    if close:
        input_file.close()

    # samples
    assert (3,) == callset['samples'].shape
    assert 'S' == callset['samples'].dtype.kind
    assert [b'NA00001', b'NA00002', b'NA00003'] == callset['samples'].tolist()

    # fixed fields
    assert (9,) == callset['variants/CHROM'].shape
    assert 'S' == callset['variants/CHROM'].dtype.kind
    assert b'19' == callset['variants/CHROM'][0]
    assert (9,) == callset['variants/POS'].shape
    assert 111 == callset['variants/POS'][0]
    assert (9,) == callset['variants/ID'].shape
    assert 'S' == callset['variants/ID'].dtype.kind
    assert b'rs6054257' == callset['variants/ID'][2]
    assert (9,) == callset['variants/REF'].shape
    assert b'A' == callset['variants/REF'][0]
    assert 'S' == callset['variants/REF'].dtype.kind
    assert (9, 3) == callset['variants/ALT'].shape
    assert b'ATG' == callset['variants/ALT'][8, 1]
    assert 'S' == callset['variants/ALT'].dtype.kind
    assert (9,) == callset['variants/QUAL'].shape
    assert 10.0 == callset['variants/QUAL'][1]
    assert (9,) == callset['variants/FILTER_PASS'].shape
    assert callset['variants/FILTER_PASS'][2]
    assert not callset['variants/FILTER_PASS'][3]
    assert (9,) == callset['variants/FILTER_q10'].shape
    assert callset['variants/FILTER_q10'][3]

    # INFO fields
    assert 3 == callset['variants/NS'][2]
    assert .5 == callset['variants/AF'][2, 0]
    assert callset['variants/DB'][2]
    assert (3, 1, -1) == tuple(callset['variants/AC'][6])

    # test calldata content
    assert (9, 3, 2) == callset['calldata/GT'].shape
    assert (0, 0) == tuple(callset['calldata/GT'][0, 0])
    assert (-1, -1) == tuple(callset['calldata/GT'][6, 2])
    assert (-1, -1) == tuple(callset['calldata/GT'][7, 2])
    assert (9, 3, 2) == callset['calldata/HQ'].shape
    assert (10, 15) == tuple(callset['calldata/HQ'][0, 0])
    assert (9, 3) == callset['calldata/DP'].shape
    assert 'S' == callset['calldata/DP'].dtype.kind
    assert (b'4', b'2', b'3') == tuple(callset['calldata/DP'][6])


def test_inputs():
    vcf_path = fixture_path('sample.vcf')

    with open(vcf_path, mode='rb') as f:
        data = f.read(-1)

    inputs = (vcf_path,
              vcf_path + '.gz',
              lambda: open(vcf_path, mode='rb'),
              lambda: gzip.open(vcf_path + '.gz', mode='rb'),
              lambda: io.BytesIO(data),
              lambda: io.BytesIO(data.replace(b'\n', b'\r')),
              lambda: io.BytesIO(data.replace(b'\n', b'\r\n')))

    chunk_length = 3
    buffer_size = 10

    for n, i in enumerate(inputs):
        _test_read_vcf_content(i, chunk_length, buffer_size)


def test_chunk_lengths():
    vcf_path = fixture_path('sample.vcf')
    chunk_lengths = 1, 2, 3, 5, 10, 20
    buffer_size = 10

    for chunk_length in chunk_lengths:
        _test_read_vcf_content(vcf_path, chunk_length, buffer_size)


def test_buffer_sizes():
    vcf_path = fixture_path('sample.vcf')
    chunk_length = 3
    buffer_sizes = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512

    for buffer_size in buffer_sizes:
        _test_read_vcf_content(vcf_path, chunk_length, buffer_size)


def test_utf8():
    vcf_path = fixture_path('sample.utf8.vcf')
    callset = read_vcf(vcf_path, fields='*')

    # samples
    assert (3,) == callset['samples'].shape
    assert 'O' == callset['samples'].dtype.kind
    assert [u'NA00001', u'Γεια σου κόσμε!', u'NA00003'] == callset['samples'].tolist()

    # CHROM
    assert (9,) == callset['variants/CHROM'].shape
    assert np.dtype(object) == callset['variants/CHROM'].dtype
    assert '19' == callset['variants/CHROM'][0]
    assert u'Njatjeta Botë!' == callset['variants/CHROM'][-2]

    # POS
    assert (9,) == callset['variants/POS'].shape
    assert 111 == callset['variants/POS'][0]

    # ID
    assert (9,) == callset['variants/ID'].shape
    assert np.dtype(object) == callset['variants/ID'].dtype
    assert 'foo' == callset['variants/ID'][0]
    assert u'¡Hola mundo!' == callset['variants/ID'][1]

    # REF
    assert (9,) == callset['variants/REF'].shape
    assert np.dtype(object) == callset['variants/REF'].dtype
    assert 'A' == callset['variants/REF'][0]

    # ALT
    assert (9, 3) == callset['variants/ALT'].shape
    assert np.dtype(object) == callset['variants/ALT'].dtype
    assert 'ATG' == callset['variants/ALT'][8, 1]

    # QUAL
    assert (9,) == callset['variants/QUAL'].shape
    assert 10.0 == callset['variants/QUAL'][1]

    # FILTER
    assert (9,) == callset['variants/FILTER_PASS'].shape
    assert callset['variants/FILTER_PASS'][2]
    assert not callset['variants/FILTER_PASS'][5]
    assert (9,) == callset[u'variants/FILTER_Helló_világ!'].shape
    assert not callset[u'variants/FILTER_Helló_világ!'][0]
    assert callset[u'variants/FILTER_Helló_világ!'][5]

    # INFO fields
    assert u'foo' == callset['variants/TEXT'][0]
    assert u'こんにちは世界' == callset['variants/TEXT'][4]

    # calldata
    assert (9, 3, 2) == callset['calldata/GT'].shape
    assert (0, 0) == tuple(callset['calldata/GT'][0, 0])
    assert (-1, -1) == tuple(callset['calldata/GT'][6, 2])
    assert (-1, -1) == tuple(callset['calldata/GT'][7, 2])
    assert (9, 3, 2) == callset['calldata/HQ'].shape
    assert (10, 15) == tuple(callset['calldata/HQ'][0, 0])
    assert (9, 3) == callset['calldata/DP'].shape
    assert (4, 2, 3) == tuple(callset['calldata/DP'][6])
    assert (u'foo', u'Hej Världen!', u'.') == tuple(callset['calldata/GTXT'][0])


def test_truncation_chrom():

    input_data = (b"#CHROM\n"
                  b"2L\n"
                  b"2R\n")

    # with and without final line terminator
    for data in (input_data, input_data[:-1]):

        for string_type in 'S10', 'object':
            input_file = io.BytesIO(data)
            callset = read_vcf(input_file, fields=['CHROM', 'samples'],
                               types={'CHROM': string_type})

            # check fields
            expected_fields = ['variants/CHROM']
            assert sorted(expected_fields) == sorted(callset.keys())

            # check data content
            a = callset['variants/CHROM']
            assert 2 == len(a)
            if string_type == 'S10':
                assert b'2L' == a[0]
                assert b'2R' == a[1]
            else:
                assert '2L' == a[0]
                assert '2R' == a[1]


def test_truncation_pos():

    input_data = (b"#CHROM\tPOS\n"
                  b"2L\t12\n"
                  b"2R\t34\n")

    # with and without final line terminator
    for data in (input_data, input_data[:-1]):

        input_file = io.BytesIO(data)
        callset = read_vcf(input_file, fields=['POS', 'samples'])

        # check fields
        expected_fields = ['variants/POS']
        assert sorted(expected_fields) == sorted(callset.keys())

        # check data content
        a = callset['variants/POS']
        assert 2 == len(a)
        assert 12 == a[0]
        assert 34 == a[1]


def test_truncation_id():

    input_data = (b"#CHROM\tPOS\tID\n"
                  b"2L\t12\tfoo\n"
                  b"2R\t34\tbar\n")

    # with and without final line terminator
    for data in (input_data, input_data[:-1]):

        for string_type in 'S10', 'object':
            input_file = io.BytesIO(data)
            callset = read_vcf(input_file, fields=['ID', 'samples'],
                               types={'ID': string_type})

            # check fields
            expected_fields = ['variants/ID']
            assert sorted(expected_fields) == sorted(callset.keys())

            # check data content
            a = callset['variants/ID']
            assert 2 == len(a)
            if string_type == 'S10':
                assert b'foo' == a[0]
                assert b'bar' == a[1]
            else:
                assert 'foo' == a[0]
                assert 'bar' == a[1]


def test_truncation_ref():

    input_data = (b"#CHROM\tPOS\tID\tREF\n"
                  b"2L\t12\tfoo\tA\n"
                  b"2R\t34\tbar\tC\n")

    # with and without final line terminator
    for data in (input_data, input_data[:-1]):

        for string_type in 'S10', 'object':
            input_file = io.BytesIO(data)
            callset = read_vcf(input_file, fields=['REF', 'samples'],
                               types={'REF': string_type})

            # check fields
            expected_fields = ['variants/REF']
            assert sorted(expected_fields) == sorted(callset.keys())

            # check data content
            a = callset['variants/REF']
            assert 2 == len(a)
            if string_type == 'S10':
                assert b'A' == a[0]
                assert b'C' == a[1]
            else:
                assert 'A' == a[0]
                assert 'C' == a[1]


def test_truncation_alt():

    input_data = (b"#CHROM\tPOS\tID\tREF\tALT\n"
                  b"2L\t12\tfoo\tA\tC\n"
                  b"2R\t34\tbar\tC\tG\n")

    # with and without final line terminator
    for data in (input_data, input_data[:-1]):

        for string_type in 'S10', 'object':
            input_file = io.BytesIO(data)
            callset = read_vcf(input_file, fields=['ALT', 'samples'], numbers=dict(ALT=1),
                               types={'ALT': string_type})

            # check fields
            expected_fields = ['variants/ALT']
            assert sorted(expected_fields) == sorted(callset.keys())

            # check data content
            a = callset['variants/ALT']
            assert 2 == len(a)
            if string_type == 'S10':
                assert b'C' == a[0]
                assert b'G' == a[1]
            else:
                assert 'C' == a[0]
                assert 'G' == a[1]


def test_truncation_qual():

    input_data = (b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\n"
                  b"2L\t12\tfoo\tA\tC\t1.2\n"
                  b"2R\t34\tbar\tC\tG\t3.4\n")

    # with and without final line terminator
    for data in (input_data, input_data[:-1]):

        input_file = io.BytesIO(data)
        callset = read_vcf(input_file, fields=['QUAL', 'samples'])

        # check fields
        expected_fields = ['variants/QUAL']
        assert sorted(expected_fields) == sorted(callset.keys())

        # check data content
        a = callset['variants/QUAL']
        assert 2 == len(a)
        assert approx(1.2) == a[0]
        assert approx(3.4) == a[1]


def test_truncation_filter():

    input_data = (b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\n"
                  b"2L\t12\tfoo\tA\tC\t1.2\t.\n"
                  b"2R\t34\tbar\tC\tG\t3.4\tPASS\n"
                  b"2R\t56\tbaz\tG\tT\t56.77\tq10,s50\n")

    # with and without final line terminator
    for data in (input_data, input_data[:-1]):

        input_file = io.BytesIO(data)
        callset = read_vcf(input_file,
                           fields=['FILTER_PASS', 'FILTER_q10', 'FILTER_s50', 'samples'])

        # check fields
        expected_fields = ['variants/FILTER_PASS', 'variants/FILTER_q10',
                           'variants/FILTER_s50']
        assert sorted(expected_fields) == sorted(callset.keys())

        # check data content
        a = callset['variants/FILTER_PASS']
        assert 3 == len(a)
        assert [False, True, False] == a.tolist()
        a = callset['variants/FILTER_q10']
        assert 3 == len(a)
        assert [False, False, True] == a.tolist()
        a = callset['variants/FILTER_s50']
        assert 3 == len(a)
        assert [False, False, True] == a.tolist()


def test_truncation_info():

    input_data = (b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
                  b"2L\t12\tfoo\tA\tC\t1.2\t.\tfoo=42;bar=1.2\n"
                  b"2R\t34\tbar\tC\tG\t3.4\tPASS\t.\n"
                  b"2R\t56\tbaz\tG\tT\t56.77\tq10,s50\t\n")

    # with and without final line terminator
    for data in (input_data, input_data[:-1]):

        input_file = io.BytesIO(data)
        callset = read_vcf(input_file,
                           fields=['foo', 'bar', 'samples'],
                           types=dict(foo='Integer', bar='Float'))

        # check fields
        expected_fields = ['variants/foo', 'variants/bar']
        assert sorted(expected_fields) == sorted(callset.keys())

        # check data content
        a = callset['variants/foo']
        assert 3 == len(a)
        assert 42 == a[0]
        assert -1 == a[1]
        assert -1 == a[2]
        a = callset['variants/bar']
        assert 3 == len(a)
        assert approx(1.2) == a[0]
        assert np.isnan(a[1])
        assert np.isnan(a[2])


def test_truncation_format():

    input_data = (b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\n"
                  b"2L\t12\tfoo\tA\tC\t1.2\t.\tfoo=42;bar=1.2\tGT:GQ\n"
                  b"2R\t34\tbar\tC\tG\t3.4\tPASS\t.\t.\n"
                  b"2R\t56\tbaz\tG\tT\t56.77\tq10,s50\t\t\n")

    # with and without final line terminator
    for data in (input_data, input_data[:-1]):

        input_file = io.BytesIO(data)
        callset = read_vcf(input_file,
                           fields=['foo', 'bar', 'samples'],
                           types=dict(foo='Integer', bar='Float'))

        # check fields
        expected_fields = ['variants/foo', 'variants/bar']
        assert sorted(expected_fields) == sorted(callset.keys())

        # check data content
        a = callset['variants/foo']
        assert 3 == len(a)
        assert 42 == a[0]
        assert -1 == a[1]
        assert -1 == a[2]
        a = callset['variants/bar']
        assert 3 == len(a)
        assert approx(1.2) == a[0]
        assert np.isnan(a[1])
        assert np.isnan(a[2])


def test_truncation_calldata():

    input_data = (b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS2\tS1\n"
                  b"2L\t12\tfoo\tA\tC\t1.2\t.\tfoo=42;bar=1.2\tGT:GQ\t0/1:12\t1/2:34\n"
                  b"2R\t34\tbar\tC\tG\t3.4\tPASS\t.\tGT\t./.\n"
                  b"2R\t56\tbaz\tG\tT\t56.77\tq10,s50\t\n")

    # with and without final line terminator
    for data in (input_data, input_data[:-1]):

        input_file = io.BytesIO(data)
        callset = read_vcf(input_file,
                           fields=['calldata/GT', 'calldata/GQ', 'samples'],
                           types={'calldata/GT': 'i1', 'calldata/GQ': 'i2'})

        # check fields
        expected_fields = ['calldata/GT', 'calldata/GQ', 'samples']
        assert sorted(expected_fields) == sorted(callset.keys())

        # check data content
        assert 2 == len(callset['samples'])
        assert ['S2', 'S1'] == callset['samples'].tolist()
        a = callset['calldata/GT']
        assert (3, 2, 2) == a.shape
        assert (0, 1) == tuple(a[0, 0])
        assert (1, 2) == tuple(a[0, 1])
        assert (-1, -1) == tuple(a[1, 0])
        assert (-1, -1) == tuple(a[1, 1])
        assert (-1, -1) == tuple(a[2, 0])
        assert (-1, -1) == tuple(a[2, 1])

        a = callset['calldata/GQ']
        assert (3, 2) == a.shape
        assert 12 == a[0, 0]
        assert 34 == a[0, 1]
        assert -1 == a[1, 0]
        assert -1 == a[1, 1]
        assert -1 == a[2, 0]
        assert -1 == a[2, 1]


def test_info_types():
    vcf_path = fixture_path('sample.vcf')

    for dtype in ('i1', 'i2', 'i4', 'i8', 'u1', 'u2', 'u4', 'u8', 'f4', 'f8', 'S10',
                  'object'):
        callset = read_vcf(vcf_path, fields=['variants/DP', 'variants/AC'],
                           types={'variants/DP': dtype, 'variants/AC': dtype},
                           numbers={'variants/AC': 3})
        assert np.dtype(dtype) == callset['variants/DP'].dtype
        assert (9,) == callset['variants/DP'].shape
        assert (9, 3) == callset['variants/AC'].shape


def test_vcf_types():

    input_data = (
        b'##INFO=<ID=foo,Number=1,Type=String,Description="Testing 123.">\n'
        b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\n"
        b"2L\t12\t.\tA\tC\t.\t.\tfoo=bar\t.\n"
    )
    callset = read_vcf(io.BytesIO(input_data), fields=['foo'])
    assert np.dtype(object) == callset['variants/foo'].dtype

    input_data = (
        b'##INFO=<ID=foo,Number=1,Type=Integer,Description="Testing 123.">\n'
        b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\n"
        b"2L\t12\t.\tA\tC\t.\t.\tfoo=42\t.\n"
    )
    callset = read_vcf(io.BytesIO(input_data), fields=['foo'])
    assert np.dtype('i4') == callset['variants/foo'].dtype

    input_data = (
        b'##INFO=<ID=foo,Number=1,Type=Float,Description="Testing 123.">\n'
        b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\n"
        b"2L\t12\t.\tA\tC\t.\t.\tfoo=42.0\t.\n"
    )
    callset = read_vcf(io.BytesIO(input_data), fields=['foo'])
    assert np.dtype('f4') == callset['variants/foo'].dtype

    input_data = (
        b'##INFO=<ID=foo,Number=1,Type=Character,Description="Testing 123.">\n'
        b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\n"
        b"2L\t12\t.\tA\tC\t.\t.\tfoo=b\t.\n"
    )
    callset = read_vcf(io.BytesIO(input_data), fields=['foo'])
    assert np.dtype('S1') == callset['variants/foo'].dtype


def test_genotype_types():

    vcf_path = fixture_path('sample.vcf')
    for dtype in 'i1', 'i2', 'i4', 'i8', 'u1', 'u2', 'u4', 'u8', 'S3', 'object':
        callset = read_vcf(vcf_path, fields=['GT'], types={'GT': dtype},
                           numbers={'GT': 2})
        assert np.dtype(dtype) == callset['calldata/GT'].dtype
        assert (9, 3, 2) == callset['calldata/GT'].shape

    # non-GT field with genotype dtype

    input_data = (
        b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\tS3\n"
        b"2L\t12\t.\tA\t.\t.\t.\t.\tCustomGT:CustomGQ\t0/0/0:11\t0/1/2:12\t././.:.\n"
        b"2L\t34\t.\tC\tT\t.\t.\t.\tCustomGT:CustomGQ\t0/1/2:22\t3/3/.:33\t.\n"
        b"3R\t45\t.\tG\tA,T\t.\t.\t.\tCustomGT:CustomGQ\t0/1:.\t5:12\t\n"
    )
    callset = read_vcf(io.BytesIO(input_data),
                       fields=['calldata/CustomGT', 'calldata/CustomGQ'],
                       numbers={'calldata/CustomGT': 3, 'calldata/CustomGQ': 1},
                       types={'calldata/CustomGT': 'genotype/i1',
                              'calldata/CustomGQ': 'i2'})

    e = np.array([[[0, 0, 0], [0, 1, 2], [-1, -1, -1]],
                  [[0, 1, 2], [3, 3, -1], [-1, -1, -1]],
                  [[0, 1, -1], [5, -1, -1], [-1, -1, -1]]], dtype='i1')
    a = callset['calldata/CustomGT']
    assert_array_equal(e, a)
    assert e.dtype == a.dtype

    e = np.array([[11, 12, -1],
                  [22, 33, -1],
                  [-1, 12, -1]], dtype='i2')
    a = callset['calldata/CustomGQ']
    assert_array_equal(e, a)
    assert e.dtype == a.dtype


def test_calldata_types():
    vcf_path = fixture_path('sample.vcf')

    for dtype in ('i1', 'i2', 'i4', 'i8', 'u1', 'u2', 'u4', 'u8', 'f4', 'f8', 'S10',
                  'object'):
        callset = read_vcf(vcf_path, fields=['HQ'], types={'HQ': dtype},
                           numbers={'HQ': 2})
        assert np.dtype(dtype) == callset['calldata/HQ'].dtype
        assert (9, 3, 2) == callset['calldata/HQ'].shape


def test_genotype_ploidy():
    vcf_path = fixture_path('sample.vcf')

    callset = read_vcf(vcf_path, fields='GT', numbers=dict(GT=1))
    gt = callset['calldata/GT']
    assert (9, 3) == gt.shape
    assert (0, 0, 0) == tuple(gt[8, :])

    callset = read_vcf(vcf_path, fields='GT', numbers=dict(GT=2))
    gt = callset['calldata/GT']
    assert (9, 3, 2) == gt.shape
    assert (0, -1) == tuple(gt[8, 0])
    assert (0, 1) == tuple(gt[8, 1])
    assert (0, 2) == tuple(gt[8, 2])

    callset = read_vcf(vcf_path, fields='GT', numbers=dict(GT=3))
    gt = callset['calldata/GT']
    assert (9, 3, 3) == gt.shape
    assert (0, -1, -1) == tuple(gt[8, 0])
    assert (0, 1, -1) == tuple(gt[8, 1])
    assert (0, 2, -1) == tuple(gt[8, 2])


def test_fills_info():
    vcf_path = fixture_path('sample.vcf')

    callset = read_vcf(vcf_path, fields='AN', numbers=dict(AN=1))
    a = callset['variants/AN']
    assert (9,) == a.shape
    assert -1 == a[0]
    assert -1 == a[1]
    assert -1 == a[2]

    callset = read_vcf(vcf_path, fields='AN', numbers=dict(AN=1), fills=dict(AN=-2))
    a = callset['variants/AN']
    assert (9,) == a.shape
    assert -2 == a[0]
    assert -2 == a[1]
    assert -2 == a[2]

    callset = read_vcf(vcf_path, fields='AN', numbers=dict(AN=1), fills=dict(AN=-1))
    a = callset['variants/AN']
    assert (9,) == a.shape
    assert -1 == a[0]
    assert -1 == a[1]
    assert -1 == a[2]


def test_fills_genotype():
    vcf_path = fixture_path('sample.vcf')

    callset = read_vcf(vcf_path, fields='GT', numbers=dict(GT=2))
    gt = callset['calldata/GT']
    assert (9, 3, 2) == gt.shape
    assert (0, -1) == tuple(gt[8, 0])
    assert (0, 1) == tuple(gt[8, 1])
    assert (0, 2) == tuple(gt[8, 2])

    callset = read_vcf(vcf_path, fields='GT', numbers=dict(GT=2), fills=dict(GT=-2))
    gt = callset['calldata/GT']
    assert (9, 3, 2) == gt.shape
    assert (0, -2) == tuple(gt[8, 0])
    assert (0, 1) == tuple(gt[8, 1])
    assert (0, 2) == tuple(gt[8, 2])

    callset = read_vcf(vcf_path, fields='GT', numbers=dict(GT=3), fills=dict(GT=-1))
    gt = callset['calldata/GT']
    assert (9, 3, 3) == gt.shape
    assert (0, -1, -1) == tuple(gt[8, 0])
    assert (0, 1, -1) == tuple(gt[8, 1])
    assert (0, 2, -1) == tuple(gt[8, 2])


def test_fills_calldata():
    vcf_path = fixture_path('sample.vcf')

    callset = read_vcf(vcf_path, fields='HQ', numbers=dict(HQ=2))
    a = callset['calldata/HQ']
    assert (9, 3, 2) == a.shape
    assert (10, 15) == tuple(a[0, 0])
    assert (-1, -1) == tuple(a[7, 0])
    assert (-1, -1) == tuple(a[8, 0])

    callset = read_vcf(vcf_path, fields='HQ', numbers=dict(HQ=2), fills=dict(HQ=-2))
    a = callset['calldata/HQ']
    assert (9, 3, 2) == a.shape
    assert (10, 15) == tuple(a[0, 0])
    assert (-2, -2) == tuple(a[7, 0])
    assert (-2, -2) == tuple(a[8, 0])

    callset = read_vcf(vcf_path, fields='HQ', numbers=dict(HQ=2), fills=dict(HQ=-1))
    a = callset['calldata/HQ']
    assert (9, 3, 2) == a.shape
    assert (10, 15) == tuple(a[0, 0])
    assert (-1, -1) == tuple(a[7, 0])
    assert (-1, -1) == tuple(a[8, 0])


def test_numbers():
    vcf_path = fixture_path('sample.vcf')

    callset = read_vcf(vcf_path, fields=['ALT'], numbers=dict(ALT=1))
    a = callset['variants/ALT']
    assert (9,) == a.shape
    assert 'A' == a[8]

    callset = read_vcf(vcf_path, fields=['ALT'], numbers=dict(ALT=2),
                       types=dict(ALT='S4'))
    a = callset['variants/ALT']
    assert (9, 2) == a.shape
    assert b'A' == a[8, 0]
    assert b'ATG' == a[8, 1]

    callset = read_vcf(vcf_path, fields=['ALT'], numbers=dict(ALT=3),
                       types=dict(ALT='S4'))
    a = callset['variants/ALT']
    assert (9, 3) == a.shape
    assert b'A' == a[8, 0]
    assert b'ATG' == a[8, 1]
    assert b'C' == a[8, 2]

    callset = read_vcf(vcf_path, fields=['AC'], numbers=dict(AC=0))
    a = callset['variants/AC']
    assert (9,) == a.shape
    assert not a[0]
    assert a[6]

    callset = read_vcf(vcf_path, fields=['AC'], numbers=dict(AC=1))
    a = callset['variants/AC']
    assert (9,) == a.shape
    assert -1 == a[0]
    assert 3 == a[6]

    callset = read_vcf(vcf_path, fields=['AC'], numbers=dict(AC=2))
    a = callset['variants/AC']
    assert (9, 2) == a.shape
    assert -1 == a[0, 0]
    assert -1 == a[0, 1]
    assert 3 == a[6, 0]
    assert 1 == a[6, 1]

    callset = read_vcf(vcf_path, fields='AF', numbers=dict(AF=1))
    a = callset['variants/AF']
    assert (9,) == a.shape
    assert 0.5 == a[2]
    assert approx(0.333) == a[4]

    callset = read_vcf(vcf_path, fields='AF', numbers=dict(AF=2))
    a = callset['variants/AF']
    assert (9, 2) == a.shape
    assert 0.5 == a[2, 0]
    assert np.isnan(a[2, 1])
    assert approx(0.333) == a[4, 0]
    assert approx(0.667) == a[4, 1]

    callset = read_vcf(vcf_path, fields=['HQ'], numbers=dict(HQ=1))
    a = callset['calldata/HQ']
    assert (9, 3) == a.shape
    assert 10 == a[0, 0]
    assert 51 == a[2, 0]
    assert -1 == a[6, 0]

    callset = read_vcf(vcf_path, fields=['HQ'], numbers=dict(HQ=2))
    a = callset['calldata/HQ']
    assert (9, 3, 2) == a.shape
    assert (10, 15) == tuple(a[0, 0])
    assert (51, 51) == tuple(a[2, 0])
    assert (-1, -1) == tuple(a[6, 0])


def test_alt_number():
    vcf_path = fixture_path('sample.vcf')

    callset = read_vcf(vcf_path, fields=['ALT', 'AC', 'AF'], alt_number=2)
    a = callset['variants/ALT']
    assert (9, 2) == a.shape
    a = callset['variants/AC']
    assert (9, 2) == a.shape
    a = callset['variants/AF']
    assert (9, 2) == a.shape

    callset = read_vcf(vcf_path, fields=['ALT', 'AC', 'AF'], alt_number=1)
    a = callset['variants/ALT']
    assert (9,) == a.shape
    a = callset['variants/AC']
    assert (9,) == a.shape
    a = callset['variants/AF']
    assert (9,) == a.shape

    callset = read_vcf(vcf_path, fields=['ALT', 'AC', 'AF'], alt_number=5)
    a = callset['variants/ALT']
    assert (9, 5) == a.shape
    a = callset['variants/AC']
    assert (9, 5) == a.shape
    a = callset['variants/AF']
    assert (9, 5) == a.shape

    # can override
    callset = read_vcf(vcf_path, fields=['ALT', 'AC', 'AF'],
                       alt_number=5, numbers={'ALT': 2, 'AC': 4})
    a = callset['variants/ALT']
    assert (9, 2) == a.shape
    a = callset['variants/AC']
    assert (9, 4) == a.shape
    a = callset['variants/AF']
    assert (9, 5) == a.shape


def test_read_region():

    for vcf_path in (fixture_path('sample.vcf.gz'),
                     fixture_path('sample.vcf')):
        for tabix in 'tabix', None, 'foobar':

            region = '19'
            callset = read_vcf(vcf_path, region=region, tabix=tabix)
            chrom = callset['variants/CHROM']
            pos = callset['variants/POS']
            assert 2 == len(chrom)
            assert isinstance(chrom, np.ndarray)
            assert np.all(chrom == '19')
            assert 2 == len(pos)
            assert_array_equal([111, 112], pos)

            region = '20'
            callset = read_vcf(vcf_path, region=region, tabix=tabix)
            chrom = callset['variants/CHROM']
            pos = callset['variants/POS']
            assert 6 == len(chrom)
            assert isinstance(chrom, np.ndarray)
            assert np.all(chrom == '20')
            assert 6 == len(pos)
            assert_array_equal([14370, 17330, 1110696, 1230237, 1234567, 1235237], pos)

            region = 'X'
            callset = read_vcf(vcf_path, region=region, tabix=tabix)
            chrom = callset['variants/CHROM']
            pos = callset['variants/POS']
            assert 1 == len(chrom)
            assert isinstance(chrom, np.ndarray)
            assert np.all(chrom == 'X')
            assert 1 == len(pos)
            assert_array_equal([10], pos)

            region = 'Y'
            callset = read_vcf(vcf_path, region=region, tabix=tabix)
            assert callset is None

            region = '20:1-100000'
            callset = read_vcf(vcf_path, region=region, tabix=tabix)
            chrom = callset['variants/CHROM']
            pos = callset['variants/POS']
            assert 2 == len(chrom)
            assert isinstance(chrom, np.ndarray)
            assert np.all(chrom == '20')
            assert 2 == len(pos)
            assert_array_equal([14370, 17330], pos)

            region = '20:1000000-1233000'
            callset = read_vcf(vcf_path, region=region, tabix=tabix)
            chrom = callset['variants/CHROM']
            pos = callset['variants/POS']
            assert 2 == len(chrom)
            assert isinstance(chrom, np.ndarray)
            assert np.all(chrom == '20')
            assert 2 == len(pos)
            assert_array_equal([1110696, 1230237], pos)

            region = '20:1233000-2000000'
            callset = read_vcf(vcf_path, region=region, tabix=tabix)
            chrom = callset['variants/CHROM']
            pos = callset['variants/POS']
            assert 2 == len(chrom)
            assert isinstance(chrom, np.ndarray)
            assert np.all(chrom == '20')
            assert 2 == len(pos)
            assert_array_equal([1234567, 1235237], pos)


def test_read_region_unsorted():
    # Test behaviour when data are not sorted by chromosome or position and tabix is
    # not available.

    fn = fixture_path('unsorted.vcf')
    tabix = None

    region = '19'
    callset = read_vcf(fn, region=region, tabix=tabix)
    chrom = callset['variants/CHROM']
    pos = callset['variants/POS']
    assert 2 == len(chrom)
    assert isinstance(chrom, np.ndarray)
    assert np.all(chrom == '19')
    assert 2 == len(pos)
    assert_array_equal([111, 112], pos)

    region = '20'
    callset = read_vcf(fn, region=region, tabix=tabix)
    chrom = callset['variants/CHROM']
    pos = callset['variants/POS']
    assert 6 == len(chrom)
    assert isinstance(chrom, np.ndarray)
    assert np.all(chrom == '20')
    assert 6 == len(pos)
    assert_array_equal([14370, 1230237, 1234567, 1235237, 17330, 1110696], pos)

    region = 'X'
    callset = read_vcf(fn, region=region, tabix=tabix)
    chrom = callset['variants/CHROM']
    pos = callset['variants/POS']
    assert 1 == len(chrom)
    assert isinstance(chrom, np.ndarray)
    assert np.all(chrom == 'X')
    assert 1 == len(pos)
    assert_array_equal([10], pos)

    region = 'Y'
    callset = read_vcf(fn, region=region, tabix=tabix)
    assert callset is None

    region = '20:1-100000'
    callset = read_vcf(fn, region=region, tabix=tabix)
    chrom = callset['variants/CHROM']
    pos = callset['variants/POS']
    assert 2 == len(chrom)
    assert isinstance(chrom, np.ndarray)
    assert np.all(chrom == '20')
    assert 2 == len(pos)
    assert_array_equal([14370, 17330], pos)

    region = '20:1000000-1233000'
    callset = read_vcf(fn, region=region, tabix=tabix)
    chrom = callset['variants/CHROM']
    pos = callset['variants/POS']
    assert 2 == len(chrom)
    assert isinstance(chrom, np.ndarray)
    assert np.all(chrom == '20')
    assert 2 == len(pos)
    assert_array_equal([1230237, 1110696], pos)

    region = '20:1233000-2000000'
    callset = read_vcf(fn, region=region, tabix=tabix)
    chrom = callset['variants/CHROM']
    pos = callset['variants/POS']
    assert 2 == len(chrom)
    assert isinstance(chrom, np.ndarray)
    assert np.all(chrom == '20')
    assert 2 == len(pos)
    assert_array_equal([1234567, 1235237], pos)


def test_read_samples():
    vcf_path = fixture_path('sample.vcf')

    for samples in ['NA00001', 'NA00003'], [0, 2], ['NA00003', 'NA00001'], [2, 'NA00001']:
        callset = read_vcf(vcf_path, fields=['samples', 'GT'], samples=samples)
        assert ['NA00001', 'NA00003'] == callset['samples'].astype('U').tolist()
        gt = callset['calldata/GT']
        assert (9, 2, 2) == gt.shape
        assert (0, 0) == tuple(gt[2, 0])
        assert (1, 1) == tuple(gt[2, 1])
        assert (1, 2) == tuple(gt[4, 0])
        assert (2, 2) == tuple(gt[4, 1])

    for samples in ['NA00002'], [1]:
        callset = read_vcf(vcf_path, fields=['samples', 'GT'], samples=samples)
        assert ['NA00002'] == callset['samples'].astype('U').tolist()
        gt = callset['calldata/GT']
        assert (9, 1, 2) == gt.shape
        assert (1, 0) == tuple(gt[2, 0])
        assert (2, 1) == tuple(gt[4, 0])


def test_read_empty():
    vcf_path = fixture_path('empty.vcf')
    callset = read_vcf(vcf_path)
    assert callset is None


def test_ann():
    vcf_path = fixture_path('ann.vcf')

    # all ANN fields
    callset = read_vcf(vcf_path, fields=['ANN'], transformers=[ANNTransformer()])
    expect_keys = sorted(['variants/ANN_Allele',
                          'variants/ANN_Annotation',
                          'variants/ANN_Annotation_Impact',
                          'variants/ANN_Gene_Name',
                          'variants/ANN_Gene_ID',
                          'variants/ANN_Feature_Type',
                          'variants/ANN_Feature_ID',
                          'variants/ANN_Transcript_BioType',
                          'variants/ANN_Rank',
                          'variants/ANN_HGVS_c',
                          'variants/ANN_HGVS_p',
                          'variants/ANN_cDNA_pos',
                          'variants/ANN_cDNA_length',
                          'variants/ANN_CDS_pos',
                          'variants/ANN_CDS_length',
                          'variants/ANN_AA_pos',
                          'variants/ANN_AA_length',
                          'variants/ANN_Distance'])
    assert expect_keys == sorted(callset.keys())
    a = callset['variants/ANN_Allele']
    assert (3,) == a.shape
    assert np.dtype('object') == a.dtype
    assert_array_equal(['T', '', 'T'], a)
    a = callset['variants/ANN_Annotation']
    assert (3,) == a.shape
    assert np.dtype('object') == a.dtype
    assert_array_equal(['intergenic_region', '', 'missense_variant'], a)
    a = callset['variants/ANN_Annotation_Impact']
    assert (3,) == a.shape
    assert np.dtype('object') == a.dtype
    assert_array_equal(['MODIFIER', '', 'MODERATE'], a)
    a = callset['variants/ANN_Gene_Name']
    assert (3,) == a.shape
    assert np.dtype('object') == a.dtype
    assert_array_equal(['AGAP004677', '', 'AGAP005273'], a)
    a = callset['variants/ANN_Gene_ID']
    assert (3,) == a.shape
    assert np.dtype('object') == a.dtype
    assert_array_equal(['AGAP004677', '', 'AGAP005273'], a)
    a = callset['variants/ANN_Feature_Type']
    assert (3,) == a.shape
    assert np.dtype('object') == a.dtype
    assert_array_equal(['intergenic_region', '', 'transcript'], a)
    a = callset['variants/ANN_Feature_ID']
    assert (3,) == a.shape
    assert np.dtype('object') == a.dtype
    assert_array_equal(['AGAP004677', '', 'AGAP005273-RA'], a)
    a = callset['variants/ANN_Transcript_BioType']
    assert np.dtype('object') == a.dtype
    assert (3,) == a.shape
    assert_array_equal(['', '', 'VectorBase'], a)
    assert np.dtype('object') == a.dtype
    a = callset['variants/ANN_Rank']
    assert (3,) == a.shape
    assert np.dtype('int8') == a.dtype
    assert_array_equal([-1, -1, 1], a[:])
    a = callset['variants/ANN_HGVS_c']
    assert (3,) == a.shape
    assert np.dtype('object') == a.dtype
    assert_array_equal(['', '', '17A>T'], a)
    a = callset['variants/ANN_HGVS_p']
    assert (3,) == a.shape
    assert np.dtype('object') == a.dtype
    assert_array_equal(['', '', 'Asp6Val'], a)
    a = callset['variants/ANN_cDNA_pos']
    assert (3,) == a.shape
    assert np.dtype('int32') == a.dtype
    assert_array_equal([-1, -1, 17], a)
    a = callset['variants/ANN_cDNA_length']
    assert (3,) == a.shape
    assert np.dtype('int32') == a.dtype
    assert_array_equal([-1, -1, 4788], a)
    a = callset['variants/ANN_CDS_pos']
    assert (3,) == a.shape
    assert np.dtype('int32') == a.dtype
    assert_array_equal([-1, -1, 17], a)
    a = callset['variants/ANN_CDS_length']
    assert (3,) == a.shape
    assert np.dtype('int32') == a.dtype
    assert_array_equal([-1, -1, 4788], a)
    a = callset['variants/ANN_AA_pos']
    assert (3,) == a.shape
    assert np.dtype('int32') == a.dtype
    assert_array_equal([-1, -1, 6], a)
    a = callset['variants/ANN_AA_length']
    assert (3,) == a.shape
    assert np.dtype('int32') == a.dtype
    assert_array_equal([-1, -1, 1596], a)
    a = callset['variants/ANN_Distance']
    assert (3,) == a.shape
    assert np.dtype('int32') == a.dtype
    assert_array_equal([3000, -1, -1], a)

    # numbers=2
    callset = read_vcf(vcf_path, fields=['ANN'], numbers={'ANN': 2},
                       transformers=[ANNTransformer()])
    a = callset['variants/ANN_Allele']
    assert (3, 2) == a.shape
    assert np.dtype('object') == a.dtype
    assert_array_equal(['T', ''], a[0])
    assert_array_equal(['', ''], a[1])
    assert_array_equal(['T', 'G'], a[2])
    a = callset['variants/ANN_cDNA_pos']
    assert (3, 2) == a.shape
    assert np.dtype('int32') == a.dtype
    assert_array_equal([-1, -1, 17], a[:, 0])
    assert_array_equal([-1, -1, 12], a[:, 1])
    a = callset['variants/ANN_cDNA_length']
    assert (3, 2) == a.shape
    assert np.dtype('int32') == a.dtype
    assert_array_equal([-1, -1, 4788], a[:, 0])
    assert_array_equal([-1, -1, 4768], a[:, 1])

    # choose fields and types
    transformers = [
        ANNTransformer(
            fields=['Allele', 'ANN_HGVS_c', 'variants/ANN_cDNA_pos'],
            types={'Allele': 'S12',
                   'ANN_HGVS_c': 'S20',
                   'variants/ANN_cDNA_pos': 'i8'})
    ]
    callset = read_vcf(vcf_path, fields=['ANN'], transformers=transformers)
    assert (sorted(['variants/ANN_Allele', 'variants/ANN_HGVS_c',
                    'variants/ANN_cDNA_pos']) == sorted(callset.keys()))
    a = callset['variants/ANN_Allele']
    assert (3,) == a.shape
    assert np.dtype('S12') == a.dtype
    assert_array_equal([b'T', b'', b'T'], a)
    a = callset['variants/ANN_HGVS_c']
    assert (3,) == a.shape
    assert np.dtype('S20') == a.dtype
    assert_array_equal([b'', b'', b'17A>T'], a)
    a = callset['variants/ANN_cDNA_pos']
    assert (3,) == a.shape
    assert np.dtype('i8') == a.dtype
    assert_array_equal([-1, -1, 17], a)


def test_format_inconsistencies():

    input_data = (
        b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS2\tS1\tS3\tS4\n"
        b"2L\t12\tfoo\tA\tC\t1.2\t.\t.\tGT:GQ\t0/1:12\t1/2\t2/3:34:67,89\t\n"
        b"2R\t34\tbar\tC\tG\t3.4\t.\t.\tGT\t./.\t\t3/3:45\t1/2:11:55,67\n"
    )

    input_file = io.BytesIO(input_data)
    callset = read_vcf(input_file, fields=['calldata/GT', 'calldata/GQ'])
    gt = callset['calldata/GT']
    assert (2, 4, 2) == gt.shape
    assert_array_equal([[0, 1], [1, 2], [2, 3], [-1, -1]], gt[0])
    assert_array_equal([[-1, -1], [-1, -1], [3, 3], [1, 2]], gt[1])
    gq = callset['calldata/GQ']
    assert (2, 4) == gq.shape
    assert_array_equal([12, -1, 34, -1], gq[0])
    assert_array_equal([-1, -1, -1, -1], gq[1])


# noinspection PyTypeChecker
def test_warnings():

    warnings.resetwarnings()
    warnings.simplefilter('error')

    # empty CHROM
    input_data = (
        b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS2\tS1\tS3\tS4\n"
        b"\t12\t.\t.\t.\t.\t.\t.\t.\t.\t.\t.\t.\n"
    )
    with pytest.warns(UserWarning):
        read_vcf(io.BytesIO(input_data))

    # empty POS
    input_data = (
        b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS2\tS1\tS3\tS4\n"
        b"2L\t\t.\t.\t.\t.\t.\t.\t.\t.\t.\t.\t.\n"
    )
    with pytest.warns(UserWarning):
        read_vcf(io.BytesIO(input_data))

    # dodgy POS
    input_data = (
        b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS2\tS1\tS3\tS4\n"
        b"2L\taaa\t.\t.\t.\t.\t.\t.\t.\t.\t.\t.\t.\n"
    )
    with pytest.warns(UserWarning):
        read_vcf(io.BytesIO(input_data))

    # dodgy POS
    input_data = (
        b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS2\tS1\tS3\tS4\n"
        b"2L\t12aaa\t.\t.\t.\t.\t.\t.\t.\t.\t.\t.\t.\n"
    )
    with pytest.warns(UserWarning):
        read_vcf(io.BytesIO(input_data))

    # dodgy QUAL
    input_data = (
        b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS2\tS1\tS3\tS4\n"
        b"2L\t12\t.\t.\t.\taaa\t.\t.\t.\t.\t.\t.\t.\n"
    )
    with pytest.warns(UserWarning):
        read_vcf(io.BytesIO(input_data))

    # dodgy QUAL
    input_data = (
        b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS2\tS1\tS3\tS4\n"
        b"2L\t12\t.\t.\t.\t1.2aaa\t.\t.\t.\t.\t.\t.\t.\n"
    )
    with pytest.warns(UserWarning):
        read_vcf(io.BytesIO(input_data))

    # empty QUAL - no warning
    input_data = (
        b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS2\tS1\tS3\tS4\n"
        b"2L\t12\t.\t.\t.\t\t.\t.\t.\t.\t.\t.\t.\n"
    )
    read_vcf(io.BytesIO(input_data))

    # empty FILTER - no warning
    input_data = (
        b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS2\tS1\tS3\tS4\n"
        b"2L\t12\t.\t.\t.\t.\t\t.\t.\t.\t.\t.\t.\n"
    )
    read_vcf(io.BytesIO(input_data))

    # empty INFO - no warning
    input_data = (
        b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS2\tS1\tS3\tS4\n"
        b"2L\t12\t.\t.\t.\t.\t.\t\t.\t.\t.\t.\t.\n"
    )
    read_vcf(io.BytesIO(input_data))

    # empty FORMAT - no warning
    input_data = (
        b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS2\tS1\tS3\tS4\n"
        b"2L\t12\t.\t.\t.\t.\t.\t.\t\t.\t.\t.\t.\n"
    )
    read_vcf(io.BytesIO(input_data))

    # dodgy calldata (integer)
    input_data = (
        b'##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS2\tS1\tS3\tS4\n"
        b"2L\t12\t.\t.\t.\t.\t.\t.\tGT\t0/1\taa/bb\t.\t.\n"
    )
    with pytest.warns(UserWarning):
        read_vcf(io.BytesIO(input_data), fields=['calldata/GT'])

    # dodgy calldata (integer)
    input_data = (
        b'##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS2\tS1\tS3\tS4\n"
        b"2L\t12\t.\t.\t.\t.\t.\t.\tGT\t0/1\t12aa/22\t.\t.\n"
    )
    with pytest.warns(UserWarning):
        read_vcf(io.BytesIO(input_data), fields=['calldata/GT'])

    # dodgy calldata (float)
    input_data = (
        b'##FORMAT=<ID=MQ,Number=1,Type=Float,Description="Mapping Quality">\n'
        b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS2\tS1\tS3\tS4\n"
        b"2L\t12\t.\t.\t.\t.\t.\t.\tMQ\t.\t12.3\taaa\t.\n"
    )
    with pytest.warns(UserWarning):
        read_vcf(io.BytesIO(input_data), fields=['calldata/MQ'])

    # dodgy calldata (float)
    input_data = (
        b'##FORMAT=<ID=MQ,Number=1,Type=Float,Description="Mapping Quality">\n'
        b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS2\tS1\tS3\tS4\n"
        b"2L\t12\t.\t.\t.\t.\t.\t.\tMQ\t.\t12.3\t34.5aaa\t.\n"
    )
    with pytest.warns(UserWarning):
        read_vcf(io.BytesIO(input_data), fields=['calldata/MQ'])

    # dodgy INFO (missing key)
    input_data = (
        b'##INFO=<ID=MQ,Number=1,Type=Float,Description="Mapping Quality">\n'
        b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS2\tS1\tS3\tS4\n"
        b"2L\t12\t.\t.\t.\t.\t.\tfoo=qux;MQ=12\t.\t.\t.\t.\t.\n"
        b"2L\t34\t.\t.\t.\t.\t.\tfoo=bar;=34;baz\t.\t.\t.\t.\t.\n"
    )
    with pytest.warns(UserWarning):
        read_vcf(io.BytesIO(input_data), fields=['variants/MQ'])

    # INFO not declared in header
    input_data = (
        b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS2\tS1\tS3\tS4\n"
        b"2L\t12\tfoo\tA\tC,T\t12.3\tPASS\tfoo=bar\tGT:GQ\t0/0:99\t0/1:12\t./.:.\t.\n"
    )
    with pytest.warns(UserWarning):
        read_vcf(io.BytesIO(input_data), fields=['variants/foo'])

    # FORMAT not declared in header
    input_data = (
        b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS2\tS1\tS3\tS4\n"
        b"2L\t12\tfoo\tA\tC,T\t12.3\tPASS\tfoo=bar\tGT:GQ\t0/0:99\t0/1:12\t./.:.\t.\n"
    )
    with pytest.warns(UserWarning):
        read_vcf(io.BytesIO(input_data), fields=['calldata/GT'])
    with pytest.warns(UserWarning):
        read_vcf(io.BytesIO(input_data), fields=['calldata/GQ'])

    warnings.resetwarnings()
    warnings.simplefilter('always')


def test_missing_headers():
    vcf_path = fixture_path('test14.vcf')

    # INFO DP not declared
    callset = read_vcf(vcf_path, fields=['DP'], types={'DP': 'String'})
    a = callset['variants/DP']
    assert '14' == a[2]  # default type is string
    callset = read_vcf(vcf_path, fields=['DP'], types={'DP': 'Integer'})
    a = callset['variants/DP']
    assert 14 == a[2]
    # what about a field which isn't present at all?
    callset = read_vcf(vcf_path, fields=['FOO'])
    assert '' == callset['variants/FOO'][2]  # default missing value for string field

    # FORMAT field DP not declared in VCF header
    callset = read_vcf(vcf_path, fields=['calldata/DP'],
                       types={'calldata/DP': 'Integer'})
    assert 1 == callset['calldata/DP'][2, 0]


def test_extra_samples():
    # more calldata samples than samples declared in header
    path = fixture_path('test48b.vcf')
    input_data = (
        b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS2\tS1\tS3\tS4\n"
        b"2L\t12\t.\t.\t.\t.\t.\t.\tGT:GQ\t0/0:34\t0/1:45\t1/1:56\t1/2:99\t2/3:101\n"
    )

    warnings.resetwarnings()
    warnings.simplefilter('error')
    with pytest.warns(UserWarning):
        read_vcf(path)
    with pytest.warns(UserWarning):
        read_vcf(io.BytesIO(input_data), fields=['calldata/GT', 'calldata/GQ'])

    warnings.resetwarnings()
    warnings.simplefilter('always')
    # try again without raising warnings to check data
    callset = read_vcf(io.BytesIO(input_data), fields=['calldata/GT', 'calldata/GQ'])
    assert (1, 4, 2) == callset['calldata/GT'].shape
    callset = read_vcf(path)
    assert (9, 2, 2) == callset['calldata/GT'].shape


# noinspection PyTypeChecker
def test_no_samples():

    input_data = (
        b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\n"
        b"2L\t12\tfoo\tA\tC,T\t12.3\tPASS\tfoo=bar\tGT:GQ\t0/0:99\t0/1:12\t./.:.\t.\n"
    )

    callset = read_vcf(io.BytesIO(input_data),
                       fields=['calldata/GT', 'calldata/GQ', 'samples', 'POS'])

    assert 'variants/POS' in callset
    assert 'samples' not in callset
    assert 'calldata/GT' not in callset
    assert 'calldata/GQ' not in callset

    h5_path = os.path.join(tempdir, 'sample.h5')
    if os.path.exists(h5_path):
        os.remove(h5_path)
    vcf_to_hdf5(io.BytesIO(input_data), h5_path,
                fields=['calldata/GT', 'calldata/GQ', 'samples', 'POS'])
    with h5py.File(h5_path, mode='r') as callset:
        assert 'variants/POS' in callset
        assert 'samples' not in callset
        assert 'calldata/GT' not in callset
        assert 'calldata/GQ' not in callset

    zarr_path = os.path.join(tempdir, 'sample.zarr')
    if os.path.exists(zarr_path):
        shutil.rmtree(zarr_path)
    vcf_to_zarr(io.BytesIO(input_data), zarr_path,
                fields=['calldata/GT', 'calldata/GQ', 'samples', 'POS'])
    callset = zarr.open_group(zarr_path, mode='r')
    assert 'variants/POS' in callset
    assert 'samples' not in callset
    assert 'calldata/GT' not in callset
    assert 'calldata/GQ' not in callset


def test_computed_fields():

    input_data = (b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\n"
                  b"2L\t2\t.\t.\t.\t.\t.\t.\t.\n"
                  b"2L\t4\t.\t.\tG\t.\t.\t.\t.\n"
                  b"2L\t12\t.\tA\t.\t.\t.\t.\t.\n"
                  b"2L\t34\t.\tC\tT\t.\t.\t.\t.\n"
                  b"3R\t45\t.\tG\tA,T\t.\t.\t.\t.\n"
                  b"3R\t47\t.\tG\tC,T,*\t.\t.\t.\t.\n"
                  b"3R\t56\t.\tG\tA,GTAC\t.\t.\t.\t.\n"
                  b"3R\t56\t.\tCATG\tC,GATG\t.\t.\t.\t.\n"
                  b"3R\t56\t.\tGTAC\tATAC,GTACTACTAC,G,GTACA,GTA\t.\t.\t.\t.\n")

    for string_dtype in 'S20', 'object':

        callset = read_vcf(io.BytesIO(input_data),
                           fields='*',
                           numbers={'ALT': 5},
                           types={'REF': string_dtype, 'ALT': string_dtype})

        a = callset['variants/ALT']
        assert (9, 5) == a.shape
        e = np.array([[b'', b'', b'', b'', b''],
                      [b'G', b'', b'', b'', b''],
                      [b'', b'', b'', b'', b''],
                      [b'T', b'', b'', b'', b''],
                      [b'A', b'T', b'', b'', b''],
                      [b'C', b'T', b'*', b'', b''],
                      [b'A', b'GTAC', b'', b'', b''],
                      [b'C', b'GATG', b'', b'', b''],
                      [b'ATAC', b'GTACTACTAC', b'G', b'GTACA', b'GTA']])
        if a.dtype.kind == 'O':
            e = e.astype('U').astype(object)
        assert_array_equal(e, a)

        a = callset['variants/numalt']
        assert (9,) == a.shape
        assert_array_equal([0, 1, 0, 1, 2, 3, 2, 2, 5], a)

        a = callset['variants/altlen']
        assert (9, 5) == a.shape
        e = np.array([[0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, -1, 0, 0],
                      [0, 3, 0, 0, 0],
                      [-3, 0, 0, 0, 0],
                      [0, 6, -3, 1, -1]])
        assert_array_equal(e, a)

        a = callset['variants/is_snp']
        assert (9,) == a.shape
        assert np.dtype(bool) == a.dtype
        assert_array_equal([False, False, False, True, True, False, False, False, False],
                           a)

        # test is_snp with reduced ALT number
        callset = read_vcf(io.BytesIO(input_data),
                           fields='*',
                           numbers={'ALT': 1},
                           types={'REF': string_dtype, 'ALT': string_dtype})

        a = callset['variants/ALT']
        assert (9,) == a.shape
        e = np.array([b'', b'G', b'', b'T', b'A', b'C', b'A', b'C', b'ATAC'])
        if a.dtype.kind == 'O':
            e = e.astype('U').astype(object)
        assert_array_equal(e, a)

        a = callset['variants/numalt']
        assert (9,) == a.shape
        assert_array_equal([0, 1, 0, 1, 2, 3, 2, 2, 5], a)

        a = callset['variants/altlen']
        assert (9,) == a.shape
        e = np.array([0, 1, 0, 0, 0, 0, 0, -3, 0])
        assert_array_equal(e, a)

        a = callset['variants/is_snp']
        assert (9,) == a.shape
        assert np.dtype(bool) == a.dtype
        assert_array_equal([False, False, False, True, True, False, False, False, False],
                           a)


def test_genotype_ac():

    input_data = (
        b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\tS3\n"
        b"2L\t12\t.\tA\t.\t.\t.\t.\tGT:GQ\t0/0/0:11\t0/1/2:12\t././.:.\n"
        b"2L\t34\t.\tC\tT\t.\t.\t.\tGT:GQ\t0/1/2:22\t3/3/.:33\t.\n"
        b"3R\t45\t.\tG\tA,T\t.\t.\t.\tGT:GQ\t0/1:.\t3:12\t\n"
        b"X\t55\t.\tG\tA,T\t.\t.\t.\tGT:GQ\t0/1/1/3/4:.\t1/1/2/2/4/4/5:12\t0/0/1/2/3/./4\n"
    )

    for t in 'i1', 'i2', 'i4', 'i8', 'u1', 'u2', 'u4', 'u8':
        callset = read_vcf(io.BytesIO(input_data),
                           fields=['calldata/GT'],
                           numbers={'calldata/GT': 4},
                           types={'calldata/GT': 'genotype_ac/' + t})
        e = np.array([[[3, 0, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0]],
                      [[1, 1, 1, 0], [0, 0, 0, 2], [0, 0, 0, 0]],
                      [[1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]],
                      [[1, 2, 0, 1], [0, 2, 2, 0], [2, 1, 1, 1]]], dtype=t)
        a = callset['calldata/GT']
        assert e.dtype == a.dtype
        assert_array_equal(e, a)

    vcf_path = fixture_path('test63.vcf')
    callset = read_vcf(vcf_path, fields='GT', numbers={'GT': 3},
                       types={'GT': 'genotype_ac/i1'})
    e = np.array([
        [(2, 0, 0), (3, 0, 0), (1, 0, 0)],
        [(0, 1, 0), (1, 1, 0), (1, 1, 1)],
        [(0, 0, 0), (0, 0, 0), (0, 0, 0)],
        [(0, 0, 0), (0, 0, 0), (0, 0, 0)],
    ])
    a = callset['calldata/GT']
    assert_array_equal(e, a)


def test_region_truncate():
    vcf_path = fixture_path('test54.vcf.gz')
    for tabix in 'tabix', None:
        callset = read_vcf(vcf_path, region='chr1:10-100', tabix=tabix)
        pos = callset['variants/POS']
        assert 2 == pos.shape[0]
        assert_array_equal([20, 30], pos)


def test_errors():

    # try to open a directory
    path = '.'
    with pytest.raises(OSError):
        read_vcf(path)

    # try to open a file that doesn't exist
    path = 'doesnotexist.vcf'
    with pytest.raises(FileNotFoundError):
        read_vcf(path)

    # try to open a file that doesn't exist
    path = 'doesnotexist.vcf.gz'
    with pytest.raises(FileNotFoundError):
        read_vcf(path)

    # file is nothing like a VCF (has no header)
    path = fixture_path('test48a.vcf')
    with pytest.raises(RuntimeError):
        read_vcf(path)


def test_dup_headers():

    warnings.resetwarnings()
    warnings.simplefilter('error')

    # dup FILTER
    input_data = b"""##fileformat=VCFv4.1
##FILTER=<ID=s50,Description="Less than 50% of samples have data">
##FILTER=<ID=s50,Description="Less than 50% of samples have data">
##INFO=<ID=DP,Number=1,Type=Integer,Description="Total Depth">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=AD,Number=A,Type=Integer,Description="Allele Depths">
##FORMAT=<ID=ZZ,Number=1,Type=String,Description="ZZ">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	test1	test2	test3	test4
chr1	1	.	A	G	.	PASS	DP=2	GT:AD	0:1,0	.:1,0	0:0,0	.:0,0
chr1	2	.	A	G	.	PASS	DP=2	GT:AD:ZZ	0:1,0:dummy	0:1,0	0:0,0	.:0,0
chr1	3	.	A	G	.	PASS	DP=2	GT:AD:ZZ	0:1,0:dummy	1:1,0	.	./.
"""
    with pytest.warns(UserWarning):
        read_vcf(io.BytesIO(input_data))

    # dup INFO
    input_data = b"""##fileformat=VCFv4.1
##FILTER=<ID=s50,Description="Less than 50% of samples have data">
##INFO=<ID=DP,Number=1,Type=Integer,Description="Total Depth">
##INFO=<ID=DP,Number=1,Type=Integer,Description="Total Depth">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=AD,Number=A,Type=Integer,Description="Allele Depths">
##FORMAT=<ID=ZZ,Number=1,Type=String,Description="ZZ">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	test1	test2	test3	test4
chr1	1	.	A	G	.	PASS	DP=2	GT:AD	0:1,0	.:1,0	0:0,0	.:0,0
chr1	2	.	A	G	.	PASS	DP=2	GT:AD:ZZ	0:1,0:dummy	0:1,0	0:0,0	.:0,0
chr1	3	.	A	G	.	PASS	DP=2	GT:AD:ZZ	0:1,0:dummy	1:1,0	.	./.
"""
    with pytest.warns(UserWarning):
        read_vcf(io.BytesIO(input_data))

    # dup FORMAT
    input_data = b"""##fileformat=VCFv4.1
##FILTER=<ID=s50,Description="Less than 50% of samples have data">
##INFO=<ID=DP,Number=1,Type=Integer,Description="Total Depth">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=AD,Number=A,Type=Integer,Description="Allele Depths">
##FORMAT=<ID=AD,Number=A,Type=Integer,Description="Allele Depths">
##FORMAT=<ID=ZZ,Number=1,Type=String,Description="ZZ">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	test1	test2	test3	test4
chr1	1	.	A	G	.	PASS	DP=2	GT:AD	0:1,0	.:1,0	0:0,0	.:0,0
chr1	2	.	A	G	.	PASS	DP=2	GT:AD:ZZ	0:1,0:dummy	0:1,0	0:0,0	.:0,0
chr1	3	.	A	G	.	PASS	DP=2	GT:AD:ZZ	0:1,0:dummy	1:1,0	.	./.
"""
    with pytest.warns(UserWarning):
        read_vcf(io.BytesIO(input_data))

    warnings.resetwarnings()
    warnings.simplefilter('always')


def test_override_vcf_type():
    vcf_path = fixture_path('test4.vcf')
    callset = read_vcf(vcf_path, fields=['MQ0FractionTest'])
    assert 0 == callset['variants/MQ0FractionTest'][2]
    callset = read_vcf(vcf_path, fields=['MQ0FractionTest'],
                       types={'MQ0FractionTest': 'Float'})
    assert approx(0.03) == callset['variants/MQ0FractionTest'][2]


def test_header_overrides_default_vcf_type():
    vcf_path = fixture_path('test176.vcf')
    callset = read_vcf(vcf_path, fields='*')
    gq = callset['calldata/GQ']
    assert 'f' == gq.dtype.kind
    assert np.isnan(gq[0, 0])
    assert approx(48.2) == gq[2, 0]
    assert approx(48.1) == gq[2, 1]
    assert approx(43.9) == gq[2, 2]
    assert approx(49.) == gq[3, 0]
    assert approx(3.) == gq[3, 1]
    assert approx(41.) == gq[3, 2]


def test_missing_calldata():
    vcf_path = fixture_path('test1.vcf')
    callset = read_vcf(vcf_path, fields='calldata/*', numbers={'AD': 2})
    gt = callset['calldata/GT']
    ad = callset['calldata/AD']
    assert (-1, -1) == tuple(gt[0, 1])
    assert (1, 0) == tuple(ad[0, 1])
    assert (-1, -1) == tuple(gt[2, 2])
    assert (-1, -1) == tuple(ad[2, 2])
    assert (-1, -1) == tuple(gt[2, 3])
    assert (-1, -1) == tuple(ad[2, 3])


def test_calldata_cleared():
    vcf_path = fixture_path('test32.vcf')
    callset = read_vcf(vcf_path, fields=['calldata/GT', 'calldata/DP', 'calldata/GQ'])
    gt = callset['calldata/GT']
    dp = callset['calldata/DP']
    gq = callset['calldata/GQ']
    assert (0, 0) == tuple(gt[0, 3])
    assert 8 == dp[0, 3]
    assert 3 == gq[0, 3]
    assert (-1, -1) == tuple(gt[1, 3])
    assert -1 == dp[1, 3]
    assert -1 == gq[1, 3]


def test_calldata_quirks():
    vcf_path = fixture_path('test1.vcf')
    callset = read_vcf(vcf_path, fields=['AD', 'GT'], numbers={'AD': 2})
    gt = callset['calldata/GT']
    ad = callset['calldata/AD']
    e = np.array([[-1, -1], [0, -1], [1, -1]])
    assert_array_equal(e, gt[:, 1])
    e = np.array([[1, 0], [1, 0], [1, 0]])
    assert_array_equal(e, ad[:, 1])


def test_vcf_to_npz():
    vcf_paths = [fixture_path(x) for x in ['sample.vcf', 'sample.vcf.gz']]
    npz_path = os.path.join(tempdir, 'sample.npz')
    region_values = None, '20', '20:10000-20000', 'Y'
    tabix_values = 'tabix', None
    samples_values = None, ['NA00001', 'NA00003']
    string_type_values = 'S10', 'object'
    param_matrix = itertools.product(vcf_paths, region_values, tabix_values,
                                     samples_values, string_type_values)
    for vcf_path, region, tabix, samples, string_type in param_matrix:
        types = {'CHROM': string_type, 'ALT': string_type, 'samples': string_type}
        expected = read_vcf(vcf_path, fields='*', alt_number=2, region=region,
                            tabix=tabix, samples=samples, types=types)
        if os.path.exists(npz_path):
            os.remove(npz_path)
        vcf_to_npz(vcf_path, npz_path, fields='*', chunk_length=2, alt_number=2,
                   region=region, tabix=tabix, samples=samples, types=types)
        if expected is None:
            assert not os.path.exists(npz_path)
        else:
            actual = np.load(npz_path, allow_pickle=True)
            for key in expected.keys():
                if expected[key].dtype.kind == 'f':
                    assert_array_almost_equal(expected[key], actual[key])
                else:
                    assert_array_equal(expected[key], actual[key])
            for key in actual.keys():
                assert key in expected
            actual.close()


def test_vcf_to_npz_exclude():
    vcf_path = fixture_path('sample.vcf')
    npz_path = os.path.join(tempdir, 'sample.npz')
    exclude = ['variants/altlen', 'ID', 'calldata/DP']
    expected = read_vcf(vcf_path, fields='*', exclude_fields=exclude)
    if os.path.exists(npz_path):
        os.remove(npz_path)
    vcf_to_npz(vcf_path, npz_path, fields='*', exclude_fields=exclude)
    actual = np.load(npz_path, allow_pickle=True)
    for key in expected.keys():
        if expected[key].dtype.kind == 'f':
            assert_array_almost_equal(expected[key], actual[key])
        else:
            assert_array_equal(expected[key], actual[key])
    for key in actual.keys():
        assert key in expected
    actual.close()


def test_vcf_to_npz_rename():
    vcf_path = fixture_path('sample.vcf')
    npz_path = os.path.join(tempdir, 'sample.npz')
    rename = {'CHROM': 'variants/chromosome',
              'variants/altlen': 'spam/eggs',
              'calldata/GT': 'foo/bar'}
    expected = read_vcf(vcf_path, fields='*', rename_fields=rename)
    if os.path.exists(npz_path):
        os.remove(npz_path)
    vcf_to_npz(vcf_path, npz_path, fields='*', rename_fields=rename)
    actual = np.load(npz_path, allow_pickle=True)
    for key in expected.keys():
        if expected[key].dtype.kind == 'f':
            assert_array_almost_equal(expected[key], actual[key])
        else:
            assert_array_equal(expected[key], actual[key])
    for key in actual.keys():
        assert key in expected
    actual.close()


def test_vcf_to_zarr():
    vcf_paths = [fixture_path(x) for x in ['sample.vcf', 'sample.vcf.gz']]
    zarr_path = os.path.join(tempdir, 'sample.zarr')
    region_values = None, '20', '20:10000-20000', 'Y'
    tabix_values = 'tabix', None
    samples_values = None, ['NA00001', 'NA00003']
    string_type_values = 'S10', 'object'
    param_matrix = itertools.product(vcf_paths, region_values, tabix_values,
                                     samples_values, string_type_values)
    for vcf_path, region, tabix, samples, string_type in param_matrix:
        types = {'CHROM': string_type, 'ALT': string_type, 'samples': string_type}
        expected = read_vcf(vcf_path, fields='*', alt_number=2, region=region,
                            tabix=tabix, samples=samples, types=types)
        if os.path.exists(zarr_path):
            shutil.rmtree(zarr_path)
        vcf_to_zarr(vcf_path, zarr_path, fields='*', alt_number=2, chunk_length=2,
                    region=region, tabix=tabix, samples=samples, types=types)
        if expected is None:
            assert not os.path.exists(zarr_path)
        else:
            actual = zarr.open_group(zarr_path, mode='r')
            for key in expected.keys():
                e = expected[key]
                a = actual[key][:]
                compare_arrays(e, a)
                assert (actual['variants/NS'].attrs['Description'] ==
                        'Number of Samples With Data')
                assert (actual['calldata/GQ'].attrs['Description'] ==
                        'Genotype Quality')
            for key in actual.keys():
                if key not in {'variants', 'calldata'}:
                    assert key in expected
            for key in actual['variants'].keys():
                assert 'variants/' + key in expected
            for key in actual['calldata'].keys():
                assert 'calldata/' + key in expected


def test_vcf_to_zarr_exclude():
    vcf_path = fixture_path('sample.vcf')
    zarr_path = os.path.join(tempdir, 'sample.zarr')
    exclude = ['variants/altlen', 'ID', 'calldata/DP']
    expected = read_vcf(vcf_path, fields='*', exclude_fields=exclude)
    if os.path.exists(zarr_path):
        shutil.rmtree(zarr_path)
    vcf_to_zarr(vcf_path, zarr_path, fields='*', exclude_fields=exclude)
    actual = zarr.open_group(zarr_path, mode='r')
    for key in expected.keys():
        e = expected[key]
        a = actual[key][:]
        compare_arrays(e, a)
    for key in actual.keys():
        if key not in {'variants', 'calldata'}:
            assert key in expected
    for key in actual['variants'].keys():
        assert 'variants/' + key in expected
    for key in actual['calldata'].keys():
        assert 'calldata/' + key in expected


def test_vcf_to_zarr_rename():
    vcf_path = fixture_path('sample.vcf')
    zarr_path = os.path.join(tempdir, 'sample.zarr')
    rename = {'CHROM': 'variants/chromosome',
              'variants/altlen': 'spam/eggs',
              'calldata/GT': 'foo/bar'}
    expected = read_vcf(vcf_path, fields='*', rename_fields=rename)
    if os.path.exists(zarr_path):
        shutil.rmtree(zarr_path)
    vcf_to_zarr(vcf_path, zarr_path, fields='*', rename_fields=rename)
    actual = zarr.open_group(zarr_path, mode='r')
    for key in expected.keys():
        e = expected[key]
        a = actual[key][:]
        compare_arrays(e, a)
    for key in actual['variants'].keys():
        assert 'variants/' + key in expected
    for key in actual['calldata'].keys():
        assert 'calldata/' + key in expected


def test_vcf_to_zarr_rename_clash():
    vcf_path = fixture_path('sample.vcf')
    zarr_path = os.path.join(tempdir, 'sample.zarr')

    # dup values
    rename = {'CHROM': 'variants/chromosome',
              'variants/altlen': 'spam/eggs',
              'calldata/GT': 'spam/eggs'}
    with pytest.raises(ValueError):
        vcf_to_zarr(vcf_path, zarr_path, fields='*', rename_fields=rename)

    # parent clash
    rename = {'CHROM': 'variants/chromosome',
              'variants/altlen': 'spam/eggs',
              'calldata/GT': 'spam'}
    with pytest.raises(ValueError):
        vcf_to_zarr(vcf_path, zarr_path, fields='*', rename_fields=rename)

    # parent clash
    rename = {'CHROM': 'variants/chromosome',
              'variants/altlen': 'spam',
              'calldata/GT': 'spam/eggs'}
    with pytest.raises(ValueError):
        vcf_to_zarr(vcf_path, zarr_path, fields='*', rename_fields=rename)


def test_vcf_to_zarr_dup_fields_case_insensitive():
    vcf_path = fixture_path('altlen.vcf')
    zarr_path = os.path.join(tempdir, 'sample.zarr')
    with pytest.raises(ValueError):
        vcf_to_zarr(vcf_path, zarr_path, fields=['ALTLEN', 'altlen'])
    with pytest.raises(ValueError):
        vcf_to_zarr(vcf_path, zarr_path, fields=['variants/ALTLEN', 'variants/altlen'])
    # should be fine if renamed
    vcf_to_zarr(vcf_path, zarr_path, fields=['ALTLEN', 'altlen'],
                rename_fields={'altlen': 'variants/spam'})


def test_vcf_to_zarr_group():
    vcf_path = fixture_path('sample.vcf.gz')
    zarr_path = os.path.join(tempdir, 'sample.zarr')
    if os.path.exists(zarr_path):
        shutil.rmtree(zarr_path)
    chroms = ['19', '20', 'X']
    for chrom in chroms:
        vcf_to_zarr(vcf_path, zarr_path, fields='*', alt_number=2, chunk_length=2,
                    region=chrom, group=chrom)
    actual = zarr.open_group(zarr_path, mode='r')
    assert chroms == sorted(actual)
    for chrom in chroms:
        assert ['calldata', 'samples', 'variants'] == sorted(actual[chrom])
        expect = read_vcf(vcf_path, fields='*', alt_number=2, region=chrom)
        for key in expect.keys():
            e = expect[key]
            a = actual[chrom][key][:]
            compare_arrays(e, a)
            assert (actual[chrom]['variants/NS'].attrs['Description'] ==
                    'Number of Samples With Data')
            assert (actual[chrom]['calldata/GQ'].attrs['Description'] ==
                    'Genotype Quality')


def test_vcf_to_zarr_string_codec():
    vcf_path = fixture_path('sample.vcf')
    zarr_path = os.path.join(tempdir, 'sample.zarr')
    types = {'CHROM': object, 'ALT': object, 'samples': object}
    expect = read_vcf(vcf_path, fields='*', alt_number=2, types=types)
    if os.path.exists(zarr_path):
        shutil.rmtree(zarr_path)
    vcf_to_zarr(vcf_path, zarr_path, fields='*', alt_number=2, chunk_length=2,
                types=types)
    actual = zarr.open_group(zarr_path, mode='r')
    for key in expect.keys():
        e = expect[key]
        a = actual[key][:]
        compare_arrays(e, a)


def test_vcf_to_zarr_ann():
    vcf_path = fixture_path('ann.vcf')
    zarr_path = os.path.join(tempdir, 'ann.zarr')
    for string_type in 'S10', 'object':
        types = {'CHROM': string_type, 'ALT': string_type, 'samples': string_type}
        transformers = [ANNTransformer(fields=['Allele', 'HGVS_c', 'AA'],
                                       types={'Allele': string_type,
                                              'HGVS_c': string_type})]
        expected = read_vcf(vcf_path, fields='*', alt_number=2, types=types,
                            transformers=transformers)
        if os.path.exists(zarr_path):
            shutil.rmtree(zarr_path)
        vcf_to_zarr(vcf_path, zarr_path, fields='*', alt_number=2, chunk_length=2,
                    types=types, transformers=transformers)
        actual = zarr.open_group(zarr_path, mode='r')
        for key in expected.keys():
            compare_arrays(expected[key], actual[key][:])


def test_vcf_to_zarr_empty():
    vcf_path = fixture_path('empty.vcf')
    zarr_path = os.path.join(tempdir, 'empty.zarr')
    vcf_to_zarr(vcf_path, zarr_path)
    assert not os.path.exists(zarr_path)


def test_vcf_to_hdf5():
    vcf_paths = [fixture_path(x) for x in ['sample.vcf', 'sample.vcf.gz']]
    h5_path = os.path.join(tempdir, 'sample.h5')
    region_values = None, '20', '20:10000-20000', 'Y'
    tabix_values = 'tabix', None
    samples_values = None, ['NA00001', 'NA00003']
    string_type_values = 'S10', 'object'
    param_matrix = itertools.product(vcf_paths, region_values, tabix_values,
                                     samples_values, string_type_values)
    for vcf_path, region, tabix, samples, string_type in param_matrix:
        types = {'CHROM': string_type, 'ALT': string_type, 'samples': string_type}
        expected = read_vcf(vcf_path, fields='*', alt_number=2, region=region,
                            tabix=tabix, samples=samples, types=types)
        if os.path.exists(h5_path):
            os.remove(h5_path)
        vcf_to_hdf5(vcf_path, h5_path, fields='*', alt_number=2, chunk_length=2,
                    region=region, tabix=tabix, samples=samples, types=types)
        if expected is None:
            assert not os.path.exists(h5_path)
        else:
            with h5py.File(h5_path, mode='r') as actual:
                for key in expected.keys():
                    expect_arr = expected[key]
                    actual_arr = actual[key]
                    if expect_arr.dtype == object:
                        # deal with recent h5py change, need to get h5py to read
                        # array elements as str objects
                        actual_arr = actual_arr.asstr()
                    compare_arrays(expect_arr, actual_arr[:])
                assert (actual['variants/NS'].attrs['Description'] ==
                        'Number of Samples With Data')
                assert (actual['calldata/GQ'].attrs['Description'] ==
                        'Genotype Quality')
                for key in actual.keys():
                    if key not in {'variants', 'calldata'}:
                        assert key in expected
                for key in actual['variants'].keys():
                    assert 'variants/' + key in expected
                for key in actual['calldata'].keys():
                    assert 'calldata/' + key in expected


def test_vcf_to_hdf5_exclude():
    vcf_path = fixture_path('sample.vcf')
    h5_path = os.path.join(tempdir, 'sample.h5')
    exclude = ['variants/altlen', 'ID', 'calldata/DP']
    expected = read_vcf(vcf_path, fields='*', exclude_fields=exclude)
    if os.path.exists(h5_path):
        os.remove(h5_path)
    vcf_to_hdf5(vcf_path, h5_path, fields='*', exclude_fields=exclude)
    with h5py.File(h5_path, mode='r') as actual:
        for key in expected.keys():
            expect_arr = expected[key]
            actual_arr = actual[key]
            if expect_arr.dtype == object:
                # deal with recent h5py change, need to get h5py to read
                # array elements as str objects
                actual_arr = actual_arr.asstr()
            compare_arrays(expect_arr, actual_arr[:])
        for key in actual.keys():
            if key not in {'variants', 'calldata'}:
                assert key in expected
        for key in actual['variants'].keys():
            assert 'variants/' + key in expected
        for key in actual['calldata'].keys():
            assert 'calldata/' + key in expected


def test_vcf_to_hdf5_rename():
    vcf_path = fixture_path('sample.vcf')
    h5_path = os.path.join(tempdir, 'sample.h5')
    rename = {'CHROM': 'variants/chromosome',
              'variants/altlen': 'spam/eggs',
              'calldata/GT': 'foo/bar'}
    expected = read_vcf(vcf_path, fields='*', rename_fields=rename)
    if os.path.exists(h5_path):
        os.remove(h5_path)
    vcf_to_hdf5(vcf_path, h5_path, fields='*', rename_fields=rename)
    with h5py.File(h5_path, mode='r') as actual:
        for key in expected.keys():
            expect_arr = expected[key]
            actual_arr = actual[key]
            if expect_arr.dtype == object:
                # deal with recent h5py change, need to get h5py to read
                # array elements as str objects
                actual_arr = actual_arr.asstr()
            compare_arrays(expect_arr, actual_arr[:])
        for key in actual['variants'].keys():
            assert 'variants/' + key in expected
        for key in actual['calldata'].keys():
            assert 'calldata/' + key in expected


def test_vcf_to_hdf5_group():
    vcf_path = fixture_path('sample.vcf.gz')
    h5_path = os.path.join(tempdir, 'sample.h5')
    if os.path.exists(h5_path):
        os.remove(h5_path)
    chroms = ['19', '20', 'X']
    for chrom in chroms:
        vcf_to_hdf5(vcf_path, h5_path, fields='*', alt_number=2, chunk_length=2,
                    region=chrom, group=chrom)
    with h5py.File(h5_path, mode='r') as actual:
        assert chroms == sorted(actual)
        for chrom in chroms:
            assert ['calldata', 'samples', 'variants'] == sorted(actual[chrom])
            expect = read_vcf(vcf_path, fields='*', alt_number=2, region=chrom)
            for key in expect.keys():
                expect_arr = expect[key]
                actual_arr = actual[chrom][key]
                if expect_arr.dtype == object:
                    # deal with recent h5py change, need to get h5py to read
                    # array elements as str objects
                    actual_arr = actual_arr.asstr()
                compare_arrays(expect_arr, actual_arr[:])
                assert (actual[chrom]['variants/NS'].attrs['Description'] ==
                        'Number of Samples With Data')
                assert (actual[chrom]['calldata/GQ'].attrs['Description'] ==
                        'Genotype Quality')


def test_vcf_to_hdf5_ann():
    vcf_path = fixture_path('ann.vcf')
    h5_path = os.path.join(tempdir, 'ann.h5')
    for string_type in 'S10', 'object':
        types = {'CHROM': string_type, 'ALT': string_type, 'samples': string_type}
        transformers = [ANNTransformer(fields=['Allele', 'HGVS_c', 'AA'],
                                       types={'Allele': string_type,
                                              'HGVS_c': string_type})]
        expected = read_vcf(vcf_path, fields='*', types=types, transformers=transformers)
        if os.path.exists(h5_path):
            os.remove(h5_path)
        vcf_to_hdf5(vcf_path, h5_path, fields='*', chunk_length=2, types=types,
                    transformers=transformers)
        with h5py.File(h5_path, mode='r') as actual:
            for key in expected.keys():
                expect_arr = expected[key]
                actual_arr = actual[key]
                if expect_arr.dtype == object:
                    # deal with recent h5py change, need to get h5py to read
                    # array elements as str objects
                    actual_arr = actual_arr.asstr()
                compare_arrays(expect_arr, actual_arr[:])


def test_vcf_to_hdf5_vlen():
    vcf_path = fixture_path('sample.vcf')
    h5_path = os.path.join(tempdir, 'sample.h5')
    fields = ['CHROM', 'ID', 'samples']
    for string_type in 'S10', 'object':
        types = {'CHROM': string_type, 'ID': string_type, 'samples': string_type}
        expect = read_vcf(vcf_path, fields=fields, alt_number=2, types=types)
        if os.path.exists(h5_path):
            os.remove(h5_path)
        vcf_to_hdf5(vcf_path, h5_path, fields=fields, alt_number=2, chunk_length=3,
                    types=types, vlen=False)
        with h5py.File(h5_path, mode='r') as actual:
            for key in expect.keys():
                if expect[key].dtype.kind == 'f':
                    assert_array_almost_equal(expect[key], actual[key][:])
                elif expect[key].dtype.kind == 'O':
                    # strings always stored as fixed length if vlen=False
                    assert 'S' == actual[key].dtype.kind
                    assert_array_equal(expect[key].astype('S'), actual[key][:])
                else:
                    assert_array_equal(expect[key], actual[key][:])


def test_vcf_to_hdf5_empty():
    vcf_path = fixture_path('empty.vcf')
    h5_path = os.path.join(tempdir, 'empty.h5')
    vcf_to_hdf5(vcf_path, h5_path)
    assert not os.path.exists(h5_path)


def to_pandas_expectation(e):
    # expect that all string fields end up as objects with nans for missing
    if e.dtype.kind == 'S':
        e = e.astype('U').astype(object)
    if e.dtype == object:
        e[e == ''] = np.nan
    return e


def check_dataframe(callset, df):
    for k in callset:
        if k.startswith('variants/'):
            group, name = k.split('/')
            e = to_pandas_expectation(callset[k])
            if e.ndim == 1:
                compare_arrays(e, df[name].values)
            elif e.ndim == 2:
                for i in range(e.shape[1]):
                    compare_arrays(e[:, i], df['%s_%s' % (name, i + 1)])


def test_vcf_to_dataframe():
    vcf_path = fixture_path('sample.vcf')
    fields = ['CHROM', 'POS', 'REF', 'ALT', 'DP', 'AC', 'GT']
    numbers = {'AC': 3}
    for string_type in 'S10', 'object':
        types = {'CHROM': string_type, 'ALT': string_type}
        callset = read_vcf(vcf_path, fields=fields, alt_number=2, numbers=numbers,
                           types=types)
        df = vcf_to_dataframe(vcf_path, fields=fields, alt_number=2, numbers=numbers,
                              chunk_length=2, types=types)
        assert (['CHROM', 'POS', 'REF', 'ALT_1', 'ALT_2', 'DP', 'AC_1', 'AC_2', 'AC_3'] ==
                df.columns.tolist())
        # always convert strings to object dtype for pandas
        assert np.dtype(object) == df['CHROM'].dtype
        assert np.dtype(object) == df['ALT_1'].dtype
        check_dataframe(callset, df)


def test_vcf_to_dataframe_all():
    vcf_path = fixture_path('sample.vcf')
    fields = '*'
    numbers = {'AC': 3}
    for string_type in 'S10', 'object':
        types = {'CHROM': string_type, 'ALT': string_type}
        callset = read_vcf(vcf_path, fields=fields, alt_number=2, numbers=numbers,
                           types=types)
        df = vcf_to_dataframe(vcf_path, fields=fields, alt_number=2, numbers=numbers,
                              chunk_length=2, types=types)
        for k in ['CHROM', 'POS', 'ID', 'REF', 'ALT_1', 'ALT_2', 'DP', 'AC_1',
                  'AC_2', 'AC_3']:
            assert k in df.columns.tolist()
        # always convert strings to object dtype for pandas
        assert np.dtype(object) == df['CHROM'].dtype
        assert np.dtype(object) == df['ALT_1'].dtype
        check_dataframe(callset, df)


def test_vcf_to_dataframe_exclude():
    vcf_path = fixture_path('sample.vcf')
    fields = '*'
    exclude = ['ALT', 'ID']
    df = vcf_to_dataframe(vcf_path, fields=fields, exclude_fields=exclude)
    for k in ['CHROM', 'POS', 'REF', 'DP', 'AC_1', 'AC_2', 'AC_3']:
        assert k in df.columns.tolist()
    for k in ['ALT_1', 'ALT_2', 'ID']:
        assert k not in df.columns.tolist()


def test_vcf_to_dataframe_ann():
    vcf_path = fixture_path('ann.vcf')
    fields = ['CHROM', 'POS', 'REF', 'ALT', 'ANN', 'DP', 'AC', 'GT']
    numbers = {'AC': 2, 'ALT': 2}
    for string_type in 'S10', 'object':
        types = {'CHROM': string_type, 'ALT': string_type}
        transformers = [ANNTransformer(fields=['Allele', 'HGVS_c', 'AA'],
                                       types={'Allele': string_type,
                                              'HGVS_c': string_type})]
        callset = read_vcf(vcf_path, fields=fields, numbers=numbers, types=types,
                           transformers=transformers)
        df = vcf_to_dataframe(vcf_path, fields=fields, numbers=numbers, chunk_length=2,
                              types=types, transformers=transformers)
        assert (['CHROM', 'POS', 'REF', 'ALT_1', 'ALT_2', 'ANN_Allele', 'ANN_HGVS_c',
                 'ANN_AA_pos', 'ANN_AA_length', 'DP', 'AC_1', 'AC_2'] ==
                df.columns.tolist())
        # always convert strings to object dtype for pandas
        assert np.dtype(object) == df['CHROM'].dtype
        assert np.dtype(object) == df['ALT_1'].dtype
        check_dataframe(callset, df)


def test_vcf_to_csv():
    vcf_path = fixture_path('sample.vcf')
    fields = ['CHROM', 'POS', 'REF', 'ALT', 'DP', 'AC', 'GT']
    numbers = {'AC': 3}
    for string_type in 'S20', 'object':
        types = {'REF': string_type, 'ALT': string_type}
        df = vcf_to_dataframe(vcf_path, fields=fields, alt_number=2, numbers=numbers,
                              types=types, chunk_length=2)
        csv_path = os.path.join(tempdir, 'test.csv')
        if os.path.exists(csv_path):
            os.remove(csv_path)
        vcf_to_csv(vcf_path, csv_path, fields=fields, alt_number=2, numbers=numbers,
                   types=types, chunk_length=2)
        import pandas
        adf = pandas.read_csv(csv_path, na_filter=True)
        assert df.columns.tolist() == adf.columns.tolist()
        for k in df.columns:
            compare_arrays(df[k].values, adf[k].values)


def test_vcf_to_csv_all():
    vcf_path = fixture_path('sample.vcf')
    fields = '*'
    df = vcf_to_dataframe(vcf_path, fields=fields)
    csv_path = os.path.join(tempdir, 'test.csv')
    if os.path.exists(csv_path):
        os.remove(csv_path)
    vcf_to_csv(vcf_path, csv_path, fields=fields)
    import pandas
    adf = pandas.read_csv(csv_path, na_filter=True)
    assert df.columns.tolist() == adf.columns.tolist()
    for k in df.columns:
        compare_arrays(df[k].values, adf[k].values)


def test_vcf_to_csv_exclude():
    vcf_path = fixture_path('sample.vcf')
    fields = '*'
    exclude = ['ALT', 'ID']
    df = vcf_to_dataframe(vcf_path, fields=fields, exclude_fields=exclude)
    csv_path = os.path.join(tempdir, 'test.csv')
    if os.path.exists(csv_path):
        os.remove(csv_path)
    vcf_to_csv(vcf_path, csv_path, fields=fields, exclude_fields=exclude)
    import pandas
    adf = pandas.read_csv(csv_path, na_filter=True)
    assert df.columns.tolist() == adf.columns.tolist()


def test_vcf_to_csv_ann():
    vcf_path = fixture_path('ann.vcf')
    fields = ['CHROM', 'POS', 'REF', 'ALT', 'DP', 'AC', 'ANN', 'GT']
    numbers = {'AC': 2, 'ALT': 2}
    for string_type in 'S20', 'object':
        types = {'CHROM': string_type, 'REF': string_type, 'ALT': string_type}
        transformers = [ANNTransformer(fields=['Allele', 'HGVS_c', 'AA'],
                                       types={'Allele': string_type,
                                              'HGVS_c': string_type})]
        df = vcf_to_dataframe(vcf_path, fields=fields, numbers=numbers, types=types,
                              chunk_length=2, transformers=transformers)
        csv_path = os.path.join(tempdir, 'test.csv')
        if os.path.exists(csv_path):
            os.remove(csv_path)
        vcf_to_csv(vcf_path, csv_path, fields=fields, numbers=numbers, types=types,
                   chunk_length=2, transformers=transformers)
        import pandas
        adf = pandas.read_csv(csv_path, na_filter=True)
        assert df.columns.tolist() == adf.columns.tolist()
        for k in df.columns:
            compare_arrays(df[k].values, adf[k].values)


def test_vcf_to_recarray():
    vcf_path = fixture_path('sample.vcf')
    fields = ['CHROM', 'POS', 'REF', 'ALT', 'DP', 'AC', 'GT']
    numbers = {'AC': 3}
    for string_type in 'S20', 'object':
        types = {'CHROM': string_type, 'REF': string_type, 'ALT': string_type}
        callset = read_vcf(vcf_path, fields=fields, alt_number=2, numbers=numbers,
                           types=types)
        a = vcf_to_recarray(vcf_path, fields=fields, alt_number=2, numbers=numbers,
                            chunk_length=2, types=types)
        assert (['CHROM', 'POS', 'REF', 'ALT_1', 'ALT_2', 'DP', 'AC_1', 'AC_2', 'AC_3']
                == list(a.dtype.names))
        assert np.dtype(string_type) == a['CHROM'].dtype
        for k in callset:
            if k.startswith('variants/'):
                group, name = k.split('/')
                e = callset[k]
                if e.ndim == 1:
                    assert_array_equal(e, a[name])
                elif e.ndim == 2:
                    for i in range(e.shape[1]):
                        assert_array_equal(e[:, i], a['%s_%s' % (name, i + 1)])
                else:
                    assert False, (k, e.ndim)


def test_vcf_to_recarray_all():
    vcf_path = fixture_path('sample.vcf')
    fields = '*'
    numbers = {'AC': 3}
    for string_type in 'S20', 'object':
        types = {'CHROM': string_type, 'REF': string_type, 'ALT': string_type}
        callset = read_vcf(vcf_path, fields=fields, alt_number=2, numbers=numbers,
                           types=types)
        a = vcf_to_recarray(vcf_path, fields=fields, alt_number=2, numbers=numbers,
                            chunk_length=2, types=types)
        for k in ['CHROM', 'POS', 'ID', 'REF', 'ALT_1', 'ALT_2', 'DP', 'AC_1',
                  'AC_2', 'AC_3']:
            assert k in a.dtype.names
        assert np.dtype(string_type) == a['CHROM'].dtype
        for k in callset:
            if k.startswith('variants/'):
                group, name = k.split('/')
                e = callset[k]
                if e.ndim == 1:
                    assert_array_equal(e, a[name])
                elif e.ndim == 2:
                    for i in range(e.shape[1]):
                        assert_array_equal(e[:, i], a['%s_%s' % (name, i + 1)])
                else:
                    assert False, (k, e.ndim)


def test_vcf_to_recarray_exclude():
    vcf_path = fixture_path('sample.vcf')
    fields = '*'
    exclude = ['ALT', 'ID']
    a = vcf_to_recarray(vcf_path, fields=fields, exclude_fields=exclude)
    for k in ['CHROM', 'POS', 'REF', 'DP', 'AC_1', 'AC_2', 'AC_3']:
        assert k in a.dtype.names
    for k in 'ALT_1', 'ALT_2', 'ALT', 'ID':
        assert k not in a.dtype.names


def test_vcf_to_recarray_ann():
    vcf_path = fixture_path('ann.vcf')
    fields = ['CHROM', 'POS', 'REF', 'ALT', 'ANN', 'DP', 'AC', 'GT']
    numbers = {'AC': 2, 'ALT': 2}
    for string_type in 'S20', 'object':
        types = {'CHROM': string_type, 'REF': string_type, 'ALT': string_type}
        transformers = [ANNTransformer(fields=['Allele', 'HGVS_c', 'AA'],
                                       types={'Allele': string_type,
                                              'HGVS_c': string_type})]
        callset = read_vcf(vcf_path, fields=fields, numbers=numbers, types=types,
                           transformers=transformers)
        a = vcf_to_recarray(vcf_path, fields=fields, numbers=numbers, chunk_length=2,
                            types=types, transformers=transformers)
        assert (['CHROM', 'POS', 'REF', 'ALT_1', 'ALT_2', 'ANN_Allele', 'ANN_HGVS_c',
                 'ANN_AA_pos', 'ANN_AA_length', 'DP', 'AC_1', 'AC_2'] ==
                list(a.dtype.names))
        assert np.dtype(string_type) == a['CHROM'].dtype
        assert np.dtype(string_type) == a['ALT_1'].dtype
        for k in callset:
            group, name = k.split('/')
            if group == 'variants':
                e = callset[k]
                if e.ndim == 1:
                    assert_array_equal(e, a[name])
                elif e.ndim == 2:
                    for i in range(e.shape[1]):
                        assert_array_equal(e[:, i], a['%s_%s' % (name, i + 1)])
                else:
                    assert False, (k, e.ndim)
            else:
                assert name not in a.dtype.names


def test_read_vcf_headers():
    vcf_path = fixture_path('sample.vcf')
    headers = read_vcf_headers(vcf_path)

    # check headers
    assert 'q10' in headers.filters
    assert 's50' in headers.filters
    assert 'AA' in headers.infos
    assert 'AC' in headers.infos
    assert 'AF' in headers.infos
    assert 'AN' in headers.infos
    assert 'DB' in headers.infos
    assert 'DP' in headers.infos
    assert 'H2' in headers.infos
    assert 'NS' in headers.infos
    assert 'DP' in headers.formats
    assert 'GQ' in headers.formats
    assert 'GT' in headers.formats
    assert 'HQ' in headers.formats
    assert ['NA00001', 'NA00002', 'NA00003'] == headers.samples
    assert '1' == headers.infos['AA']['Number']
    assert 'String' == headers.infos['AA']['Type']
    assert 'Ancestral Allele' == headers.infos['AA']['Description']
    assert '2' == headers.formats['HQ']['Number']
    assert 'Integer' == headers.formats['HQ']['Type']
    assert 'Haplotype Quality' == headers.formats['HQ']['Description']
