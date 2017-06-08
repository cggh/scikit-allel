# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import io
import os
import shutil
import itertools
import gzip


import zarr
import h5py
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import *
from allel.io_vcf_read import read_vcf_chunks, read_vcf, vcf_to_zarr, vcf_to_hdf5, \
    vcf_to_npz


def test_read_vcf_chunks():
    fn = 'fixture/sample.vcf'
    headers, chunks = read_vcf_chunks(fn, fields='*', chunk_length=4, buffer_size=100)

    # check headers
    assert_in('q10', headers.filters)
    assert_in('s50', headers.filters)
    assert_in('AA', headers.infos)
    assert_in('AC', headers.infos)
    assert_in('AF', headers.infos)
    assert_in('AN', headers.infos)
    assert_in('DB', headers.infos)
    assert_in('DP', headers.infos)
    assert_in('H2', headers.infos)
    assert_in('NS', headers.infos)
    assert_in('DP', headers.formats)
    assert_in('GQ', headers.formats)
    assert_in('GT', headers.formats)
    assert_in('HQ', headers.formats)
    eq_(['NA00001', 'NA00002', 'NA00003'], headers.samples)
    eq_('1', headers.infos['AA']['Number'])
    eq_('String', headers.infos['AA']['Type'])
    eq_('Ancestral Allele', headers.infos['AA']['Description'])
    eq_('2', headers.formats['HQ']['Number'])
    eq_('Integer', headers.formats['HQ']['Type'])
    eq_('Haplotype Quality', headers.formats['HQ']['Description'])

    # check chunk lengths
    chunks = list(chunks)
    eq_(3, len(chunks))
    eq_(4, chunks[0]['variants/POS'].shape[0])
    eq_(4, chunks[1]['variants/POS'].shape[0])
    eq_(1, chunks[2]['variants/POS'].shape[0])

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
        # FORMAT fields
        'calldata/GT',
        'calldata/GQ',
        'calldata/HQ',
        'calldata/DP',
    ]
    for chunk in chunks:
        assert_list_equal(sorted(expected_fields), sorted(chunk.keys()))


def test_read_vcf_fields_all():
    fn = 'fixture/sample.vcf'
    callset = read_vcf(fn, fields='*', chunk_length=4, buffer_size=100)
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
        # FORMAT fields
        'calldata/GT',
        'calldata/GQ',
        'calldata/HQ',
        'calldata/DP',
    ]
    assert_list_equal(sorted(expected_fields), sorted(callset.keys()))


def test_read_vcf_fields_default():
    fn = 'fixture/sample.vcf'
    callset = read_vcf(fn, chunk_length=3, buffer_size=30)
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
    assert_list_equal(sorted(expected_fields), sorted(callset.keys()))


def test_read_vcf_fields_all_variants():
    fn = 'fixture/sample.vcf'
    callset = read_vcf(fn, fields='variants/*', chunk_length=2, buffer_size=20)
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
    ]
    assert_list_equal(sorted(expected_fields), sorted(callset.keys()))


def test_read_vcf_fields_info():
    fn = 'fixture/sample.vcf'
    callset = read_vcf(fn, fields='INFO', chunk_length=5, buffer_size=10)
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
    assert_list_equal(sorted(expected_fields), sorted(callset.keys()))


def test_read_vcf_fields_filter():
    fn = 'fixture/sample.vcf'
    callset = read_vcf(fn, fields='FILTER', chunk_length=1, buffer_size=2)
    expected_fields = [
        'variants/FILTER_PASS',
        'variants/FILTER_q10',
        'variants/FILTER_s50',
    ]
    assert_list_equal(sorted(expected_fields), sorted(callset.keys()))


def test_read_vcf_fields_all_calldata():
    fn = 'fixture/sample.vcf'
    callset = read_vcf(fn, fields='calldata/*', chunk_length=6, buffer_size=1000)
    expected_fields = [
        'calldata/GT',
        'calldata/GQ',
        'calldata/HQ',
        'calldata/DP',
    ]
    assert_list_equal(sorted(expected_fields), sorted(callset.keys()))


def test_read_vcf_fields_selected():
    fn = 'fixture/sample.vcf'

    # without samples
    callset = read_vcf(fn, fields=['CHROM', 'variants/POS', 'AC', 'variants/AF', 'GT',
                                   'calldata/HQ', 'FILTER_q10'],
                       chunk_length=4, buffer_size=100)
    expected_fields = [
        'variants/CHROM',
        'variants/POS',
        'variants/FILTER_q10',
        'variants/AC',
        'variants/AF',
        # FORMAT fields
        'calldata/GT',
        'calldata/HQ',
    ]
    assert_list_equal(sorted(expected_fields), sorted(callset.keys()))

    # with samples
    callset = read_vcf(fn, fields=['CHROM', 'variants/POS', 'AC', 'variants/AF', 'GT',
                                   'calldata/HQ', 'FILTER_q10', 'samples'],
                       chunk_length=4, buffer_size=100)
    expected_fields = [
        'samples',
        'variants/CHROM',
        'variants/POS',
        'variants/FILTER_q10',
        'variants/AC',
        'variants/AF',
        # FORMAT fields
        'calldata/GT',
        'calldata/HQ',
    ]
    assert_list_equal(sorted(expected_fields), sorted(callset.keys()))


def test_read_vcf_content():
    fn = 'fixture/sample.vcf'
    data = open(fn, mode='rb').read(-1)

    inputs = (fn,
              fn + '.gz',
              lambda: open(fn, mode='rb'),
              lambda: gzip.open(fn + '.gz', mode='rb'),
              lambda: io.BytesIO(data),
              lambda: io.BytesIO(data.replace(b'\n', b'\r')),
              lambda: io.BytesIO(data.replace(b'\n', b'\r\n')))

    chunk_lengths = 1, 2, 3, 4, 5, 6, 8, 10, 12, 20, 1000

    buffer_sizes = 2, 10, 20, 30, 50, 100, 1000, 10000, 100000

    for input, chunk_length, buffer_size in itertools.product(inputs,
                                                              chunk_lengths,
                                                              buffer_sizes):
        print(repr(input), chunk_length, buffer_size)

        if isinstance(input, str):
            input_file = input
        else:
            input_file = input()

        callset = read_vcf(input_file,
                           fields='*',
                           chunk_length=chunk_length,
                           buffer_size=buffer_size,
                           types={'ALT': 'S3', 'calldata/DP': 'S3'})

        # fixed fields
        print(callset['variants/CHROM'])
        print(callset['variants/POS'])
        eq_((9,), callset['variants/CHROM'].shape)
        eq_(b'19', callset['variants/CHROM'][0])
        eq_((9,), callset['variants/POS'].shape)
        eq_(111, callset['variants/POS'][0])
        eq_((9,), callset['variants/ID'].shape)
        eq_(b'rs6054257', callset['variants/ID'][2])
        eq_((9,), callset['variants/REF'].shape)
        eq_(b'A', callset['variants/REF'][0])
        eq_((9, 3), callset['variants/ALT'].shape)
        eq_(b'ATG', callset['variants/ALT'][8, 1])
        eq_((9,), callset['variants/QUAL'].shape)
        eq_(10.0, callset['variants/QUAL'][1])
        eq_((9,), callset['variants/FILTER_PASS'].shape)
        eq_(True, callset['variants/FILTER_PASS'][2])
        eq_(False, callset['variants/FILTER_PASS'][3])
        eq_((9,), callset['variants/FILTER_q10'].shape)
        eq_(True, callset['variants/FILTER_q10'][3])
        # TODO special fields
        # eq_(2, callset['variants/num_alleles'][0])
        # eq_(False, callset['variants/is_snp'][5])

        # INFO fields
        eq_(3, callset['variants/NS'][2])
        eq_(.5, callset['variants/AF'][2, 0])
        eq_(True, callset['variants/DB'][2])
        eq_((3, 1, -1), tuple(callset['variants/AC'][6]))

        # test calldata content
        eq_((9, 3, 2), callset['calldata/GT'].shape)
        eq_((0, 0), tuple(callset['calldata/GT'][0, 0]))
        eq_((-1, -1), tuple(callset['calldata/GT'][6, 2]))
        eq_((-1, -1), tuple(callset['calldata/GT'][7, 2]))
        eq_((9, 3, 2), callset['calldata/HQ'].shape)
        eq_((10, 10), tuple(callset['calldata/HQ'][0, 0]))
        eq_((9, 3), callset['calldata/DP'].shape)
        eq_((b'4', b'2', b'3'), tuple(callset['calldata/DP'][6]))

        # TODO test GT as int16, int32, int64, S3

        # TODO special fields?
        # eq_(True, a[0]['NA00001']['is_called'])
        # eq_(True, a[0]['NA00001']['is_phased'])


def test_vcf_truncation_chrom():

    input_data = (b"#CHROM\n"
                  b"2L\n"
                  b"2R\n")

    # with and without final line terminator
    for data in (input_data, input_data[:-1]):

        input_file = io.BytesIO(data)
        callset = read_vcf(input_file, fields=['CHROM', 'samples'])

        # check fields
        expected_fields = ['variants/CHROM', 'samples']
        assert_list_equal(sorted(expected_fields), sorted(callset.keys()))

        # check data content
        eq_(0, len(callset['samples']))
        a = callset['variants/CHROM']
        eq_(2, len(a))
        eq_(b'2L', a[0])
        eq_(b'2R', a[1])


def test_vcf_truncation_pos():

    input_data = (b"#CHROM\tPOS\n"
                  b"2L\t12\n"
                  b"2R\t34\n")

    # with and without final line terminator
    for data in (input_data, input_data[:-1]):

        input_file = io.BytesIO(data)
        callset = read_vcf(input_file, fields=['POS', 'samples'])

        # check fields
        expected_fields = ['variants/POS', 'samples']
        assert_list_equal(sorted(expected_fields), sorted(callset.keys()))

        # check data content
        eq_(0, len(callset['samples']))
        a = callset['variants/POS']
        eq_(2, len(a))
        eq_(12, a[0])
        eq_(34, a[1])


def test_vcf_truncation_id():

    input_data = (b"#CHROM\tPOS\tID\n"
                  b"2L\t12\tfoo\n"
                  b"2R\t34\tbar\n")

    # with and without final line terminator
    for data in (input_data, input_data[:-1]):

        input_file = io.BytesIO(data)
        callset = read_vcf(input_file, fields=['ID', 'samples'])

        # check fields
        expected_fields = ['variants/ID', 'samples']
        assert_list_equal(sorted(expected_fields), sorted(callset.keys()))

        # check data content
        eq_(0, len(callset['samples']))
        a = callset['variants/ID']
        eq_(2, len(a))
        eq_(b'foo', a[0])
        eq_(b'bar', a[1])


def test_vcf_truncation_ref():

    input_data = (b"#CHROM\tPOS\tID\tREF\n"
                  b"2L\t12\tfoo\tA\n"
                  b"2R\t34\tbar\tC\n")

    # with and without final line terminator
    for data in (input_data, input_data[:-1]):

        input_file = io.BytesIO(data)
        callset = read_vcf(input_file, fields=['REF', 'samples'])

        # check fields
        expected_fields = ['variants/REF', 'samples']
        assert_list_equal(sorted(expected_fields), sorted(callset.keys()))

        # check data content
        eq_(0, len(callset['samples']))
        a = callset['variants/REF']
        eq_(2, len(a))
        eq_(b'A', a[0])
        eq_(b'C', a[1])


def test_vcf_truncation_alt():

    input_data = (b"#CHROM\tPOS\tID\tREF\tALT\n"
                  b"2L\t12\tfoo\tA\tC\n"
                  b"2R\t34\tbar\tC\tG\n")

    # with and without final line terminator
    for data in (input_data, input_data[:-1]):

        input_file = io.BytesIO(data)
        callset = read_vcf(input_file, fields=['ALT', 'samples'], numbers=dict(ALT=1))

        # check fields
        expected_fields = ['variants/ALT', 'samples']
        assert_list_equal(sorted(expected_fields), sorted(callset.keys()))

        # check data content
        eq_(0, len(callset['samples']))
        a = callset['variants/ALT']
        eq_(2, len(a))
        eq_(b'C', a[0])
        eq_(b'G', a[1])


def test_vcf_truncation_qual():

    input_data = (b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\n"
                  b"2L\t12\tfoo\tA\tC\t1.2\n"
                  b"2R\t34\tbar\tC\tG\t3.4\n")

    # with and without final line terminator
    for data in (input_data, input_data[:-1]):

        input_file = io.BytesIO(data)
        callset = read_vcf(input_file, fields=['QUAL', 'samples'])

        # check fields
        expected_fields = ['variants/QUAL', 'samples']
        assert_list_equal(sorted(expected_fields), sorted(callset.keys()))

        # check data content
        eq_(0, len(callset['samples']))
        a = callset['variants/QUAL']
        eq_(2, len(a))
        assert_almost_equal(1.2, a[0], places=6)
        assert_almost_equal(3.4, a[1], places=6)


def test_vcf_truncation_filter():

    input_data = (b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\n"
                  b"2L\t12\tfoo\tA\tC\t1.2\t.\n"
                  b"2R\t34\tbar\tC\tG\t3.4\tPASS\n"
                  b"2R\t56\tbaz\tG\tT\t56.77\tq10,s50\n")

    # with and without final line terminator
    for data in (input_data, input_data[:-1]):

        input_file = io.BytesIO(data)
        callset = read_vcf(input_file, fields=['FILTER_PASS', 'FILTER_q10', 'FILTER_s50'])

        # check fields
        expected_fields = ['variants/FILTER_PASS', 'variants/FILTER_q10',
                           'variants/FILTER_s50']
        assert_list_equal(sorted(expected_fields), sorted(callset.keys()))

        # check data content
        a = callset['variants/FILTER_PASS']
        eq_(3, len(a))
        assert_list_equal([False, True, False], a.tolist())
        a = callset['variants/FILTER_q10']
        eq_(3, len(a))
        assert_list_equal([False, False, True], a.tolist())
        a = callset['variants/FILTER_s50']
        eq_(3, len(a))
        assert_list_equal([False, False, True], a.tolist())


def test_vcf_truncation_info():

    input_data = (b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
                  b"2L\t12\tfoo\tA\tC\t1.2\t.\tfoo=42;bar=1.2\n"
                  b"2R\t34\tbar\tC\tG\t3.4\tPASS\t.\n"
                  b"2R\t56\tbaz\tG\tT\t56.77\tq10,s50\t\n")

    # with and without final line terminator
    for data in (input_data, input_data[:-1]):

        input_file = io.BytesIO(data)
        callset = read_vcf(input_file,
                           fields=['foo', 'bar'],
                           types=dict(foo='Integer', bar='Float'))

        # check fields
        expected_fields = ['variants/foo', 'variants/bar']
        assert_list_equal(sorted(expected_fields), sorted(callset.keys()))

        # check data content
        a = callset['variants/foo']
        eq_(3, len(a))
        eq_(42, a[0])
        eq_(-1, a[1])
        eq_(-1, a[2])
        a = callset['variants/bar']
        eq_(3, len(a))
        assert_almost_equal(1.2, a[0], places=6)
        assert np.isnan(a[1])
        assert np.isnan(a[2])


def test_vcf_truncation_format():

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
        expected_fields = ['variants/foo', 'variants/bar', 'samples']
        assert_list_equal(sorted(expected_fields), sorted(callset.keys()))

        # check data content
        eq_(0, len(callset['samples']))
        a = callset['variants/foo']
        eq_(3, len(a))
        eq_(42, a[0])
        eq_(-1, a[1])
        eq_(-1, a[2])
        a = callset['variants/bar']
        eq_(3, len(a))
        assert_almost_equal(1.2, a[0], places=6)
        assert np.isnan(a[1])
        assert np.isnan(a[2])


def test_vcf_truncation_calldata():

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
        assert_list_equal(sorted(expected_fields), sorted(callset.keys()))

        # check data content
        eq_(2, len(callset['samples']))
        assert_list_equal([b'S2', b'S1'], callset['samples'].tolist())
        a = callset['calldata/GT']
        eq_((3, 2, 2), a.shape)
        eq_((0, 1), tuple(a[0, 0]))
        eq_((1, 2), tuple(a[0, 1]))
        eq_((-1, -1), tuple(a[1, 0]))
        eq_((-1, -1), tuple(a[1, 1]))
        eq_((-1, -1), tuple(a[2, 0]))
        eq_((-1, -1), tuple(a[2, 1]))

        a = callset['calldata/GQ']
        eq_((3, 2), a.shape)
        eq_(12, a[0, 0])
        eq_(34, a[0, 1])
        eq_(-1, a[1, 0])
        eq_(-1, a[1, 1])
        eq_(-1, a[2, 0])
        eq_(-1, a[2, 1])


def test_vcf_to_npz():
    fn = 'fixture/sample.vcf'
    expect = read_vcf(fn)
    npz_fn = 'temp/sample.npz'
    if os.path.exists(npz_fn):
        os.remove(npz_fn)
    vcf_to_npz(fn, npz_fn, chunk_length=2)
    actual = np.load(npz_fn)
    for key in expect.keys():
        if expect[key].dtype.kind == 'f':
            assert_array_almost_equal(expect[key], actual[key])
        else:
            assert_array_equal(expect[key], actual[key])


def test_vcf_to_zarr():
    fn = 'fixture/sample.vcf'
    expect = read_vcf(fn)
    zarr_path = 'temp/sample.zarr'
    if os.path.exists(zarr_path):
        shutil.rmtree(zarr_path)
    vcf_to_zarr(fn, zarr_path, chunk_length=2)
    actual = zarr.open_group(zarr_path, mode='r')
    for key in expect.keys():
        if expect[key].dtype.kind == 'f':
            assert_array_almost_equal(expect[key], actual[key][:])
        else:
            assert_array_equal(expect[key], actual[key][:])


def test_vcf_to_hdf5():
    fn = 'fixture/sample.vcf'
    expect = read_vcf(fn)
    h5_fn = 'temp/sample.h5'
    if os.path.exists(h5_fn):
        os.remove(h5_fn)
    vcf_to_hdf5(fn, h5_fn, chunk_length=2)
    with h5py.File(h5_fn, mode='r') as actual:
        for key in expect.keys():
            if expect[key].dtype.kind == 'f':
                assert_array_almost_equal(expect[key], actual[key][:])
            else:
                assert_array_equal(expect[key], actual[key][:])


# TODO test types

# TODO test numbers

# TODO test fills
