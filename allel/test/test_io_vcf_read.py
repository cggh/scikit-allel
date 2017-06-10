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
from nose.tools import assert_almost_equal, eq_, assert_in, assert_list_equal
from allel.io_vcf_read import read_vcf_chunks, read_vcf, vcf_to_zarr, vcf_to_hdf5, \
    vcf_to_npz, debug


def test_read_vcf_chunks():
    fn = 'fixture/sample.vcf'

    for n_threads in 1, 2:
        headers, it = read_vcf_chunks(fn, fields='*', chunk_length=4, block_length=2,
                                      buffer_size=100, n_threads=n_threads)

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
        chunks = [chunk for chunk, _, _, _ in it]
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


def _test_read_vcf_content(input, chunk_length, buffer_size, n_threads, block_length):

    if isinstance(input, str):
        input_file = input
    else:
        input_file = input()

    callset = read_vcf(input_file,
                       fields='*',
                       chunk_length=chunk_length,
                       buffer_size=buffer_size,
                       types={'ALT': 'S3', 'calldata/DP': 'S3'},
                       block_length=block_length,
                       n_threads=n_threads)

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
    eq_((10, 15), tuple(callset['calldata/HQ'][0, 0]))
    eq_((9, 3), callset['calldata/DP'].shape)
    eq_((b'4', b'2', b'3'), tuple(callset['calldata/DP'][6]))

    # TODO test GT as int16, int32, int64, S3

    # TODO special fields?
    # eq_(True, a[0]['NA00001']['is_called'])
    # eq_(True, a[0]['NA00001']['is_phased'])


def test_read_vcf_content_inputs():
    fn = 'fixture/sample.vcf'

    data = open(fn, mode='rb').read(-1)

    inputs = (fn,
              fn + '.gz',
              lambda: open(fn, mode='rb'),
              lambda: gzip.open(fn + '.gz', mode='rb'),
              lambda: io.BytesIO(data),
              lambda: io.BytesIO(data.replace(b'\n', b'\r')),
              lambda: io.BytesIO(data.replace(b'\n', b'\r\n')))

    chunk_length = 3
    block_length = 2
    buffer_size = 10
    n_threads = 1

    for input in inputs:
        _test_read_vcf_content(input, chunk_length, buffer_size, n_threads, block_length)


def test_read_vcf_content_chunk_block_lengths():
    fn = 'fixture/sample.vcf'
    input = fn
    chunk_lengths = 1, 2, 3, 5, 10, 20
    block_lengths = 1, 2, 3, 5, 10, 20
    buffer_size = 10
    n_threadses = 1, 2

    for chunk_length, n_threads, block_length in itertools.product(
            chunk_lengths, n_threadses, block_lengths):
        _test_read_vcf_content(input, chunk_length, buffer_size, n_threads, block_length)


def test_read_vcf_content_buffer_size():
    fn = 'fixture/sample.vcf'
    input = fn
    chunk_length = 3
    block_length = 2
    buffer_sizes = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512
    n_threadses = 1, 2

    for n_threads, buffer_size in itertools.product(
            n_threadses, buffer_sizes):
        _test_read_vcf_content(input, chunk_length, buffer_size, n_threads, block_length)


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
    for n_threads in 1, 2:
        expect = read_vcf(fn)
        npz_fn = 'temp/sample.npz'
        if os.path.exists(npz_fn):
            os.remove(npz_fn)
        vcf_to_npz(fn, npz_fn, chunk_length=2, n_threads=n_threads)
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
    for n_threads in 1, 2:
        if os.path.exists(zarr_path):
            shutil.rmtree(zarr_path)
        vcf_to_zarr(fn, zarr_path, chunk_length=2, n_threads=n_threads)
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
    for n_threads in 1, 2:
        if os.path.exists(h5_fn):
            os.remove(h5_fn)
        vcf_to_hdf5(fn, h5_fn, chunk_length=2, n_threads=n_threads)
        with h5py.File(h5_fn, mode='r') as actual:
            for key in expect.keys():
                if expect[key].dtype.kind == 'f':
                    assert_array_almost_equal(expect[key], actual[key][:])
                else:
                    assert_array_equal(expect[key], actual[key][:])


def test_read_vcf_info_types():
    fn = 'fixture/sample.vcf'

    for dtype in 'i1', 'i2', 'i4', 'i8', 'u1', 'u2', 'u4', 'u8', 'f4', 'f8':
        callset = read_vcf(fn, fields=['variants/DP'], types={'variants/DP': dtype})
        eq_(np.dtype(dtype), callset['variants/DP'].dtype)
        eq_((9,), callset['variants/DP'].shape)


def test_read_vcf_genotype_types():
    fn = 'fixture/sample.vcf'

    for dtype in 'i1', 'i2', 'i4', 'i8', 'u1', 'u2', 'u4', 'u8':
        callset = read_vcf(fn, fields=['GT'], types={'GT': dtype})
        eq_(np.dtype(dtype), callset['calldata/GT'].dtype)
        eq_((9, 3, 2), callset['calldata/GT'].shape)


def test_read_vcf_calldata_types():
    fn = 'fixture/sample.vcf'

    for dtype in 'i1', 'i2', 'i4', 'i8', 'u1', 'u2', 'u4', 'u8', 'f4', 'f8':
        callset = read_vcf(fn, fields=['HQ'], types={'HQ': dtype})
        eq_(np.dtype(dtype), callset['calldata/HQ'].dtype)
        eq_((9, 3, 2), callset['calldata/HQ'].shape)


def test_genotype_ploidy():
    fn = 'fixture/sample.vcf'

    callset = read_vcf(fn, fields='GT', numbers=dict(GT=1))
    gt = callset['calldata/GT']
    eq_((9, 3), gt.shape)
    eq_((0, 0, 0), tuple(gt[8, :]))

    callset = read_vcf(fn, fields='GT', numbers=dict(GT=2))
    gt = callset['calldata/GT']
    eq_((9, 3, 2), gt.shape)
    eq_((0, -1), tuple(gt[8, 0]))
    eq_((0, 1), tuple(gt[8, 1]))
    eq_((0, 2), tuple(gt[8, 2]))

    callset = read_vcf(fn, fields='GT', numbers=dict(GT=3))
    gt = callset['calldata/GT']
    eq_((9, 3, 3), gt.shape)
    eq_((0, -1, -1), tuple(gt[8, 0]))
    eq_((0, 1, -1), tuple(gt[8, 1]))
    eq_((0, 2, -1), tuple(gt[8, 2]))


def test_fills_info():
    fn = 'fixture/sample.vcf'

    callset = read_vcf(fn, fields='AN', numbers=dict(AN=1))
    a = callset['variants/AN']
    eq_((9,), a.shape)
    eq_(-1, a[0])
    eq_(-1, a[1])
    eq_(-1, a[2])

    callset = read_vcf(fn, fields='AN', numbers=dict(AN=1), fills=dict(AN=-2))
    a = callset['variants/AN']
    eq_((9,), a.shape)
    eq_(-2, a[0])
    eq_(-2, a[1])
    eq_(-2, a[2])

    callset = read_vcf(fn, fields='AN', numbers=dict(AN=1), fills=dict(AN=-1))
    a = callset['variants/AN']
    eq_((9,), a.shape)
    eq_(-1, a[0])
    eq_(-1, a[1])
    eq_(-1, a[2])


def test_fills_genotype():
    fn = 'fixture/sample.vcf'

    callset = read_vcf(fn, fields='GT', numbers=dict(GT=2))
    gt = callset['calldata/GT']
    eq_((9, 3, 2), gt.shape)
    eq_((0, -1), tuple(gt[8, 0]))
    eq_((0, 1), tuple(gt[8, 1]))
    eq_((0, 2), tuple(gt[8, 2]))

    callset = read_vcf(fn, fields='GT', numbers=dict(GT=2), fills=dict(GT=-2))
    gt = callset['calldata/GT']
    eq_((9, 3, 2), gt.shape)
    eq_((0, -2), tuple(gt[8, 0]))
    eq_((0, 1), tuple(gt[8, 1]))
    eq_((0, 2), tuple(gt[8, 2]))

    callset = read_vcf(fn, fields='GT', numbers=dict(GT=3), fills=dict(GT=-1))
    gt = callset['calldata/GT']
    eq_((9, 3, 3), gt.shape)
    eq_((0, -1, -1), tuple(gt[8, 0]))
    eq_((0, 1, -1), tuple(gt[8, 1]))
    eq_((0, 2, -1), tuple(gt[8, 2]))


def test_fills_calldata():
    fn = 'fixture/sample.vcf'

    callset = read_vcf(fn, fields='HQ', numbers=dict(HQ=2))
    a = callset['calldata/HQ']
    eq_((9, 3, 2), a.shape)
    eq_((10, 15), tuple(a[0, 0]))
    eq_((-1, -1), tuple(a[7, 0]))
    eq_((-1, -1), tuple(a[8, 0]))

    callset = read_vcf(fn, fields='HQ', numbers=dict(HQ=2), fills=dict(HQ=-2))
    a = callset['calldata/HQ']
    eq_((9, 3, 2), a.shape)
    eq_((10, 15), tuple(a[0, 0]))
    eq_((-2, -2), tuple(a[7, 0]))
    eq_((-2, -2), tuple(a[8, 0]))

    callset = read_vcf(fn, fields='HQ', numbers=dict(HQ=2), fills=dict(HQ=-1))
    a = callset['calldata/HQ']
    eq_((9, 3, 2), a.shape)
    eq_((10, 15), tuple(a[0, 0]))
    eq_((-1, -1), tuple(a[7, 0]))
    eq_((-1, -1), tuple(a[8, 0]))


def test_numbers():
    fn = 'fixture/sample.vcf'

    callset = read_vcf(fn, fields=['ALT'], numbers=dict(ALT=1))
    a = callset['variants/ALT']
    eq_((9,), a.shape)
    eq_(b'A', a[8])

    callset = read_vcf(fn, fields=['ALT'], numbers=dict(ALT=2), types=dict(ALT='S4'))
    a = callset['variants/ALT']
    eq_((9, 2), a.shape)
    eq_(b'A', a[8, 0])
    eq_(b'ATG', a[8, 1])

    callset = read_vcf(fn, fields=['ALT'], numbers=dict(ALT=3), types=dict(ALT='S4'))
    a = callset['variants/ALT']
    eq_((9, 3), a.shape)
    eq_(b'A', a[8, 0])
    eq_(b'ATG', a[8, 1])
    eq_(b'C', a[8, 2])

    callset = read_vcf(fn, fields=['AC'], numbers=dict(AC=0))
    a = callset['variants/AC']
    eq_((9,), a.shape)
    eq_(False, a[0])
    eq_(True, a[6])

    callset = read_vcf(fn, fields=['AC'], numbers=dict(AC=1))
    a = callset['variants/AC']
    eq_((9,), a.shape)
    eq_(-1, a[0])
    eq_(3, a[6])

    callset = read_vcf(fn, fields=['AC'], numbers=dict(AC=2))
    a = callset['variants/AC']
    eq_((9, 2), a.shape)
    eq_(-1, a[0, 0])
    eq_(-1, a[0, 1])
    eq_(3, a[6, 0])
    eq_(1, a[6, 1])

    callset = read_vcf(fn, fields='AF', numbers=dict(AF=1))
    a = callset['variants/AF']
    eq_((9,), a.shape)
    eq_(0.5, a[2])
    assert_almost_equal(0.333, a[4])

    callset = read_vcf(fn, fields='AF', numbers=dict(AF=2))
    a = callset['variants/AF']
    eq_((9, 2), a.shape)
    eq_(0.5, a[2, 0])
    assert np.isnan(a[2, 1])
    assert_almost_equal(0.333, a[4, 0])
    assert_almost_equal(0.667, a[4, 1])

    callset = read_vcf(fn, fields=['HQ'], numbers=dict(HQ=1))
    a = callset['calldata/HQ']
    eq_((9, 3), a.shape)
    eq_(10, a[0, 0])
    eq_(51, a[2, 0])
    eq_(-1, a[6, 0])

    callset = read_vcf(fn, fields=['HQ'], numbers=dict(HQ=2))
    a = callset['calldata/HQ']
    eq_((9, 3, 2), a.shape)
    eq_((10, 15), tuple(a[0, 0]))
    eq_((51, 51), tuple(a[2, 0]))
    eq_((-1, -1), tuple(a[6, 0]))
