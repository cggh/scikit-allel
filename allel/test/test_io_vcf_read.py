# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import io


from nose.tools import *
from allel.io_vcf_read import read_vcf_chunks, read_vcf


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
    callset = read_vcf(fn, fields='*', chunk_length=4, buffer_size=10,
                       types={'ALT': 'S3', 'calldata/DP': 'S3'})

    # fixed fields
    eq_(b'19', callset['variants/CHROM'][0])
    eq_(111, callset['variants/POS'][0])
    eq_(b'rs6054257', callset['variants/ID'][2])
    eq_(b'A', callset['variants/REF'][0])
    eq_(b'ATG', callset['variants/ALT'][8, 1])
    eq_(10.0, callset['variants/QUAL'][1])
    eq_(True, callset['variants/FILTER_PASS'][2])
    eq_(False, callset['variants/FILTER_PASS'][3])
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
    eq_((0, 0), tuple(callset['calldata/GT'][0, 0]))
    eq_((-1, -1), tuple(callset['calldata/GT'][6, 2]))
    eq_((-1, -1), tuple(callset['calldata/GT'][7, 2]))
    eq_((10, 10), tuple(callset['calldata/HQ'][0, 0]))
    eq_((b'4', b'2', b'3'), tuple(callset['calldata/DP'][6]))

    # TODO test GT as int16, int32, int64, S3

    # TODO special fields?
    # eq_(True, a[0]['NA00001']['is_called'])
    # eq_(True, a[0]['NA00001']['is_phased'])


def test_vcf_truncation():

    input_data = b"""#CHROM\n2L\n"""
    input_file = io.BytesIO(input_data)
    callset = read_vcf(input_file, fields=['CHROM', 'POS', 'samples'])

    # check fields
    expected_fields = ['variants/CHROM', 'variants/POS', 'samples']
    assert_list_equal(sorted(expected_fields), sorted(callset.keys()))

    # check data content
    chrom = callset['variants/CHROM']
    pos = callset['variants/POS']
    eq_(1, len(chrom))
    eq_(1, len(pos))
    eq_(b'2L', chrom[0])
    eq_(-1, pos[0])


# TODO test types

# TODO test numbers

# TODO test fills
