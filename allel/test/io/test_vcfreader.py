from allel.io.vcfreader import VCFReader
import gzip
import os


def fixture_path(fn):
    return os.path.join(os.path.dirname(__file__), os.pardir, 'data', fn)


def test_parse_region():
    chrom, begin, end = VCFReader.parse_region('chr1:10-100')
    assert chrom == b'chr1'
    assert begin == 9
    assert end == 100

    chrom, begin, end = VCFReader.parse_region('chr1:10')
    assert chrom == b'chr1'
    assert begin == 9
    assert end == 10

    chrom, begin, end = VCFReader.parse_region('chr1')
    assert chrom == b'chr1'
    assert begin == 0
    assert end == 2147483647


def test_decompress():
    vcf_reader = VCFReader(
        fixture_path('test54.vcf.gz'), fixture_path('test54.vcf.gz.tbi'), 'chr1:1-100'
    )
    gzip_reader = gzip.open(fixture_path('test54.vcf.gz'), 'rb')
    buffer_vcf, buffer_gzip = bytearray(2 ** 14), bytearray(2 ** 14)
    while True:
        num_vcf = vcf_reader.readinto(buffer_vcf)
        num_gzip = gzip_reader.readinto(buffer_gzip)
        assert num_vcf == num_gzip
        if num_gzip > 0:
            assert buffer_vcf == buffer_gzip
        else:
            break
