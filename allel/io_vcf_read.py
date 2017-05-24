# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import gzip
import io
import sys


from allel.opt.io_vcf_read import iter_vcf


def debug(*msg):
    print(*msg, file=sys.stderr)
    sys.stderr.flush()


def read_vcf(path, buffer_size=2**15, block_size=2**15, temp_max_size=2**15):
    """TODO"""

    if isinstance(path, str) and path.endswith('gz'):
        # assume gzip-compatible compression
        with gzip.open(path, mode='rb') as binary_file:
            return _read_vcf(binary_file, buffer_size=buffer_size, block_size=block_size,
                             temp_max_size=temp_max_size)

    elif isinstance(path, str):
        # assume no compression
        with open(path, mode='rb', buffering=0) as binary_file:
            return _read_vcf(binary_file, buffer_size=buffer_size, block_size=block_size,
                             temp_max_size=temp_max_size)

    else:
        # assume some other binary file-like object
        binary_file = path
        return _read_vcf(binary_file, buffer_size=buffer_size, block_size=block_size,
                         temp_max_size=temp_max_size)


def _read_vcf(fileobj, buffer_size, block_size, temp_max_size):
    headers, samples = _read_vcf_headers(fileobj)
    # debug(samples)
    return iter_vcf(fileobj, buffer_size=buffer_size, block_size=block_size,
                    temp_max_size=temp_max_size, n_samples=len(samples))


def _binary_readline(binary_file):
    line = []
    c = binary_file.read(1)
    while c and c != b'\n':
        line.append(c)
        c = binary_file.read(1)
    return b''.join(line)


def _read_vcf_headers(binary_file):

    # setup
    headers = []
    samples = None

    # read first header line
    header = _binary_readline(binary_file)

    while header and header[0] == ord(b'#'):
        # debug('found header', header)

        headers.append(header)

        if header.startswith(b'#CHROM'):

            # parse out samples
            samples = header.split(b'\t')[9:]
            break

        else:

            # read next header
            header = _binary_readline(binary_file)

    # check if we saw the mandatory header line or not
    if samples is None:
        raise RuntimeError('VCF file is missing mandatory header line ("#CHROM...")')

    return headers, samples
