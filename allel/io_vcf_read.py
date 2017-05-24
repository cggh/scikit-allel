# -*- coding: utf-8 -*-
"""
TODO:

* DONE Fix missing value in trailing item
* DONE Inintial implementation of vcf_to_npz
* DONE Initial implementation of vcf_to_hdf5
* DONE Add samples to output of read_vcf and store in vcf_to_... functions
* DONE Initial implementation of vcf_to_zarr
* DONE Parse FILTERS from header
* DONE Return filters as separate arrays in read_vcf
* DONE Store filters as separate arrays/datasets in vcf_to_... functions
* Parse INFO fields
* Parse other FORMAT fields
* Read from region via tabix
* Read from region via scanning
* Progress logging in vcf_to_... functions
* User-controlled dtypes
* User-controlled fill values
* User-controlled arities
* User-specified samples to parse

"""
from __future__ import absolute_import, print_function, division
import gzip
import sys
import itertools
import os
import re
from collections import namedtuple


import numpy as np


from allel.opt.io_vcf_read import iter_vcf


def debug(*msg):
    print(*msg, file=sys.stderr)
    sys.stderr.flush()


DEFAULT_BUFFER_SIZE = 2**15
DEFAULT_CHUNK_LENGTH = 2**15
DEFAULT_CHUNK_WIDTH = 2**6
DEFAULT_TEMP_SIZE = 2**15


def read_vcf(path,
             fields=None,
             buffer_size=DEFAULT_BUFFER_SIZE,
             chunk_length=DEFAULT_CHUNK_LENGTH,
             temp_max_size=DEFAULT_TEMP_SIZE):
    """Read data from a VCF file into NumPy arrays.

    Parameters
    ----------
    path : str
        TODO
    fields : sequence of str
        TODO
    buffer_size : int
        TODO
    chunk_length : int
        TODO
    temp_max_size : int
        TODO

    Returns
    -------
    data : dict[str -> ndarray]
        A dictionary holding arrays.

    """

    # setup
    headers, chunks = read_vcf_chunks(path=path, fields=fields, buffer_size=buffer_size,
                                      chunk_length=chunk_length, temp_max_size=temp_max_size)

    # read all chunks into a list
    chunks = list(chunks)

    # setup output
    output = dict()
    output['samples'] = np.array(headers.samples)

    if chunks:

        # find array keys
        keys = sorted(chunks[0].keys())

        # concatenate chunks
        for k in keys:
            output[k] = np.concatenate([chunk[k] for chunk in chunks], axis=0)

    return output


def vcf_to_npz(input_path, output_path,
               compressed=True,
               overwrite=False,
               fields=None,
               buffer_size=DEFAULT_BUFFER_SIZE,
               chunk_length=DEFAULT_CHUNK_LENGTH,
               temp_max_size=DEFAULT_TEMP_SIZE):
    """TODO"""

    # guard condition
    if not overwrite and os.path.exists(output_path):
        # TODO right exception class?
        raise ValueError('file exists at path %r; use overwrite=True to replace' % output_path)

    # read all data into memory
    data = read_vcf(path=input_path, fields=fields, buffer_size=buffer_size,
                    chunk_length=chunk_length, temp_max_size=temp_max_size)

    # setup save function
    if compressed:
        savez = np.savez_compressed
    else:
        savez = np.savez

    # save as npz
    savez(output_path, **data)


def vcf_to_hdf5(input_path, output_path,
                group='/',
                compression='gzip',
                compression_opts=1,
                shuffle=False,
                overwrite=False,
                fields=None,
                buffer_size=DEFAULT_BUFFER_SIZE,
                chunk_length=DEFAULT_CHUNK_LENGTH,
                chunk_width=DEFAULT_CHUNK_WIDTH,
                temp_max_size=DEFAULT_TEMP_SIZE):
    """TODO"""

    import h5py

    with h5py.File(output_path, mode='a') as h5f:

        # setup offset for loading
        offset = 0

        # obtain root group that data will be stored into
        root = h5f.require_group(group)

        # ensure sub-groups
        root.require_group('variants')
        root.require_group('calldata')

        # setup chunk iterator
        headers, chunks = read_vcf_chunks(input_path, fields=fields, buffer_size=buffer_size,
                                          chunk_length=chunk_length, temp_max_size=temp_max_size)
        # TODO this won't be necessary when using generators
        chunks = iter(chunks)

        # store samples
        name = 'samples'
        if name in root[group]:
            if overwrite:
                del root[group][name]
            else:
                # TODO right exception class?
                raise ValueError('dataset exists at path %r; use overwrite=True to replace' % name)
        root[group].create_dataset(name, data=np.array(headers.samples), chunks=None)

        # read first chunk
        chunk = next(chunks, None)

        # handle no input
        if chunk is None:
            raise RuntimeError('input file has no data?')

        # setup datasets
        keys = sorted(chunk.keys())
        for k in keys:

            # obtain initial data
            data = chunk[k]

            # determine chunk shape
            if data.ndim == 1:
                chunk_shape = (chunk_length,)
            else:
                chunk_shape = (chunk_length, min(chunk_width, data.shape[1])) + data.shape[2:]

            # create dataset
            group, name = k.split('/')
            if name in root[group]:
                if overwrite:
                    del root[group][name]
                else:
                    # TODO right exception class?
                    raise ValueError('dataset exists at path %r; use overwrite=True to replace' % k)

            shape = (0,) + data.shape[1:]
            maxshape = (None,) + data.shape[1:]
            root[group].create_dataset(name, shape=shape, maxshape=maxshape,
                                       chunks=chunk_shape, dtype=data.dtype,
                                       compression=compression,
                                       compression_opts=compression_opts, shuffle=shuffle)

        # reconstitute chunks iterator
        chunks = itertools.chain([chunk], chunks)

        # load chunks
        for chunk_index, chunk in enumerate(chunks):

            # compute length of current chunk
            current_chunk_length = chunk[keys[0]].shape[0]

            # new length of all arrays after loading this chunk
            new_length = offset + current_chunk_length

            # load arrays
            for k in keys:

                # data to be loaded
                data = chunk[k]

                # obtain dataset
                dataset = root[k]

                # ensure dataset is large enough
                if dataset.shape[0] < new_length:
                    dataset.resize(new_length, axis=0)

                # store the data
                dataset[offset:new_length, ...] = data

            # move offset
            offset = new_length


def vcf_to_zarr(input_path, output_path,
                group='/',
                compressor='default',
                fill_value=0,
                order='C',
                overwrite=False,
                fields=None,
                buffer_size=DEFAULT_BUFFER_SIZE,
                chunk_length=DEFAULT_CHUNK_LENGTH,
                chunk_width=DEFAULT_CHUNK_WIDTH,
                temp_max_size=DEFAULT_TEMP_SIZE):
    """TODO"""

    import zarr

    # open root group
    root = zarr.open_group(output_path, mode='a', path=group)

    # ensure sub-groups
    root.require_group('variants')
    root.require_group('calldata')

    # setup chunk iterator
    headers, chunks = read_vcf_chunks(input_path, fields=fields, buffer_size=buffer_size,
                                      chunk_length=chunk_length, temp_max_size=temp_max_size)
    # TODO this won't be necessary when using generators
    chunks = iter(chunks)

    # store samples
    root[group].create_dataset('samples', data=np.array(headers.samples), compressor=None,
                               overwrite=overwrite)

    # read first chunk
    chunk = next(chunks, None)

    # handle no input
    if chunk is None:
        raise RuntimeError('input file has no data?')

    # setup datasets
    keys = sorted(chunk.keys())
    for k in keys:

        # obtain initial data
        data = chunk[k]

        # determine chunk shape
        if data.ndim == 1:
            chunk_shape = (chunk_length,)
        else:
            chunk_shape = (chunk_length, min(chunk_width, data.shape[1])) + data.shape[2:]

        # create dataset
        shape = (0,) + data.shape[1:]
        root.create_dataset(k, shape=shape, chunks=chunk_shape, dtype=data.dtype,
                            compressor=compressor, overwrite=overwrite, fill_value=fill_value,
                            order=order)

    # reconstitute chunks iterator
    chunks = itertools.chain([chunk], chunks)

    # load chunks
    for chunk_index, chunk in enumerate(chunks):

        # load arrays
        for k in keys:

            # append data
            root[k].append(chunk[k], axis=0)


def read_vcf_chunks(path,
                    fields=None,
                    buffer_size=DEFAULT_BUFFER_SIZE,
                    chunk_length=DEFAULT_CHUNK_LENGTH,
                    temp_max_size=DEFAULT_TEMP_SIZE):
    """TODO"""

    if isinstance(path, str) and path.endswith('gz'):
        # assume gzip-compatible compression
        with gzip.open(path, mode='rb') as binary_file:
            return _read_vcf(binary_file, fields=fields, buffer_size=buffer_size,
                             chunk_length=chunk_length, temp_max_size=temp_max_size)

    elif isinstance(path, str):
        # assume no compression
        with open(path, mode='rb', buffering=0) as binary_file:
            return _read_vcf(binary_file, fields=fields, buffer_size=buffer_size,
                             chunk_length=chunk_length, temp_max_size=temp_max_size)

    else:
        # assume some other binary file-like object
        binary_file = path
        return _read_vcf(binary_file, fields=fields, buffer_size=buffer_size,
                         chunk_length=chunk_length, temp_max_size=temp_max_size)


FIXED_FIELDS = (
    b'variants/CHROM',
    b'variants/POS',
    b'variants/ID',
    b'variants/REF',
    b'variants/ALT',
    b'variants/QUAL',
)


def normalize_fields(fields, headers):

    # setup normalized fields
    normed_fields = set()

    for f in fields:

        # ensure bytes
        if isinstance(f, str):
            f = f.encode('ascii')

        # special-case: user requests all variants fields
        if f == b'variants' or f == b'variants*' or f == b'variants/*':

            # all fixed fields
            for k in FIXED_FIELDS:
                normed_fields.add(k)

            # all FILTER fields
            normed_fields.add(b'variants/FILTER_PASS')
            for k in headers.filters:
                normed_fields.add(b'variants/FILTER_' + k)

            # all INFO fields
            for k in headers.infos:
                normed_fields.add(b'variants/' + k)

            continue

        # special-case: user requests all calldata fields
        if f == b'calldata' or f == b'calldata*' or f == b'calldata/*':
            
            # all FORMAT fields
            for k in headers.formats:
                normed_fields.add(b'calldata/' + k)

            continue

        # normalize prefix
        if not f.startswith(b'variants/') and not f.startswith(b'calldata/'):
            # assume variants field
            f = b'variants/' + f

        if f in FIXED_FIELDS:
            # no need for further checks
            normed_fields.add(f)

        # special-case: user requests all filters
        elif f == b'variants/FILTER' or f == b'variants/FILTER*':

            # add all filter fields
            normed_fields.add(b'variants/FILTER_PASS')
            for k in sorted(headers.filters):
                normed_fields.add(b'variants/FILTER_' + k)

        # special-case: user requests all INFO fields
        elif f == b'variants/INFO' or f == b'variants/INFO*':
            pass
            # TODO

        elif f.startswith(b'variants/'):

            # check it's declared in INFO header
            k = f.split(b'/')[1]
            if k not in headers.infos:
                raise ValueError('field not declared in INFO header: %r' % str(f, 'ascii'))
            normed_fields.add(f)

        elif f.startswith(b'calldata/'):

            # check it's declared in FORMAT header
            k = f.split(b'/')[1]
            if k not in headers.formats:
                raise ValueError('field not declared in FORMAT header: %r' % str(f, 'ascii'))
            normed_fields.add(f)

        else:
            raise RuntimeError('unexpected field: %r' + f)

    return normed_fields


def _read_vcf(fileobj, fields, buffer_size, chunk_length, temp_max_size):

    # read VCF headers
    headers = read_vcf_headers(fileobj)

    # setup fields to read
    if fields is None:

        # choose default fields to parse
        fields = list(FIXED_FIELDS)
        # add in all FILTER fields by default
        for f in sorted(headers.filters):
            fields.append(b'variants/FILTER_' + f)
        # add in GT field by default
        fields.append(b'calldata/GT')

    else:
        fields = normalize_fields(fields, headers)

    # setup chunks iterator
    chunks = iter_vcf(fileobj, buffer_size=buffer_size, chunk_length=chunk_length,
                      temp_max_size=temp_max_size, headers=headers, fields=fields)

    return headers, chunks


def _binary_readline(binary_file):
    line = []
    c = binary_file.read(1)
    while c and c != b'\n':
        line.append(c)
        c = binary_file.read(1)
    return b''.join(line)


# pre-compile some regular expressions
re_filter_header = \
    re.compile(b'##FILTER=<ID=([^,]+),Description="([^"]+)">')
re_info_header = \
    re.compile(b'##INFO=<ID=([^,]+),Number=([^,]+),Type=([^,]+),Description="([^"]+)">')
re_format_header = \
    re.compile(b'##FORMAT=<ID=([^,]+),Number=([^,]+),Type=([^,]+),Description="([^"]+)">')


VCFHeaders = namedtuple('VCFHeaders', ['headers', 'filters', 'infos', 'formats', 'samples'])


def read_vcf_headers(binary_file):

    # setup
    headers = []
    samples = None
    filters = dict()
    infos = dict()
    formats = dict()

    # read first header line
    header = _binary_readline(binary_file)

    while header and header[0] == ord(b'#'):
        # debug('found header', header)

        headers.append(header)

        if header.startswith(b'##FILTER'):

            match = re_filter_header.match(header)
            if match is None:
                raise RuntimeError('bad FILTER header: %r' % header)
            else:
                k, d = match.groups()
                filters[k] = {
                    'ID': k,
                    'Description': d
                }

        elif header.startswith(b'##INFO'):

            match = re_info_header.match(header)
            if match is None:
                raise RuntimeError('bad INFO header: %r' % header)
            else:
                k, n, t, d = match.groups()
                infos[k] = {
                    'ID': k,
                    'Number': n,
                    'Type': t,
                    'Description': d
                }

        elif header.startswith(b'##FORMAT'):

            match = re_format_header.match(header)
            if match is None:
                raise RuntimeError('bad FORMAT header: %r' % header)
            else:
                k, n, t, d = match.groups()
                formats[k] = {
                    'ID': k,
                    'Number': n,
                    'Type': t,
                    'Description': d
                }

        elif header.startswith(b'#CHROM'):

            # parse out samples
            samples = header.split(b'\t')[9:]
            break

        # read next header line
        header = _binary_readline(binary_file)

    # check if we saw the mandatory header line or not
    if samples is None:
        raise RuntimeError('VCF file is missing mandatory header line ("#CHROM...")')

    return VCFHeaders(headers, filters, infos, formats, samples)
