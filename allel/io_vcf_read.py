# -*- coding: utf-8 -*-
"""
TODO:

* DONE Fix missing value in trailing item
* DONE Inintial implementation of vcf_to_npz
* DONE Initial implementation of vcf_to_hdf5
* DONE Add samples to output of read_vcf and store in vcf_to_... functions
* Implement vcf_to_zarr
* Parse FILTERS from header
* Return filters as separate arrays in read_vcf
* Store filters as separate arrays/datasets in vcf_to_... functions
* Parse INFO fields
* Parse other FORMAT fields
* Read from region via tabix
* Read from region via scanning
* Progress logging in vcf_to_... functions
* User-controlled dtypes
* User-controlled fill values
* User-controlled arities

"""
from __future__ import absolute_import, print_function, division
import gzip
import sys
import itertools
import os


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
             buffer_size=DEFAULT_BUFFER_SIZE,
             chunk_length=DEFAULT_CHUNK_LENGTH,
             temp_max_size=DEFAULT_TEMP_SIZE):
    """Read data from a VCF file into NumPy arrays.

    Parameters
    ----------
    path : str
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
    samples, chunks = read_vcf_chunks(path=path, buffer_size=buffer_size,
                                      chunk_length=chunk_length, temp_max_size=temp_max_size)

    # read all chunks into a list
    chunks = list(chunks)

    # setup output
    output = dict()
    output['samples'] = np.array(samples)

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
               buffer_size=DEFAULT_BUFFER_SIZE,
               chunk_length=DEFAULT_CHUNK_LENGTH,
               temp_max_size=DEFAULT_TEMP_SIZE):
    """TODO"""

    # guard condition
    if not overwrite and os.path.exists(output_path):
        # TODO right exception class?
        raise ValueError('file exists at path %r; use overwrite=True to replace' % output_path)

    # read all data into memory
    data = read_vcf(path=input_path, buffer_size=buffer_size, chunk_length=chunk_length,
                    temp_max_size=temp_max_size)

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
        samples, chunks = read_vcf_chunks(input_path, buffer_size=buffer_size,
                                          chunk_length=chunk_length,
                                          temp_max_size=temp_max_size)
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
        root[group].create_dataset(name, data=np.array(samples))

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

            shape = data.shape
            maxshape = (None,) + shape[1:]
            root[group].create_dataset(name, shape=data.shape, maxshape=maxshape,
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


def read_vcf_chunks(path,
                    buffer_size=DEFAULT_BUFFER_SIZE,
                    chunk_length=DEFAULT_CHUNK_LENGTH,
                    temp_max_size=DEFAULT_TEMP_SIZE):
    """TODO"""

    if isinstance(path, str) and path.endswith('gz'):
        # assume gzip-compatible compression
        with gzip.open(path, mode='rb') as binary_file:
            return _read_vcf(binary_file, buffer_size=buffer_size, chunk_length=chunk_length,
                             temp_max_size=temp_max_size)

    elif isinstance(path, str):
        # assume no compression
        with open(path, mode='rb', buffering=0) as binary_file:
            return _read_vcf(binary_file, buffer_size=buffer_size, chunk_length=chunk_length,
                             temp_max_size=temp_max_size)

    else:
        # assume some other binary file-like object
        binary_file = path
        return _read_vcf(binary_file, buffer_size=buffer_size, chunk_length=chunk_length,
                         temp_max_size=temp_max_size)


def _read_vcf(fileobj, buffer_size, chunk_length, temp_max_size):
    headers, samples = _read_vcf_headers(fileobj)
    # debug(samples)
    chunks = iter_vcf(fileobj, buffer_size=buffer_size, chunk_length=chunk_length,
                      temp_max_size=temp_max_size, n_samples=len(samples))
    return samples, chunks


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
