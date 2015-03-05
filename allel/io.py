# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import csv
from datetime import date
import itertools
from operator import itemgetter
import subprocess
import gzip
import logging
from allel.compat import zip_longest, PY2, text_type, range, unquote_plus


import numpy as np


import allel


logger = logging.getLogger(__name__)
debug = logger.debug


VCF_FIXED_FIELDS = 'CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO'


def write_vcf(path, variants, rename=None, number=None, description=None,
              fill=None, write_header=True):
    with open(path, 'w') as vcf_file:
        if write_header:
            write_vcf_header(vcf_file, variants=variants, rename=rename,
                             number=number, description=description)
        write_vcf_data(vcf_file, variants=variants, rename=rename, fill=fill)


def write_vcf_header(vcf_file, variants, rename, number, description):
    if rename is None:
        rename = dict()
    if number is None:
        number = dict()
    if description is None:
        description = dict()

    # write file format version
    print('##fileformat=VCFv4.1', file=vcf_file)

    # write today's date
    today = date.today().strftime('%Y%m%d')
    print('##fileDate=%s' % today, file=vcf_file)

    # write source
    print('##source=scikit-allel-%s' % allel.__version__, file=vcf_file)

    names = variants.dtype.names
    info_names = [n for n in names
                  if not n.upper().startswith('FILTER_') and
                  not n.upper() in VCF_FIXED_FIELDS]
    info_ids = [rename[n] if n in rename else n
                for n in info_names]

    # write INFO headers, sorted by ID
    for name, vcf_id in sorted(zip(info_names, info_ids), key=itemgetter(1)):
        col = variants[name]

        # determine VCF Number
        if name in number:
            vcf_number = number[name]
        else:
            if col.ndim == 1 and col.dtype.kind == 'b':
                # Flag
                vcf_number = 0
            elif col.ndim == 1:
                vcf_number = 1
            elif col.ndim == 2:
                vcf_number = col.shape[1]
            else:
                raise NotImplementedError('only columns with 1 or two '
                                          'dimensions are supported')

        # determine VCF Type
        kind = col.dtype.kind
        if kind == 'b':
            vcf_type = 'Flag'
        elif kind in 'ui':
            vcf_type = 'Integer'
        elif kind == 'f':
            vcf_type = 'Float'
        else:
            vcf_type = 'String'

        # determine VCF Description
        if name in description:
            vcf_description = description[name]
        else:
            vcf_description = ''

        # construct INFO header line
        header_line = '##INFO=<ID=%s,Number=%s,Type=%s,Description="%s">'\
            % (vcf_id, vcf_number, vcf_type, vcf_description)
        print(header_line, file=vcf_file)

    filter_names = [n for n in names if n.upper().startswith('FILTER_')]
    filter_ids = [rename[n] if n in rename else n[7:]
                  for n in filter_names]

    # write FILTER headers, sorted by ID
    for name, vcf_id in sorted(zip(filter_names, filter_ids),
                               key=itemgetter(1)):

        # determine VCF Description
        if name in description:
            vcf_description = description[name]
        else:
            vcf_description = ''

        # construct FILTER header line
        header_line = '##FILTER=<ID=%s,Description="%s">'\
            % (vcf_id, vcf_description)
        print(header_line, file=vcf_file)

    # write column names
    line = '#' + '\t'.join(VCF_FIXED_FIELDS)
    print(line, file=vcf_file)


def write_vcf_data(vcf_file, variants, rename, fill):
    if rename is None:
        rename = dict()
    if fill is None:
        fill = dict()

    # find the fixed columns, allowing for case insensitive naming in the
    # input array
    col_chrom = None
    col_pos = None
    col_id = None
    col_ref = None
    col_alt = None
    col_qual = None
    for n in variants.dtype.names:
        if n.upper() == 'CHROM':
            col_chrom = variants[n]
        elif n.upper() == 'POS':
            col_pos = variants[n]
        elif n.upper() == 'ID':
            col_id = variants[n]
        elif n.upper() == 'REF':
            col_ref = variants[n]
        elif n.upper() == 'ALT':
            col_alt = variants[n]
        elif n.upper() == 'QUAL':
            col_qual = variants[n]

    # check for required columns
    if col_chrom is None:
        raise ValueError('CHROM column not found')
    if col_pos is None:
        raise ValueError('POS column not found')

    # pad optional columns
    dot = itertools.repeat('.')
    if col_id is None:
        col_id = dot
    if col_ref is None:
        col_ref = dot
    if col_alt is None:
        col_alt = dot
    if col_qual is None:
        col_qual = dot

    # find FILTER columns
    filter_names = [n for n in variants.dtype.names
                    if n.upper().startswith('FILTER_')]
    filter_ids = [rename[n] if n in rename else n[7:]
                  for n in filter_names]
    filter_cols = [variants[n] for n in filter_names]
    # sort by ID
    if filter_names:
        filters = sorted(zip(filter_names, filter_ids, filter_cols),
                         key=itemgetter(1))
        filter_names, filter_ids, filter_cols = zip(*filters)

    # find INFO columns
    info_names = [n for n in variants.dtype.names
                  if not n.upper().startswith('FILTER_') and
                  not n.upper() in VCF_FIXED_FIELDS]
    info_ids = [rename[n] if n in rename else n
                for n in info_names]
    info_cols = [variants[n] for n in info_names]
    # sort by ID
    if info_names:
        infos = sorted(zip(info_names, info_ids, info_cols),
                       key=itemgetter(1))
        info_names, info_ids, info_cols = zip(*infos)

    # setup writer
    writer = csv.writer(vcf_file, delimiter='\t', lineterminator='\n')

    # zip up data as rows
    rows = zip(col_chrom, col_pos, col_id, col_ref, col_alt, col_qual)
    filter_rows = zip(*filter_cols)
    info_rows = zip(*info_cols)

    for row, filter_row, info_row in zip_longest(rows, filter_rows, info_rows):

        # unpack main row
        chrom, pos, id, ref, alt, qual = row
        chrom = _vcf_value_str(chrom)
        pos = _vcf_value_str(pos)
        id = _vcf_value_str(id)
        ref = _vcf_value_str(ref)
        alt = _vcf_value_str(alt, fill=fill.get('ALT', None))
        qual = _vcf_value_str(qual)

        # construct FILTER value
        if filter_row is not None:
            flt = [i for i, v in zip(filter_ids, filter_row) if v]
            if flt:
                flt = ';'.join(flt)
            else:
                flt = 'PASS'
        else:
            flt = '.'

        # construct INFO value
        if info_row is not None:
            info_vals = [_vcf_info_str(n, i, v, fill)
                         for n, i, v in zip(info_names, info_ids, info_row)]
            info_vals = [x for x in info_vals if x is not None]
            info = ';'.join(info_vals)
        else:
            info = '.'

        # repack
        row = chrom, pos, id, ref, alt, qual, flt, info
        writer.writerow(row)


def _vcf_value_str(o, fill=None):
    if isinstance(o, bytes) and not PY2:
        return str(o, encoding='ascii')
    elif isinstance(o, (tuple, list, np.ndarray)):
        if fill is None:
            t = [_vcf_value_str(x) for x in o]
        else:
            t = [_vcf_value_str(x) for x in o if x != fill]
        return ','.join(t)
    else:
        return str(o)


def _vcf_info_str(name, id, value, fill):
    if isinstance(value, (bool, np.bool_)):
        if bool(value):
            return id
        else:
            return None
    else:
        return '%s=%s' % (id, _vcf_value_str(value, fill=fill.get(name, None)))


def write_fasta(path, sequences, names, mode='w', width=80):
    """Write nucleotide sequences stored as numpy arrays to a FASTA file.

    Parameters
    ----------

    path : string
        File path.
    sequences : sequence of arrays
        One or more ndarrays of dtype 'S1' containing the sequences.
    names : sequence of strings
        Names of the sequences.
    mode : string, optional
        Use 'a' to append to an existing file.
    width : int, optional
        Maximum line width.

    """

    # check inputs
    if isinstance(sequences, np.ndarray):
        # single sequence
        sequences = [sequences]
        names = [names]
    if len(sequences) != len(names):
        raise ValueError('must provide the same number of sequences and names')
    for sequence in sequences:
        if sequence.dtype != np.dtype('S1'):
            raise ValueError('expected S1 dtype, found %r' % sequence.dtype)

    # force binary mode
    mode = 'ab' if 'a' in mode else 'wb'

    # write to file
    with open(path, mode=mode) as fasta:
        for name, sequence in zip(names, sequences):
            # force bytes
            if isinstance(name, text_type):
                name = name.encode('ascii')
            header = b'>' + name + b'\n'
            fasta.write(header)
            for i in range(0, sequence.size, width):
                line = sequence[i:i+width].tostring() + b'\n'
                fasta.write(line)


def gff3_parse_attributes(attributes_string):
    """Parse a string of GFF3 attributes ('key=value' pairs delimited by ';')
    and return a dictionary.

    """

    attributes = dict()
    if not PY2:
        # convert to ascii string to enable conversion of escaped characters
        attributes_string = str(attributes_string, encoding='ascii')
    fields = attributes_string.split(';')
    for f in fields:
        if '=' in f:
            key, value = f.split('=')
            key = unquote_plus(key).strip()
            value = unquote_plus(value.strip())
            if not PY2:
                # convert back to bytes
                value = value.encode('ascii')
            attributes[key] = value
        elif len(f) > 0:
            # not strictly kosher
            attributes[unquote_plus(f).strip()] = True
    return attributes


def iter_gff3(path, attributes=None, region=None, score_fill=-1,
              phase_fill=-1, attributes_fill=b'.'):

    # prepare fill values for attributes
    if attributes is not None:
        if isinstance(attributes_fill, (list, tuple)):
            if len(attributes != len(attributes_fill)):
                raise ValueError('number of fills does not match attributes')
        else:
            attributes_fill = [attributes_fill] * len(attributes)

    # open input stream
    needs_closing = False
    if region is not None:
        cmd = ['tabix', path, region]
        buffer = subprocess.Popen(cmd, stdout=subprocess.PIPE).stdout
    elif path.endswith('.gz') or path.endswith('.bgz'):
        buffer = gzip.open(path, mode='rb')
        needs_closing = True
    else:
        buffer = open(path, mode='rb')
        needs_closing = True
    debug(buffer)

    try:
        for line in buffer:
            if line[0] == b'>':
                # assume begin embedded FASTA
                raise StopIteration
            if line[0] == b'#':
                # skip comment lines
                continue
            vals = line.split(b'\t')
            if len(vals) == 9:
                # unpack for processing
                fseqid, fsource, ftype, fstart, fend, fscore, fstrand, \
                    fphase, fattrs = vals
                # convert numerics
                fstart = int(fstart)
                fend = int(fend)
                if fscore == b'.':
                    fscore = score_fill
                else:
                    fscore = float(fscore)
                if fphase == b'.':
                    fphase = phase_fill
                else:
                    fphase = int(fphase)
                rec = (fseqid, fsource, ftype, fstart, fend, fscore,
                       fstrand, fphase)
                if attributes is not None:
                    dattrs = gff3_parse_attributes(fattrs)
                    vattrs = tuple(
                        dattrs.get(k, f)
                        for k, f in zip(attributes, attributes_fill)
                    )
                    rec += vattrs
                yield rec

    finally:
        if needs_closing:
            buffer.close()
