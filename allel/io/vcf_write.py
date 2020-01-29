# -*- coding: utf-8 -*-
import csv
from datetime import date
import itertools
from operator import itemgetter
import logging


import numpy as np
import zarr


import allel


logger = logging.getLogger(__name__)
debug = logger.debug


VCF_FIXED_FIELDS = 'CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO'

CALLDATA_CALLSET_GROUP = 'calldata'
GENOTYPE_CALLSET_KEY = 'GT'
SAMPLES_CALLSET_KEY = 'samples'

NORMALIZED_SAMPLE_NAME_PREFIX = 'SAMPLE_'


def normalize_callset(callset):

    if hasattr(callset, 'keys'):
        names = list()
        new_callset = dict()
        for k in list(callset.keys()):
            a = callset[k]
            if k.startswith('calldata/'):
                continue
            if k == 'samples':
                continue
            if k.startswith('variants/'):
                k = k[9:]
            names.append(k)
            new_callset[k] = a
        callset = new_callset
    elif hasattr(callset, 'dtype') and callset.dtype.names:
        names = list(callset.dtype.names)
    elif isinstance(callset, zarr.hierarchy.Group):
        names = list()
        new_callset = dict()

        for group in callset.group_keys():
            if group == CALLDATA_CALLSET_GROUP:
                continue

            for key in callset[group].array_keys():
                names.append(key)
                new_callset[key] = callset[group][key]
        
        gt = callset.get(CALLDATA_CALLSET_GROUP + '/' + GENOTYPE_CALLSET_KEY)
        if gt:
            samples = callset.get(SAMPLES_CALLSET_KEY, [])
            n_gt_cols = gt.shape[1]
            if len(samples) != n_gt_cols:
                raise ValueError('number of sample names and genotype columns mismatch')

            for i in range(n_gt_cols):
                sample_name = samples[i]
                names.append(_filterable_sample_name(sample_name))
                new_callset[sample_name] = gt[:,i,:]

        callset = new_callset
    else:
        raise ValueError('callset should be dict or recarray, found %r' % callset)

    return names, callset


def write_vcf(path, callset, rename=None, number=None, description=None,
              fill=None, write_header=True):
    """Preliminary support for writing a VCF file. Currently does not support sample data.
    Needs further work."""

    names, callset = normalize_callset(callset)

    with open(path, 'w') as vcf_file:
        if write_header:
            write_vcf_header(vcf_file, names, callset=callset, rename=rename,
                             number=number, description=description)
        write_vcf_data(vcf_file, names, callset=callset, rename=rename, fill=fill)


def write_vcf_header(vcf_file, names, callset, rename, number, description):
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

    info_names = [n for n in names
                  if not n.upper().startswith('FILTER_') and
                  not n.upper() in VCF_FIXED_FIELDS and
                  not _is_sample_name(n)]
    info_ids = [rename[n] if n in rename else n
                for n in info_names]

    # write INFO headers, sorted by ID
    for name, vcf_id in sorted(zip(info_names, info_ids), key=itemgetter(1)):
        col = callset[name]

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

    # reconstruct sample names
    sample_names = _filter_sample_names(names)

    # write column names
    columns = list(VCF_FIXED_FIELDS) + sample_names
    line = '#' + '\t'.join(columns)
    print(line, file=vcf_file)


# noinspection PyShadowingBuiltins
def write_vcf_data(vcf_file, names, callset, rename, fill):
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
    for n in names:
        if n.upper() == 'CHROM':
            col_chrom = callset[n]
        elif n.upper() == 'POS':
            col_pos = callset[n]
        elif n.upper() == 'ID':
            col_id = callset[n]
        elif n.upper() == 'REF':
            col_ref = callset[n]
        elif n.upper() == 'ALT':
            col_alt = callset[n]
        elif n.upper() == 'QUAL':
            col_qual = callset[n]

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
    filter_names = [n for n in names
                    if n.upper().startswith('FILTER_')]
    filter_ids = [rename[n] if n in rename else n[7:]
                  for n in filter_names]
    filter_cols = [callset[n] for n in filter_names]
    # sort by ID
    if filter_names:
        filters = sorted(zip(filter_names, filter_ids, filter_cols),
                         key=itemgetter(1))
        filter_names, filter_ids, filter_cols = zip(*filters)

    # find INFO columns
    info_names = [n for n in names
                  if not n.upper().startswith('FILTER_') and
                  not n.upper() in VCF_FIXED_FIELDS and
                  not _is_sample_name(n)]
    info_ids = [rename[n] if n in rename else n
                for n in info_names]
    info_cols = [callset[n] for n in info_names]
    # sort by ID
    if info_names:
        infos = sorted(zip(info_names, info_ids, info_cols),
                       key=itemgetter(1))
        info_names, info_ids, info_cols = zip(*infos)

    # genotype columns
    sample_names = _filter_sample_names(names)
    gt_cols = [callset[n] for n in sample_names]

    # setup writer
    writer = csv.writer(vcf_file, delimiter='\t', lineterminator='\n')

    # zip up data as rows
    rows = zip(col_chrom, col_pos, col_id, col_ref, col_alt, col_qual, *gt_cols)
    filter_rows = zip(*filter_cols)
    info_rows = zip(*info_cols)

    for row, filter_row, info_row in itertools.zip_longest(rows, filter_rows, info_rows):

        # unpack main row
        chrom, pos, id, ref, alt, qual, *gts = row
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

        # construct genotype value
        gts = ['/'.join(map(str, gt)) for gt in gts]

        # repack
        row = chrom, pos, id, ref, alt, qual, flt, info, *gts
        writer.writerow(row)


def _vcf_value_str(o, fill=None):
    if isinstance(o, bytes):
        return str(o, encoding='ascii')
    elif isinstance(o, (tuple, list, np.ndarray)):
        if fill is None:
            t = [_vcf_value_str(x) for x in o]
        else:
            t = [_vcf_value_str(x) for x in o if x != fill]
        return ','.join(t)
    else:
        return str(o)


# noinspection PyShadowingBuiltins
def _vcf_info_str(name, id, value, fill):
    if isinstance(value, (bool, np.bool_)):
        if bool(value):
            return id
        else:
            return None
    else:
        return '%s=%s' % (id, _vcf_value_str(value, fill=fill.get(name, None)))


def _filter_sample_names(names):
    return [n[len(NORMALIZED_SAMPLE_NAME_PREFIX):] for n in names 
            if _is_sample_name(n)]


def _is_sample_name(name):
    return name.startswith(NORMALIZED_SAMPLE_NAME_PREFIX)


def _filterable_sample_name(sample_name):
    return NORMALIZED_SAMPLE_NAME_PREFIX + sample_name