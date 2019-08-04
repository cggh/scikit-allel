# -*- coding: utf-8 -*-
import subprocess
import gzip
from urllib.parse import unquote_plus


import numpy as np


def gff3_parse_attributes(attributes_string):
    """Parse a string of GFF3 attributes ('key=value' pairs delimited by ';')
    and return a dictionary."""

    attributes = dict()
    fields = attributes_string.split(';')
    for f in fields:
        if '=' in f:
            key, value = f.split('=')
            key = unquote_plus(key).strip()
            value = unquote_plus(value.strip())
            attributes[key] = value
        elif len(f) > 0:
            # not strictly kosher
            attributes[unquote_plus(f).strip()] = True
    return attributes


def iter_gff3(path, attributes=None, region=None, score_fill=-1,
              phase_fill=-1, attributes_fill='.', tabix='tabix'):
    """Iterate over records in a GFF3 file.

    Parameters
    ----------
    path : string
        Path to input file.
    attributes : list of strings, optional
        List of columns to extract from the "attributes" field.
    region : string, optional
        Genome region to extract. If given, file must be position
        sorted, bgzipped and tabix indexed. Tabix must also be installed
        and on the system path.
    score_fill : int, optional
        Value to use where score field has a missing value.
    phase_fill : int, optional
        Value to use where phase field has a missing value.
    attributes_fill : object or list of objects, optional
        Value(s) to use where attribute field(s) have a missing value.
    tabix : string
        Tabix command.

    Returns
    -------
    Iterator

    """

    # prepare fill values for attributes
    if attributes is not None:
        attributes = list(attributes)
        if isinstance(attributes_fill, (list, tuple)):
            if len(attributes) != len(attributes_fill):
                raise ValueError('number of fills does not match attributes')
        else:
            attributes_fill = [attributes_fill] * len(attributes)

    # open input stream
    if region is not None:
        cmd = [tabix, path, region]
        buffer = subprocess.Popen(cmd, stdout=subprocess.PIPE).stdout
    elif path.endswith('.gz') or path.endswith('.bgz'):
        buffer = gzip.open(path, mode='rb')
    else:
        buffer = open(path, mode='rb')

    try:
        for line in buffer:
            if line[0] == b'>':
                # assume begin embedded FASTA
                return
            if line[0] == b'#':
                # skip comment lines
                continue
            vals = line.split(b'\t')
            if len(vals) == 9:
                # unpack for processing
                fseqid, fsource, ftype, fstart, fend, fscore, fstrand, fphase, fattrs = vals
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
                fseqid = str(fseqid, 'ascii')
                fsource = str(fsource, 'ascii')
                ftype = str(ftype, 'ascii')
                fstrand = str(fstrand, 'ascii')
                fattrs = str(fattrs, 'ascii')
                rec = (fseqid, fsource, ftype, fstart, fend, fscore, fstrand, fphase)
                if attributes is not None:
                    dattrs = gff3_parse_attributes(fattrs)
                    vattrs = tuple(
                        dattrs.get(k, f)
                        for k, f in zip(attributes, attributes_fill)
                    )
                    rec += vattrs
                yield rec

    finally:
        buffer.close()


# TODO dry docstrings


def gff3_to_recarray(path, attributes=None, region=None, score_fill=-1,
                     phase_fill=-1, attributes_fill='.', tabix='tabix', dtype=None):
    """Load data from a GFF3 into a NumPy recarray.

    Parameters
    ----------
    path : string
        Path to input file.
    attributes : list of strings, optional
        List of columns to extract from the "attributes" field.
    region : string, optional
        Genome region to extract. If given, file must be position
        sorted, bgzipped and tabix indexed. Tabix must also be installed
        and on the system path.
    score_fill : int, optional
        Value to use where score field has a missing value.
    phase_fill : int, optional
        Value to use where phase field has a missing value.
    attributes_fill : object or list of objects, optional
        Value(s) to use where attribute field(s) have a missing value.
    tabix : string, optional
        Tabix command.
    dtype : dtype, optional
        Override dtype.

    Returns
    -------
    np.recarray

    """

    # read records
    recs = list(iter_gff3(path, attributes=attributes, region=region,
                          score_fill=score_fill, phase_fill=phase_fill,
                          attributes_fill=attributes_fill, tabix=tabix))

    if not recs:
        return None

    # determine dtype
    if dtype is None:
        dtype = [('seqid', object),
                 ('source', object),
                 ('type', object),
                 ('start', int),
                 ('end', int),
                 ('score', float),
                 ('strand', object),
                 ('phase', int)]
        if attributes:
            for n in attributes:
                dtype.append((n, object))

    a = np.rec.fromrecords(recs, dtype=dtype)
    return a


def gff3_to_dataframe(path, attributes=None, region=None, score_fill=-1,
                      phase_fill=-1, attributes_fill='.', tabix='tabix', **kwargs):
    """Load data from a GFF3 into a pandas DataFrame.

    Parameters
    ----------
    path : string
        Path to input file.
    attributes : list of strings, optional
        List of columns to extract from the "attributes" field.
    region : string, optional
        Genome region to extract. If given, file must be position
        sorted, bgzipped and tabix indexed. Tabix must also be installed
        and on the system path.
    score_fill : int, optional
        Value to use where score field has a missing value.
    phase_fill : int, optional
        Value to use where phase field has a missing value.
    attributes_fill : object or list of objects, optional
        Value(s) to use where attribute field(s) have a missing value.
    tabix : string, optional
        Tabix command.

    Returns
    -------
    pandas.DataFrame

    """

    import pandas

    # read records
    recs = list(iter_gff3(path, attributes=attributes, region=region,
                          score_fill=score_fill, phase_fill=phase_fill,
                          attributes_fill=attributes_fill, tabix=tabix))

    # load into pandas
    columns = ['seqid', 'source', 'type', 'start', 'end', 'score', 'strand', 'phase']
    if attributes:
        columns += list(attributes)
    df = pandas.DataFrame.from_records(recs, columns=columns, **kwargs)

    return df
