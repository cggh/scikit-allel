# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import itertools


# third-party imports
import numpy as np


# internal imports
from allel.compat import PY2
from allel.util import check_ndim
from allel.io import write_vcf, recarray_from_hdf5_group, recarray_to_hdf5_group, iter_gff3
from .arrays import ArrayBase
from .indexes import SortedIndex, SortedMultiIndex


class RecArrayBase(ArrayBase):

    @classmethod
    def _check_values(cls, data):
        check_ndim(data, 1)
        if not data.dtype.names:
            raise ValueError('expected recarray')

    @property
    def names(self):
        """Column names."""
        return self.dtype.names

    def __getitem__(self, item):
        s = super(RecArrayBase, self).__getitem__(item)
        if isinstance(item, (slice, list, np.ndarray, type(Ellipsis))):
            return type(self)(s)
        return s

    def _display_items(self, threshold, edgeitems):
        if threshold is None:
            threshold = self.shape[0]

        # ensure sensible edgeitems
        edgeitems = min(edgeitems, threshold // 2)

        # determine indices of items to show
        if self.shape[0] > threshold:
            indices = list(range(edgeitems))
            indices += list(range(self.shape[0] - edgeitems, self.shape[0], 1))
        else:
            indices = list(range(self.shape[0]))

        # convert to stringy thingy
        tmp = self[indices]
        items = [[repr(x) for x in row] for row in tmp]

        # insert ellipsis
        if self.shape[0] > threshold:
            indices = (
                indices[:edgeitems] + ['...'] + indices[-edgeitems:]
            )
            items = items[:edgeitems] + [['...']] + items[-edgeitems:]

        return indices, items

    def to_html(self, threshold=6, edgeitems=3, caption=None):
        indices, items = self._display_items(threshold, edgeitems)
        # N.B., table captions don't render in jupyter notebooks on GitHub,
        # so put caption outside table element
        if caption is None:
            caption = '%s(%s, dtype=%s)\n' % (type(self).__name__, self.shape, self.dtype)
        # sanitize caption
        caption = caption.replace('<', '&lt;')
        caption = caption.replace('\n', '<br/>')
        html = caption
        html += '<table>'
        html += '<tr><th></th>'
        html += ''.join(['<th style="text-align: center">%s</th>' % n
                         for n in self.dtype.names])
        html += '</tr>'
        for row_index, row in zip(indices, items):
            if row_index == '...':
                html += '<tr><td>...</td><td colspan="%s"></td></tr>' % self.shape[1]
            else:
                html += '<tr><th style="text-align: center">%s</th>' % row_index
                html += ''.join(['<td style="text-align: center">%s</td>' % item
                                 for item in row])
                html += '</tr>'
        html += '</table>'
        return html

    def __str__(self):
        return str(self.values)

    def _repr_html_(self):
        return self.to_html()

    def display(self, threshold=6, edgeitems=3, caption=None):
        html = self.to_html(threshold, edgeitems, caption)
        from IPython.display import display_html
        display_html(html, raw=True)

    def displayall(self, caption=None):
        self.display(threshold=None, caption=caption)

    @classmethod
    def from_hdf5_group(cls, *args, **kwargs):
        a = recarray_from_hdf5_group(*args, **kwargs)
        return cls(a, copy=False)

    def to_hdf5_group(self, parent, name, **kwargs):
        return recarray_to_hdf5_group(self, parent, name, **kwargs)

    def eval(self, expression, vm='python'):
        """Evaluate an expression against the table columns.

        Parameters
        ----------
        expression : string
            Expression to evaluate.
        vm : {'numexpr', 'python'}
            Virtual machine to use.

        Returns
        -------
        result : ndarray

        """

        if vm == 'numexpr':
            import numexpr as ne
            return ne.evaluate(expression, local_dict=self)
        else:
            if PY2:
                # locals must be a mapping
                m = {k: self[k] for k in self.dtype.names}
            else:
                m = self
            return eval(expression, dict(), m)

    def query(self, expression, vm='python'):
        """Evaluate expression and then use it to extract rows from the table.

        Parameters
        ----------
        expression : string
            Expression to evaluate.
        vm : {'numexpr', 'python'}
            Virtual machine to use.

        Returns
        -------
        result : structured array

        """

        condition = self.eval(expression, vm=vm)
        return self.compress(condition)


class VariantTable(RecArrayBase):
    """Table (catalogue) of variants.

    Parameters
    ----------
    data : array_like, structured, shape (n_variants,)
        Variant records.
    index : string or pair of strings, optional
        Names of columns to use for positional index, e.g., 'POS' if table
        contains a 'POS' column and records from a single chromosome/contig,
        or ('CHROM', 'POS') if table contains records from multiple
        chromosomes/contigs.
    **kwargs : keyword arguments, optional
        Further keyword arguments are passed through to
        :func:`numpy.rec.array`.

    Examples
    --------
    Instantiate a table from existing data::

        >>> import allel
        >>> records = [[b'chr1', 2, 35, 4.5, (1, 2)],
        ...            [b'chr1', 7, 12, 6.7, (3, 4)],
        ...            [b'chr2', 3, 78, 1.2, (5, 6)],
        ...            [b'chr2', 9, 22, 4.4, (7, 8)],
        ...            [b'chr3', 6, 99, 2.8, (9, 10)]]
        >>> dtype = [('CHROM', 'S4'),
        ...          ('POS', 'u4'),
        ...          ('DP', int),
        ...          ('QD', float),
        ...          ('AC', (int, 2))]
        >>> vt = allel.VariantTable(records, dtype=dtype,
        ...                         index=('CHROM', 'POS'))
        >>> vt.names
        ('CHROM', 'POS', 'DP', 'QD', 'AC')
        >>> vt.n_variants
        5

    Access a column::

        >>> vt['DP']
        array([35, 12, 78, 22, 99])

    Access multiple columns::

        >>> vt[['DP', 'QD']]  # doctest: +ELLIPSIS
        VariantTable((5,), dtype=(numpy.record, [('DP', '<i8'), ('QD', '<f8...
        [(35, 4.5) (12, 6.7) (78, 1.2) (22, 4.4) (99, 2.8)]

    Access a row::

        >>> vt[2]
        (b'chr2', 3, 78, 1.2, array([5, 6]))

    Access multiple rows::

        >>> vt[2:4]  # doctest: +ELLIPSIS
        VariantTable((2,), dtype=(numpy.record, [('CHROM', 'S4'), ('POS', '...
        [(b'chr2', 3, 78, 1.2, array([5, 6])) (b'chr2', 9, 22, 4.4, array([...

    Evaluate expressions against the table::

        >>> vt.eval('DP > 30')
        array([ True, False,  True, False,  True], dtype=bool)
        >>> vt.eval('(DP > 30) & (QD > 4)')
        array([ True, False, False, False, False], dtype=bool)
        >>> vt.eval('DP * 2')
        array([ 70,  24, 156,  44, 198])

    Query the table::

        >>> vt.query('DP > 30')  # doctest: +ELLIPSIS
        VariantTable((3,), dtype=(numpy.record, [('CHROM', 'S4'), ('POS', '...
        [(b'chr1', 2, 35, 4.5, array([1, 2])) (b'chr2', 3, 78, 1.2, array([...
         (b'chr3', 6, 99, 2.8, array([ 9, 10]))]
        >>> vt.query('(DP > 30) & (QD > 4)')  # doctest: +ELLIPSIS
        VariantTable((1,), dtype=(numpy.record, [('CHROM', 'S4'), ('POS', '...
        [(b'chr1', 2, 35, 4.5, array([1, 2]))]

    Use the index to query variants::

        >>> vt.query_region(b'chr2', 1, 10)  # doctest: +ELLIPSIS
        VariantTable((2,), dtype=(numpy.record, [('CHROM', 'S4'), ('POS', '...
        [(b'chr2', 3, 78, 1.2, array([5, 6])) (b'chr2', 9, 22, 4.4, array([...

    """

    def __init__(self, data, index=None, copy=False, **kwargs):
        super(VariantTable, self).__init__(data, copy=copy, **kwargs)
        self.set_index(index)

    @property
    def n_variants(self):
        """Number of variants (length of first dimension)."""
        return self.shape[0]

    # noinspection PyAttributeOutsideInit
    def set_index(self, index):
        """Set or reset the index.

        Parameters
        ----------
        index : string or pair of strings, optional
            Names of columns to use for positional index, e.g., 'POS' if table
            contains a 'POS' column and records from a single
            chromosome/contig, or ('CHROM', 'POS') if table contains records
            from multiple chromosomes/contigs.

        """
        if index is None:
            pass
        elif isinstance(index, str):
            index = SortedIndex(self[index], copy=False)
        elif isinstance(index, (tuple, list)) and len(index) == 2:
            index = SortedMultiIndex(self[index[0]], self[index[1]],
                                     copy=False)
        else:
            raise ValueError('invalid index argument, expected string or '
                             'pair of strings, found %s' % repr(index))
        self.index = index

    def query_position(self, chrom=None, position=None):
        """Query the table, returning row or rows matching the given genomic
        position.

        Parameters
        ----------
        chrom : string, optional
            Chromosome/contig.
        position : int, optional
            Position (1-based).

        Returns
        -------
        result : row or VariantTable

        """

        if self.index is None:
            raise ValueError('no index has been set')
        if isinstance(self.index, SortedIndex):
            # ignore chrom
            loc = self.index.locate_key(position)
        else:
            loc = self.index.locate_key(chrom, position)
        return self[loc]

    def query_region(self, chrom=None, start=None, stop=None):
        """Query the table, returning row or rows within the given genomic
        region.

        Parameters
        ----------
        chrom : string, optional
            Chromosome/contig.
        start : int, optional
            Region start position (1-based).
        stop : int, optional
            Region stop position (1-based).

        Returns
        -------
        result : VariantTable

        """
        if self.index is None:
            raise ValueError('no index has been set')
        if isinstance(self.index, SortedIndex):
            # ignore chrom
            loc = self.index.locate_range(start, stop)
        else:
            loc = self.index.locate_range(chrom, start, stop)
        return self[loc]

    def to_vcf(self, path, rename=None, number=None, description=None,
               fill=None, write_header=True):
        r"""Write to a variant call format (VCF) file.

        Parameters
        ----------
        path : string
            File path.
        rename : dict, optional
            Rename these columns in the VCF.
        number : dict, optional
            Override the number specified in INFO headers.
        description : dict, optional
            Descriptions for the INFO and FILTER headers.
        fill : dict, optional
            Fill values used for missing data in the table.
        write_header : bool, optional
            If True write VCF header.

        Examples
        --------
        Setup a variant table to write out::

            >>> import allel
            >>> chrom = [b'chr1', b'chr1', b'chr2', b'chr2', b'chr3']
            >>> pos = [2, 6, 3, 8, 1]
            >>> ids = ['a', 'b', 'c', 'd', 'e']
            >>> ref = [b'A', b'C', b'T', b'G', b'N']
            >>> alt = [(b'T', b'.'),
            ...        (b'G', b'.'),
            ...        (b'A', b'C'),
            ...        (b'C', b'A'),
            ...        (b'X', b'.')]
            >>> qual = [1.2, 2.3, 3.4, 4.5, 5.6]
            >>> filter_qd = [True, True, True, False, False]
            >>> filter_dp = [True, False, True, False, False]
            >>> dp = [12, 23, 34, 45, 56]
            >>> qd = [12.3, 23.4, 34.5, 45.6, 56.7]
            >>> flg = [True, False, True, False, True]
            >>> ac = [(1, -1), (3, -1), (5, 6), (7, 8), (9, -1)]
            >>> xx = [(1.2, 2.3), (3.4, 4.5), (5.6, 6.7), (7.8, 8.9),
            ...       (9.0, 9.9)]
            >>> columns = [chrom, pos, ids, ref, alt, qual, filter_dp,
            ...            filter_qd, dp, qd, flg, ac, xx]
            >>> records = list(zip(*columns))
            >>> dtype = [('CHROM', 'S4'),
            ...          ('POS', 'u4'),
            ...          ('ID', 'S1'),
            ...          ('REF', 'S1'),
            ...          ('ALT', ('S1', 2)),
            ...          ('qual', 'f4'),
            ...          ('filter_dp', bool),
            ...          ('filter_qd', bool),
            ...          ('dp', int),
            ...          ('qd', float),
            ...          ('flg', bool),
            ...          ('ac', (int, 2)),
            ...          ('xx', (float, 2))]
            >>> vt = allel.VariantTable(records, dtype=dtype)

        Now write out to VCF and inspect the result::

            >>> rename = {'dp': 'DP', 'qd': 'QD', 'filter_qd': 'QD'}
            >>> fill = {'ALT': b'.', 'ac': -1}
            >>> number = {'ac': 'A'}
            >>> description = {'ac': 'Allele counts', 'filter_dp': 'Low depth'}
            >>> vt.to_vcf('example.vcf', rename=rename, fill=fill,
            ...           number=number, description=description)
            >>> print(open('example.vcf').read())  # doctest: +ELLIPSIS
            ##fileformat=VCFv4.1
            ##fileDate=...
            ##source=...
            ##INFO=<ID=DP,Number=1,Type=Integer,Description="">
            ##INFO=<ID=QD,Number=1,Type=Float,Description="">
            ##INFO=<ID=ac,Number=A,Type=Integer,Description="Allele counts">
            ##INFO=<ID=flg,Number=0,Type=Flag,Description="">
            ##INFO=<ID=xx,Number=2,Type=Float,Description="">
            ##FILTER=<ID=QD,Description="">
            ##FILTER=<ID=dp,Description="Low depth">
            #CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO
            chr1	2	a	A	T	1.2	QD;dp	DP=12;QD=12.3;ac=1;flg;xx=...
            chr1	6	b	C	G	2.3	QD	DP=23;QD=23.4;ac=3;xx=3.4,4.5
            chr2	3	c	T	A,C	3.4	QD;dp	DP=34;QD=34.5;ac=5,6;flg;x...
            chr2	8	d	G	C,A	4.5	PASS	DP=45;QD=45.6;ac=7,8;xx=7...
            chr3	1	e	N	X	5.6	PASS	DP=56;QD=56.7;ac=9;flg;xx=...

        """

        write_vcf(path, variants=self, rename=rename, number=number,
                  description=description, fill=fill,
                  write_header=write_header)


def sample_to_haplotype_selection(indices, ploidy):
    return [(i * ploidy) + n for i in indices for n in range(ploidy)]


class FeatureTable(RecArrayBase):
    """Table of genomic features (e.g., genes, exons, etc.).

    Parameters
    ----------
    data : array_like, structured, shape (n_variants,)
        Variant records.
    copy : bool, optional
        If True, make a copy of `data`.
    **kwargs : keyword arguments, optional
        Further keyword arguments are passed through to
        :func:`numpy.rec.array`.

    """

    def __init__(self, data, copy=False, **kwargs):
        super(FeatureTable, self).__init__(data, copy=copy, **kwargs)

    @property
    def n_features(self):
        """Number of features (length of first dimension)."""
        return self.shape[0]

    def to_mask(self, size, start_name='start', stop_name='end'):
        """Construct a mask array where elements are True if the fall within
        features in the table.

        Parameters
        ----------

        size : int
            Size of chromosome/contig.
        start_name : string, optional
            Name of column with start coordinates.
        stop_name : string, optional
            Name of column with stop coordinates.

        Returns
        -------

        mask : ndarray, bool

        """
        m = np.zeros(size, dtype=bool)
        for start, stop in self[[start_name, stop_name]]:
            m[start-1:stop] = True
        return m

    @staticmethod
    def from_gff3(path, attributes=None, region=None, score_fill=-1, phase_fill=-1,
                  attributes_fill=b'.', dtype=None):
        """Read a feature table from a GFF3 format file.

        Parameters
        ----------
        path : string
            File path.
        attributes : list of strings, optional
            List of columns to extract from the "attributes" field.
        region : string, optional
            Genome region to extract. If given, file must be position
            sorted, bgzipped and tabix indexed. Tabix must also be installed
            and on the system path.
        score_fill : object, optional
            Value to use where score field has a missing value.
        phase_fill : object, optional
            Value to use where phase field has a missing value.
        attributes_fill : object or list of objects, optional
            Value(s) to use where attribute field(s) have a missing value.
        dtype : numpy dtype, optional
            Manually specify a dtype.

        Returns
        -------
        ft : FeatureTable

        """

        # setup iterator
        recs = iter_gff3(path, attributes=attributes, region=region,
                         score_fill=score_fill, phase_fill=phase_fill,
                         attributes_fill=attributes_fill)

        # determine dtype from sample of initial records
        if dtype is None:
            names = 'seqid', 'source', 'type', 'start', 'end', 'score', \
                    'strand', 'phase'
            if attributes is not None:
                names += tuple(attributes)
            recs_sample = list(itertools.islice(recs, 1000))
            a = np.rec.array(recs_sample, names=names)
            dtype = a.dtype
            recs = itertools.chain(recs_sample, recs)

        a = np.fromiter(recs, dtype=dtype)
        ft = FeatureTable(a, copy=False)
        return ft
