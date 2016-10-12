# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


# third-party imports
import numpy as np


# internal imports
from allel.io import recarray_from_hdf5_group, recarray_to_hdf5_group
from allel.util import check_ndim, asarray_ndim
from allel.compat import PY2


class ArrayWrapper(object):
    """Abstract base class that delegates to a wrapped array-like object."""

    @classmethod
    def check_values(cls, data):
        if not hasattr(data, 'shape') or not hasattr(data, 'dtype'):
            raise TypeError('values must be array-like')

    def __init__(self, data):
        self.check_values(data)
        self._values = data

    @property
    def values(self):
        """The underlying array of values."""
        return self._values

    @property
    def caption(self):
        return '<%s shape=%s dtype=%s>' % (type(self).__name__, self.shape, self.dtype)

    def __repr__(self):
        return self.caption + '\n' + str(self)

    def __str__(self):
        return str(self.values)

    def __getattr__(self, item):
        if item in {'__array_struct__', '__array_interface__'}:
            # don't pass these through because we want to use __array__ to control numpy
            # behaviour
            raise AttributeError
        return getattr(self.values, item)

    def __getitem__(self, item):
        return self.values[item]

    def __setitem__(self, item, value):
        self.values[item] = value

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)

    def __array__(self, *args):
        v = self.values[:]
        print('__array__', type(v), repr(v))
        a = np.asarray(v)
        if args:
            a = a.astype(args[0])
        return a


def arr1d_to_html(indices, items, caption):
    # N.B., table captions don't render in jupyter notebooks on GitHub,
    # so put caption outside table element

    # sanitize caption
    caption = caption.replace('<', '&lt;')
    html = caption

    # build table
    html += '<table>'
    html += '<tr>'
    for i in indices:
        html += '<th style="text-align: center">%s</th>' % i
    html += '</tr>'
    html += '<tr>'
    for item in items:
        html += '<td style="text-align: center">%s</td>' % item
    html += '</tr>'
    html += '</table>'

    return html


def arr2d_to_html(row_indices, col_indices, items, caption):
    # N.B., table captions don't render in jupyter notebooks on GitHub,
    # so put caption outside table element

    # sanitize caption
    caption = caption.replace('<', '&lt;')
    html = caption

    # build table
    html += '<table>'
    html += '<tr><th></th>'
    for i in col_indices:
        html += '<th style="text-align: center">%s</th>' % i
    html += '</tr>'
    for row_index, row in zip(row_indices, items):
        if row_index == ' ... ':
            html += '<tr><th style="text-align: center">...</th>' \
                    '<td style="text-align: center" colspan=%s>...</td></tr>' % \
                    (len(col_indices) + 1)
        else:
            html += '<tr><th style="text-align: center">%s</th>' % row_index
            for item in row:
                html += '<td style="text-align: center">%s</td>' % item
            html += '</tr>'
    html += '</table>'

    return html


def recarr_to_html(names, indices, items, caption):
    # N.B., table captions don't render in jupyter notebooks on GitHub,
    # so put caption outside table element

    # sanitize caption
    caption = caption.replace('<', '&lt;')
    html = caption

    # build table
    html += '<table>'
    html += '<tr><th></th>'
    for n in names:
        html += '<th style="text-align: center">%s</th>' % n
    html += '</tr>'
    for row_index, row in zip(indices, items):
        if row_index == ' ... ':
            html += '<tr><th style="text-align: center">...</th>' \
                    '<td style="text-align: center" colspan=%s>...</td></tr>' % \
                    (len(names) + 1)
        else:
            html += '<tr><th style="text-align: center">%s</th>' % row_index
            for item in row:
                html += '<td style="text-align: center">%s</td>' % item
            html += '</tr>'
    html += '</table>'

    return html


class DisplayableArray(ArrayWrapper):

    def __str__(self):
        return self.to_str()

    def _repr_html_(self):
        return self.to_html()


# noinspection PyAbstractClass
class DisplayAs1D(DisplayableArray):

    def get_display_items(self, threshold=10, edgeitems=5):

        # ensure threshold
        if threshold is None:
            threshold = self.shape[0]

        # ensure sensible edgeitems
        edgeitems = min(edgeitems, threshold // 2)

        # determine indices of items to show
        if self.shape[0] > threshold:
            indices = (
                list(range(edgeitems)) + [' ... '] +
                list(range(self.shape[0] - edgeitems, self.shape[0], 1))
            )
            head = self[:edgeitems].str_items()
            tail = self[-edgeitems:].str_items()
            items = head + [' ... '] + tail
        else:
            indices = list(range(self.shape[0]))
            items = self[:].str_items()

        return indices, items

    def to_str(self, threshold=10, edgeitems=5):
        _, items = self.get_display_items(threshold, edgeitems)
        s = ' '.join(items)
        return s

    def to_html(self, threshold=10, edgeitems=5, caption=None):
        indices, items = self.get_display_items(threshold, edgeitems)
        if caption is None:
            caption = self.caption
        return arr1d_to_html(indices, items, caption)

    def display(self, threshold=10, edgeitems=5, caption=None):
        html = self.to_html(threshold, edgeitems, caption)
        from IPython.display import display_html
        display_html(html, raw=True)

    def displayall(self, caption=None):
        self.display(threshold=None, caption=caption)


# noinspection PyAbstractClass
class DisplayAs2D(DisplayableArray):

    def get_display_items(self, row_threshold, col_threshold, row_edgeitems,
                          col_edgeitems):

        # ensure threshold
        if row_threshold is None:
            row_threshold = self.shape[0]
        if col_threshold is None:
            col_threshold = self.shape[1]

        # ensure sensible edgeitems
        row_edgeitems = min(row_edgeitems, row_threshold // 2)
        col_edgeitems = min(col_edgeitems, col_threshold // 2)

        # determine row indices of items to show
        if self.shape[0] > row_threshold:
            row_indices = (
                list(range(row_edgeitems)) + [' ... '] +
                list(range(self.shape[0] - row_edgeitems, self.shape[0], 1))
            )
            head = self[:row_edgeitems].str_items()
            tail = self[-row_edgeitems:].str_items()
            items = head + [' ... '] + tail
        else:
            row_indices = list(range(self.shape[0]))
            items = self[:].str_items()

        # determine col indices of items to show
        if self.shape[1] > col_threshold:
            col_indices = (
                list(range(col_edgeitems)) + [' ... '] +
                list(range(self.shape[1] - col_edgeitems, self.shape[1], 1))
            )
            items = [
                row if row == ' ... ' else
                (row[:col_edgeitems] + [' ... '] + row[-col_edgeitems:])
                for row in items
            ]
        else:
            col_indices = list(range(self.shape[1]))
            # items unchanged

        return row_indices, col_indices, items

    def to_str(self, row_threshold=6, col_threshold=10, row_edgeitems=3, col_edgeitems=5):
        _, _, items = self.get_display_items(row_threshold, col_threshold, row_edgeitems,
                                             col_edgeitems)
        s = ''
        for row in items:
            if row == ' ... ':
                s += row + '\n'
            else:
                s += ' '.join(row) + '\n'
        return s

    def to_html(self, row_threshold=6, col_threshold=10, row_edgeitems=3, col_edgeitems=5,
                caption=None):
        row_indices, col_indices, items = self.get_display_items(
            row_threshold, col_threshold, row_edgeitems, col_edgeitems
        )
        if caption is None:
            caption = self.caption
        return arr2d_to_html(row_indices, col_indices, items, caption)

    def display(self, row_threshold=6, col_threshold=10, row_edgeitems=3,
                col_edgeitems=5, caption=None):
        html = self.to_html(row_threshold, col_threshold, row_edgeitems, col_edgeitems,
                            caption)
        from IPython.display import display_html
        display_html(html, raw=True)

    def displayall(self, caption=None):
        self.display(row_threshold=None, col_threshold=None, caption=caption)


class DisplayAsTable(DisplayableArray):

    @property
    def names(self):
        """Column names."""
        return self.dtype.names

    def str_items(self):
        tmp = self[:]
        items = [[repr(x) for x in row] for row in tmp]
        return items

    def get_display_items(self, threshold=6, edgeitems=3):

        # ensure threshold
        if threshold is None:
            threshold = self.shape[0]

        # ensure sensible edgeitems
        edgeitems = min(edgeitems, threshold // 2)

        # determine indices of items to show
        if self.shape[0] > threshold:
            indices = (
                list(range(edgeitems)) + [' ... '] +
                list(range(self.shape[0] - edgeitems, self.shape[0], 1))
            )
            head = self[:edgeitems].str_items()
            tail = self[-edgeitems:].str_items()
            items = head + [' ... '] + tail
        else:
            indices = list(range(self.shape[0]))
            items = self[:].str_items()

        return indices, items

    def to_str(self, threshold=6, edgeitems=3):
        _, items = self.get_display_items(threshold, edgeitems)
        s = ' '.join(items)
        return s

    def to_html(self, threshold=6, edgeitems=3, caption=None):
        indices, items = self.get_display_items(threshold, edgeitems)
        if caption is None:
            caption = self.caption
        return recarr_to_html(self.names, indices, items, caption)

    def display(self, threshold=6, edgeitems=3, caption=None):
        html = self.to_html(threshold, edgeitems, caption)
        from IPython.display import display_html
        display_html(html, raw=True)

    def displayall(self, caption=None):
        self.display(threshold=None, caption=caption)

    def __str__(self):
        # stick with default string output of values
        return str(self.values)


class NumpyArrayWrapper(ArrayWrapper):
    """Abstract base class that wraps a NumPy array."""

    def __init__(self, data, copy=False, **kwargs):
        values = np.array(data, copy=copy, **kwargs)
        super(NumpyArrayWrapper, self).__init__(values)

    def __eq__(self, other):
        return self.values == other

    def __ne__(self, other):
        return self.values != other

    def __lt__(self, other):
        return self.values < other

    def __gt__(self, other):
        return self.values > other

    def __le__(self, other):
        return self.values <= other

    def __ge__(self, other):
        return self.values >= other

    def __abs__(self):
        return abs(self.values)

    def __add__(self, other):
        return self.values + other

    def __and__(self, other):
        return self.values & other

    def __div__(self, other):
        return self.values.__div__(other)

    def __floordiv__(self, other):
        return self.values // other

    def __inv__(self):
        return ~self.values

    def __invert__(self):
        return ~self.values

    def __lshift__(self, other):
        return self.values << other

    def __mod__(self, other):
        return self.values % other

    def __mul__(self, other):
        return self.values * other

    def __neg__(self):
        return -self.values

    def __or__(self, other):
        return self.values | other

    def __pos__(self):
        return +self.values

    def __pow__(self, other):
        return self.values ** other

    def __rshift__(self, other):
        return self.values >> other

    def __sub__(self, other):
        return self.values - other

    def __truediv__(self, other):
        return self.values.__truediv__(other)

    def __xor__(self, other):
        return self.values ^ other

    def copy(self, *args, **kwargs):
        data = self.values.copy(*args, **kwargs)
        # can always wrap this as sub-class type
        return type(self)(data)


class NumpyRecArrayWrapper(NumpyArrayWrapper, DisplayAsTable):

    @classmethod
    def _check_values(cls, data):
        check_ndim(data, 1)
        if not data.dtype.names:
            raise ValueError('expected recarray')

    # noinspection PyMissingConstructor
    def __init__(self, data, copy=False, **kwargs):
        values = np.rec.array(data, copy=copy, **kwargs)
        self._check_values(values)
        self._values = values

    def __getitem__(self, item):
        s = self.values[item]
        if isinstance(item, (slice, list, np.ndarray, type(Ellipsis))):
            return type(self)(s)
        return s

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

    def compress(self, condition, axis=0):
        out = self.values.compress(condition, axis=axis)
        if axis == 0:
            out = type(self)(out)
        return out

    def take(self, indices, axis=0):
        out = self.values.take(indices, axis=axis)
        if axis == 0:
            out = type(self)(out)
        return out

    def concatenate(self, *others, **kwargs):
        """Concatenate arrays."""
        out = super(NumpyRecArrayWrapper, self).concatenate(*others, **kwargs)
        axis = kwargs.get('axis', 0)
        if axis == 0:
            out = type(self)(out.view(self.dtype))
        return out
