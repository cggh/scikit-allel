# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import bisect


# third-party imports
import numpy as np


# internal imports
from .arrays import ArrayBase
from allel.util import check_ndim, asarray_ndim, check_dim0_aligned


class SortedIndex(ArrayBase):
    """Index of sorted values, e.g., positions from a single chromosome or
    contig.

    Parameters
    ----------
    data : array_like
        Values in ascending order.
    **kwargs : keyword arguments
        All keyword arguments are passed through to :func:`numpy.array`.

    Notes
    -----
    Values must be given in ascending order, although duplicate values
    may be present (i.e., values must be monotonically increasing).

    Examples
    --------

    >>> import allel
    >>> idx = allel.SortedIndex([2, 5, 14, 15, 42, 42, 77], dtype='i4')
    >>> idx.dtype
    dtype('int32')
    >>> idx.ndim
    1
    >>> idx.shape
    (7,)
    >>> idx.is_unique
    False

    """

    @classmethod
    def _check_values(cls, values):
        check_ndim(values, 1)
        # check sorted ascending
        if np.any(values[:-1] > values[1:]):
            raise ValueError('values must be monotonically increasing')

    def __init__(self, data, copy=False, **kwargs):
        super(SortedIndex, self).__init__(self, data, copy=copy, **kwargs)
        self._is_unique = None

    @property
    def is_unique(self):
        """True if no duplicate entries."""
        if self._unique is None:
            self._is_unique = ~np.any(self[:-1] == self[1:])
        return self._is_unique

    def locate_key(self, key):
        """Get index location for the requested key.

        Parameters
        ----------
        key : int
            Value to locate.

        Returns
        -------
        loc : int or slice
            Location of `key` (will be slice if there are duplicate entries).

        Examples
        --------

        >>> import allel
        >>> idx = allel.SortedIndex([3, 6, 6, 11])
        >>> idx.locate_key(3)
        0
        >>> idx.locate_key(11)
        3
        >>> idx.locate_key(6)
        slice(1, 3, None)
        >>> try:
        ...     idx.locate_key(2)
        ... except KeyError as e:
        ...     print(e)
        ...
        2

        """

        left = bisect.bisect_left(self, key)
        right = bisect.bisect_right(self, key)
        diff = right - left
        if diff == 0:
            raise KeyError(key)
        elif diff == 1:
            return left
        else:
            return slice(left, right)

    def locate_intersection(self, other):
        """Locate the intersection with another array.

        Parameters
        ----------
        other : array_like, int
            Array of values to intersect.

        Returns
        -------
        loc : ndarray, bool
            Boolean array with location of intersection.
        loc_other : ndarray, bool
            Boolean array with location in `other` of intersection.

        Examples
        --------

        >>> import allel
        >>> idx1 = allel.SortedIndex([3, 6, 11, 20, 35])
        >>> idx2 = allel.SortedIndex([4, 6, 20, 39])
        >>> loc1, loc2 = idx1.locate_intersection(idx2)
        >>> loc1
        array([False,  True, False,  True, False], dtype=bool)
        >>> loc2
        array([False,  True,  True, False], dtype=bool)
        >>> idx1[loc1]
        SortedIndex((2,), dtype=int64)
        [ 6 20]
        >>> idx2[loc2]
        SortedIndex((2,), dtype=int64)
        [ 6 20]

        """

        # check inputs
        other = SortedIndex(other, copy=False)

        # find intersection
        assume_unique = self.is_unique and other.is_unique
        loc = np.in1d(self, other, assume_unique=assume_unique)
        loc_other = np.in1d(other, self, assume_unique=assume_unique)

        return loc, loc_other

    def locate_keys(self, keys, strict=True):
        """Get index locations for the requested keys.

        Parameters
        ----------
        keys : array_like, int
            Array of keys to locate.
        strict : bool, optional
            If True, raise KeyError if any keys are not found in the index.

        Returns
        -------
        loc : ndarray, bool
            Boolean array with location of values.

        Examples
        --------

        >>> import allel
        >>> idx1 = allel.SortedIndex([3, 6, 11, 20, 35])
        >>> idx2 = allel.SortedIndex([4, 6, 20, 39])
        >>> loc = idx1.locate_keys(idx2, strict=False)
        >>> loc
        array([False,  True, False,  True, False], dtype=bool)
        >>> idx1[loc]
        SortedIndex((2,), dtype=int64)
        [ 6 20]

        """

        # check inputs
        keys = SortedIndex(keys, copy=False)

        # find intersection
        loc, found = self.locate_intersection(keys)

        if strict and np.any(~found):
            raise KeyError(keys[~found])

        return loc

    def intersect(self, other):
        """Intersect with `other` sorted index.

        Parameters
        ----------
        other : array_like, int
            Array of values to intersect with.

        Returns
        -------
        out : SortedIndex
            Values in common.

        Examples
        --------

        >>> import allel
        >>> idx1 = allel.SortedIndex([3, 6, 11, 20, 35])
        >>> idx2 = allel.SortedIndex([4, 6, 20, 39])
        >>> idx1.intersect(idx2)
        SortedIndex((2,), dtype=int64)
        [ 6 20]

        """

        loc = self.locate_keys(other, strict=False)
        return np.compress(loc, self)

    def locate_range(self, start=None, stop=None):
        """Locate slice of index containing all entries within `start` and
        `stop` values **inclusive**.

        Parameters
        ----------
        start : int, optional
            Start value.
        stop : int, optional
            Stop value.

        Returns
        -------
        loc : slice
            Slice object.

        Examples
        --------

        >>> import allel
        >>> idx = allel.SortedIndex([3, 6, 11, 20, 35])
        >>> loc = idx.locate_range(4, 32)
        >>> loc
        slice(1, 4, None)
        >>> idx[loc]
        SortedIndex((3,), dtype=int64)
        [ 6 11 20]

        """

        # locate start and stop indices
        if start is None:
            start_index = 0
        else:
            start_index = bisect.bisect_left(self, start)
        if stop is None:
            stop_index = len(self)
        else:
            stop_index = bisect.bisect_right(self, stop)

        if stop_index - start_index == 0:
            raise KeyError(start, stop)

        loc = slice(start_index, stop_index)
        return loc

    def intersect_range(self, start=None, stop=None):
        """Intersect with range defined by `start` and `stop` values
        **inclusive**.

        Parameters
        ----------
        start : int, optional
            Start value.
        stop : int, optional
            Stop value.

        Returns
        -------
        idx : SortedIndex

        Examples
        --------

        >>> import allel
        >>> idx = allel.SortedIndex([3, 6, 11, 20, 35])
        >>> idx.intersect_range(4, 32)
        SortedIndex((3,), dtype=int64)
        [ 6 11 20]

        """

        try:
            loc = self.locate_range(start=start, stop=stop)
        except KeyError:
            return self[0:0]
        else:
            return self[loc]

    def locate_intersection_ranges(self, starts, stops):
        """Locate the intersection with a set of ranges.

        Parameters
        ----------
        starts : array_like, int
            Range start values.
        stops : array_like, int
            Range stop values.

        Returns
        -------
        loc : ndarray, bool
            Boolean array with location of entries found.
        loc_ranges : ndarray, bool
            Boolean array with location of ranges containing one or more
            entries.

        Examples
        --------

        >>> import allel
        >>> import numpy as np
        >>> idx = allel.SortedIndex([3, 6, 11, 20, 35])
        >>> ranges = np.array([[0, 2], [6, 17], [12, 15], [31, 35],
        ...                    [100, 120]])
        >>> starts = ranges[:, 0]
        >>> stops = ranges[:, 1]
        >>> loc, loc_ranges = idx.locate_intersection_ranges(starts, stops)
        >>> loc
        array([False,  True,  True, False,  True], dtype=bool)
        >>> loc_ranges
        array([False,  True, False,  True, False], dtype=bool)
        >>> idx[loc]
        SortedIndex((3,), dtype=int64)
        [ 6 11 35]
        >>> ranges[loc_ranges]
        array([[ 6, 17],
               [31, 35]])

        """

        # check inputs
        starts = asarray_ndim(starts, 1)
        stops = asarray_ndim(stops, 1)
        check_dim0_aligned(starts, stops)

        # find indices of start and stop values in idx
        start_indices = np.searchsorted(self, starts)
        stop_indices = np.searchsorted(self, stops, side='right')

        # find intervals overlapping at least one value
        loc_ranges = start_indices < stop_indices

        # find values within at least one interval
        loc = np.zeros(self.shape, dtype=np.bool)
        for i, j in zip(start_indices[loc_ranges], stop_indices[loc_ranges]):
            loc[i:j] = True

        return loc, loc_ranges

    def locate_ranges(self, starts, stops, strict=True):
        """Locate items within the given ranges.

        Parameters
        ----------
        starts : array_like, int
            Range start values.
        stops : array_like, int
            Range stop values.
        strict : bool, optional
            If True, raise KeyError if any ranges contain no entries.

        Returns
        -------
        loc : ndarray, bool
            Boolean array with location of entries found.

        Examples
        --------

        >>> import allel
        >>> import numpy as np
        >>> idx = allel.SortedIndex([3, 6, 11, 20, 35])
        >>> ranges = np.array([[0, 2], [6, 17], [12, 15], [31, 35],
        ...                    [100, 120]])
        >>> starts = ranges[:, 0]
        >>> stops = ranges[:, 1]
        >>> loc = idx.locate_ranges(starts, stops, strict=False)
        >>> loc
        array([False,  True,  True, False,  True], dtype=bool)
        >>> idx[loc]
        SortedIndex((3,), dtype=int64)
        [ 6 11 35]

        """

        loc, found = self.locate_intersection_ranges(starts, stops)

        if strict and np.any(~found):
            raise KeyError(starts[~found], stops[~found])

        return loc

    def intersect_ranges(self, starts, stops):
        """Intersect with a set of ranges.

        Parameters
        ----------
        starts : array_like, int
            Range start values.
        stops : array_like, int
            Range stop values.

        Returns
        -------
        idx : SortedIndex

        Examples
        --------

        >>> import allel
        >>> import numpy as np
        >>> idx = allel.SortedIndex([3, 6, 11, 20, 35])
        >>> ranges = np.array([[0, 2], [6, 17], [12, 15], [31, 35],
        ...                    [100, 120]])
        >>> starts = ranges[:, 0]
        >>> stops = ranges[:, 1]
        >>> idx.intersect_ranges(starts, stops)
        SortedIndex((3,), dtype=int64)
        [ 6 11 35]

        """

        loc = self.locate_ranges(starts, stops, strict=False)
        return np.compress(loc, self)


class UniqueIndex(ArrayBase):
    """Array of unique values (e.g., variant or sample identifiers).

    Parameters
    ----------
    data : array_like
        Values.
    **kwargs : keyword arguments
        All keyword arguments are passed through to :func:`numpy.array`.

    Notes
    -----
    This class represents an arbitrary set of unique values, e.g., sample or
    variant identifiers.

    There is no need for values to be sorted. However, all values must be
    unique within the array, and must be hashable objects.

    Examples
    --------

    >>> import allel
    >>> idx = allel.UniqueIndex(['A', 'C', 'B', 'F'])
    >>> idx.dtype
    dtype('<U1')
    >>> idx.ndim
    1
    >>> idx.shape
    (4,)

    """

    @classmethod
    def _check_values(cls, obj):
        check_ndim(obj, 1)
        # check unique
        # noinspection PyTupleAssignmentBalance
        _, counts = np.unique(obj, return_counts=True)
        if np.any(counts > 1):
            raise ValueError('values are not unique')

    def __init__(self, data, copy=False, **kwargs):
        super(UniqueIndex, self).__init__(data, copy=copy, **kwargs)
        self.lookup = {v: i for i, v in enumerate(data)}

    def locate_key(self, key):
        """Get index location for the requested key.

        Parameters
        ----------
        key : object
            Key to locate.

        Returns
        -------
        loc : int
            Location of `key`.

        Examples
        --------

        >>> import allel
        >>> idx = allel.UniqueIndex(['A', 'C', 'B', 'F'])
        >>> idx.locate_key('A')
        0
        >>> idx.locate_key('B')
        2
        >>> try:
        ...     idx.locate_key('X')
        ... except KeyError as e:
        ...     print(e)
        ...
        'X'

        """

        return self.lookup[key]

    def locate_intersection(self, other):
        """Locate the intersection with another array.

        Parameters
        ----------
        other : array_like
            Array to intersect.

        Returns
        -------
        loc : ndarray, bool
            Boolean array with location of intersection.
        loc_other : ndarray, bool
            Boolean array with location in `other` of intersection.

        Examples
        --------

        >>> import allel
        >>> idx1 = allel.UniqueIndex(['A', 'C', 'B', 'F'])
        >>> idx2 = allel.UniqueIndex(['X', 'F', 'G', 'C', 'Z'])
        >>> loc1, loc2 = idx1.locate_intersection(idx2)
        >>> loc1
        array([False,  True, False,  True], dtype=bool)
        >>> loc2
        array([False,  True, False,  True, False], dtype=bool)
        >>> idx1[loc1]
        UniqueIndex((2,), dtype=<U1)
        ['C' 'F']
        >>> idx2[loc2]
        UniqueIndex((2,), dtype=<U1)
        ['F' 'C']

        """

        # check inputs
        other = UniqueIndex(other)

        # find intersection
        assume_unique = True
        loc = np.in1d(self, other, assume_unique=assume_unique)
        loc_other = np.in1d(other, self, assume_unique=assume_unique)

        return loc, loc_other

    def locate_keys(self, keys, strict=True):
        """Get index locations for the requested keys.

        Parameters
        ----------
        keys : array_like
            Array of keys to locate.
        strict : bool, optional
            If True, raise KeyError if any keys are not found in the index.

        Returns
        -------
        loc : ndarray, bool
            Boolean array with location of keys.

        Examples
        --------

        >>> import allel
        >>> idx = allel.UniqueIndex(['A', 'C', 'B', 'F'])
        >>> idx.locate_keys(['F', 'C'])
        array([False,  True, False,  True], dtype=bool)
        >>> idx.locate_keys(['X', 'F', 'G', 'C', 'Z'], strict=False)
        array([False,  True, False,  True], dtype=bool)

        """

        # check inputs
        keys = UniqueIndex(keys)

        # find intersection
        loc, found = self.locate_intersection(keys)

        if strict and np.any(~found):
            raise KeyError(keys[~found])

        return loc

    def intersect(self, other):
        """Intersect with `other`.

        Parameters
        ----------
        other : array_like
            Array to intersect.

        Returns
        -------
        out : UniqueIndex

        Examples
        --------

        >>> import allel
        >>> idx1 = allel.UniqueIndex(['A', 'C', 'B', 'F'])
        >>> idx2 = allel.UniqueIndex(['X', 'F', 'G', 'C', 'Z'])
        >>> idx1.intersect(idx2)
        UniqueIndex((2,), dtype=<U1)
        ['C' 'F']
        >>> idx2.intersect(idx1)
        UniqueIndex((2,), dtype=<U1)
        ['F' 'C']

        """

        loc = self.locate_keys(other, strict=False)
        return np.compress(loc, self)


class SortedMultiIndex(object):
    """Two-level index of sorted values, e.g., variant positions from two or
    more chromosomes/contigs.

    Parameters
    ----------
    l1 : array_like
        First level values in ascending order.
    l2 : array_like
        Second level values, in ascending order within each sub-level.
    copy : bool, optional
        If True, inputs will be copied into new arrays.

    Examples
    --------

    >>> import allel
    >>> chrom = ['chr1', 'chr1', 'chr2', 'chr2', 'chr2', 'chr3']
    >>> pos = [1, 4, 2, 5, 5, 3]
    >>> idx = allel.SortedMultiIndex(chrom, pos)
    >>> len(idx)
    6

    """

    def __init__(self, l1, l2, copy=False):
        l1 = SortedIndex(l1, copy=copy)
        l2 = np.array(l2, copy=copy)
        check_ndim(l2, 1)
        check_dim0_aligned(l1, l2)
        self.l1 = l1
        self.l2 = l2

    def __repr__(self):
        s = ('SortedMultiIndex(%s)\n' % len(self))
        return s

    def __str__(self):
        s = ('SortedMultiIndex(%s)\n' % len(self))
        return s

    def locate_key(self, k1, k2=None):
        """
        Get index location for the requested key.

        Parameters
        ----------
        k1 : object
            Level 1 key.
        k2 : object, optional
            Level 2 key.

        Returns
        -------
        loc : int or slice
            Location of requested key (will be slice if there are duplicate
            entries).

        Examples
        --------

        >>> import allel
        >>> chrom = ['chr1', 'chr1', 'chr2', 'chr2', 'chr2', 'chr3']
        >>> pos = [1, 4, 2, 5, 5, 3]
        >>> idx = allel.SortedMultiIndex(chrom, pos)
        >>> idx.locate_key('chr1')
        slice(0, 2, None)
        >>> idx.locate_key('chr1', 4)
        1
        >>> idx.locate_key('chr2', 5)
        slice(3, 5, None)
        >>> try:
        ...     idx.locate_key('chr3', 4)
        ... except KeyError as e:
        ...     print(e)
        ...
        ('chr3', 4)

        """

        loc1 = self.l1.locate_key(k1)
        if k2 is None:
            return loc1
        if isinstance(loc1, slice):
            offset = loc1.start
            try:
                loc2 = SortedIndex(self.l2[loc1], copy=False).locate_key(k2)
            except KeyError:
                # reraise with more information
                raise KeyError(k1, k2)
            else:
                if isinstance(loc2, slice):
                    loc = slice(offset + loc2.start, offset + loc2.stop)
                else:
                    # assume singleton
                    loc = offset + loc2
        else:
            # singleton match in l1
            v = self.l2[loc1]
            if v == k2:
                loc = loc1
            else:
                raise KeyError(k1, k2)
        return loc

    def locate_range(self, key, start=None, stop=None):
        """Locate slice of index containing all entries within the range
        `key`:`start`-`stop` **inclusive**.

        Parameters
        ----------
        key : object
            Level 1 key value.
        start : object, optional
            Level 2 start value.
        stop : object, optional
            Level 2 stop value.

        Returns
        -------
        loc : slice
            Slice object.

        Examples
        --------

        >>> import allel
        >>> chrom = ['chr1', 'chr1', 'chr2', 'chr2', 'chr2', 'chr3']
        >>> pos = [1, 4, 2, 5, 5, 3]
        >>> idx = allel.SortedMultiIndex(chrom, pos)
        >>> idx.locate_range('chr1')
        slice(0, 2, None)
        >>> idx.locate_range('chr1', 1, 4)
        slice(0, 2, None)
        >>> idx.locate_range('chr2', 3, 7)
        slice(3, 5, None)
        >>> try:
        ...     idx.locate_range('chr3', 4, 9)
        ... except KeyError as e:
        ...     print(e)
        ('chr3', 4, 9)

        """

        loc1 = self.l1.locate_key(key)
        if start is None and stop is None:
            loc = loc1
        elif isinstance(loc1, slice):
            offset = loc1.start
            idx = SortedIndex(self.l2[loc1], copy=False)
            try:
                loc2 = idx.locate_range(start, stop)
            except KeyError:
                raise KeyError(key, start, stop)
            else:
                loc = slice(offset + loc2.start, offset + loc2.stop)
        else:
            # singleton match in l1
            v = self.l2[loc1]
            if start <= v <= stop:
                loc = loc1
            else:
                raise KeyError(key, start, stop)
        # ensure slice is always returned
        if not isinstance(loc, slice):
            loc = slice(loc, loc + 1)
        return loc

    def __len__(self):
        return len(self.l1)
