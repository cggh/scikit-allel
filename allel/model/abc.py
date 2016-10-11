# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


# third-party imports
import numpy as np


class Wrapper(object):
    """Abstract base class that delegates everything to a wrapped object."""

    def __init__(self, values):
        self._values = values

    @property
    def values(self):
        """The underlying array of values."""
        return self._values

    def __getattr__(self, item):
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
        a = np.asarray(self.values)
        if args:
            a = a.astype(args[0])
        return a

    def __str__(self):
        return str(self.values)

    def __repr__(self):
        return repr(self.values)

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
