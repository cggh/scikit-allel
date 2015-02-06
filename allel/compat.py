# -*- coding: utf-8 -*-
# flake8: noqa
from __future__ import absolute_import, print_function, division


import sys
import itertools


PY2 = sys.version_info[0] == 2


if PY2:
    range = xrange
    map = itertools.imap
    string_types = basestring,
    text_type = unicode
    binary_type = str
    zip = itertools.izip
    reduce = reduce
else:
    range = range
    map = map
    string_types = str,
    text_type = str
    binary_type = bytes
    zip = zip
    import functools
    reduce = functools.reduce
