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
    zip = itertools.izip
else:
    range = range
    map = map
    string_types = str,
    zip = zip


def force_bytes(o):
    if PY2:
        return o
    elif isinstance(o, str):
        return o.encode('ascii')
    elif isinstance(o, list):
        return [force_bytes(x) for x in o]
    elif isinstance(o, tuple):
        return tuple(force_bytes(x) for x in o)
    else:
        return o
