# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


from nose.tools import eq_ as eq  # noqa
import numpy as np


def aeq(expect, actual):
    assert np.array_equal(expect, actual), \
        '\nExpect:\n%r\nActual:\n%r\n' % (expect, actual)
