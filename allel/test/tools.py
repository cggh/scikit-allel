# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np


def assert_array_equal(expect, actual):
    expect = np.asarray(expect)
    actual = np.asarray(actual)
    assert np.array_equal(expect, actual), \
        '\nExpect:\n%r\nActual:\n%r\n' % (expect, actual)


def assert_array_close(expect, actual):
    expect = np.asarray(expect)
    actual = np.asarray(actual)
    assert np.allclose(expect, actual), \
        '\nExpect:\n%r\nActual:\n%r\n' % (expect, actual)
