# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np
from nose.tools import assert_true


def assert_array_equal(expect, actual):
    expect = np.asarray(expect[:])
    actual = np.asarray(actual[:])
    assert_true(
        np.array_equal(expect, actual),
        '\nExpect:\n%r\nActual:\n%r\n' % (expect, actual)
    )


def assert_array_close(expect, actual):
    expect = np.asarray(expect[:])
    actual = np.asarray(actual[:])
    assert_true(
        np.allclose(expect, actual),
        '\nExpect:\n%r\nActual:\n%r\n' % (expect, actual)
    )


def assert_array_nanclose(expect, actual):
    expect = np.asarray(expect[:])
    actual = np.asarray(actual[:])
    ein = np.isnan(expect)
    ain = np.isnan(actual)
    assert_true(
        np.array_equal(ein, ain),
        '\nExpect isnan:\n%r\nActual isnan:\n%r\n' % (ein, ain)
    )
    assert_true(
        np.allclose(expect[~ein], actual[~ain]),
        '\nExpect:\n%r\nActual:\n%r\n' % (expect, actual)
    )
