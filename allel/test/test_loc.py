# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np
from allel.test.tools import eq, aeq
import allel.loc


def test_query_position():
    f = allel.loc.query_position
    pos = np.array([3, 6, 11])
    eq(0, f(pos, 3))
    eq(1, f(pos, 6))
    eq(2, f(pos, 11))
    eq(None, f(pos, 1))
    eq(None, f(pos, 7))
    eq(None, f(pos, 12))


# TODO test_query_positions():
# TODO test_query_interval():
# TODO test_query_intervals():