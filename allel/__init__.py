# -*- coding: utf-8 -*-
# flake8: noqa
from __future__ import absolute_import, print_function, division


__version__ = '0.4.0.dev0'


from allel.constants import *
from allel.model import GenotypeArray, HaplotypeArray, PositionIndex, \
    LabelIndex
from allel.util import windowed_count, windowed_density
