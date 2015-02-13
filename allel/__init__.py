# -*- coding: utf-8 -*-
# flake8: noqa
from __future__ import absolute_import, print_function, division


__version__ = '0.4.0'


from allel.constants import *
from allel.model import GenotypeArray, HaplotypeArray, PositionIndex, \
    LabelIndex
from allel.stats import windowed_statistic, windowed_nnz, \
    windowed_mean_per_base, windowed_nnz_per_base, windowed_nucleotide_diversity
