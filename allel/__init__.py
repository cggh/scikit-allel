# -*- coding: utf-8 -*-
# flake8: noqa


__version__ = '0.15.0.dev4'


import allel.model as model
from allel.model import GenotypeArray, HaplotypeArray, AlleleCountsArray, \
    FeatureTable, VariantTable, SortedIndex, SortedMultiIndex, UniqueIndex

try:
    # noinspection PyUnresolvedReferences
    import bcolz
except ImportError:
    pass
else:
    import allel.bcolz as bcolz
    from allel.bcolz import GenotypeCArray, HaplotypeCArray, \
        AlleleCountsCArray, AlleleCountsCTable, VariantCTable, FeatureCTable

import allel.stats as stats
import allel.plot as plot
import allel.io as io
import allel.constants as constants
