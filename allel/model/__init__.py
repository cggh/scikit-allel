# -*- coding: utf-8 -*-
# flake8: noqa
from __future__ import absolute_import, print_function, division


from allel.model.ndarray import GenotypeArray, HaplotypeArray, \
    AlleleCountsArray, FeatureTable, VariantTable, SortedIndex, \
    SortedMultiIndex, UniqueIndex, create_allele_mapping, \
    locate_fixed_differences, locate_private_alleles

from allel.model.chunked import ChunkedArray, GenotypeChunkedArray, \
    AlleleCountsChunkedArray  # TODO remaining imports

try:
    # noinspection PyUnresolvedReferences
    import bcolz
except ImportError:
    pass
else:
    from allel.model.bcolz import GenotypeCArray, HaplotypeCArray, \
        AlleleCountsCArray, AlleleCountsCTable, VariantCTable, FeatureCTable
