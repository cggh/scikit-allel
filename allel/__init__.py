# -*- coding: utf-8 -*-
# flake8: noqa


from allel import model
from allel.model.ndarray import GenotypeArray, HaplotypeArray, \
    AlleleCountsArray, VariantTable, FeatureTable, SortedIndex, \
    SortedMultiIndex, UniqueIndex
from allel.model.chunked import GenotypeChunkedArray, HaplotypeChunkedArray,\
    AlleleCountsChunkedArray, VariantChunkedTable, FeatureChunkedTable, \
    AlleleCountsChunkedTable
from allel.model.dask import GenotypeDaskArray, HaplotypeDaskArray, \
    AlleleCountsDaskArray
from allel.model.bcolz import GenotypeCArray, HaplotypeCArray, \
    AlleleCountsCArray, VariantCTable, FeatureCTable, AlleleCountsCTable
from allel import stats
from allel import plot
from allel import io
from allel import chunked
from allel import constants
from allel import util

__version__ = '0.20.0.feature_dask'
