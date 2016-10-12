# -*- coding: utf-8 -*-
"""This module provides alternative implementations of array
classes defined in the :mod:`allel.model.ndarray` module, using
`dask.array <http://dask.pydata.org/en/latest/array.html>`_ as the
computational engine.

Dask uses blocked algorithms and task scheduling to break up work into
smaller pieces, allowing computation over large datasets. It also uses
lazy evaluation, meaning that multiple operations can be chained together
into a task graph, reducing total memory requirements for intermediate
results, and only the tasks required to generate the requested
part of the final data set will be executed.

This module is experimental, if you find a bug please `raise an issue on GitHub
<https://github.com/cggh/scikit-allel/issues/new>`_.

This module requires Dask >= 0.7.6.

"""
from __future__ import absolute_import, print_function, division


import numpy as np
import dask.array as da


from allel.compat import copy_method_doc
from .ndarray import GenotypeArray, HaplotypeArray, AlleleCountsArray


__all__ = ['GenotypeDaskArray', 'HaplotypeDaskArray', 'AlleleCountsDaskArray']



def ensure_dask_array(data, chunks=None):
    if isinstance(data, da.Array):
        if chunks:
            data = data.rechunk(chunks)
        return data
    else:
        data = ensure_array_like(data)
        chunks = get_chunks(data, chunks)
        return da.from_array(data, chunks=chunks)







# noinspection PyAbstractClass
class HaplotypeDaskArray(DaskArrayAug):
    """Dask haplotype array.

    To instantiate from an existing array-like object,
    use :func:`HaplotypeDaskArray.from_array`.

    """


# noinspection PyAbstractClass
class AlleleCountsDaskArray(DaskArrayAug):
    """Dask allele counts array.

    To instantiate from an existing array-like object,
    use :func:`AlleleCountsDaskArray.from_array`.

    """

    @staticmethod
    def check_input_data(x):
        if len(x.shape) != 2:
            raise ValueError('expected 2 dimensions')
        # don't check dtype now as it forces compute()

    def __getitem__(self, *args):
        out = super(AlleleCountsDaskArray, self).__getitem__(*args)
        if hasattr(out, 'shape') and len(self.shape) == len(out.shape) \
                and self.shape[1] == out.shape[1]:
            # dimensionality and allele indices preserved
            out = view_subclass(out, AlleleCountsDaskArray)
        return out

    def compute(self, **kwargs):
        a = super(AlleleCountsDaskArray, self).compute(**kwargs)
        h = AlleleCountsArray(a)
        return h

    def _repr_html_(self):
        return self[:6].compute().to_html_str(caption=repr(self))

    @property
    def n_variants(self):
        return self.shape[0]

    @property
    def n_alleles(self):
        return self.shape[1]

    def _method(self, method_name, chunks=None, drop_axis=None, **kwargs):
        if chunks is None:
            # no shape change
            chunks = self.chunks

        def f(block):
            ac = AlleleCountsArray(block)
            method = getattr(ac, method_name)
            return method(**kwargs)
        out = self.map_blocks(f, chunks=chunks, drop_axis=drop_axis)

        return out

    def _method_drop_axis1(self, method_name, **kwargs):
        chunks = self.chunks[:1]
        return self._method(method_name, chunks=chunks, drop_axis=1, **kwargs)

    def to_frequencies(self, fill=np.nan):
        return self._method('to_frequencies', chunks=self.chunks, fill=fill)

    def allelism(self):
        return self._method_drop_axis1('allelism')

    def max_allele(self):
        return self._method_drop_axis1('max_allele')

    def is_variant(self):
        return self._method_drop_axis1('is_variant')

    def is_non_variant(self):
        return self._method_drop_axis1('is_non_variant')

    def is_segregating(self):
        return self._method_drop_axis1('is_segregating')

    def is_non_segregating(self, allele=None):
        return self._method_drop_axis1('is_non_segregating', allele=allele)

    def is_singleton(self, allele=1):
        return self._method_drop_axis1('is_singleton', allele=allele)

    def is_doubleton(self, allele=1):
        return self._method_drop_axis1('is_doubleton', allele=allele)

    def is_biallelic(self):
        return self._method_drop_axis1('is_biallelic')

    def is_biallelic_01(self, min_mac=None):
        return self._method_drop_axis1('is_biallelic_01', min_mac=min_mac)

    def _count(self, method_name, **kwargs):
        method = getattr(self, method_name)
        # result is scalar, might as well compute now (also helps tests)
        return method(**kwargs).sum().compute()[()]

    def count_variant(self):
        return self._count('is_variant')

    def count_non_variant(self):
        return self._count('is_non_variant')

    def count_segregating(self):
        return self._count('is_segregating')

    def count_non_segregating(self, allele=None):
        return self._count('is_non_segregating', allele=allele)

    def count_singleton(self, allele=1):
        return self._count('is_singleton', allele=allele)

    def count_doubleton(self, allele=1):
        return self._count('is_doubleton', allele=allele)

    def map_alleles(self, mapping):

        def f(block, bmapping):
            ac = AlleleCountsArray(block)
            return ac.map_alleles(bmapping)

        # obtain dask array
        mapping = da.from_array(mapping, chunks=(self.chunks[0], None))

        # map blocks
        out = da.map_blocks(f, self, mapping, chunks=self.chunks)
        return view_subclass(out, AlleleCountsDaskArray)

    def concatenate(self, others, axis=0):
        pass


copy_method_doc(AlleleCountsDaskArray.allelism,
                AlleleCountsArray.allelism)
copy_method_doc(AlleleCountsDaskArray.max_allele,
                AlleleCountsArray.max_allele)
copy_method_doc(AlleleCountsDaskArray.map_alleles,
                AlleleCountsArray.map_alleles)
copy_method_doc(AlleleCountsDaskArray.to_frequencies,
                AlleleCountsArray.to_frequencies)
copy_method_doc(AlleleCountsDaskArray.is_variant,
                AlleleCountsArray.is_variant)
copy_method_doc(AlleleCountsDaskArray.is_non_variant,
                AlleleCountsArray.is_non_variant)
copy_method_doc(AlleleCountsDaskArray.is_segregating,
                AlleleCountsArray.is_segregating)
copy_method_doc(AlleleCountsDaskArray.is_non_segregating,
                AlleleCountsArray.is_non_segregating)
copy_method_doc(AlleleCountsDaskArray.is_singleton,
                AlleleCountsArray.is_singleton)
copy_method_doc(AlleleCountsDaskArray.is_doubleton,
                AlleleCountsArray.is_doubleton)
copy_method_doc(AlleleCountsDaskArray.is_biallelic,
                AlleleCountsArray.is_biallelic)
copy_method_doc(AlleleCountsDaskArray.is_biallelic_01,
                AlleleCountsArray.is_biallelic_01)
