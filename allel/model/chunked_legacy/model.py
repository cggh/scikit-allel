# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import operator
from collections import namedtuple


import numpy as np


from allel.compat import string_types, copy_method_doc, integer_types
from allel.model.ndarray import GenotypeArray, HaplotypeArray, \
    AlleleCountsArray, recarray_to_html_str, recarray_display
from .backend_base import Backend, check_equal_length, get_column_names



class HaplotypeChunkedArray(ChunkedArray):
    """TODO

    """

    def to_genotypes(self, ploidy=2, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        # check ploidy is compatible
        if (self.n_haplotypes % ploidy) > 0:
            raise ValueError('incompatible ploidy')

        # build output
        def mapper(block):
            return block.to_genotypes(ploidy)

        out = backend.map_blocks(self, mapper, **kwargs)
        return GenotypeChunkedArray(out)

    def is_called(self, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))
        return backend.op_scalar(self, operator.ge, 0, **kwargs)

    def is_missing(self, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))
        return backend.op_scalar(self, operator.lt, 0, **kwargs)

    def is_ref(self, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))
        return backend.op_scalar(self, operator.eq, 0, **kwargs)

    def is_alt(self, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))
        return backend.op_scalar(self, operator.gt, 0, **kwargs)

    def is_call(self, allele, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))
        return backend.op_scalar(self, operator.eq, allele, **kwargs)

    def count_called(self, axis=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_called()

        out = backend.sum(self, axis=axis, mapper=mapper, **kwargs)
        return out

    def count_missing(self, axis=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_missing()

        out = backend.sum(self, axis=axis, mapper=mapper, **kwargs)
        return out

    def count_ref(self, axis=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_ref()

        out = backend.sum(self, axis=axis, mapper=mapper, **kwargs)
        return out

    def count_alt(self, axis=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_alt()

        out = backend.sum(self, axis=axis, mapper=mapper, **kwargs)
        return out

    def count_call(self, allele, axis=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_call(allele)

        out = backend.sum(self, axis=axis, mapper=mapper, **kwargs)
        return out

    def count_alleles(self, max_allele=None, subpop=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        # if max_allele not specified, count all alleles
        if max_allele is None:
            max_allele = self.max()

        def mapper(block):
            return block.count_alleles(max_allele=max_allele, subpop=subpop)

        out = backend.map_blocks(self, mapper, **kwargs)
        return AlleleCountsChunkedArray(out)

    def count_alleles_subpops(self, subpops, max_allele=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        if max_allele is None:
            max_allele = self.max()

        def mapper(block):
            return block.count_alleles_subpops(subpops, max_allele=max_allele)

        out = backend.dict_map_blocks(self, mapper, **kwargs)
        for k, v in out.items():
            out[k] = AlleleCountsChunkedArray(v)
        return out

    def map_alleles(self, mapping, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        # check inputs
        check_equal_length(self, mapping)

        # define mapping function
        def mapper(block, bmapping):
            return block.map_alleles(bmapping, copy=False)

        # execute map
        domain = (self, mapping)
        kwargs.setdefault('dtype', self.dtype)  # TODO needed?
        out = backend.map_blocks(domain, mapper, **kwargs)
        return HaplotypeChunkedArray(out)


# copy docstrings
copy_method_doc(HaplotypeChunkedArray.to_genotypes,
                HaplotypeArray.to_genotypes)
copy_method_doc(HaplotypeChunkedArray.count_alleles,
                HaplotypeArray.count_alleles)
copy_method_doc(HaplotypeChunkedArray.count_alleles_subpops,
                HaplotypeArray.count_alleles_subpops)
copy_method_doc(HaplotypeChunkedArray.map_alleles, HaplotypeArray.map_alleles)


class AlleleCountsChunkedArray(ChunkedArray):

    def __init__(self, data):
        self.check_input_data(data)
        super(AlleleCountsChunkedArray, self).__init__(data)

    @staticmethod
    def check_input_data(data):
        check_array_like(data, 2)
        if data.dtype.kind not in 'ui':
            raise TypeError('integer dtype required')

    def __getitem__(self, *args):
        out = self.data.__getitem__(*args)
        if is_array_like(out) and len(self.shape) == len(out.shape) and \
                out.shape[1] == self.shape[1]:
            out = AlleleCountsArray(out)
        return out

    def _repr_html_(self):
        return self[:6].to_html_str(caption=repr(self))

    @property
    def n_variants(self):
        """Number of variants (length of first array dimension)."""
        return self.shape[0]

    @property
    def n_alleles(self):
        """Number of alleles (length of second array dimension)."""
        return self.shape[1]

    def to_frequencies(self, fill=np.nan, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.to_frequencies(fill=fill)

        out = backend.map_blocks(self, mapper, **kwargs)
        return ChunkedArray(out)

    def allelism(self, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.allelism()

        out = backend.map_blocks(self, mapper, **kwargs)
        return ChunkedArray(out)

    def max_allele(self, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.max_allele()

        out = backend.map_blocks(self, mapper, **kwargs)
        return ChunkedArray(out)

    def is_variant(self, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_variant()

        out = backend.map_blocks(self, mapper, **kwargs)
        return ChunkedArray(out)

    def is_non_variant(self, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_non_variant()

        out = backend.map_blocks(self, mapper, **kwargs)
        return ChunkedArray(out)

    def is_segregating(self, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_segregating()

        out = backend.map_blocks(self, mapper, **kwargs)
        return ChunkedArray(out)

    def is_non_segregating(self, allele=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_non_segregating(allele=allele)

        out = backend.map_blocks(self, mapper, **kwargs)
        return ChunkedArray(out)

    def is_singleton(self, allele=1, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_singleton(allele=allele)

        out = backend.map_blocks(self, mapper, **kwargs)
        return ChunkedArray(out)

    def is_doubleton(self, allele=1, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_doubleton(allele=allele)

        out = backend.map_blocks(self, mapper, **kwargs)
        return ChunkedArray(out)

    def count_variant(self, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_variant()

        out = backend.sum(self, mapper=mapper, **kwargs)
        return out

    def count_non_variant(self, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_non_variant()

        out = backend.sum(self, mapper=mapper, **kwargs)
        return out

    def count_segregating(self, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_segregating()

        out = backend.sum(self, mapper=mapper, **kwargs)
        return out

    def count_non_segregating(self, allele=None, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_non_segregating(allele=allele)

        out = backend.sum(self, mapper=mapper, **kwargs)
        return out

    def count_singleton(self, allele=1, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_singleton(allele=allele)

        out = backend.sum(self, mapper=mapper, **kwargs)
        return out

    def count_doubleton(self, allele=1, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        def mapper(block):
            return block.is_doubleton(allele=allele)

        out = backend.sum(self, mapper=mapper, **kwargs)
        return out

    def map_alleles(self, mapping, **kwargs):
        backend = get_backend(kwargs.pop('backend', None))

        # check inputs
        check_equal_length(self, mapping)

        # define mapping function
        def mapper(block, bmapping):
            return block.map_alleles(bmapping)

        # execute map
        domain = (self, mapping)
        kwargs.setdefault('dtype', self.dtype)  # TODO needed?
        out = backend.map_blocks(domain, mapper, **kwargs)
        return AlleleCountsChunkedArray(out)


copy_method_doc(AlleleCountsChunkedArray.allelism, AlleleCountsArray.allelism)
copy_method_doc(AlleleCountsChunkedArray.max_allele,
                AlleleCountsArray.max_allele)
copy_method_doc(AlleleCountsChunkedArray.map_alleles,
                AlleleCountsArray.map_alleles)


