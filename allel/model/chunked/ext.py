# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np


from allel.compat import copy_method_doc, string_types
from allel.model.ndarray import GenotypeArray, HaplotypeArray, \
    AlleleCountsArray
from allel.model.chunked import core as chunked


class GenotypeChunkedArray(chunked.Array):

    def __init__(self, data):
        super(GenotypeChunkedArray, self).__init__(data)
        self._check_input_data(self.data)
        self._mask = None

    @staticmethod
    def _check_input_data(data):
        if len(data.shape) != 3:
            raise ValueError('expected 3 dimensions')
        if data.dtype.kind not in 'ui':
            raise TypeError('expected integer dtype')

    def __getitem__(self, *args):
        out = super(GenotypeChunkedArray, self).__getitem__(*args)
        if hasattr(out, 'shape') \
                and len(self.shape) == len(out.shape) \
                and self.shape[2] == out.shape[2]:
            # dimensionality and ploidy preserved
            out = GenotypeArray(out)
            if self.mask is not None:
                # attempt to slice mask too
                m = self.mask.__getitem__(*args)
                out.mask = m
        return out

    def _repr_html_(self):
        return self[:6].to_html_str(caption=repr(self))

    @property
    def n_variants(self):
        return self.shape[0]

    @property
    def n_samples(self):
        return self.shape[1]

    @property
    def ploidy(self):
        return self.shape[2]

    @property
    def n_calls(self):
        return self.shape[0] * self.shape[1]

    @property
    def n_allele_calls(self):
        return self.shape[0] * self.shape[1] * self.shape[2]

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask):

        # check input
        if not hasattr(mask, 'shape'):
            mask = np.asarray(mask)
        if mask.shape != self.shape[:2]:
            raise ValueError('mask has incorrect shape')

        # store
        self._mask = chunked.Array(mask)

    def fill_masked(self, value=-1, **kwargs):
        def f(block):
            return block.fill_masked(value=value)
        out = self.apply(f, **kwargs)
        return GenotypeChunkedArray(out)

    def compress(self, condition, axis=0, **kwargs):
        out = super(GenotypeChunkedArray, self).compress(condition, axis=axis,
                                                         **kwargs)
        if self.mask is not None:
            out.mask = self.mask.compress(condition, axis=axis, **kwargs)
        return out

    def take(self, indices, axis=0, **kwargs):
        out = super(GenotypeChunkedArray, self).take(indices, axis=axis,
                                                     **kwargs)
        if self.mask is not None:
            out.mask = self.mask.take(indices, axis=axis, **kwargs)
        return out

    def subset(self, sel0, sel1, **kwargs):
        out = super(GenotypeChunkedArray, self).subset(sel0, sel1, **kwargs)
        if self.mask is not None:
            out.mask = self.mask.subset(sel0, sel1, **kwargs)
        return out

    def is_called(self, **kwargs):
        def f(block):
            return block.is_called()
        out = self.apply(f, **kwargs)
        return chunked.Array(out)

    def is_missing(self, **kwargs):
        def f(block):
            return block.is_missing()
        out = self.apply(f, **kwargs)
        return chunked.Array(out)

    def is_hom(self, allele=None, **kwargs):
        def f(block):
            return block.is_hom(allele=allele)
        out = self.apply(f, **kwargs)
        return chunked.Array(out)

    def is_hom_ref(self, **kwargs):
        def f(block):
            return block.is_hom_ref()
        out = self.apply(f, **kwargs)
        return chunked.Array(out)

    def is_hom_alt(self, **kwargs):
        def f(block):
            return block.is_hom_alt()
        out = self.apply(f, **kwargs)
        return chunked.Array(out)

    def is_het(self, allele=None, **kwargs):
        def f(block):
            return block.is_het(allele=allele)
        out = self.apply(f, **kwargs)
        return chunked.Array(out)

    def is_call(self, call, **kwargs):
        def f(block):
            return block.is_call(call)
        out = self.apply(f, **kwargs)
        return chunked.Array(out)

    def count_called(self, axis=None, **kwargs):
        def mapper(block):
            return block.is_called()
        out = self.sum(axis=axis, mapper=mapper, **kwargs)
        return out

    def count_missing(self, axis=None, **kwargs):
        def mapper(block):
            return block.is_missing()
        out = self.sum(axis=axis, mapper=mapper, **kwargs)
        return out

    def count_hom(self, allele=None, axis=None, **kwargs):
        def mapper(block):
            return block.is_hom(allele=allele)
        out = self.sum(axis=axis, mapper=mapper, **kwargs)
        return out

    def count_hom_ref(self, axis=None, **kwargs):
        def mapper(block):
            return block.is_hom_ref()
        out = self.sum(axis=axis, mapper=mapper, **kwargs)
        return out

    def count_hom_alt(self, axis=None, **kwargs):
        def mapper(block):
            return block.is_hom_alt()
        out = self.sum(axis=axis, mapper=mapper, **kwargs)
        return out

    def count_het(self, axis=None, **kwargs):
        def mapper(block):
            return block.is_het()
        out = self.sum(axis=axis, mapper=mapper, **kwargs)
        return out

    def count_call(self, call, axis=None, **kwargs):
        def mapper(block):
            return block.is_call(call)
        out = self.sum(axis=axis, mapper=mapper, **kwargs)
        return out

    def to_haplotypes(self, **kwargs):
        def f(block):
            return block.to_haplotypes()
        out = self.apply(f, **kwargs)
        return HaplotypeChunkedArray(out)

    def to_n_ref(self, fill=0, dtype='i1', **kwargs):
        def f(block):
            return block.to_n_ref(fill=fill, dtype=dtype)
        out = self.apply(f, dtype=dtype, **kwargs)
        return chunked.Array(out)

    def to_n_alt(self, fill=0, dtype='i1', **kwargs):
        def f(block):
            return block.to_n_alt(fill=fill, dtype=dtype)
        out = self.apply(f, dtype=dtype, **kwargs)
        return chunked.Array(out)

    def to_allele_counts(self, alleles=None, **kwargs):
        # determine alleles to count
        if alleles is None:
            m = self.max()
            alleles = list(range(m+1))

        def f(block):
            return block.to_allele_counts(alleles)
        out = self.apply(f, **kwargs)
        return chunked.Array(out)

    def to_packed(self, boundscheck=True, **kwargs):
        def f(block):
            return block.to_packed(boundscheck=boundscheck)
        out = self.apply(f, **kwargs)
        return chunked.Array(out)

    @staticmethod
    def from_packed(packed, **kwargs):
        def f(block):
            return GenotypeArray.from_packed(block)
        out = chunked.apply(packed, f, **kwargs)
        return GenotypeChunkedArray(out)

    def count_alleles(self, max_allele=None, subpop=None, **kwargs):
        # if max_allele not specified, count all alleles
        if max_allele is None:
            max_allele = self.max()

        def f(block):
            return block.count_alleles(max_allele=max_allele, subpop=subpop)
        out = self.apply(f, **kwargs)
        return AlleleCountsChunkedArray(out)

    def count_alleles_subpops(self, subpops, max_allele=None, **kwargs):
        if max_allele is None:
            max_allele = self.max()

        def f(block):
            return block.count_alleles_subpops(subpops, max_allele=max_allele)
        out = chunked.apply(self, f, create='table', **kwargs)
        return AlleleCountsChunkedTable(out)

    def to_gt(self, phased=False, max_allele=None, **kwargs):
        def f(block):
            return block.to_gt(phased=phased, max_allele=max_allele)
        out = self.apply(f, **kwargs)
        return chunked.Array(out)

    def map_alleles(self, mapping, **kwargs):
        def f(block, bmapping):
            return block.map_alleles(bmapping, copy=False)
        domain = (self, mapping)
        out = chunked.apply(domain, f, **kwargs)
        return GenotypeChunkedArray(out)


# copy docstrings
copy_method_doc(GenotypeChunkedArray.fill_masked, GenotypeArray.fill_masked)
copy_method_doc(GenotypeChunkedArray.subset, GenotypeArray.subset)
copy_method_doc(GenotypeChunkedArray.is_called, GenotypeArray.is_called)
copy_method_doc(GenotypeChunkedArray.is_missing, GenotypeArray.is_missing)
copy_method_doc(GenotypeChunkedArray.is_hom, GenotypeArray.is_hom)
copy_method_doc(GenotypeChunkedArray.is_hom_ref, GenotypeArray.is_hom_ref)
copy_method_doc(GenotypeChunkedArray.is_hom_alt, GenotypeArray.is_hom_alt)
copy_method_doc(GenotypeChunkedArray.is_het, GenotypeArray.is_het)
copy_method_doc(GenotypeChunkedArray.is_call, GenotypeArray.is_call)
copy_method_doc(GenotypeChunkedArray.to_haplotypes,
                GenotypeArray.to_haplotypes)
copy_method_doc(GenotypeChunkedArray.to_n_ref, GenotypeArray.to_n_ref)
copy_method_doc(GenotypeChunkedArray.to_n_alt, GenotypeArray.to_n_alt)
copy_method_doc(GenotypeChunkedArray.to_allele_counts,
                GenotypeArray.to_allele_counts)
copy_method_doc(GenotypeChunkedArray.to_packed, GenotypeArray.to_packed)
GenotypeChunkedArray.from_packed.__doc__ = GenotypeArray.from_packed.__doc__
copy_method_doc(GenotypeChunkedArray.count_alleles,
                GenotypeArray.count_alleles)
copy_method_doc(GenotypeChunkedArray.count_alleles_subpops,
                GenotypeArray.count_alleles_subpops)
copy_method_doc(GenotypeChunkedArray.to_gt, GenotypeArray.to_gt)
copy_method_doc(GenotypeChunkedArray.map_alleles, GenotypeArray.map_alleles)
copy_method_doc(GenotypeChunkedArray.hstack, GenotypeArray.hstack)
copy_method_doc(GenotypeChunkedArray.vstack, GenotypeArray.vstack)


class HaplotypeChunkedArray(chunked.Array):

    def __init__(self, data):
        super(HaplotypeChunkedArray, self).__init__(data)
        self._check_input_data(self.data)

    @staticmethod
    def _check_input_data(data):
        if len(data.shape) != 2:
            raise ValueError('expected 2 dimensions')
        if data.dtype.kind not in 'ui':
            raise TypeError('expected integer dtype')

    def __getitem__(self, *args):
        out = super(HaplotypeChunkedArray, self).__getitem__(*args)
        if hasattr(out, 'shape') and len(self.shape) == len(out.shape):
            # dimensionality preserved
            out = HaplotypeArray(out)
        return out

    def _repr_html_(self):
        return self[:6].to_html_str(caption=repr(self))

    @property
    def n_variants(self):
        return self.shape[0]

    @property
    def n_haplotypes(self):
        return self.shape[1]

    def to_genotypes(self, ploidy=2, **kwargs):

        # check ploidy is compatible
        if (self.n_haplotypes % ploidy) > 0:
            raise ValueError('incompatible ploidy')

        # build output
        def f(block):
            return block.to_genotypes(ploidy)

        out = self.apply(f, **kwargs)
        return GenotypeChunkedArray(out)

    def is_called(self, **kwargs):
        return self.__ge__(0, **kwargs)

    def is_missing(self, **kwargs):
        return self.__lt__(0, **kwargs)

    def is_ref(self, **kwargs):
        return self.__eq__(0, **kwargs)

    def is_alt(self, **kwargs):
        return self.__gt__(0, **kwargs)

    def is_call(self, allele, **kwargs):
        return self.__eq__(allele, **kwargs)

    def count_called(self, axis=None, **kwargs):
        def mapper(block):
            return block.is_called()
        out = self.sum(axis=axis, mapper=mapper, **kwargs)
        return out

    def count_missing(self, axis=None, **kwargs):
        def mapper(block):
            return block.is_missing()
        out = self.sum(axis=axis, mapper=mapper, **kwargs)
        return out

    def count_ref(self, axis=None, **kwargs):
        def mapper(block):
            return block.is_ref()
        out = self.sum(axis=axis, mapper=mapper, **kwargs)
        return out

    def count_alt(self, axis=None, **kwargs):
        def mapper(block):
            return block.is_alt()
        out = self.sum(axis=axis, mapper=mapper, **kwargs)
        return out

    def count_call(self, allele, axis=None, **kwargs):
        def mapper(block):
            return block.is_call(allele)
        out = self.sum(axis=axis, mapper=mapper, **kwargs)
        return out

    def count_alleles(self, max_allele=None, subpop=None, **kwargs):
        # if max_allele not specified, count all alleles
        if max_allele is None:
            max_allele = self.max()

        def f(block):
            return block.count_alleles(max_allele=max_allele, subpop=subpop)
        out = self.apply(f, **kwargs)
        return AlleleCountsChunkedArray(out)

    def count_alleles_subpops(self, subpops, max_allele=None, **kwargs):
        if max_allele is None:
            max_allele = self.max()

        def f(block):
            return block.count_alleles_subpops(subpops, max_allele=max_allele)
        out = chunked.apply(self, f, create='table', **kwargs)
        return AlleleCountsChunkedTable(out)

    def map_alleles(self, mapping, **kwargs):
        def f(block, bmapping):
            return block.map_alleles(bmapping, copy=False)
        domain = (self, mapping)
        out = chunked.apply(domain, f, **kwargs)
        return HaplotypeChunkedArray(out)


# copy docstrings
copy_method_doc(HaplotypeChunkedArray.to_genotypes,
                HaplotypeArray.to_genotypes)
copy_method_doc(HaplotypeChunkedArray.count_alleles,
                HaplotypeArray.count_alleles)
copy_method_doc(HaplotypeChunkedArray.count_alleles_subpops,
                HaplotypeArray.count_alleles_subpops)
copy_method_doc(HaplotypeChunkedArray.map_alleles, HaplotypeArray.map_alleles)


class AlleleCountsChunkedArray(chunked.Array):

    def __init__(self, data):
        super(AlleleCountsChunkedArray, self).__init__(data)
        self._check_input_data(self.data)

    @staticmethod
    def _check_input_data(data):
        if len(data.shape) != 2:
            raise ValueError('expected 2 dimensions')
        if data.dtype.kind not in 'ui':
            raise TypeError('expected integer dtype')

    def __getitem__(self, *args):
        out = super(AlleleCountsChunkedArray, self).__getitem__(*args)
        if hasattr(out, 'shape') and len(self.shape) == len(out.shape) and \
                out.shape[1] == self.shape[1]:
            # dimensionality and allele indices preserved
            out = AlleleCountsArray(out)
        return out

    def _repr_html_(self):
        return self[:6].to_html_str(caption=repr(self))

    @property
    def n_variants(self):
        return self.shape[0]

    @property
    def n_alleles(self):
        return self.shape[1]

    def to_frequencies(self, fill=np.nan, **kwargs):
        def f(block):
            return block.to_frequencies(fill=fill)
        out = self.apply(f, **kwargs)
        return chunked.Array(out)

    def allelism(self, **kwargs):
        def f(block):
            return block.allelism()
        out = self.apply(f, **kwargs)
        return chunked.Array(out)

    def max_allele(self, **kwargs):
        def f(block):
            return block.max_allele()
        out = self.apply(f, **kwargs)
        return chunked.Array(out)

    def is_variant(self, **kwargs):
        def f(block):
            return block.is_variant()
        out = self.apply(f, **kwargs)
        return chunked.Array(out)

    def is_non_variant(self, **kwargs):
        def f(block):
            return block.is_non_variant()
        out = self.apply(f, **kwargs)
        return chunked.Array(out)

    def is_segregating(self, **kwargs):
        def f(block):
            return block.is_segregating()
        out = self.apply(f, **kwargs)
        return chunked.Array(out)

    def is_non_segregating(self, allele=None, **kwargs):
        def f(block):
            return block.is_non_segregating(allele=allele)
        out = self.apply(f, **kwargs)
        return chunked.Array(out)

    def is_singleton(self, allele=1, **kwargs):
        def f(block):
            return block.is_singleton(allele=allele)
        out = self.apply(f, **kwargs)
        return chunked.Array(out)

    def is_doubleton(self, allele=1, **kwargs):
        def f(block):
            return block.is_doubleton(allele=allele)
        out = self.apply(f, **kwargs)
        return chunked.Array(out)

    def count_variant(self, **kwargs):
        def mapper(block):
            return block.is_variant()
        out = self.sum(mapper=mapper, **kwargs)
        return out

    def count_non_variant(self, **kwargs):
        def mapper(block):
            return block.is_non_variant()
        out = self.sum(mapper=mapper, **kwargs)
        return out

    def count_segregating(self, **kwargs):
        def mapper(block):
            return block.is_segregating()
        out = self.sum(mapper=mapper, **kwargs)
        return out

    def count_non_segregating(self, allele=None, **kwargs):
        def mapper(block):
            return block.is_non_segregating(allele=allele)
        out = self.sum(mapper=mapper, **kwargs)
        return out

    def count_singleton(self, allele=1, **kwargs):
        def mapper(block):
            return block.is_singleton(allele=allele)
        out = self.sum(mapper=mapper, **kwargs)
        return out

    def count_doubleton(self, allele=1, **kwargs):
        def mapper(block):
            return block.is_doubleton(allele=allele)
        out = self.sum(mapper=mapper, **kwargs)
        return out

    def map_alleles(self, mapping, **kwargs):
        def f(block, bmapping):
            return block.map_alleles(bmapping)
        domain = (self, mapping)
        out = chunked.apply(domain, f, **kwargs)
        return AlleleCountsChunkedArray(out)


copy_method_doc(AlleleCountsChunkedArray.allelism, AlleleCountsArray.allelism)
copy_method_doc(AlleleCountsChunkedArray.max_allele,
                AlleleCountsArray.max_allele)
copy_method_doc(AlleleCountsChunkedArray.map_alleles,
                AlleleCountsArray.map_alleles)


class VariantChunkedTable(chunked.Table):
    # TODO
    pass


class FeatureChunkedTable(chunked.Table):
    # TODO
    pass


class AlleleCountsChunkedTable(chunked.Table):

    def __getitem__(self, item):
        out = super(AlleleCountsChunkedTable, self).__getitem__(item)
        if isinstance(item, string_types):
            out = AlleleCountsChunkedArray(out)
        return out

    # TODO
