# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np
import numexpr as ne


from allel.constants import *


def _check_genotype_array_data(obj):
    obj = np.asarray(obj)

    # check dtype
    if obj.dtype.kind not in 'ui':
        raise TypeError('integer dtype required')

    # check dimensionality
    if obj.ndim != 3:
        raise TypeError('array with 3 dimensions required')

    return obj


class GenotypeArray(np.ndarray):
    """TODO

    """

    def __new__(cls, data):
        obj = _check_genotype_array_data(data)
        obj = obj.view(cls)
        return obj

    def __array_finalize__(self, obj):

        # called after constructor
        if obj is None:
            return

        # called after slice (new-from-template)
        if isinstance(obj, GenotypeArray):
            return

        # called after view
        _check_genotype_array_data(obj)

    def __getslice__(self, *args, **kwargs):
        s = np.ndarray.__getslice__(self, *args, **kwargs)
        if hasattr(s, 'ndim') and s.ndim == 3:
            return s
        else:
            # slice with reduced dimensionality returns plain ndarray
            return np.asarray(s)

    def __getitem__(self, *args, **kwargs):
        s = np.ndarray.__getitem__(self, *args, **kwargs)
        if hasattr(s, 'ndim') and s.ndim == 3:
            return s
        else:
            # slice with reduced dimensionality returns plain ndarray
            return np.asarray(s)

    @property
    def n_variants(self):
        """TODO"""
        return self.shape[0]

    @property
    def n_samples(self):
        """TODO"""
        return self.shape[1]

    @property
    def ploidy(self):
        """TODO"""
        return self.shape[2]

    def __repr__(self):
        s = super(GenotypeArray, self).__repr__()
        return s[:-1] + ', n_variants=%s, n_samples=%s, ploidy=%s)' % \
                        self.shape

    def is_called(self):
        """TODO

        """

        # special case diploid
        if self.ploidy == DIPLOID:
            allele1 = self[..., 0]  # noqa
            allele2 = self[..., 1]  # noqa
            ex = '(allele1 >= 0) & (allele2 >= 0)'
            out = ne.evaluate(ex)

        # general ploidy case
        else:
            out = np.all(self >= 0, axis=DIM_PLOIDY)

        return out

    def is_missing(self):
        """TODO

        """

        # special case diploid
        if self.ploidy == DIPLOID:
            allele1 = self[..., 0]  # noqa
            allele2 = self[..., 1]  # noqa
            # call is missing if either allele is missing
            ex = '(allele1 < 0) | (allele2 < 0)'
            out = ne.evaluate(ex)

        # general ploidy case
        else:
            # call is missing if any allele is missing
            out = np.any(self < 0, axis=DIM_PLOIDY)

        return out

    def is_hom(self, allele=None):
        """TODO

        """

        # special case diploid
        if self.ploidy == DIPLOID:
            allele1 = self[..., 0]  # noqa
            allele2 = self[..., 1]  # noqa
            if allele is None:
                ex = '(allele1 >= 0) & (allele1  == allele2)'
            else:
                ex = '(allele1 == {0}) & (allele2 == {0})'.format(allele)
            out = ne.evaluate(ex)

        # general ploidy case
        else:
            if allele is None:
                allele1 = self[..., 0, None]  # noqa
                other_alleles = self[..., 1:]  # noqa
                ex = '(allele1 >= 0) & (allele1 == other_alleles)'
                out = np.all(ne.evaluate(ex), axis=DIM_PLOIDY)
            else:
                out = np.all(self == allele, axis=DIM_PLOIDY)

        return out

    def is_hom_ref(self):
        """TODO

        """

        return self.is_hom(allele=0)

    def is_hom_alt(self):
        """TODO

        """

        # special case diploid
        if self.ploidy == DIPLOID:
            allele1 = self[..., 0]  # noqa
            allele2 = self[..., 1]  # noqa
            ex = '(allele1 > 0) & (allele1  == allele2)'
            out = ne.evaluate(ex)

        # general ploidy case
        else:
            allele1 = self[..., 0, None]  # noqa
            other_alleles = self[..., 1:]  # noqa
            ex = '(allele1 > 0) & (allele1 == other_alleles)'
            out = np.all(ne.evaluate(ex), axis=DIM_PLOIDY)

        return out

    def is_het(self):
        """TODO

        """

        # special case diploid
        if self.ploidy == DIPLOID:
            allele1 = self[..., 0]  # noqa
            allele2 = self[..., 1]  # noqa
            ex = '(allele1 >= 0) & (allele2  >= 0) & (allele1 != allele2)'
            out = ne.evaluate(ex)

        # general ploidy case
        else:
            allele1 = self[..., 0, None]  # noqa
            other_alleles = self[..., 1:]  # noqa
            out = np.all(self >= 0, axis=DIM_PLOIDY) \
                & np.any(allele1 != other_alleles, axis=DIM_PLOIDY)

        return out

    def is_call(self, call):
        """TODO

        """

        # special case diploid
        if self.ploidy == DIPLOID:
            if not len(call) == DIPLOID:
                raise ValueError('invalid call: %r', call)
            allele1 = self[..., 0]  # noqa
            allele2 = self[..., 1]  # noqa
            ex = '(allele1 == {0}) & (allele2  == {1})'.format(*call)
            out = ne.evaluate(ex)

        # general ploidy case
        else:
            if not len(call) == self.ploidy:
                raise ValueError('invalid call: %r', call)
            call = np.asarray(call)[None, None, :]
            out = np.all(self == call, axis=DIM_PLOIDY)

        return out

    def count_called(self, axis=None):
        """TODO

        """

        b = self.is_called()
        return np.sum(b, axis=axis)

    def count_missing(self, axis=None):
        """TODO

        """

        b = self.is_missing()
        return np.sum(b, axis=axis)

    def count_hom(self, allele=None, axis=None):
        """TODO

        """

        b = self.is_hom(allele=allele)
        return np.sum(b, axis=axis)

    def count_hom_ref(self, axis=None):
        """TODO

        """

        b = self.is_hom_ref()
        return np.sum(b, axis=axis)

    def count_hom_alt(self, axis=None):
        """TODO

        """

        b = self.is_hom_alt()
        return np.sum(b, axis=axis)

    def count_het(self, axis=None):
        """TODO

        """

        b = self.is_het()
        return np.sum(b, axis=axis)

    def count_call(self, call, axis=None):
        """TODO

        """

        b = self.is_call(call=call)
        return np.sum(b, axis=axis)
