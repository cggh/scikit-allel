# -*- coding: utf-8 -*-
import collections
import bisect


# third-party imports
import numpy as np


# internal imports
from allel.util import check_integer_dtype, check_shape, check_dtype, ignore_invalid, \
    check_dim0_aligned, check_ploidy, check_ndim, asarray_ndim, check_dim1_aligned
from allel.compat import copy_method_doc, memoryview_safe
from allel.io.vcf_write import write_vcf
from allel.io.gff import gff3_to_recarray
from allel.io.util import recarray_from_hdf5_group, recarray_to_hdf5_group
from allel.abc import ArrayWrapper, DisplayAs1D, DisplayAs2D, DisplayAsTable
from allel.opt.model import genotype_array_pack_diploid, genotype_array_unpack_diploid, \
    genotype_array_count_alleles, genotype_array_count_alleles_masked, \
    genotype_array_count_alleles_subpop, genotype_array_count_alleles_subpop_masked, \
    genotype_array_to_allele_counts, genotype_array_to_allele_counts_masked, \
    haplotype_array_count_alleles, haplotype_array_count_alleles_subpop, \
    haplotype_array_map_alleles, allele_counts_array_map_alleles
from .generic import index_genotype_vector, compress_genotypes, \
    take_genotypes, concatenate_genotypes, index_genotype_array, subset_genotype_array, \
    index_haplotype_array, compress_haplotype_array, take_haplotype_array, \
    subset_haplotype_array, concatenate_haplotype_array, index_allele_counts_array, \
    compress_allele_counts_array, take_allele_counts_array, concatenate_allele_counts_array,\
    index_genotype_ac_array, index_genotype_ac_vector, compress_genotype_ac, \
    take_genotype_ac, concatenate_genotype_ac, subset_genotype_ac_array


__all__ = ['Genotypes', 'GenotypeArray', 'GenotypeVector', 'HaplotypeArray', 'AlleleCountsArray',
           'GenotypeAlleleCounts', 'GenotypeAlleleCountsArray', 'GenotypeAlleleCountsVector',
           'SortedIndex', 'UniqueIndex', 'SortedMultiIndex', 'VariantTable', 'FeatureTable',
           'ChromPosIndex']


# noinspection PyTypeChecker
_total_slice = slice(None)


def subset(data, sel0, sel1):
    """Apply selections on first and second axes."""

    # check inputs
    data = np.asarray(data)
    if data.ndim < 2:
        raise ValueError('data must have 2 or more dimensions')
    sel0 = asarray_ndim(sel0, 1, allow_none=True)
    sel1 = asarray_ndim(sel1, 1, allow_none=True)

    # ensure indices
    if sel0 is not None and sel0.dtype.kind == 'b':
        sel0, = np.nonzero(sel0)
    if sel1 is not None and sel1.dtype.kind == 'b':
        sel1, = np.nonzero(sel1)

    # ensure leading dimension indices can be broadcast correctly
    if sel0 is not None and sel1 is not None:
        sel0 = sel0[:, np.newaxis]

    # deal with None arguments
    if sel0 is None:
        sel0 = _total_slice
    if sel1 is None:
        sel1 = _total_slice

    return data[sel0, sel1]


class NumpyArrayWrapper(ArrayWrapper):
    """Abstract base class that wraps a NumPy array."""

    def __init__(self, data, copy=False, **kwargs):
        values = np.array(data, copy=copy, **kwargs)
        super(NumpyArrayWrapper, self).__init__(values)


class NumpyRecArrayWrapper(DisplayAsTable):

    def __init__(self, data, copy=False, **kwargs):
        if isinstance(data, dict):
            names = kwargs.get('names', sorted(data.keys()))
            arrays = [data[n] for n in names]
            values = np.rec.fromarrays(arrays, names=names)
        else:
            values = np.rec.array(data, copy=copy, **kwargs)
            check_ndim(values, 1)
        if not values.dtype.names:
            raise ValueError('expected recarray')
        super(NumpyRecArrayWrapper, self).__init__(values)

    def __getitem__(self, item):
        s = self.values[item]
        if isinstance(item, (slice, list, np.ndarray, type(Ellipsis))):
            return type(self)(s)
        return s

    @property
    def names(self):
        return tuple(self.dtype.names)

    @classmethod
    def fromdict(cls, data, **kwargs):
        names = kwargs.get('names', sorted(data.keys()))
        arrays = [data[n] for n in names]
        a = np.rec.fromarrays(arrays, names=names)
        return cls(a, copy=False)

    @classmethod
    def fromarrays(cls, *args, **kwargs):
        a = np.rec.fromarrays(*args, **kwargs)
        return cls(a, copy=False)

    @classmethod
    def fromrecords(cls, *args, **kwargs):
        a = np.rec.fromrecords(*args, **kwargs)
        return cls(a, copy=False)

    @classmethod
    def from_hdf5_group(cls, *args, **kwargs):
        a = recarray_from_hdf5_group(*args, **kwargs)
        return cls(a, copy=False)

    def to_hdf5_group(self, parent, name, **kwargs):
        return recarray_to_hdf5_group(self, parent, name, **kwargs)

    def eval(self, expression, vm='python'):
        """Evaluate an expression against the table columns.

        Parameters
        ----------
        expression : string
            Expression to evaluate.
        vm : {'numexpr', 'python'}
            Virtual machine to use.

        Returns
        -------
        result : ndarray

        """

        if vm == 'numexpr':
            import numexpr as ne
            return ne.evaluate(expression, local_dict=self)
        else:
            return eval(expression, dict(), self)

    def query(self, expression, vm='python'):
        """Evaluate expression and then use it to extract rows from the table.

        Parameters
        ----------
        expression : string
            Expression to evaluate.
        vm : {'numexpr', 'python'}
            Virtual machine to use.

        Returns
        -------
        result : structured array

        """

        condition = self.eval(expression, vm=vm)
        return self.compress(condition)

    def copy(self, *args, **kwargs):
        data = self.values.copy(*args, **kwargs)
        # can always wrap this as sub-class type
        return type(self)(data)

    def compress(self, condition, axis=0, out=None):
        out = self.values.compress(condition, axis=axis, out=out)
        if axis == 0:
            out = type(self)(out)
        return out

    def take(self, indices, axis=0, out=None, mode='raise'):
        out = self.values.take(indices, axis=axis, out=out, mode=mode)
        if axis == 0:
            out = type(self)(out)
        return out

    def concatenate(self, others):
        """Concatenate arrays."""
        if not isinstance(others, (list, tuple)):
            others = others,
        tup = (self.values,) + tuple(o.values for o in others)
        out = np.concatenate(tup, axis=0)
        out = type(self)(out)
        return out


class Genotypes(NumpyArrayWrapper):
    """Base class for wrapping a NumPy array of genotype calls.

    See Also
    --------
    GenotypeArray, GenotypeVector

    """

    def __init__(self, data, copy=False, **kwargs):
        super(Genotypes, self).__init__(data, copy=copy, **kwargs)
        check_integer_dtype(self.values)
        self._mask = None
        self._is_phased = None

    @property
    def ploidy(self):
        """Sample ploidy."""
        return self.shape[-1]

    @property
    def n_allele_calls(self):
        """Total number of allele calls."""
        return np.prod(self.shape)

    @property
    def n_calls(self):
        """Total number of genotype calls."""
        return self.n_allele_calls // self.ploidy

    @property
    def mask(self):
        """A boolean mask, indicating genotype calls that should be filtered (i.e.,
        excluded) from genotype and allele counting operations.

        Examples
        --------
        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 1], [1, 1]],
        ...                          [[0, 2], [-1, -1]]], dtype='i1')
        >>> g
        <GenotypeArray shape=(3, 2, 2) dtype=int8>
        0/0 0/1
        0/1 1/1
        0/2 ./.
        >>> g.count_called()
        5
        >>> g.count_alleles()
        <AlleleCountsArray shape=(3, 3) dtype=int32>
        3 1 0
        1 3 0
        1 0 1
        >>> v = g[:, 1]
        >>> v
        <GenotypeVector shape=(3, 2) dtype=int8>
        0/1 1/1 ./.
        >>> v.is_called()
        array([ True,  True, False])
        >>> mask = [[True, False], [False, True], [False, False]]
        >>> g.mask = mask
        >>> g
        <GenotypeArray shape=(3, 2, 2) dtype=int8>
        ./. 0/1
        0/1 ./.
        0/2 ./.
        >>> g.count_called()
        3
        >>> g.count_alleles()
        <AlleleCountsArray shape=(3, 3) dtype=int32>
        1 1 0
        1 1 0
        1 0 1
        >>> v = g[:, 1]
        >>> v
        <GenotypeVector shape=(3, 2) dtype=int8>
        0/1 ./. ./.
        >>> v.is_called()
        array([ True, False, False])

        Notes
        -----
        This is a lightweight genotype call mask and **not** a mask in the
        sense of a numpy masked array. This means that the mask will only be
        taken into account by the genotype and allele counting methods of this
        class, and is ignored by any of the generic methods on the ndarray
        class or by any numpy ufuncs.

        """
        return self._mask

    @mask.setter
    def mask(self, mask):
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            check_shape(mask, self.shape[:-1])
        self._mask = mask

    @property
    def is_phased(self):
        """A Boolean array indicating which genotype calls are phased and which are not.

        Examples
        --------
        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 1], [1, 1]],
        ...                          [[0, 2], [-1, -1]]], dtype='i1')
        >>> g
        <GenotypeArray shape=(3, 2, 2) dtype=int8>
        0/0 0/1
        0/1 1/1
        0/2 ./.
        >>> g.is_phased = [[True, True], [False, True], [False, False]]
        >>> g
        <GenotypeArray shape=(3, 2, 2) dtype=int8>
        0|0 0|1
        0/1 1|1
        0/2 ./.

        """
        return self._is_phased

    @is_phased.setter
    def is_phased(self, is_phased):
        if is_phased is not None:
            is_phased = np.asarray(is_phased, dtype=bool)
            check_shape(is_phased, self.shape[:-1])
        self._is_phased = is_phased

    def fill_masked(self, value=-1, copy=True):
        """Fill masked genotype calls with a given value.

        Parameters
        ----------
        value : int, optional
            The fill value.
        copy : bool, optional
            If False, modify the array in place.

        Returns
        -------
        g : GenotypeArray

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 1], [1, 1]],
        ...                          [[0, 2], [-1, -1]]], dtype='i1')
        >>> mask = [[True, False], [False, True], [False, False]]
        >>> g.mask = mask
        >>> g.fill_masked().values
        array([[[-1, -1],
                [ 0,  1]],
               [[ 0,  1],
                [-1, -1]],
               [[ 0,  2],
                [-1, -1]]], dtype=int8)

        """

        if self.mask is None:
            raise ValueError('no mask is set')

        # apply the mask
        data = np.array(self.values, copy=copy)
        data[self.mask, ...] = value

        if copy:
            out = type(self)(data)  # wrap
            out.is_phased = self.is_phased
            # don't set mask because it has been filled in
        else:
            out = self
            out.mask = None  # reset mask

        return out

    def is_called(self):
        """Find non-missing genotype calls.

        Returns
        -------
        out : ndarray, bool, shape (n_variants, n_samples)
            Array where elements are True if the genotype call matches the
            condition.

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 1], [1, 1]],
        ...                          [[0, 2], [-1, -1]]])
        >>> g.is_called()
        array([[ True,  True],
               [ True,  True],
               [ True, False]])
        >>> v = g[:, 1]
        >>> v
        <GenotypeVector shape=(3, 2) dtype=int64>
        0/1 1/1 ./.
        >>> v.is_called()
        array([ True,  True, False])

        """

        out = np.all(self.values >= 0, axis=-1)

        # handle mask
        if self.mask is not None:
            out &= ~self.mask

        return out

    def is_missing(self):
        """Find missing genotype calls.

        Returns
        -------
        out : ndarray, bool, shape (n_variants, n_samples)
            Array where elements are True if the genotype call matches the
            condition.

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 1], [1, 1]],
        ...                          [[0, 2], [-1, -1]]])
        >>> g.is_missing()
        array([[False, False],
               [False, False],
               [False,  True]])
        >>> v = g[:, 1]
        >>> v
        <GenotypeVector shape=(3, 2) dtype=int64>
        0/1 1/1 ./.
        >>> v.is_missing()
        array([False, False,  True])

        """

        out = np.any(self.values < 0, axis=-1)

        # handle mask
        if self.mask is not None:
            out |= self.mask

        return out

    def is_hom(self, allele=None):
        """Find genotype calls that are homozygous.

        Parameters
        ----------
        allele : int, optional
            Allele index.

        Returns
        -------
        out : ndarray, bool, shape (n_variants, n_samples)
            Array where elements are True if the genotype call matches the
            condition.

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 1], [1, 1]],
        ...                          [[2, 2], [-1, -1]]])
        >>> g.is_hom()
        array([[ True, False],
               [False,  True],
               [ True, False]])
        >>> g.is_hom(allele=1)
        array([[False, False],
               [False,  True],
               [False, False]])
        >>> v = g[:, 0]
        >>> v
        <GenotypeVector shape=(3, 2) dtype=int64>
        0/0 0/1 2/2
        >>> v.is_hom()
        array([ True, False,  True])

        """

        if allele is None:
            allele1 = self.values[..., 0, np.newaxis]
            other_alleles = self.values[..., 1:]
            tmp = (allele1 >= 0) & (allele1 == other_alleles)
            out = np.all(tmp, axis=-1)
        else:
            out = np.all(self.values == allele, axis=-1)

        # handle mask
        if self.mask is not None:
            out &= ~self.mask

        return out

    def is_hom_ref(self):
        """Find genotype calls that are homozygous for the reference allele.

        Returns
        -------
        out : ndarray, bool, shape (n_variants, n_samples)
            Array where elements are True if the genotype call matches the
            condition.

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 1], [1, 1]],
        ...                          [[0, 2], [-1, -1]]])
        >>> g.is_hom_ref()
        array([[ True, False],
               [False, False],
               [False, False]])
        >>> v = g[:, 0]
        >>> v
        <GenotypeVector shape=(3, 2) dtype=int64>
        0/0 0/1 0/2
        >>> v.is_hom_ref()
        array([ True, False, False])

        """

        return self.is_hom(allele=0)

    def is_hom_alt(self):
        """Find genotype calls that are homozygous for any alternate (i.e.,
        non-reference) allele.

        Returns
        -------
        out : ndarray, bool, shape (n_variants, n_samples)
            Array where elements are True if the genotype call matches the
            condition.

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 1], [1, 1]],
        ...                          [[2, 2], [-1, -1]]])
        >>> g.is_hom_alt()
        array([[False, False],
               [False,  True],
               [ True, False]])
        >>> v = g[:, 1]
        >>> v
        <GenotypeVector shape=(3, 2) dtype=int64>
        0/1 1/1 ./.
        >>> v.is_hom_alt()
        array([False,  True, False])

        """

        allele1 = self.values[..., 0, np.newaxis]
        other_alleles = self.values[..., 1:]
        tmp = (allele1 > 0) & (allele1 == other_alleles)
        out = np.all(tmp, axis=-1)

        # handle mask
        if self.mask is not None:
            out &= ~self.mask

        return out

    def is_het(self, allele=None):
        """Find genotype calls that are heterozygous.

        Returns
        -------
        out : ndarray, bool, shape (n_variants, n_samples)
            Array where elements are True if the genotype call matches the
            condition.
        allele : int, optional
            Heterozygous allele.

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 1], [1, 1]],
        ...                          [[0, 2], [-1, -1]]])
        >>> g.is_het()
        array([[False,  True],
               [ True, False],
               [ True, False]])
        >>> g.is_het(2)
        array([[False, False],
               [False, False],
               [ True, False]])
        >>> v = g[:, 0]
        >>> v
        <GenotypeVector shape=(3, 2) dtype=int64>
        0/0 0/1 0/2
        >>> v.is_het()
        array([False,  True,  True])

        """

        allele1 = self.values[..., 0, np.newaxis]  # type: np.ndarray
        other_alleles = self.values[..., 1:]  # type: np.ndarray
        out = np.all(self.values >= 0, axis=-1) & np.any(allele1 != other_alleles, axis=-1)
        if allele is not None:
            out &= np.any(self.values == allele, axis=-1)

        # handle mask
        if self.mask is not None:
            out &= ~self.mask

        return out

    def is_call(self, call):
        """Locate genotypes with a given call.

        Parameters
        ----------
        call : array_like, int, shape (ploidy,)
            The genotype call to find.

        Returns
        -------
        out : ndarray, bool, shape (n_variants, n_samples)
            Array where elements are True if the genotype is `call`.

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 1], [1, 1]],
        ...                          [[0, 2], [-1, -1]]])
        >>> g.is_call((0, 2))
        array([[False, False],
               [False, False],
               [ True, False]])
        >>> v = g[:, 0]
        >>> v
        <GenotypeVector shape=(3, 2) dtype=int64>
        0/0 0/1 0/2
        >>> v.is_call((0, 2))
        array([False, False,  True])

        """

        # guard conditions
        if not len(call) == self.shape[-1]:
            raise ValueError('invalid call ploidy: %s', repr(call))

        if self.ndim == 2:
            call = np.asarray(call)[np.newaxis, :]
        else:
            call = np.asarray(call)[np.newaxis, np.newaxis, :]
        out = np.all(self.values == call, axis=-1)

        # handle mask
        if self.mask is not None:
            out &= ~self.mask

        return out

    def count_called(self, axis=None):
        """Count called genotypes.

        Parameters
        ----------
        axis : int, optional
            Axis over which to count, or None to perform overall count.

        """
        b = self.is_called()
        return np.sum(b, axis=axis)

    def count_missing(self, axis=None):
        """Count missing genotypes.

        Parameters
        ----------
        axis : int, optional
            Axis over which to count, or None to perform overall count.

        """
        b = self.is_missing()
        return np.sum(b, axis=axis)

    def count_hom(self, allele=None, axis=None):
        """Count homozygous genotypes.

        Parameters
        ----------
        allele : int, optional
            Allele index.
        axis : int, optional
            Axis over which to count, or None to perform overall count.

        """
        b = self.is_hom(allele=allele)
        return np.sum(b, axis=axis)

    def count_hom_ref(self, axis=None):
        """Count homozygous reference genotypes.

        Parameters
        ----------
        axis : int, optional
            Axis over which to count, or None to perform overall count.

        """
        b = self.is_hom_ref()
        return np.sum(b, axis=axis)

    def count_hom_alt(self, axis=None):
        """Count homozygous alternate genotypes.

        Parameters
        ----------
        axis : int, optional
            Axis over which to count, or None to perform overall count.

        """
        b = self.is_hom_alt()
        return np.sum(b, axis=axis)

    def count_het(self, allele=None, axis=None):
        """Count heterozygous genotypes.

        Parameters
        ----------
        allele : int, optional
            Allele index.
        axis : int, optional
            Axis over which to count, or None to perform overall count.

        """
        b = self.is_het(allele=allele)
        return np.sum(b, axis=axis)

    def count_call(self, call, axis=None):
        """Count genotypes with a given call.

        Parameters
        ----------
        call : array_like, int, shape (ploidy,)
            The genotype call to find.
        axis : int, optional
            Axis over which to count, or None to perform overall count.

        """
        b = self.is_call(call=call)
        return np.sum(b, axis=axis)

    def to_n_ref(self, fill=0, dtype='i1'):
        """Transform each genotype call into the number of
        reference alleles.

        Parameters
        ----------
        fill : int, optional
            Use this value to represent missing calls.
        dtype : dtype, optional
            Output dtype.

        Returns
        -------
        out : ndarray, int8, shape (n_variants, n_samples)
            Array of ref alleles per genotype call.

        Notes
        -----
        By default this function returns 0 for missing genotype calls
        **and** for homozygous non-reference genotype calls. Use the
        `fill` argument to change how missing calls are represented.

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 2], [1, 1]],
        ...                          [[2, 2], [-1, -1]]])
        >>> g.to_n_ref()
        array([[2, 1],
               [1, 0],
               [0, 0]], dtype=int8)
        >>> g.to_n_ref(fill=-1)
        array([[ 2,  1],
               [ 1,  0],
               [ 0, -1]], dtype=int8)
        >>> v = g[:, 0]
        >>> v
        <GenotypeVector shape=(3, 2) dtype=int64>
        0/0 0/2 2/2
        >>> v.to_n_ref()
        array([2, 1, 0], dtype=int8)

        """

        # count number of alternate alleles
        out = np.empty(self.shape[:-1], dtype=dtype)
        np.sum(self.values == 0, axis=-1, out=out)

        # fill missing calls
        if fill != 0:
            m = self.is_missing()
            out[m] = fill

        # handle mask
        if self.mask is not None:
            out[self.mask] = fill

        return out

    def to_n_alt(self, fill=0, dtype='i1'):
        """Transform each genotype call into the number of
        non-reference alleles.

        Parameters
        ----------
        fill : int, optional
            Use this value to represent missing calls.
        dtype : dtype, optional
            Output dtype.

        Returns
        -------
        out : ndarray, int, shape (n_variants, n_samples)
            Array of non-ref alleles per genotype call.

        Notes
        -----
        This function simply counts the number of non-reference
        alleles, it makes no distinction between different alternate
        alleles.

        By default this function returns 0 for missing genotype calls
        **and** for homozygous reference genotype calls. Use the
        `fill` argument to change how missing calls are represented.

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 2], [1, 1]],
        ...                          [[2, 2], [-1, -1]]])
        >>> g.to_n_alt()
        array([[0, 1],
               [1, 2],
               [2, 0]], dtype=int8)
        >>> g.to_n_alt(fill=-1)
        array([[ 0,  1],
               [ 1,  2],
               [ 2, -1]], dtype=int8)
        >>> v = g[:, 0]
        >>> v
        <GenotypeVector shape=(3, 2) dtype=int64>
        0/0 0/2 2/2
        >>> v.to_n_alt()
        array([0, 1, 2], dtype=int8)

        """

        # count number of alternate alleles
        out = np.empty(self.shape[:-1], dtype=dtype)
        np.sum(self.values > 0, axis=-1, out=out)

        # fill missing calls
        if fill != 0:
            m = self.is_missing()
            out[m] = fill

        # handle mask
        if self.mask is not None:
            out[self.mask] = fill

        return out

    def to_allele_counts(self, max_allele=None):
        """Transform genotype calls into allele counts per call.

        Parameters
        ----------
        max_allele : int, optional
            Highest allele index. Provide this value to speed up computation.
        dtype : dtype, optional
            Output dtype.

        Returns
        -------
        out : ndarray, uint8, shape (n_variants, n_samples, len(alleles))
            Array of allele counts per call.

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 2], [1, 1]],
        ...                          [[2, 2], [-1, -1]]])
        >>> g.to_allele_counts()
        <GenotypeAlleleCountsArray shape=(3, 2, 3) dtype=uint8>
        2:0:0 1:1:0
        1:0:1 0:2:0
        0:0:2 0:0:0
        >>> v = g[:, 0]
        >>> v
        <GenotypeVector shape=(3, 2) dtype=int64>
        0/0 0/2 2/2
        >>> v.to_allele_counts()
        <GenotypeAlleleCountsVector shape=(3, 3) dtype=uint8>
        2:0:0 1:0:1 0:0:2

        """

        # determine alleles to count
        if max_allele is None:
            max_allele = self.max()

        # use optimisations
        values = memoryview_safe(self.values)
        mask = memoryview_safe(self.mask).view(dtype='u1') if self.mask is not None else None

        # handle genotype vector case as an array of one genotype
        if values.ndim == 2:
            is_vector = True
            values = np.expand_dims(values, 1)
            if mask is not None:
                mask = np.expand_dims(mask, 1)
        else:
            is_vector = False

        if mask is None:
            out = genotype_array_to_allele_counts(values, max_allele)
        else:
            out = genotype_array_to_allele_counts_masked(values, mask, max_allele)

        if is_vector:
            out = np.squeeze(out, axis=1)
            out = GenotypeAlleleCountsVector(out)
        else:
            out = GenotypeAlleleCountsArray(out)

        return out

    def to_gt(self, max_allele=None):
        """Convert genotype calls to VCF-style string representation.

        Returns
        -------
        gt : ndarray, string, shape (n_variants, n_samples)

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 2], [1, 1]],
        ...                          [[1, 2], [2, 1]],
        ...                          [[2, 2], [-1, -1]]])
        >>> g.to_gt()
        chararray([[b'0/0', b'0/1'],
               [b'0/2', b'1/1'],
               [b'1/2', b'2/1'],
               [b'2/2', b'./.']],
              dtype='|S3')
        >>> v = g[:, 0]
        >>> v
        <GenotypeVector shape=(4, 2) dtype=int64>
        0/0 0/2 1/2 2/2
        >>> v.to_gt()
        chararray([b'0/0', b'0/2', b'1/2', b'2/2'],
              dtype='|S3')
        >>> g.is_phased = np.ones(g.shape[:-1])
        >>> g.to_gt()
        chararray([[b'0|0', b'0|1'],
               [b'0|2', b'1|1'],
               [b'1|2', b'2|1'],
               [b'2|2', b'.|.']],
              dtype='|S3')
        >>> v = g[:, 0]
        >>> v
        <GenotypeVector shape=(4, 2) dtype=int64>
        0|0 0|2 1|2 2|2
        >>> v.to_gt()
        chararray([b'0|0', b'0|2', b'1|2', b'2|2'],
              dtype='|S3')

        """

        # how many characters needed per allele call?
        if max_allele is None:
            max_allele = np.max(self)
        if max_allele <= 0:
            max_allele = 1
        nchar = int(np.floor(np.log10(max_allele))) + 1

        # convert to string
        a = self.astype((np.string_, nchar)).view(np.chararray)

        # recode missing alleles
        a[self < 0] = b'.'
        if self.mask is not None:
            a[self.mask] = b'.'

        # determine allele call separator
        if self.is_phased is None:
            sep = b'/'
        else:
            sep = np.empty(self.shape[:-1], dtype='S1').view(np.chararray)
            sep[self.is_phased] = b'|'
            sep[~self.is_phased] = b'/'

        # join via separator, coping with any ploidy
        gt = a[..., 0]
        for i in range(1, self.ploidy):
            gt = gt + sep + a[..., i]

        return gt

    def copy(self, *args, **kwargs):
        data = self.values.copy(*args, **kwargs)
        out = type(self)(data)
        if self.mask is not None:
            out.mask = self.mask.copy()
        if self.is_phased is not None:
            out.is_phased = self.is_phased.copy()
        return out

    def map_alleles(self, mapping, copy=True):
        """Transform alleles via a mapping.

        Parameters
        ----------
        mapping : ndarray, int8, shape (n_variants, max_allele)
            An array defining the allele mapping for each variant.
        copy : bool, optional
            If True, return a new array; if False, apply mapping in place
            (only applies for arrays with dtype int8; all other dtypes
            require a copy).

        Returns
        -------
        gm : GenotypeArray

        Examples
        --------

        >>> import allel
        >>> import numpy as np
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 2], [1, 1]],
        ...                          [[1, 2], [2, 1]],
        ...                          [[2, 2], [-1, -1]]], dtype='i1')
        >>> mapping = np.array([[1, 2, 0],
        ...                     [2, 0, 1],
        ...                     [2, 1, 0],
        ...                     [0, 2, 1]], dtype='i1')
        >>> g.map_alleles(mapping)
        <GenotypeArray shape=(4, 2, 2) dtype=int8>
        1/1 1/2
        2/1 0/0
        1/0 0/1
        1/1 ./.
        >>> v = g[:, 0]
        >>> v
        <GenotypeVector shape=(4, 2) dtype=int8>
        0/0 0/2 1/2 2/2
        >>> v.map_alleles(mapping)
        <GenotypeVector shape=(4, 2) dtype=int8>
        1/1 2/1 1/0 1/1

        Notes
        -----
        If a mask has been set, it is ignored by this function.

        For arrays with dtype int8 an optimised implementation is used which is
        faster and uses far less memory. It is recommended to convert arrays to
        dtype int8 where possible before calling this method.

        See Also
        --------
        create_allele_mapping

        """

        h = self.to_haplotypes()
        hm = h.map_alleles(mapping, copy=copy)
        if self.ndim == 2:
            gm = GenotypeVector(hm)
        else:
            gm = hm.to_genotypes(ploidy=self.ploidy)
        return gm

    def to_haplotypes(self, copy=False):
        """Reshape a genotype array to view it as haplotypes.

        Parameters
        ----------
        copy : bool, optional
            If True, copy data.

        Returns
        -------
        h : HaplotypeArray, shape (n_variants, n_samples * ploidy)
            Haplotype array.

        Notes
        -----
        If genotype calls are unphased, the haplotypes returned by
        this function will bear no resemblance to the true haplotypes.

        Examples
        --------
        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 1], [1, 1]],
        ...                          [[0, 2], [-1, -1]]])
        >>> g.to_haplotypes()
        <HaplotypeArray shape=(3, 4) dtype=int64>
        0 0 0 1
        0 1 1 1
        0 2 . .
        >>> v = g[:, 0]
        >>> v
        <GenotypeVector shape=(3, 2) dtype=int64>
        0/0 0/1 0/2
        >>> v.to_haplotypes()
        <HaplotypeArray shape=(3, 2) dtype=int64>
        0 0
        0 1
        0 2

        """
        # implement in sub-class
        raise NotImplementedError

    def compress(self, condition, axis=0, out=None):
        """Return selected slices of an array along given axis.

        Parameters
        ----------
        condition : array_like, bool
            Array that selects which entries to return. N.B., if len(condition)
            is less than the size of the given axis, then output is truncated to the length
            of the condition array.
        axis : int, optional
            Axis along which to take slices. If None, work on the flattened array.
        out : ndarray, optional
            Output array.  Its type is preserved and it must be of the right
            shape to hold the output.

        Returns
        -------
        out : Genotypes
            A copy of the array without the slices along axis for which `condition`
            is false.

        Examples
        --------
        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 1], [1, 1]],
        ...                          [[0, 2], [-1, -1]]])
        >>> g.compress([True, False, True], axis=0)
        <GenotypeArray shape=(2, 2, 2) dtype=int64>
        0/0 0/1
        0/2 ./.
        >>> v = g[:, 0]
        >>> v
        <GenotypeVector shape=(3, 2) dtype=int64>
        0/0 0/1 0/2
        >>> v.compress([True, False, True], axis=0)
        <GenotypeVector shape=(2, 2) dtype=int64>
        0/0 0/2

        """
        # implement in sub-class
        raise NotImplementedError

    def take(self, indices, axis=0, out=None, mode='raise'):
        """Take elements from an array along an axis.

        This function does the same thing as "fancy" indexing (indexing arrays
        using arrays); however, it can be easier to use if you need elements
        along a given axis.

        Parameters
        ----------
        indices : array_like
            The indices of the values to extract.
        axis : int, optional
            The axis over which to select values.
        out : ndarray, optional
            If provided, the result will be placed in this array. It should
            be of the appropriate shape and dtype.
        mode : {'raise', 'wrap', 'clip'}, optional
            Specifies how out-of-bounds indices will behave.

            * 'raise' -- raise an error (default)
            * 'wrap' -- wrap around
            * 'clip' -- clip to the range

            'clip' mode means that all indices that are too large are replaced
            by the index that addresses the last element along that axis. Note
            that this disables indexing with negative numbers.

        Returns
        -------
        subarray : ndarray

        Examples
        --------
        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 1], [1, 1]],
        ...                          [[0, 2], [-1, -1]]])
        >>> g.take([0, 2], axis=0)
        <GenotypeArray shape=(2, 2, 2) dtype=int64>
        0/0 0/1
        0/2 ./.
        >>> v = g[:, 0]
        >>> v
        <GenotypeVector shape=(3, 2) dtype=int64>
        0/0 0/1 0/2
        >>> v.take([0, 2], axis=0)
        <GenotypeVector shape=(2, 2) dtype=int64>
        0/0 0/2

        """
        # implement in sub-class
        raise NotImplementedError

    def concatenate(self, others, axis=0):
        """Join a sequence of arrays along an existing axis.

        Parameters
        ----------
        others : sequence of array_like
            The arrays must have the same shape, except in the dimension
            corresponding to `axis` (the first, by default).
        axis : int, optional
            The axis along which the arrays will be joined.  Default is 0.

        Returns
        -------
        res : ndarray
            The concatenated array.

        Examples
        --------
        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 1], [1, 1]],
        ...                          [[0, 2], [-1, -1]]])
        >>> g.concatenate([g], axis=0)
        <GenotypeArray shape=(6, 2, 2) dtype=int64>
        0/0 0/1
        0/1 1/1
        0/2 ./.
        0/0 0/1
        0/1 1/1
        0/2 ./.
        >>> g.concatenate([g], axis=1)
        <GenotypeArray shape=(3, 4, 2) dtype=int64>
        0/0 0/1 0/0 0/1
        0/1 1/1 0/1 1/1
        0/2 ./. 0/2 ./.
        >>> v1 = g[:, 0]
        >>> v2 = g[:, 1]
        >>> v1.concatenate([v2], axis=0)
        <GenotypeVector shape=(6, 2) dtype=int64>
        0/0 0/1 0/2 0/1 1/1 ./.

        """
        # implement in sub-class
        raise NotImplementedError


class GenotypeVector(Genotypes, DisplayAs1D):
    """Array of genotype calls for a sequence of variants or samples.

    Parameters
    ----------
    data : array_like, int, shape (n_variants, n_samples, ploidy)
        Genotype data.
    copy : bool, optional
        If True, make a copy of `data`.
    **kwargs : keyword arguments
        All keyword arguments are passed through to :func:`numpy.array`.

    See Also
    --------
    Genotypes, GenotypeArray, HaplotypeArray, GenotypeAlleleCountsArray

    """

    def __init__(self, data, copy=False, **kwargs):
        super(GenotypeVector, self).__init__(data, copy=copy, **kwargs)
        check_ndim(self.values, 2)

    def __getitem__(self, item):
        return index_genotype_vector(self, item, type(self))

    def compress(self, condition, axis=0, out=None):
        return compress_genotypes(self, condition=condition, axis=axis, wrap_axes={0},
                                  cls=type(self), compress=np.compress, out=out)

    def take(self, indices, axis=0, out=None, mode='raise'):
        return take_genotypes(self, indices=indices, axis=axis, wrap_axes={0}, cls=type(self),
                              take=np.take, out=out, mode=mode)

    def concatenate(self, others, axis=0):
        return concatenate_genotypes(self, others=others, axis=axis, wrap_axes={0},
                                     cls=type(self), concatenate=np.concatenate)

    def to_haplotypes(self, copy=False):
        return HaplotypeArray(self.values, copy=copy)

    def str_items(self):
        gt = self.to_gt()
        out = [str(x, 'ascii') for x in gt]
        return out

    def to_str(self, threshold=10, edgeitems=5):
        _, items = self.get_display_items(threshold, edgeitems)
        s = ' '.join(items)
        return s


copy_method_doc(GenotypeVector.compress, Genotypes.compress)
copy_method_doc(GenotypeVector.take, Genotypes.take)
copy_method_doc(GenotypeVector.concatenate, Genotypes.concatenate)
copy_method_doc(GenotypeVector.to_haplotypes, Genotypes.to_haplotypes)


def _normalize_subpop_arg(subpop, n):
    subpop = asarray_ndim(subpop, 1, allow_none=True, dtype=np.int64)
    if subpop is not None:
        if np.any(subpop >= n):
            raise ValueError('index out of bounds')
        if np.any(subpop < 0):
            raise ValueError('negative indices not supported')
        subpop = memoryview_safe(subpop)
    return subpop


class GenotypeArray(Genotypes, DisplayAs2D):
    """Array of discrete genotype calls for a matrix of variants and samples.

    Parameters
    ----------
    data : array_like, int, shape (n_variants, n_samples, ploidy)
        Genotype data.
    copy : bool, optional
        If True, make a copy of `data`.
    **kwargs : keyword arguments
        All keyword arguments are passed through to :func:`numpy.array`.

    Notes
    -----
    This class represents data on discrete genotype calls as a
    3-dimensional numpy array of integers. By convention, the first
    dimension corresponds to the variants genotyped, the second
    dimension corresponds to the samples genotyped, and the third
    dimension corresponds to the ploidy of the samples.

    Each integer within the array corresponds to an **allele index**,
    where 0 is the reference allele, 1 is the first alternate allele,
    2 is the second alternate allele, ... and -1 (or any other
    negative integer) is a missing allele call. A single byte integer
    dtype (int8) can represent up to 127 distinct alleles, which is
    usually sufficient.  The actual alleles (i.e., the alternate
    nucleotide sequences) and the physical positions of the variants
    within the genome of an organism are stored in separate arrays,
    discussed elsewhere.

    Arrays of this class can store either **phased or unphased**
    genotype calls. If the genotypes are phased (i.e., haplotypes have
    been resolved) then individual haplotypes can be extracted by
    converting to a :class:`HaplotypeArray` then indexing the second
    dimension. If the genotype calls are unphased then the ordering of
    alleles along the third (ploidy) dimension is arbitrary. N.B.,
    this means that an unphased diploid heterozygous call could be
    stored as (0, 1) or equivalently as (1, 0).

    A genotype array can store genotype calls with any ploidy > 1. For
    haploid calls, use a :class:`HaplotypeArray`. Note that genotype
    arrays are not capable of storing calls for samples with differing
    or variable ploidy, see :class:`GenotypeAlleleCountsArray` instead.

    With genotype data on large numbers of variants and/or samples,
    storing the genotype calls in memory as an uncompressed numpy
    array if integers may be impractical. For working with large
    arrays of genotype data, see the :mod:`allel.model.chunked` and
    :mod:`allel.model.dask` modules.

    Examples
    --------

    Instantiate a genotype array::

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 1], [1, 1]],
        ...                          [[0, 2], [-1, -1]]], dtype='i1')
        >>> g.dtype
        dtype('int8')
        >>> g.ndim
        3
        >>> g.shape
        (3, 2, 2)
        >>> g.n_variants
        3
        >>> g.n_samples
        2
        >>> g.ploidy
        2
        >>> g
        <GenotypeArray shape=(3, 2, 2) dtype=int8>
        0/0 0/1
        0/1 1/1
        0/2 ./.

    Genotype calls for a single variant at all samples can be obtained
    by indexing the first dimension, e.g.::

        >>> g[1]
        <GenotypeVector shape=(2, 2) dtype=int8>
        0/1 1/1

    Genotype calls for a single sample at all variants can be obtained
    by indexing the second dimension, e.g.::

        >>> g[:, 1]
        <GenotypeVector shape=(3, 2) dtype=int8>
        0/1 1/1 ./.

    A genotype call for a single sample at a single variant can be
    obtained by indexing the first and second dimensions, e.g.::

        >>> g[1, 0]
        array([0, 1], dtype=int8)

    A genotype array can store polyploid calls, e.g.::

        >>> g = allel.GenotypeArray([[[0, 0, 0], [0, 0, 1]],
        ...                          [[0, 1, 1], [1, 1, 1]],
        ...                          [[0, 1, 2], [-1, -1, -1]]],
        ...                         dtype='i1')
        >>> g.ploidy
        3
        >>> g
        <GenotypeArray shape=(3, 2, 3) dtype=int8>
        0/0/0 0/0/1
        0/1/1 1/1/1
        0/1/2 ././.

    See Also
    --------
    Genotypes, GenotypeVector, HaplotypeArray, GenotypeAlleleCountsArray

    """

    def __init__(self, data, copy=False, **kwargs):
        super(GenotypeArray, self).__init__(data, copy=copy, **kwargs)
        check_ndim(self.values, 3)

    def __getitem__(self, item):
        return index_genotype_array(self, item, array_cls=type(self),
                                    vector_cls=GenotypeVector)

    @property
    def n_variants(self):
        """Number of variants."""
        return self.shape[0]

    @property
    def n_samples(self):
        """Number of samples."""
        return self.shape[1]

    def compress(self, condition, axis=0, out=None):
        return compress_genotypes(self, condition=condition, axis=axis, wrap_axes={0, 1},
                                  cls=type(self), compress=np.compress, out=out)

    def take(self, indices, axis=0, out=None, mode='raise'):
        return take_genotypes(self, indices=indices, axis=axis, wrap_axes={0, 1},
                              cls=type(self), take=np.take, out=out, mode=mode)

    def subset(self, sel0=None, sel1=None):
        """Make a sub-selection of variants and samples.

        Parameters
        ----------
        sel0 : array_like
            Boolean array or array of indices selecting variants.
        sel1 : array_like
            Boolean array or array of indices selecting samples.

        Returns
        -------
        out : GenotypeArray

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1], [1, 1]],
        ...                          [[0, 1], [1, 1], [1, 2]],
        ...                          [[0, 2], [-1, -1], [-1, -1]]])
        >>> g.subset([0, 1], [0, 2])
        <GenotypeArray shape=(2, 2, 2) dtype=int64>
        0/0 1/1
        0/1 1/2

        See Also
        --------
        Genotypes.take, Genotypes.compress

        """
        return subset_genotype_array(self, sel0, sel1, cls=type(self), subset=subset)

    def concatenate(self, others, axis=0):
        return concatenate_genotypes(self, others=others, axis=axis, wrap_axes={0, 1},
                                     cls=type(self), concatenate=np.concatenate)

    def to_haplotypes(self, copy=False):
        # reshape, preserving size of variants dimension
        newshape = (self.shape[0], -1)
        data = np.reshape(self, newshape)
        h = HaplotypeArray(data, copy=copy)
        return h

    def str_items(self):
        gt = self.to_gt()
        n = gt.dtype.itemsize
        out = [[str(x, 'ascii').rjust(n) for x in row] for row in gt]
        return out

    def to_packed(self, boundscheck=True):
        """Pack diploid genotypes into a single byte for each genotype,
        using the left-most 4 bits for the first allele and the right-most 4
        bits for the second allele. Allows single byte encoding of diploid
        genotypes for variants with up to 15 alleles.

        Parameters
        ----------
        boundscheck : bool, optional
            If False, do not check that minimum and maximum alleles are
            compatible with bit-packing.

        Returns
        -------
        packed : ndarray, uint8, shape (n_variants, n_samples)
            Bit-packed genotype array.

        Notes
        -----
        If a mask has been set, it is ignored by this function.

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 2], [1, 1]],
        ...                          [[2, 2], [-1, -1]]], dtype='i1')
        >>> g.to_packed()
        array([[  0,   1],
               [  2,  17],
               [ 34, 239]], dtype=uint8)

        """

        check_ploidy(self.ploidy, 2)

        if boundscheck:
            amx = self.max()
            if amx > 14:
                raise ValueError('max allele for packing is 14, found %s' % amx)
            amn = self.min()
            if amn < -1:
                raise ValueError('min allele for packing is -1, found %s' % amn)

        # pack data
        values = memoryview_safe(self.values)
        packed = genotype_array_pack_diploid(values)

        return packed

    @classmethod
    def from_packed(cls, packed):
        """Unpack diploid genotypes that have been bit-packed into single
        bytes.

        Parameters
        ----------
        packed : ndarray, uint8, shape (n_variants, n_samples)
            Bit-packed diploid genotype array.

        Returns
        -------
        g : GenotypeArray, shape (n_variants, n_samples, 2)
            Genotype array.

        Examples
        --------

        >>> import allel
        >>> import numpy as np
        >>> packed = np.array([[0, 1],
        ...                    [2, 17],
        ...                    [34, 239]], dtype='u1')
        >>> allel.GenotypeArray.from_packed(packed)
        <GenotypeArray shape=(3, 2, 2) dtype=int8>
        0/0 0/1
        0/2 1/1
        2/2 ./.

        """

        # check arguments
        packed = np.asarray(packed)
        check_ndim(packed, 2)
        check_dtype(packed, 'u1')
        packed = memoryview_safe(packed)

        data = genotype_array_unpack_diploid(packed)
        return cls(data)

    # noinspection PyShadowingBuiltins
    def to_sparse(self, format='csr', **kwargs):
        """Convert into a sparse matrix.

        Parameters
        ----------
        format : {'coo', 'csc', 'csr', 'dia', 'dok', 'lil'}
            Sparse matrix format.
        kwargs : keyword arguments
            Passed through to sparse matrix constructor.

        Returns
        -------
        m : scipy.sparse.spmatrix
            Sparse matrix

        Notes
        -----

        If a mask has been set, it is ignored by this function.

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 0]],
        ...                          [[0, 1], [0, 1]],
        ...                          [[1, 1], [0, 0]],
        ...                          [[0, 0], [-1, -1]]], dtype='i1')
        >>> m = g.to_sparse(format='csr')
        >>> m
        <4x4 sparse matrix of type '<class 'numpy.int8'>'
            with 6 stored elements in Compressed Sparse Row format>
        >>> m.data
        array([ 1,  1,  1,  1, -1, -1], dtype=int8)
        >>> m.indices
        array([1, 3, 0, 1, 2, 3], dtype=int32)
        >>> m.indptr
        array([0, 0, 2, 4, 6], dtype=int32)

        """

        h = self.to_haplotypes()
        m = h.to_sparse(format=format, **kwargs)
        return m

    @staticmethod
    def from_sparse(m, ploidy, order=None, out=None):
        """Construct a genotype array from a sparse matrix.

        Parameters
        ----------
        m : scipy.sparse.spmatrix
            Sparse matrix
        ploidy : int
            The sample ploidy.
        order : {'C', 'F'}, optional
            Whether to store data in C (row-major) or Fortran (column-major)
            order in memory.
        out : ndarray, shape (n_variants, n_samples), optional
            Use this array as the output buffer.

        Returns
        -------
        g : GenotypeArray, shape (n_variants, n_samples, ploidy)
            Genotype array.

        Examples
        --------

        >>> import allel
        >>> import numpy as np
        >>> import scipy.sparse
        >>> data = np.array([ 1,  1,  1,  1, -1, -1], dtype=np.int8)
        >>> indices = np.array([1, 3, 0, 1, 2, 3], dtype=np.int32)
        >>> indptr = np.array([0, 0, 2, 4, 6], dtype=np.int32)
        >>> m = scipy.sparse.csr_matrix((data, indices, indptr))
        >>> g = allel.GenotypeArray.from_sparse(m, ploidy=2)
        >>> g
        <GenotypeArray shape=(4, 2, 2) dtype=int8>
        0/0 0/0
        0/1 0/1
        1/1 0/0
        0/0 ./.

        """

        h = HaplotypeArray.from_sparse(m, order=order, out=out)
        g = h.to_genotypes(ploidy=ploidy)
        return g

    def haploidify_samples(self):
        """Construct a pseudo-haplotype for each sample by randomly
        selecting an allele from each genotype call.

        Returns
        -------
        h : HaplotypeArray

        Notes
        -----
        If a mask has been set, it is ignored by this function.

        Examples
        --------

        >>> import allel
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 2], [1, 1]],
        ...                          [[1, 2], [2, 1]],
        ...                          [[2, 2], [-1, -1]]])
        >>> g.haploidify_samples()
        <HaplotypeArray shape=(4, 2) dtype=int64>
        0 1
        0 1
        1 1
        2 .
        >>> g = allel.GenotypeArray([[[0, 0, 0], [0, 0, 1]],
        ...                          [[0, 1, 1], [1, 1, 1]],
        ...                          [[0, 1, 2], [-1, -1, -1]]])
        >>> g.haploidify_samples()
        <HaplotypeArray shape=(3, 2) dtype=int64>
        0 0
        1 1
        2 .

        """

        # N.B., this implementation is obscure and uses more memory than
        # necessary, TODO review

        # define the range of possible indices, e.g., diploid => (0, 1)
        index_range = np.arange(0, self.ploidy, dtype='u1')

        # create a random index for each genotype call
        indices = np.random.choice(index_range, size=self.n_calls, replace=True)

        # reshape genotype data so it's suitable for passing to np.choose
        # by merging the variants and samples dimensions
        choices = self.reshape(-1, self.ploidy).T

        # now use random indices to haploidify
        data = np.choose(indices, choices)

        # reshape the haploidified data to restore the variants and samples
        # dimensions
        data = data.reshape((self.n_variants, self.n_samples))

        # view as haplotype array
        h = HaplotypeArray(data, copy=False)

        return h

    def count_alleles(self, max_allele=None, subpop=None):
        """Count the number of calls of each allele per variant.

        Parameters
        ----------
        max_allele : int, optional
            The highest allele index to count. Alleles above this will be
            ignored.
        subpop : sequence of ints, optional
            Indices of samples to include in count.

        Returns
        -------
        ac : AlleleCountsArray

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 2], [1, 1]],
        ...                          [[2, 2], [-1, -1]]])
        >>> g.count_alleles()
        <AlleleCountsArray shape=(3, 3) dtype=int32>
        3 1 0
        1 2 1
        0 0 2
        >>> g.count_alleles(max_allele=1)
        <AlleleCountsArray shape=(3, 2) dtype=int32>
        3 1
        1 2
        0 0

        """

        # check inputs
        subpop = _normalize_subpop_arg(subpop, self.shape[1])

        # determine alleles to count
        if max_allele is None:
            max_allele = self.max()

        # use optimisations
        values = memoryview_safe(self.values)
        mask = memoryview_safe(self.mask).view(dtype='u1') if self.mask is not None else None
        if subpop is None and mask is None:
            ac = genotype_array_count_alleles(values, max_allele)
        elif subpop is None:
            ac = genotype_array_count_alleles_masked(values, mask, max_allele)
        elif mask is None:
            ac = genotype_array_count_alleles_subpop(values, max_allele, subpop)
        else:
            ac = genotype_array_count_alleles_subpop_masked(values, mask, max_allele, subpop)

        return AlleleCountsArray(ac, copy=False)

    def count_alleles_subpops(self, subpops, max_allele=None):
        """Count alleles for multiple subpopulations simultaneously.

        Parameters
        ----------
        subpops : dict (string -> sequence of ints)
            Mapping of subpopulation names to sample indices.
        max_allele : int, optional
            The highest allele index to count. Alleles above this will be
            ignored.

        Returns
        -------
        out : dict (string -> AlleleCountsArray)
            A mapping of subpopulation names to allele counts arrays.

        """

        if max_allele is None:
            max_allele = self.max()

        out = {name: self.count_alleles(max_allele=max_allele, subpop=subpop)
               for name, subpop in subpops.items()}

        return out


copy_method_doc(GenotypeArray.compress, Genotypes.compress)
copy_method_doc(GenotypeArray.take, Genotypes.take)
copy_method_doc(GenotypeArray.concatenate, Genotypes.concatenate)
copy_method_doc(GenotypeArray.to_haplotypes, Genotypes.to_haplotypes)


class HaplotypeArray(NumpyArrayWrapper, DisplayAs2D):
    """Array of haplotypes.

    Parameters
    ----------
    data : array_like, int, shape (n_variants, n_haplotypes)
        Haplotype data.
    **kwargs : keyword arguments
        All keyword arguments are passed through to :func:`numpy.array`.

    Notes
    -----
    This class represents haplotype data as a 2-dimensional numpy
    array of integers. By convention the first dimension corresponds
    to the variants genotyped, the second dimension corresponds to the
    haplotypes.

    Each integer within the array corresponds to an **allele index**,
    where 0 is the reference allele, 1 is the first alternate allele,
    2 is the second alternate allele, ... and -1 (or any other
    negative integer) is a missing allele call.

    If adjacent haplotypes originate from the same sample, then a
    haplotype array can also be viewed as a genotype array. However,
    this is not a requirement.

    With data on large numbers of variants and/or haplotypes,
    storing the data in memory as an uncompressed numpy
    array if integers may be impractical. For working with large
    arrays of haplotype data, see the :mod:`allel.model.chunked` and
    :mod:`allel.model.dask` modules.

    Examples
    --------

    Instantiate a haplotype array::

        >>> import allel
        >>> h = allel.HaplotypeArray([[0, 0, 0, 1],
        ...                           [0, 1, 1, 1],
        ...                           [0, 2, -1, -1]], dtype='i1')
        >>> h.dtype
        dtype('int8')
        >>> h.ndim
        2
        >>> h.shape
        (3, 4)
        >>> h.n_variants
        3
        >>> h.n_haplotypes
        4
        >>> h
        <HaplotypeArray shape=(3, 4) dtype=int8>
        0 0 0 1
        0 1 1 1
        0 2 . .

    Allele calls for a single variant at all haplotypes can be obtained
    by indexing the first dimension, e.g.::

        >>> h[1]
        array([0, 1, 1, 1], dtype=int8)

    A single haplotype can be obtained by indexing the second
    dimension, e.g.::

        >>> h[:, 1]
        array([0, 1, 2], dtype=int8)

    An allele call for a single haplotype at a single variant can be
    obtained by indexing the first and second dimensions, e.g.::

        >>> h[1, 0]
        0

    View haplotypes as diploid genotypes::

        >>> h.to_genotypes(ploidy=2)
        <GenotypeArray shape=(3, 2, 2) dtype=int8>
        0/0 0/1
        0/1 1/1
        0/2 ./.

    See Also
    --------
    Genotypes, GenotypeArray, GenotypeVector

    """

    def __init__(self, data, copy=False, **kwargs):
        super(HaplotypeArray, self).__init__(data, copy=copy, **kwargs)
        check_integer_dtype(self.values)
        check_ndim(self.values, 2)

    @property
    def n_variants(self):
        """Number of variants."""
        return self.shape[0]

    @property
    def n_haplotypes(self):
        """Number of haplotypes."""
        return self.shape[1]

    def __getitem__(self, item):
        return index_haplotype_array(self, item, type(self))

    def compress(self, condition, axis=0, out=None):
        """Return selected slices of an array along given axis.

        Parameters
        ----------
        condition : array_like, bool
            Array that selects which entries to return. N.B., if len(condition)
            is less than the size of the given axis, then output is truncated to the length
            of the condition array.
        axis : int, optional
            Axis along which to take slices. If None, work on the flattened array.
        out : ndarray, optional
            Output array.  Its type is preserved and it must be of the right
            shape to hold the output.

        Returns
        -------
        out : HaplotypeArray
            A copy of the array without the slices along axis for which `condition`
            is false.

        Examples
        --------
        >>> import allel
        >>> h = allel.HaplotypeArray([[0, 0, 0, 1],
        ...                           [0, 1, 1, 1],
        ...                           [0, 2, -1, -1]], dtype='i1')
        >>> h.compress([True, False, True], axis=0)
        <HaplotypeArray shape=(2, 4) dtype=int8>
        0 0 0 1
        0 2 . .
        >>> h.compress([True, False, True, False], axis=1)
        <HaplotypeArray shape=(3, 2) dtype=int8>
        0 0
        0 1
        0 .

        """
        return compress_haplotype_array(self, condition, axis=axis, cls=type(self),
                                        compress=np.compress, out=out)

    def take(self, indices, axis=0, out=None, mode='raise'):
        """Take elements from an array along an axis.

        This function does the same thing as "fancy" indexing (indexing arrays
        using arrays); however, it can be easier to use if you need elements
        along a given axis.

        Parameters
        ----------
        indices : array_like
            The indices of the values to extract.
        axis : int, optional
            The axis over which to select values.
        out : ndarray, optional
            If provided, the result will be placed in this array. It should
            be of the appropriate shape and dtype.
        mode : {'raise', 'wrap', 'clip'}, optional
            Specifies how out-of-bounds indices will behave.

            * 'raise' -- raise an error (default)
            * 'wrap' -- wrap around
            * 'clip' -- clip to the range

            'clip' mode means that all indices that are too large are replaced
            by the index that addresses the last element along that axis. Note
            that this disables indexing with negative numbers.

        Returns
        -------
        subarray : ndarray

        Examples
        --------
        >>> import allel
        >>> h = allel.HaplotypeArray([[0, 0, 0, 1],
        ...                           [0, 1, 1, 1],
        ...                           [0, 2, -1, -1]], dtype='i1')
        >>> h.take([0, 2], axis=0)
        <HaplotypeArray shape=(2, 4) dtype=int8>
        0 0 0 1
        0 2 . .
        >>> h.take([0, 2], axis=1)
        <HaplotypeArray shape=(3, 2) dtype=int8>
        0 0
        0 1
        0 .

        """

        return take_haplotype_array(self, indices, axis=axis, cls=type(self), take=np.take,
                                    out=out, mode=mode)

    def subset(self, sel0=None, sel1=None):
        """Make a sub-selection of variants and haplotypes.

        Parameters
        ----------
        sel0 : array_like
            Boolean array or array of indices selecting variants.
        sel1 : array_like
            Boolean array or array of indices selecting haplotypes.

        Returns
        -------
        out : HaplotypeArray

        See Also
        --------
        HaplotypeArray.take, HaplotypeArray.compress

        """
        return subset_haplotype_array(self, sel0, sel1, cls=type(self), subset=subset)

    def concatenate(self, others, axis=0):
        """Join a sequence of arrays along an existing axis.

        Parameters
        ----------
        others : sequence of array_like
            The arrays must have the same shape, except in the dimension
            corresponding to `axis` (the first, by default).
        axis : int, optional
            The axis along which the arrays will be joined.  Default is 0.

        Returns
        -------
        res : ndarray
            The concatenated array.

        Examples
        --------
        >>> import allel
        >>> h = allel.HaplotypeArray([[0, 0, 0, 1],
        ...                           [0, 1, 1, 1],
        ...                           [0, 2, -1, -1]], dtype='i1')
        >>> h.concatenate([h], axis=0)
        <HaplotypeArray shape=(6, 4) dtype=int8>
        0 0 0 1
        0 1 1 1
        0 2 . .
        0 0 0 1
        0 1 1 1
        0 2 . .
        >>> h.concatenate([h], axis=1)
        <HaplotypeArray shape=(3, 8) dtype=int8>
        0 0 0 1 0 0 0 1
        0 1 1 1 0 1 1 1
        0 2 . . 0 2 . .

        """
        return concatenate_haplotype_array(self, others, axis=axis, cls=type(self),
                                           concatenate=np.concatenate)

    def str_items(self):
        values = self.values
        max_allele = np.max(values)
        if max_allele <= 0:
            max_allele = 1
        n = int(np.floor(np.log10(max_allele))) + 1
        t = values.astype((np.string_, n))
        # recode missing alleles
        t[values < 0] = b'.'
        out = [[str(x, 'ascii').rjust(n) for x in row] for row in t]
        return out

    def is_called(self):
        return self.values >= 0

    def is_missing(self):
        return self.values < 0

    def is_ref(self):
        return self.values == 0

    def is_alt(self, allele=None):
        if allele is None:
            return self.values > 0
        else:
            return self.values == allele

    def is_call(self, allele):
        return self.values == allele

    def count_called(self, axis=None):
        b = self.is_called()
        return np.sum(b, axis=axis)

    def count_missing(self, axis=None):
        b = self.is_missing()
        return np.sum(b, axis=axis)

    def count_ref(self, axis=None):
        b = self.is_ref()
        return np.sum(b, axis=axis)

    def count_alt(self, axis=None):
        b = self.is_alt()
        return np.sum(b, axis=axis)

    def count_call(self, allele, axis=None):
        b = self.is_call(allele=allele)
        return np.sum(b, axis=axis)

    def to_genotypes(self, ploidy, copy=False):
        """Reshape a haplotype array to view it as genotypes by restoring the
        ploidy dimension.

        Parameters
        ----------
        ploidy : int
            The sample ploidy.
        copy : bool, optional
            If True, make a copy of data.

        Returns
        -------
        g : ndarray, int, shape (n_variants, n_samples, ploidy)
            Genotype array (sharing same underlying buffer).
        copy : bool, optional
            If True, copy the data.

        Examples
        --------

        >>> import allel
        >>> h = allel.HaplotypeArray([[0, 0, 0, 1],
        ...                           [0, 1, 1, 1],
        ...                           [0, 2, -1, -1]], dtype='i1')
        >>> h.to_genotypes(ploidy=2)
        <GenotypeArray shape=(3, 2, 2) dtype=int8>
        0/0 0/1
        0/1 1/1
        0/2 ./.

        """

        # check ploidy is compatible
        if (self.shape[1] % ploidy) > 0:
            raise ValueError('incompatible ploidy')

        # reshape
        newshape = (self.shape[0], -1, ploidy)
        data = self.reshape(newshape)

        # wrap
        g = GenotypeArray(data, copy=copy)

        return g

    # noinspection PyShadowingBuiltins
    def to_sparse(self, format='csr', **kwargs):
        """Convert into a sparse matrix.

        Parameters
        ----------
        format : {'coo', 'csc', 'csr', 'dia', 'dok', 'lil'}
            Sparse matrix format.
        kwargs : keyword arguments
            Passed through to sparse matrix constructor.

        Returns
        -------
        m : scipy.sparse.spmatrix
            Sparse matrix

        Examples
        --------

        >>> import allel
        >>> h = allel.HaplotypeArray([[0, 0, 0, 0],
        ...                           [0, 1, 0, 1],
        ...                           [1, 1, 0, 0],
        ...                           [0, 0, -1, -1]], dtype='i1')
        >>> m = h.to_sparse(format='csr')
        >>> m
        <4x4 sparse matrix of type '<class 'numpy.int8'>'
            with 6 stored elements in Compressed Sparse Row format>
        >>> m.data
        array([ 1,  1,  1,  1, -1, -1], dtype=int8)
        >>> m.indices
        array([1, 3, 0, 1, 2, 3], dtype=int32)
        >>> m.indptr
        array([0, 0, 2, 4, 6], dtype=int32)

        """

        import scipy.sparse

        # check arguments
        f = {
            'bsr': scipy.sparse.bsr_matrix,
            'coo': scipy.sparse.coo_matrix,
            'csc': scipy.sparse.csc_matrix,
            'csr': scipy.sparse.csr_matrix,
            'dia': scipy.sparse.dia_matrix,
            'dok': scipy.sparse.dok_matrix,
            'lil': scipy.sparse.lil_matrix
        }
        if format not in f:
            raise ValueError('invalid format: %r' % format)

        # create sparse matrix
        m = f[format](self, **kwargs)

        return m

    @staticmethod
    def from_sparse(m, order=None, out=None):
        """Construct a haplotype array from a sparse matrix.

        Parameters
        ----------
        m : scipy.sparse.spmatrix
            Sparse matrix
        order : {'C', 'F'}, optional
            Whether to store data in C (row-major) or Fortran (column-major)
            order in memory.
        out : ndarray, shape (n_variants, n_samples), optional
            Use this array as the output buffer.

        Returns
        -------
        h : HaplotypeArray, shape (n_variants, n_haplotypes)
            Haplotype array.

        Examples
        --------

        >>> import allel
        >>> import numpy as np
        >>> import scipy.sparse
        >>> data = np.array([ 1,  1,  1,  1, -1, -1], dtype=np.int8)
        >>> indices = np.array([1, 3, 0, 1, 2, 3], dtype=np.int32)
        >>> indptr = np.array([0, 0, 2, 4, 6], dtype=np.int32)
        >>> m = scipy.sparse.csr_matrix((data, indices, indptr))
        >>> h = allel.HaplotypeArray.from_sparse(m)
        >>> h
        <HaplotypeArray shape=(4, 4) dtype=int8>
        0 0 0 0
        0 1 0 1
        1 1 0 0
        0 0 . .

        """

        import scipy.sparse

        # check arguments
        if not scipy.sparse.isspmatrix(m):
            raise ValueError('not a sparse matrix: %r' % m)

        # convert to dense array
        data = m.toarray(order=order, out=out)

        # wrap
        h = HaplotypeArray(data)

        return h

    def count_alleles(self, max_allele=None, subpop=None):
        """Count the number of calls of each allele per variant.

        Parameters
        ----------
        max_allele : int, optional
            The highest allele index to count. Alleles greater than this
            index will be ignored.
        subpop : array_like, int, optional
            Indices of haplotypes to include.

        Returns
        -------
        ac : AlleleCountsArray, int, shape (n_variants, n_alleles)

        Examples
        --------

        >>> import allel
        >>> h = allel.HaplotypeArray([[0, 0, 0, 1],
        ...                           [0, 1, 1, 1],
        ...                           [0, 2, -1, -1]], dtype='i1')
        >>> ac = h.count_alleles()
        >>> ac
        <AlleleCountsArray shape=(3, 3) dtype=int32>
        3 1 0
        1 3 0
        1 0 1

        """

        # check inputs
        subpop = _normalize_subpop_arg(subpop, self.shape[1])

        # determine alleles to count
        if max_allele is None:
            max_allele = self.max()

        # use optimisations
        values = memoryview_safe(self.values)
        if subpop is None:
            ac = haplotype_array_count_alleles(values, max_allele)

        else:
            ac = haplotype_array_count_alleles_subpop(values, max_allele, subpop)

        return AlleleCountsArray(ac, copy=False)

    def count_alleles_subpops(self, subpops, max_allele=None):
        """Count alleles for multiple subpopulations simultaneously.

        Parameters
        ----------
        subpops : dict (string -> sequence of ints)
            Mapping of subpopulation names to sample indices.
        max_allele : int, optional
            The highest allele index to count. Alleles above this will be
            ignored.

        Returns
        -------
        out : dict (string -> AlleleCountsArray)
            A mapping of subpopulation names to allele counts arrays.

        """

        if max_allele is None:
            max_allele = self.max()

        out = {name: self.count_alleles(max_allele=max_allele, subpop=subpop)
               for name, subpop in subpops.items()}

        return out

    def map_alleles(self, mapping, copy=True):
        """Transform alleles via a mapping.

        Parameters
        ----------
        mapping : ndarray, int8, shape (n_variants, max_allele)
            An array defining the allele mapping for each variant.
        copy : bool, optional
            If True, return a new array; if False, apply mapping in place
            (only applies for arrays with dtype int8; all other dtypes
            require a copy).

        Returns
        -------
        hm : HaplotypeArray

        Examples
        --------
        >>> import allel
        >>> import numpy as np
        >>> h = allel.HaplotypeArray([[0, 0, 0, 1],
        ...                           [0, 1, 1, 1],
        ...                           [0, 2, -1, -1]], dtype='i1')
        >>> mapping = np.array([[1, 2, 0],
        ...                     [2, 0, 1],
        ...                     [2, 1, 0]], dtype='i1')
        >>> h.map_alleles(mapping)
        <HaplotypeArray shape=(3, 4) dtype=int8>
        1 1 1 2
        2 0 0 0
        2 0 . .

        Notes
        -----
        For arrays with dtype int8 an optimised implementation is used which is
        faster and uses far less memory. It is recommended to convert arrays to
        dtype int8 where possible before calling this method.

        See Also
        --------
        allel.model.util.create_allele_mapping

        """

        # check inputs
        mapping = asarray_ndim(mapping, 2)
        check_dim0_aligned(self, mapping)

        # use optimisation
        mapping = np.asarray(mapping, dtype=self.dtype)
        mapping = memoryview_safe(mapping)
        values = memoryview_safe(self.values)
        data = haplotype_array_map_alleles(values, mapping, copy=copy)

        return HaplotypeArray(data, copy=False)

    def prefix_argsort(self):
        """Return indices that would sort the haplotypes by prefix."""
        return np.lexsort(self.values[::-1])

    def distinct(self):
        """Return sets of indices for each distinct haplotype."""

        # setup collection
        d = collections.defaultdict(set)

        # iterate over haplotypes
        for i in range(self.shape[1]):

            # hash the haplotype
            k = hash(self.values[:, i].tobytes())

            # collect
            d[k].add(i)

        # extract sets, sorted by most common
        return sorted(d.values(), key=len, reverse=True)

    def distinct_counts(self):
        """Return counts for each distinct haplotype."""

        # hash the haplotypes
        k = [hash(self.values[:, i].tobytes()) for i in range(self.shape[1])]

        # count and sort
        # noinspection PyArgumentList
        counts = sorted(collections.Counter(k).values(), reverse=True)

        return np.asarray(counts)

    def distinct_frequencies(self):
        """Return frequencies for each distinct haplotype."""

        c = self.distinct_counts()
        n = self.shape[1]
        return c / n


class AlleleCountsArray(NumpyArrayWrapper, DisplayAs2D):
    """Array of allele counts.

    Parameters
    ----------
    data : array_like, int, shape (n_variants, n_alleles)
        Allele counts data.
    copy : bool, optional
        If True, make a copy of `data`.
    **kwargs : keyword arguments
        All keyword arguments are passed through to :func:`numpy.array`.

    Notes
    -----
    This class represents allele counts as a 2-dimensional numpy
    array of integers. By convention the first dimension corresponds
    to the variants genotyped, the second dimension corresponds to the
    alleles counted. The integer recorded is the **total count** of
    the given allele at the given variant, across all individuals.

    Examples
    --------

    Obtain allele counts from a genotype array:

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 1], [1, 1]],
        ...                          [[0, 2], [-1, -1]]], dtype='i1')
        >>> ac = g.count_alleles()
        >>> ac
        <AlleleCountsArray shape=(3, 3) dtype=int32>
        3 1 0
        1 3 0
        1 0 1
        >>> ac.dtype
        dtype('int32')
        >>> ac.shape
        (3, 3)
        >>> ac.n_variants
        3
        >>> ac.n_alleles
        3

    Allele counts for a single variant can be obtained by indexing the first
    dimension, e.g.::

        >>> ac[1]
        array([1, 3, 0], dtype=int32)

    Allele counts for a specific allele can be obtained by indexing the
    second dimension, e.g., reference allele counts:

        >>> ac[:, 0]
        array([3, 1, 1], dtype=int32)

    Calculate the total number of alleles called for each variant:

        >>> import numpy as np
        >>> an = np.sum(ac, axis=1)
        >>> an
        array([4, 4, 2])

    Add allele counts from two populations:

        >>> ac + ac
        <AlleleCountsArray shape=(3, 3) dtype=int32>
        6 2 0
        2 6 0
        2 0 2

    """

    def __init__(self, data, copy=False, **kwargs):
        super(AlleleCountsArray, self).__init__(data, copy=copy, **kwargs)
        check_integer_dtype(self.values)
        check_ndim(self.values, 2)

    def __add__(self, other):
        ret = super(AlleleCountsArray, self).__add__(other)
        if hasattr(ret, 'shape') and ret.shape == self.shape:
            ret = AlleleCountsArray(ret)
        return ret

    def __sub__(self, other):
        ret = super(AlleleCountsArray, self).__sub__(other)
        if hasattr(ret, 'shape') and ret.shape == self.shape:
            ret = AlleleCountsArray(ret)
        return ret

    @property
    def n_variants(self):
        """Number of variants."""
        return self.shape[0]

    @property
    def n_alleles(self):
        """Number of alleles."""
        return self.shape[1]

    def __getitem__(self, item):
        return index_allele_counts_array(self, item, type(self))

    def compress(self, condition, axis=0, out=None):
        return compress_allele_counts_array(self, condition, axis=axis, cls=type(self),
                                            compress=np.compress, out=out)

    def take(self, indices, axis=0, out=None, mode='raise'):
        return take_allele_counts_array(self, indices, axis=axis, cls=type(self),
                                        take=np.take, out=out, mode=mode)

    def concatenate(self, others, axis=0):
        return concatenate_allele_counts_array(self, others, axis=axis, cls=type(self),
                                               concatenate=np.concatenate)

    def str_items(self):
        values = self.values
        max_allele = np.max(values)
        if max_allele <= 0:
            max_allele = 1
        n = int(np.floor(np.log10(max_allele))) + 1
        t = values.astype((np.string_, n))
        out = [[str(x, 'ascii').rjust(n) for x in row] for row in t]
        return out

    def to_frequencies(self, fill=np.nan):
        """Compute allele frequencies.

        Parameters
        ----------
        fill : float, optional
            Value to use when number of allele calls is 0.

        Returns
        -------
        af : ndarray, float, shape (n_variants, n_alleles)

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 2], [1, 1]],
        ...                          [[2, 2], [-1, -1]]])
        >>> ac = g.count_alleles()
        >>> ac.to_frequencies()
        array([[0.75, 0.25, 0.  ],
               [0.25, 0.5 , 0.25],
               [0.  , 0.  , 1.  ]])

        """

        an = np.sum(self, axis=1)[:, None]
        with ignore_invalid():
            af = np.where(an > 0, self / an, fill)

        return af

    def allelism(self):
        """Determine the number of distinct alleles observed for each variant.

        Returns
        -------
        n : ndarray, int, shape (n_variants,)
            Allelism array.

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 2], [1, 1]],
        ...                          [[2, 2], [-1, -1]]])
        >>> ac = g.count_alleles()
        >>> ac.allelism()
        array([2, 3, 1])

        """

        return np.sum(self.values > 0, axis=1)

    def max_allele(self):
        """Return the highest allele index for each variant.

        Returns
        -------
        n : ndarray, int, shape (n_variants,)
            Allele index array.

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 1]],
        ...                          [[0, 2], [1, 1]],
        ...                          [[2, 2], [-1, -1]]])
        >>> ac = g.count_alleles()
        >>> ac.max_allele()
        array([1, 2, 2], dtype=int8)

        """

        out = np.empty(self.shape[0], dtype='i1')
        out.fill(-1)
        for i in range(self.shape[1]):
            d = self.values[:, i] > 0
            out[d] = i
        return out

    def is_variant(self):
        """Find variants with at least one non-reference allele call.

        Returns
        -------
        out : ndarray, bool, shape (n_variants,)
            Boolean array where elements are True if variant matches the
            condition.

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 0]],
        ...                          [[0, 0], [0, 1]],
        ...                          [[0, 2], [1, 1]],
        ...                          [[2, 2], [-1, -1]]])
        >>> ac = g.count_alleles()
        >>> ac.is_variant()
        array([False,  True,  True,  True])

        """

        t = self.values[:, 1:] > 0  # type: np.ndarray
        return np.any(t, axis=1)

    def is_non_variant(self):
        """Find variants with no non-reference allele calls.

        Returns
        -------
        out : ndarray, bool, shape (n_variants,)
            Boolean array where elements are True if variant matches the
            condition.

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 0]],
        ...                          [[0, 0], [0, 1]],
        ...                          [[0, 2], [1, 1]],
        ...                          [[2, 2], [-1, -1]]])
        >>> ac = g.count_alleles()
        >>> ac.is_non_variant()
        array([ True, False, False, False])

        """

        t = self.values[:, 1:] == 0  # type: np.ndarray
        return np.all(t, axis=1)

    def is_segregating(self):
        """Find segregating variants (where more than one allele is observed).

        Returns
        -------
        out : ndarray, bool, shape (n_variants,)
            Boolean array where elements are True if variant matches the
            condition.

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 0]],
        ...                          [[0, 0], [0, 1]],
        ...                          [[0, 2], [1, 1]],
        ...                          [[2, 2], [-1, -1]]])
        >>> ac = g.count_alleles()
        >>> ac.is_segregating()
        array([False,  True,  True, False])

        """

        return self.allelism() > 1

    def is_non_segregating(self, allele=None):
        """Find non-segregating variants (where at most one allele is
        observed).

        Parameters
        ----------
        allele : int, optional
            Allele index.

        Returns
        -------
        out : ndarray, bool, shape (n_variants,)
            Boolean array where elements are True if variant matches the
            condition.

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 0]],
        ...                          [[0, 0], [0, 1]],
        ...                          [[0, 2], [1, 1]],
        ...                          [[2, 2], [-1, -1]]])
        >>> ac = g.count_alleles()
        >>> ac.is_non_segregating()
        array([ True, False, False,  True])
        >>> ac.is_non_segregating(allele=2)
        array([False, False, False,  True])

        """

        if allele is None:
            return self.allelism() <= 1
        else:
            return (self.allelism() == 1) & (self.values[:, allele] > 0)

    def is_singleton(self, allele):
        """Find variants with a single call for the given allele.

        Parameters
        ----------
        allele : int, optional
            Allele index.

        Returns
        -------
        out : ndarray, bool, shape (n_variants,)
            Boolean array where elements are True if variant matches the
            condition.

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 0]],
        ...                          [[0, 0], [0, 1]],
        ...                          [[1, 1], [1, 2]],
        ...                          [[2, 2], [-1, -1]]])
        >>> ac = g.count_alleles()
        >>> ac.is_singleton(allele=1)
        array([False,  True, False, False])
        >>> ac.is_singleton(allele=2)
        array([False, False,  True, False])

        """

        return self.values[:, allele] == 1

    def is_doubleton(self, allele):
        """Find variants with exactly two calls for the given allele.

        Parameters
        ----------
        allele : int, optional
            Allele index.

        Returns
        -------
        out : ndarray, bool, shape (n_variants,)
            Boolean array where elements are True if variant matches the
            condition.

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 0]],
        ...                          [[0, 0], [1, 1]],
        ...                          [[1, 1], [1, 2]],
        ...                          [[2, 2], [-1, -1]]])
        >>> ac = g.count_alleles()
        >>> ac.is_doubleton(allele=1)
        array([False,  True, False, False])
        >>> ac.is_doubleton(allele=2)
        array([False, False, False,  True])

        """

        return self.values[:, allele] == 2

    def is_biallelic(self):
        """Find biallelic variants.

        Returns
        -------
        out : ndarray, bool, shape (n_variants,)
            Boolean array where elements are True if variant matches the
            condition.

        """
        return self.allelism() == 2

    def is_biallelic_01(self, min_mac=None):
        """Find variants biallelic for the reference (0) and first alternate
        (1) allele.

        Parameters
        ----------
        min_mac : int, optional
            Minimum minor allele count.

        Returns
        -------
        out : ndarray, bool, shape (n_variants,)
            Boolean array where elements are True if variant matches the
            condition.

        """
        loc = self.is_biallelic() & (self.max_allele() == 1)
        if min_mac is not None:
            # noinspection PyAugmentAssignment
            loc = loc & (self.values[:, :2].min(axis=1) >= min_mac)
        return loc

    def count_variant(self):
        return np.sum(self.is_variant())

    def count_non_variant(self):
        return np.sum(self.is_non_variant())

    def count_segregating(self):
        return np.sum(self.is_segregating())

    def count_non_segregating(self, allele=None):
        return np.sum(self.is_non_segregating(allele=allele))

    def count_singleton(self, allele=1):
        return np.sum(self.is_singleton(allele=allele))

    def count_doubleton(self, allele=1):
        return np.sum(self.is_doubleton(allele=allele))

    def map_alleles(self, mapping, max_allele=None):
        """Transform alleles via a mapping.

        Parameters
        ----------
        mapping : ndarray, int8, shape (n_variants, max_allele)
            An array defining the allele mapping for each variant.
        max_allele : int, optional
            Highest allele index expected in the output. If not provided
            will be determined from maximum value in `mapping`.

        Returns
        -------
        ac : AlleleCountsArray

        Examples
        --------

        >>> import allel
        >>> g = allel.GenotypeArray([[[0, 0], [0, 0]],
        ...                          [[0, 0], [0, 1]],
        ...                          [[0, 2], [1, 1]],
        ...                          [[2, 2], [-1, -1]]])
        >>> ac = g.count_alleles()
        >>> ac
        <AlleleCountsArray shape=(4, 3) dtype=int32>
        4 0 0
        3 1 0
        1 2 1
        0 0 2
        >>> mapping = [[1, 0, 2],
        ...            [1, 0, 2],
        ...            [2, 1, 0],
        ...            [1, 2, 0]]
        >>> ac.map_alleles(mapping)
        <AlleleCountsArray shape=(4, 3) dtype=int32>
        0 4 0
        1 3 0
        1 2 1
        2 0 0

        See Also
        --------
        create_allele_mapping

        """

        # ensure correct dimensionality and matching dtype
        mapping = asarray_ndim(mapping, 2, dtype=self.dtype)
        check_dim0_aligned(self, mapping)
        check_dim1_aligned(self, mapping)

        # use optimisation
        out = allele_counts_array_map_alleles(self.values, mapping, max_allele)

        # wrap and return
        return type(self)(out)


class GenotypeAlleleCounts(NumpyArrayWrapper):
    """Base class for wrapping a NumPy array of genotype calls, stored as allele counts per
    call.

    See Also
    --------
    GenotypeAlleleCountsArray, GenotypeAlleleCountsVector

    """

    def __init__(self, data, copy=False, **kwargs):
        super(GenotypeAlleleCounts, self).__init__(data, copy=copy, **kwargs)
        check_integer_dtype(self.values)

    def is_missing(self):
        return np.sum(self.values, axis=-1) == 0

    def is_called(self):
        return np.sum(self.values, axis=-1) > 0

    # noinspection PyAugmentAssignment
    def is_hom(self, allele=None):
        out = np.sum(self.values > 0, axis=-1) == 1
        if allele is not None:
            out = out & (self.values[..., allele] > 0)
        return out

    def is_hom_ref(self):
        return self.is_hom(0)

    # noinspection PyAugmentAssignment
    def is_hom_alt(self):
        out = np.sum(self.values > 0, axis=-1) == 1
        out = out & (self.values[..., 0] == 0)
        return out

    # noinspection PyAugmentAssignment
    def is_het(self, allele=None):
        out = np.sum(self.values > 0, axis=-1) > 1
        if allele is not None:
            out = out & (self.values[..., allele] > 0)
        return out

    def to_frequencies(self, fill=np.nan):
        an = np.sum(self, axis=-1)[..., np.newaxis]
        with ignore_invalid():
            af = np.where(an > 0, self / an, fill)
        return af

    def allelism(self):
        return np.sum(self.values > 0, axis=-1)

    def max_allele(self):
        out = np.empty(self.shape[:-1], dtype='i1')
        out.fill(-1)
        for i in range(self.shape[-1]):
            d = self.values[..., i] > 0
            out[d] = i
        return out

    def is_variant(self):
        t = self.values[..., 1:] > 0  # type: np.ndarray
        return np.any(t, axis=-1)

    def is_non_variant(self):
        t = self.values[..., 1:] == 0  # type: np.ndarray
        return np.all(t, axis=-1)

    def is_segregating(self):
        return self.allelism() > 1

    def is_non_segregating(self, allele=None):
        if allele is None:
            return self.allelism() <= 1
        else:
            return (self.allelism() == 1) & (self.values[:, allele] > 0)

    def is_biallelic(self):
        return self.allelism() == 2

    def is_biallelic_01(self):
        loc = self.is_biallelic() & (self.max_allele() == 1)
        return loc

    def to_gt(self, max_count=None):

        # how many characters needed per allele?
        if max_count is None:
            max_count = np.max(self)
        nchar = int(np.floor(np.log10(max_count))) + 1

        # convert to string
        a = self.astype((np.string_, nchar)).view(np.chararray)

        # determine allele count separator
        sep = b':'

        # join via separator
        gt = a[..., 0]
        for i in range(1, self.shape[-1]):
            gt = gt + sep + a[..., i]

        return gt

    def compress(self, condition, axis=0, out=None):
        # implement in sub-class
        raise NotImplementedError

    def take(self, indices, axis=0, out=None, mode='raise'):
        # implement in sub-class
        raise NotImplementedError

    def concatenate(self, others, axis=0):
        # implement in sub-class
        raise NotImplementedError


class GenotypeAlleleCountsVector(GenotypeAlleleCounts, DisplayAs1D):
    """Array of genotype calls for a sequence of variants or samples, stored as allele
    counts per call.

    Parameters
    ----------
    data : array_like, int, shape (n_variants, n_samples, ploidy)
        Genotype data.
    copy : bool, optional
        If True, make a copy of `data`.
    **kwargs : keyword arguments
        All keyword arguments are passed through to :func:`numpy.array`.

    See Also
    --------
    GenotypeAlleleCounts, GenotypeAlleleCountsArray, GenotypeVector

    """

    def __init__(self, data, copy=False, **kwargs):
        super(GenotypeAlleleCountsVector, self).__init__(data, copy=copy, **kwargs)
        check_ndim(self.values, 2)

    def __getitem__(self, item):
        return index_genotype_ac_vector(self, item, cls=type(self))

    @property
    def n_calls(self):
        """Number of variants."""
        return self.shape[0]

    @property
    def n_alleles(self):
        """Number of alleles."""
        return self.shape[1]

    def compress(self, condition, axis=0, out=None):
        return compress_genotype_ac(self, condition=condition, axis=axis, wrap_axes={0},
                                    cls=type(self), compress=np.compress, out=out)

    def take(self, indices, axis=0, out=None, mode='raise'):
        return take_genotype_ac(self, indices=indices, axis=axis, wrap_axes={0},
                                cls=type(self), take=np.take, out=out, mode=mode)

    def concatenate(self, others, axis=0):
        return concatenate_genotype_ac(self, others=others, axis=axis, wrap_axes={0},
                                       cls=type(self), concatenate=np.concatenate)

    def str_items(self):
        gt = self.to_gt()
        out = [str(x, 'ascii') for x in gt]
        return out

    def to_str(self, threshold=10, edgeitems=5):
        _, items = self.get_display_items(threshold, edgeitems)
        s = ' '.join(items)
        return s


class GenotypeAlleleCountsArray(GenotypeAlleleCounts, DisplayAs2D):
    """Array of genotype calls for a matrix of variants and samples, stored as allele counts
    per call.

    Parameters
    ----------
    data : array_like, int, shape (n_variants, n_samples, ploidy)
        Genotype data.
    copy : bool, optional
        If True, make a copy of `data`.
    **kwargs : keyword arguments
        All keyword arguments are passed through to :func:`numpy.array`.

    Notes
    -----
    This class provides an alternative representation of genotype calls, allowing for
    variable copy number (effective ploidy) between chromosomes and/or genome regions.
    Genotype calls are represented as a 3-dimensional array of integers. By convention, the
    first dimension corresponds to the variants genotyped, the second dimension corresponds
    to the samples genotyped, and the third dimension corresponds to the alleles genotyped
    in index order. Each integer in the array records the **count** for the given allele at
    the given variant in the given sample.

    Examples
    --------
    Instantiate an array::

        >>> import allel
        >>> g = allel.GenotypeAlleleCountsArray([[[2, 0, 0], [0, 2, 0]],
        ...                                      [[1, 1, 0], [0, 1, 1]],
        ...                                      [[0, 0, 4], [0, 0, 0]]],
        ...                                     dtype='u1')
        >>> g.dtype
        dtype('uint8')
        >>> g.ndim
        3
        >>> g.shape
        (3, 2, 3)
        >>> g.n_variants
        3
        >>> g.n_samples
        2
        >>> g.n_alleles
        3
        >>> g
        <GenotypeAlleleCountsArray shape=(3, 2, 3) dtype=uint8>
        2:0:0 0:2:0
        1:1:0 0:1:1
        0:0:4 0:0:0

    Genotype calls for a single variant at all samples can be obtained
    by indexing the first dimension, e.g.::

        >>> g[1]
        <GenotypeAlleleCountsVector shape=(2, 3) dtype=uint8>
        1:1:0 0:1:1

    Genotype calls for a single sample at all variants can be obtained
    by indexing the second dimension, e.g.::

        >>> g[:, 1]
        <GenotypeAlleleCountsVector shape=(3, 3) dtype=uint8>
        0:2:0 0:1:1 0:0:0

    A genotype call for a single sample at a single variant can be
    obtained by indexing the first and second dimensions, e.g.::

        >>> g[1, 0]
        array([1, 1, 0], dtype=uint8)

    Copy number (effective ploidy) may vary between calls::

        >>> cn = g.sum(axis=2)
        >>> cn
        array([[2, 2],
               [2, 2],
               [4, 0]], dtype=uint64)

    See Also
    --------
    GenotypeAlleleCounts, GenotypeAlleleCountsVector, GenotypeArray

    """

    def __init__(self, data, copy=False, **kwargs):
        super(GenotypeAlleleCountsArray, self).__init__(data, copy=copy, **kwargs)
        check_ndim(self.values, 3)

    def __getitem__(self, item):
        return index_genotype_ac_array(self, item, array_cls=type(self),
                                       vector_cls=GenotypeAlleleCountsVector)

    @property
    def n_variants(self):
        """Number of variants."""
        return self.shape[0]

    @property
    def n_samples(self):
        """Number of samples."""
        return self.shape[1]

    @property
    def n_alleles(self):
        """Number of alleles."""
        return self.shape[2]

    def count_alleles(self, subpop=None):

        # deal with subpop
        subpop = _normalize_subpop_arg(subpop, self.shape[1])
        if subpop is not None:
            g = self.take(subpop, axis=1).values
        else:
            g = self.values

        out = np.empty((g.shape[0], g.shape[2]), dtype='i4')
        g.sum(axis=1, out=out)
        out = AlleleCountsArray(out)
        return out

    def compress(self, condition, axis=0, out=None):
        return compress_genotype_ac(self, condition=condition, axis=axis, wrap_axes={0, 1},
                                    cls=type(self), compress=np.compress, out=out)

    def take(self, indices, axis=0, out=None, mode='raise'):
        return take_genotype_ac(self, indices=indices, axis=axis, wrap_axes={0, 1},
                                cls=type(self), take=np.take, out=out, mode=mode)

    def concatenate(self, others, axis=0):
        return concatenate_genotype_ac(self, others=others, axis=axis, wrap_axes={0, 1},
                                       cls=type(self), concatenate=np.concatenate)

    def subset(self, sel0=None, sel1=None):
        return subset_genotype_ac_array(self, sel0, sel1, cls=type(self), subset=subset)

    def str_items(self):
        gt = self.to_gt()
        n = gt.dtype.itemsize
        out = [[str(x, 'ascii').rjust(n) for x in row] for row in gt]
        return out


class SortedIndex(NumpyArrayWrapper, DisplayAs1D):
    """Index of sorted values, e.g., positions from a single chromosome or
    contig.

    Parameters
    ----------
    data : array_like
        Values in ascending order.
    **kwargs : keyword arguments
        All keyword arguments are passed through to :func:`numpy.array`.

    Notes
    -----
    Values must be given in ascending order, although duplicate values
    may be present (i.e., values must be monotonically increasing).

    Examples
    --------

    >>> import allel
    >>> idx = allel.SortedIndex([2, 5, 8, 14, 15, 23, 42, 42, 61, 77, 103], dtype='i4')
    >>> idx
    <SortedIndex shape=(11,) dtype=int32>
    [2, 5, 8, 14, 15, ..., 42, 42, 61, 77, 103]
    >>> idx.dtype
    dtype('int32')
    >>> idx.ndim
    1
    >>> idx.shape
    (11,)
    >>> idx.is_unique
    False

    See Also
    --------
    SortedMultiIndex, UniqueIndex

    """

    def __init__(self, data, copy=False, **kwargs):
        super(SortedIndex, self).__init__(data, copy=copy, **kwargs)
        check_ndim(self.values, 1)
        # check sorted ascending
        t = self.values[:-1] > self.values[1:]  # type: np.ndarray
        if np.any(t):
            raise ValueError('values must be monotonically increasing')
        self._is_unique = None

    @property
    def is_unique(self):
        """True if no duplicate entries."""
        if self._is_unique is None:
            t = self.values[:-1] == self.values[1:]  # type: np.ndarray
            self._is_unique = ~np.any(t)
        return self._is_unique

    def __getitem__(self, item):
        s = self.values[item]
        if isinstance(item, (slice, list, np.ndarray, type(Ellipsis))):
            return type(self)(s)
        return s

    def compress(self, condition, axis=0, out=None):
        out = self.values.compress(condition, axis=axis, out=out)
        if axis == 0:
            out = type(self)(out)
        return out

    def take(self, indices, axis=0, out=None, mode='raise'):
        out = self.values.take(indices, axis=axis, out=out, mode=mode)
        if axis == 0:
            out = type(self)(out)
        return out

    def locate_key(self, key):
        """Get index location for the requested key.

        Parameters
        ----------
        key : object
            Value to locate.

        Returns
        -------
        loc : int or slice
            Location of `key` (will be slice if there are duplicate entries).

        Examples
        --------

        >>> import allel
        >>> idx = allel.SortedIndex([3, 6, 6, 11])
        >>> idx.locate_key(3)
        0
        >>> idx.locate_key(11)
        3
        >>> idx.locate_key(6)
        slice(1, 3, None)
        >>> try:
        ...     idx.locate_key(2)
        ... except KeyError as e:
        ...     print(e)
        ...
        2

        """

        left = bisect.bisect_left(self, key)
        right = bisect.bisect_right(self, key)
        diff = right - left
        if diff == 0:
            raise KeyError(key)
        elif diff == 1:
            return left
        else:
            return slice(left, right)

    def locate_intersection(self, other):
        """Locate the intersection with another array.

        Parameters
        ----------
        other : array_like, int
            Array of values to intersect.

        Returns
        -------
        loc : ndarray, bool
            Boolean array with location of intersection.
        loc_other : ndarray, bool
            Boolean array with location in `other` of intersection.

        Examples
        --------

        >>> import allel
        >>> idx1 = allel.SortedIndex([3, 6, 11, 20, 35])
        >>> idx2 = allel.SortedIndex([4, 6, 20, 39])
        >>> loc1, loc2 = idx1.locate_intersection(idx2)
        >>> loc1
        array([False,  True, False,  True, False])
        >>> loc2
        array([False,  True,  True, False])
        >>> idx1[loc1]
        <SortedIndex shape=(2,) dtype=int64>
        [6, 20]
        >>> idx2[loc2]
        <SortedIndex shape=(2,) dtype=int64>
        [6, 20]

        """

        # check inputs
        other = SortedIndex(other, copy=False)

        # find intersection
        assume_unique = self.is_unique and other.is_unique
        loc = np.in1d(self, other, assume_unique=assume_unique)
        loc_other = np.in1d(other, self, assume_unique=assume_unique)

        return loc, loc_other

    def locate_keys(self, keys, strict=True):
        """Get index locations for the requested keys.

        Parameters
        ----------
        keys : array_like
            Array of keys to locate.
        strict : bool, optional
            If True, raise KeyError if any keys are not found in the index.

        Returns
        -------
        loc : ndarray, bool
            Boolean array with location of values.

        Examples
        --------

        >>> import allel
        >>> idx1 = allel.SortedIndex([3, 6, 11, 20, 35])
        >>> idx2 = allel.SortedIndex([4, 6, 20, 39])
        >>> loc = idx1.locate_keys(idx2, strict=False)
        >>> loc
        array([False,  True, False,  True, False])
        >>> idx1[loc]
        <SortedIndex shape=(2,) dtype=int64>
        [6, 20]

        """

        # check inputs
        keys = SortedIndex(keys, copy=False)

        # find intersection
        loc, found = self.locate_intersection(keys)

        if strict and np.any(~found):
            raise KeyError(keys[~found])

        return loc

    def intersect(self, other):
        """Intersect with `other` sorted index.

        Parameters
        ----------
        other : array_like, int
            Array of values to intersect with.

        Returns
        -------
        out : SortedIndex
            Values in common.

        Examples
        --------

        >>> import allel
        >>> idx1 = allel.SortedIndex([3, 6, 11, 20, 35])
        >>> idx2 = allel.SortedIndex([4, 6, 20, 39])
        >>> idx1.intersect(idx2)
        <SortedIndex shape=(2,) dtype=int64>
        [6, 20]

        """

        loc = self.locate_keys(other, strict=False)
        return self.compress(loc, axis=0)

    def locate_range(self, start=None, stop=None):
        """Locate slice of index containing all entries within `start` and
        `stop` values **inclusive**.

        Parameters
        ----------
        start : int, optional
            Start value.
        stop : int, optional
            Stop value.

        Returns
        -------
        loc : slice
            Slice object.

        Examples
        --------

        >>> import allel
        >>> idx = allel.SortedIndex([3, 6, 11, 20, 35])
        >>> loc = idx.locate_range(4, 32)
        >>> loc
        slice(1, 4, None)
        >>> idx[loc]
        <SortedIndex shape=(3,) dtype=int64>
        [6, 11, 20]

        """

        # locate start and stop indices
        if start is None:
            start_index = 0
        else:
            start_index = bisect.bisect_left(self, start)
        if stop is None:
            stop_index = len(self)
        else:
            stop_index = bisect.bisect_right(self, stop)

        if stop_index - start_index == 0:
            raise KeyError(start, stop)

        loc = slice(start_index, stop_index)
        return loc

    def intersect_range(self, start=None, stop=None):
        """Intersect with range defined by `start` and `stop` values
        **inclusive**.

        Parameters
        ----------
        start : int, optional
            Start value.
        stop : int, optional
            Stop value.

        Returns
        -------
        idx : SortedIndex

        Examples
        --------

        >>> import allel
        >>> idx = allel.SortedIndex([3, 6, 11, 20, 35])
        >>> idx.intersect_range(4, 32)
        <SortedIndex shape=(3,) dtype=int64>
        [6, 11, 20]

        """

        try:
            loc = self.locate_range(start=start, stop=stop)
        except KeyError:
            return self.values[0:0]
        else:
            return self[loc]

    def locate_intersection_ranges(self, starts, stops):
        """Locate the intersection with a set of ranges.

        Parameters
        ----------
        starts : array_like, int
            Range start values.
        stops : array_like, int
            Range stop values.

        Returns
        -------
        loc : ndarray, bool
            Boolean array with location of entries found.
        loc_ranges : ndarray, bool
            Boolean array with location of ranges containing one or more
            entries.

        Examples
        --------

        >>> import allel
        >>> import numpy as np
        >>> idx = allel.SortedIndex([3, 6, 11, 20, 35])
        >>> ranges = np.array([[0, 2], [6, 17], [12, 15], [31, 35],
        ...                    [100, 120]])
        >>> starts = ranges[:, 0]
        >>> stops = ranges[:, 1]
        >>> loc, loc_ranges = idx.locate_intersection_ranges(starts, stops)
        >>> loc
        array([False,  True,  True, False,  True])
        >>> loc_ranges
        array([False,  True, False,  True, False])
        >>> idx[loc]
        <SortedIndex shape=(3,) dtype=int64>
        [6, 11, 35]
        >>> ranges[loc_ranges]
        array([[ 6, 17],
               [31, 35]])

        """

        # check inputs
        starts = asarray_ndim(starts, 1)
        stops = asarray_ndim(stops, 1)
        check_dim0_aligned(starts, stops)

        # find indices of start and stop values in idx
        start_indices = np.searchsorted(self, starts)
        stop_indices = np.searchsorted(self, stops, side='right')

        # find intervals overlapping at least one value
        loc_ranges = start_indices < stop_indices

        # find values within at least one interval
        loc = np.zeros(self.shape, dtype=np.bool)
        for i, j in zip(start_indices[loc_ranges], stop_indices[loc_ranges]):
            loc[i:j] = True

        return loc, loc_ranges

    def locate_ranges(self, starts, stops, strict=True):
        """Locate items within the given ranges.

        Parameters
        ----------
        starts : array_like, int
            Range start values.
        stops : array_like, int
            Range stop values.
        strict : bool, optional
            If True, raise KeyError if any ranges contain no entries.

        Returns
        -------
        loc : ndarray, bool
            Boolean array with location of entries found.

        Examples
        --------

        >>> import allel
        >>> import numpy as np
        >>> idx = allel.SortedIndex([3, 6, 11, 20, 35])
        >>> ranges = np.array([[0, 2], [6, 17], [12, 15], [31, 35],
        ...                    [100, 120]])
        >>> starts = ranges[:, 0]
        >>> stops = ranges[:, 1]
        >>> loc = idx.locate_ranges(starts, stops, strict=False)
        >>> loc
        array([False,  True,  True, False,  True])
        >>> idx[loc]
        <SortedIndex shape=(3,) dtype=int64>
        [6, 11, 35]

        """

        loc, found = self.locate_intersection_ranges(starts, stops)

        if strict and np.any(~found):
            raise KeyError(starts[~found], stops[~found])

        return loc

    def intersect_ranges(self, starts, stops):
        """Intersect with a set of ranges.

        Parameters
        ----------
        starts : array_like, int
            Range start values.
        stops : array_like, int
            Range stop values.

        Returns
        -------
        idx : SortedIndex

        Examples
        --------

        >>> import allel
        >>> import numpy as np
        >>> idx = allel.SortedIndex([3, 6, 11, 20, 35])
        >>> ranges = np.array([[0, 2], [6, 17], [12, 15], [31, 35],
        ...                    [100, 120]])
        >>> starts = ranges[:, 0]
        >>> stops = ranges[:, 1]
        >>> idx.intersect_ranges(starts, stops)
        <SortedIndex shape=(3,) dtype=int64>
        [6, 11, 35]

        """

        loc = self.locate_ranges(starts, stops, strict=False)
        return self.compress(loc, axis=0)


class UniqueIndex(NumpyArrayWrapper, DisplayAs1D):
    """Array of unique values (e.g., variant or sample identifiers).

    Parameters
    ----------
    data : array_like
        Values.
    **kwargs : keyword arguments
        All keyword arguments are passed through to :func:`numpy.array`.

    Notes
    -----
    This class represents an arbitrary set of unique values, e.g., sample or
    variant identifiers.

    There is no need for values to be sorted. However, all values must be
    unique within the array, and must be hashable objects.

    Examples
    --------

    >>> import allel
    >>> idx = allel.UniqueIndex(['A', 'C', 'B', 'F'])
    >>> idx
    <UniqueIndex shape=(4,) dtype=object>
    ['A', 'C', 'B', 'F']
    >>> idx.dtype
    dtype('O')
    >>> idx.ndim
    1
    >>> idx.shape
    (4,)

    See Also
    --------
    SortedIndex, SortedMultiIndex

    """

    def __init__(self, data, copy=False, dtype=object, **kwargs):
        super(UniqueIndex, self).__init__(data, copy=copy, dtype=dtype, **kwargs)
        check_ndim(self.values, 1)
        # check unique
        # noinspection PyTupleAssignmentBalance
        _, counts = np.unique(self.values, return_counts=True)
        t = counts > 1  # type: np.ndarray
        if np.any(t):
            raise ValueError('values are not unique')
        self.lookup = {v: i for i, v in enumerate(self.values)}

    def __getitem__(self, item):
        s = self.values[item]
        if isinstance(item, (slice, list, np.ndarray, type(Ellipsis))):
            return type(self)(s)
        return s

    def compress(self, condition, axis=0, out=None):
        out = self.values.compress(condition, axis=axis, out=out)
        if axis == 0:
            out = type(self)(out)
        return out

    def take(self, indices, axis=0, out=None, mode='raise'):
        out = self.values.take(indices, axis=axis, out=out, mode=mode)
        if axis == 0:
            out = type(self)(out)
        return out

    def locate_key(self, key):
        """Get index location for the requested key.

        Parameters
        ----------
        key : object
            Key to locate.

        Returns
        -------
        loc : int
            Location of `key`.

        Examples
        --------

        >>> import allel
        >>> idx = allel.UniqueIndex(['A', 'C', 'B', 'F'])
        >>> idx.locate_key('A')
        0
        >>> idx.locate_key('B')
        2
        >>> try:
        ...     idx.locate_key('X')
        ... except KeyError as e:
        ...     print(e)
        ...
        'X'

        """

        return self.lookup[key]

    def locate_intersection(self, other):
        """Locate the intersection with another array.

        Parameters
        ----------
        other : array_like
            Array to intersect.

        Returns
        -------
        loc : ndarray, bool
            Boolean array with location of intersection.
        loc_other : ndarray, bool
            Boolean array with location in `other` of intersection.

        Examples
        --------

        >>> import allel
        >>> idx1 = allel.UniqueIndex(['A', 'C', 'B', 'F'], dtype=object)
        >>> idx2 = allel.UniqueIndex(['X', 'F', 'G', 'C', 'Z'], dtype=object)
        >>> loc1, loc2 = idx1.locate_intersection(idx2)
        >>> loc1
        array([False,  True, False,  True])
        >>> loc2
        array([False,  True, False,  True, False])
        >>> idx1[loc1]
        <UniqueIndex shape=(2,) dtype=object>
        ['C', 'F']
        >>> idx2[loc2]
        <UniqueIndex shape=(2,) dtype=object>
        ['F', 'C']

        """

        # check inputs
        other = UniqueIndex(other)

        # find intersection
        assume_unique = True
        loc = np.in1d(self, other, assume_unique=assume_unique)
        loc_other = np.in1d(other, self, assume_unique=assume_unique)

        return loc, loc_other

    def locate_keys(self, keys, strict=True):
        """Get index locations for the requested keys.

        Parameters
        ----------
        keys : array_like
            Array of keys to locate.
        strict : bool, optional
            If True, raise KeyError if any keys are not found in the index.

        Returns
        -------
        loc : ndarray, bool
            Boolean array with location of keys.

        Examples
        --------

        >>> import allel
        >>> idx = allel.UniqueIndex(['A', 'C', 'B', 'F'])
        >>> idx.locate_keys(['F', 'C'])
        array([False,  True, False,  True])
        >>> idx.locate_keys(['X', 'F', 'G', 'C', 'Z'], strict=False)
        array([False,  True, False,  True])

        """

        # check inputs
        keys = UniqueIndex(keys)

        # find intersection
        loc, found = self.locate_intersection(keys)

        if strict and np.any(~found):
            raise KeyError(keys[~found])

        return loc

    def intersect(self, other):
        """Intersect with `other`.

        Parameters
        ----------
        other : array_like
            Array to intersect.

        Returns
        -------
        out : UniqueIndex

        Examples
        --------

        >>> import allel
        >>> idx1 = allel.UniqueIndex(['A', 'C', 'B', 'F'], dtype=object)
        >>> idx2 = allel.UniqueIndex(['X', 'F', 'G', 'C', 'Z'], dtype=object)
        >>> idx1.intersect(idx2)
        <UniqueIndex shape=(2,) dtype=object>
        ['C', 'F']
        >>> idx2.intersect(idx1)
        <UniqueIndex shape=(2,) dtype=object>
        ['F', 'C']

        """

        loc = self.locate_keys(other, strict=False)
        return self.compress(loc, axis=0)


# noinspection PyMissingConstructor
class SortedMultiIndex(DisplayAs1D):
    """Two-level index of sorted values, e.g., variant positions from two or
    more chromosomes/contigs.

    Parameters
    ----------
    l1 : array_like
        First level values in ascending order.
    l2 : array_like
        Second level values, in ascending order within each sub-level.
    copy : bool, optional
        If True, inputs will be copied into new arrays.

    Examples
    --------

    >>> import allel
    >>> chrom = ['chr1', 'chr1', 'chr2', 'chr2', 'chr2', 'chr3']
    >>> pos = [1, 4, 2, 5, 5, 3]
    >>> idx = allel.SortedMultiIndex(chrom, pos)
    >>> idx
    <SortedMultiIndex shape=(6,), dtype=<U4/int64>
    chr1:1 chr1:4 chr2:2 chr2:5 chr2:5 chr3:3
    >>> len(idx)
    6

    See Also
    --------
    SortedIndex, UniqueIndex

    """

    def __init__(self, l1, l2, copy=False):
        l1 = SortedIndex(l1, copy=copy)
        l2 = np.array(l2, copy=copy)
        check_ndim(l2, 1)
        check_dim0_aligned(l1, l2)
        self.l1 = l1
        self.l2 = l2

    def __repr__(self):
        s = '<SortedMultiIndex shape=(%s,), dtype=%s/%s>' % \
            (len(self), self.l1.dtype, self.l2.dtype)
        s += '\n' + str(self)
        return s

    def str_items(self):
        return ['%s:%s' % (x, y) for x, y in zip(self.l1, self.l2)]

    def to_str(self, threshold=10, edgeitems=5):
        _, items = self.get_display_items(threshold, edgeitems)
        s = ' '.join(items)
        return s

    def __len__(self):
        return len(self.l1)

    def __getitem__(self, item):
        l1 = self.l1[item]
        l2 = self.l2[item]
        if isinstance(item, int):
            return l1, l2
        else:
            return SortedMultiIndex(l1, l2, copy=False)

    def compress(self, condition, axis=0, out=None):
        if out is not None:
            raise NotImplementedError('out argument not supported')
        l1 = self.l1.compress(condition, axis=axis)
        l2 = self.l2.compress(condition, axis=axis)
        return SortedMultiIndex(l1, l2, copy=False)

    def take(self, indices, axis=0, out=None, mode='raise'):
        if out is not None:
            raise NotImplementedError('out argument not supported')
        l1 = self.l1.take(indices, axis=axis, mode=mode)
        l2 = self.l2.take(indices, axis=axis, mode=mode)
        return SortedMultiIndex(l1, l2, copy=False)

    @property
    def shape(self):
        return len(self),

    def locate_key(self, k1, k2=None):
        """
        Get index location for the requested key.

        Parameters
        ----------
        k1 : object
            Level 1 key.
        k2 : object, optional
            Level 2 key.

        Returns
        -------
        loc : int or slice
            Location of requested key (will be slice if there are duplicate
            entries).

        Examples
        --------

        >>> import allel
        >>> chrom = ['chr1', 'chr1', 'chr2', 'chr2', 'chr2', 'chr3']
        >>> pos = [1, 4, 2, 5, 5, 3]
        >>> idx = allel.SortedMultiIndex(chrom, pos)
        >>> idx.locate_key('chr1')
        slice(0, 2, None)
        >>> idx.locate_key('chr1', 4)
        1
        >>> idx.locate_key('chr2', 5)
        slice(3, 5, None)
        >>> try:
        ...     idx.locate_key('chr3', 4)
        ... except KeyError as e:
        ...     print(e)
        ...
        ('chr3', 4)

        """

        loc1 = self.l1.locate_key(k1)
        if k2 is None:
            return loc1
        if isinstance(loc1, slice):
            offset = loc1.start
            try:
                loc2 = SortedIndex(self.l2[loc1], copy=False).locate_key(k2)
            except KeyError:
                # reraise with more information
                raise KeyError(k1, k2)
            else:
                if isinstance(loc2, slice):
                    loc = slice(offset + loc2.start, offset + loc2.stop)
                else:
                    # assume singleton
                    loc = offset + loc2
        else:
            # singleton match in l1
            v = self.l2[loc1]
            if v == k2:
                loc = loc1
            else:
                raise KeyError(k1, k2)
        return loc

    def locate_range(self, key, start=None, stop=None):
        """Locate slice of index containing all entries within the range
        `key`:`start`-`stop` **inclusive**.

        Parameters
        ----------
        key : object
            Level 1 key value.
        start : object, optional
            Level 2 start value.
        stop : object, optional
            Level 2 stop value.

        Returns
        -------
        loc : slice
            Slice object.

        Examples
        --------

        >>> import allel
        >>> chrom = ['chr1', 'chr1', 'chr2', 'chr2', 'chr2', 'chr3']
        >>> pos = [1, 4, 2, 5, 5, 3]
        >>> idx = allel.SortedMultiIndex(chrom, pos)
        >>> idx.locate_range('chr1')
        slice(0, 2, None)
        >>> idx.locate_range('chr1', 1, 4)
        slice(0, 2, None)
        >>> idx.locate_range('chr2', 3, 7)
        slice(3, 5, None)
        >>> try:
        ...     idx.locate_range('chr3', 4, 9)
        ... except KeyError as e:
        ...     print(e)
        ('chr3', 4, 9)

        """

        loc1 = self.l1.locate_key(key)
        if start is None and stop is None:
            loc = loc1
        elif isinstance(loc1, slice):
            offset = loc1.start
            idx = SortedIndex(self.l2[loc1], copy=False)
            try:
                loc2 = idx.locate_range(start, stop)
            except KeyError:
                raise KeyError(key, start, stop)
            else:
                loc = slice(offset + loc2.start, offset + loc2.stop)
        else:
            # singleton match in l1
            v = self.l2[loc1]
            if start <= v <= stop:
                loc = loc1
            else:
                raise KeyError(key, start, stop)
        # ensure slice is always returned
        if not isinstance(loc, slice):
            loc = slice(loc, loc + 1)
        return loc


# noinspection PyMissingConstructor
class ChromPosIndex(DisplayAs1D):
    """Two-level index of variant positions from two or more chromosomes/contigs.

    Parameters
    ----------
    chrom : array_like
        Chromosome values.
    pos : array_like
        Position values, in ascending order within each chromosome.
    copy : bool, optional
        If True, inputs will be copied into new arrays.

    Notes
    -----
    Chromosomes do not need to be sorted, but all values for a given chromosome must be
    grouped together, and all positions for a given chromosome must be sorted in ascending
    order.

    Examples
    --------

    >>> import allel
    >>> chrom = ['chr2', 'chr2', 'chr1', 'chr1', 'chr1', 'chr3']
    >>> pos = [1, 4, 2, 5, 5, 3]
    >>> idx = allel.ChromPosIndex(chrom, pos)
    >>> idx
    <ChromPosIndex shape=(6,), dtype=<U4/int64>
    chr2:1 chr2:4 chr1:2 chr1:5 chr1:5 chr3:3
    >>> len(idx)
    6

    See Also
    --------
    SortedIndex, UniqueIndex

    """

    def __init__(self, chrom, pos, copy=False):
        chrom = np.array(chrom, copy=copy)
        pos = np.array(pos, copy=copy)
        check_ndim(chrom, 1)
        check_ndim(pos, 1)
        check_dim0_aligned(chrom, pos)
        self.chrom = chrom
        self.pos = pos
        # cache mapping of chromosomes to regions
        self.chrom_ranges = dict()

    def __repr__(self):
        s = '<ChromPosIndex shape=(%s,), dtype=%s/%s>' % \
            (len(self), self.chrom.dtype, self.pos.dtype)
        s += '\n' + str(self)
        return s

    def str_items(self):
        return ['%s:%s' % (x, y) for x, y in zip(self.chrom, self.pos)]

    def to_str(self, threshold=10, edgeitems=5):
        _, items = self.get_display_items(threshold, edgeitems)
        s = ' '.join(items)
        return s

    def __len__(self):
        return len(self.chrom)

    def __getitem__(self, item):
        l1 = self.chrom[item]
        l2 = self.pos[item]
        if isinstance(item, int):
            return l1, l2
        else:
            return ChromPosIndex(l1, l2, copy=False)

    def compress(self, condition, axis=0, out=None):
        if out is not None:
            raise NotImplementedError('out argument not supported')
        l1 = self.chrom.compress(condition, axis=axis)
        l2 = self.pos.compress(condition, axis=axis)
        return ChromPosIndex(l1, l2, copy=False)

    def take(self, indices, axis=0, out=None, mode='raise'):
        if out is not None:
            raise NotImplementedError('out argument not supported')
        l1 = self.chrom.take(indices, axis=axis, mode=mode)
        l2 = self.pos.take(indices, axis=axis, mode=mode)
        return ChromPosIndex(l1, l2, copy=False)

    @property
    def shape(self):
        return len(self),

    def locate_key(self, chrom, pos=None):
        """
        Get index location for the requested key.

        Parameters
        ----------
        chrom : object
            Chromosome or contig.
        pos : int, optional
            Position within chromosome or contig.

        Returns
        -------
        loc : int or slice
            Location of requested key (will be slice if there are duplicate
            entries).

        Examples
        --------

        >>> import allel
        >>> chrom = ['chr2', 'chr2', 'chr1', 'chr1', 'chr1', 'chr3']
        >>> pos = [1, 4, 2, 5, 5, 3]
        >>> idx = allel.ChromPosIndex(chrom, pos)
        >>> idx.locate_key('chr1')
        slice(2, 5, None)
        >>> idx.locate_key('chr2', 4)
        1
        >>> idx.locate_key('chr1', 5)
        slice(3, 5, None)
        >>> try:
        ...     idx.locate_key('chr3', 4)
        ... except KeyError as e:
        ...     print(e)
        ...
        ('chr3', 4)

        """

        if pos is None:
            # we just want the region for a chromosome
            if chrom in self.chrom_ranges:
                # return previously cached result
                return self.chrom_ranges[chrom]
            else:
                loc_chrom = np.nonzero(self.chrom == chrom)[0]
                if len(loc_chrom) == 0:
                    raise KeyError(chrom)
                slice_chrom = slice(min(loc_chrom), max(loc_chrom) + 1)
                # cache the result
                self.chrom_ranges[chrom] = slice_chrom
                return slice_chrom

        else:
            slice_chrom = self.locate_key(chrom)
            pos_chrom = SortedIndex(self.pos[slice_chrom])
            try:
                idx_within_chrom = pos_chrom.locate_key(pos)
            except KeyError:
                raise KeyError(chrom, pos)
            if isinstance(idx_within_chrom, slice):
                return slice(slice_chrom.start + idx_within_chrom.start,
                             slice_chrom.start + idx_within_chrom.stop)
            else:
                return slice_chrom.start + idx_within_chrom

    def locate_range(self, chrom, start=None, stop=None):
        """Locate slice of index containing all entries within the range
        `key`:`start`-`stop` **inclusive**.

        Parameters
        ----------
        chrom : object
            Chromosome or contig.
        start : int, optional
            Position start value.
        stop : int, optional
            Position stop value.

        Returns
        -------
        loc : slice
            Slice object.

        Examples
        --------

        >>> import allel
        >>> chrom = ['chr2', 'chr2', 'chr1', 'chr1', 'chr1', 'chr3']
        >>> pos = [1, 4, 2, 5, 5, 3]
        >>> idx = allel.ChromPosIndex(chrom, pos)
        >>> idx.locate_range('chr1')
        slice(2, 5, None)
        >>> idx.locate_range('chr2', 1, 4)
        slice(0, 2, None)
        >>> idx.locate_range('chr1', 3, 7)
        slice(3, 5, None)
        >>> try:
        ...     idx.locate_range('chr3', 4, 9)
        ... except KeyError as e:
        ...     print(e)
        ('chr3', 4, 9)

        """

        slice_chrom = self.locate_key(chrom)

        if start is None and stop is None:
            return slice_chrom

        else:

            pos_chrom = SortedIndex(self.pos[slice_chrom])
            try:
                slice_within_chrom = pos_chrom.locate_range(start, stop)
            except KeyError:
                raise KeyError(chrom, start, stop)
            loc = slice(slice_chrom.start + slice_within_chrom.start,
                        slice_chrom.start + slice_within_chrom.stop)
            return loc


class VariantTable(NumpyRecArrayWrapper):
    """Table (catalogue) of variants.

    Parameters
    ----------
    data : array_like, structured, shape (n_variants,)
        Variant records.
    index : string or pair of strings, optional
        Names of columns to use for positional index, e.g., 'POS' if table
        contains a 'POS' column and records from a single chromosome/contig,
        or ('CHROM', 'POS') if table contains records from multiple
        chromosomes/contigs.
    **kwargs : keyword arguments, optional
        Further keyword arguments are passed through to
        :func:`numpy.rec.array`.

    Examples
    --------
    Instantiate a table from existing data::

        >>> import allel
        >>> records = [[b'chr1', 2, 35, 4.5, (1, 2)],
        ...            [b'chr1', 7, 12, 6.7, (3, 4)],
        ...            [b'chr2', 3, 78, 1.2, (5, 6)],
        ...            [b'chr2', 9, 22, 4.4, (7, 8)],
        ...            [b'chr3', 6, 99, 2.8, (9, 10)]]
        >>> dtype = [('CHROM', 'S4'),
        ...          ('POS', 'u4'),
        ...          ('DP', int),
        ...          ('QD', float),
        ...          ('AC', (int, 2))]
        >>> vt = allel.VariantTable(records, dtype=dtype,
        ...                         index=('CHROM', 'POS'))
        >>> vt.names
        ('CHROM', 'POS', 'DP', 'QD', 'AC')
        >>> vt.n_variants
        5

    Access a column::

        >>> vt['DP']
        array([35, 12, 78, 22, 99])

    Access multiple columns::

        >>> vt[['DP', 'QD']]
        <VariantTable shape=(5,) dtype=(numpy.record, {'names':['DP','QD'], ...
        [(35, 4.5) (12, 6.7) (78, 1.2) (22, 4.4) (99, 2.8)]

    Access a row::

        >>> vt[2]
        (b'chr2', 3, 78, 1.2, [5, 6])

    Access multiple rows::

        >>> vt[2:4]
        <VariantTable shape=(2,) dtype=(numpy.record, [('CHROM', 'S4'), ('POS', '<u4'), ...
        [(b'chr2', 3, 78, 1.2, [5, 6]) (b'chr2', 9, 22, 4.4, ...

    Evaluate expressions against the table::

        >>> vt.eval('DP > 30')
        array([ True, False,  True, False,  True])
        >>> vt.eval('(DP > 30) & (QD > 4)')
        array([ True, False, False, False, False])
        >>> vt.eval('DP * 2')
        array([ 70,  24, 156,  44, 198])

    Query the table::

        >>> vt.query('DP > 30')
        <VariantTable shape=(3,) dtype=(numpy.record, [('CHROM', 'S4'), ('POS', '<u4'), ...
        [(b'chr1', 2, 35,  4.5, [ 1,  2]) (b'chr2', 3, 78,  1.2, ...
         (b'chr3', 6, 99,  2.8, [ 9, 10])]
        >>> vt.query('(DP > 30) & (QD > 4)')
        <VariantTable shape=(1,) dtype=(numpy.record, [('CHROM', 'S4'), ('POS', '<u4'), ...
        [(b'chr1', 2, 35, 4.5, [1, 2])]

    Use the index to query variants::

        >>> vt.query_region(b'chr2', 1, 10)
        <VariantTable shape=(2,) dtype=(numpy.record, [('CHROM', 'S4'), ('POS', '<u4'), ...
        [(b'chr2', 3, 78,  1.2, [5, 6]) (b'chr2', 9, 22,  4.4, ...

    """

    def __init__(self, data, index=None, copy=False, **kwargs):
        super(VariantTable, self).__init__(data, copy=copy, **kwargs)
        self.set_index(index)

    @property
    def n_variants(self):
        """Number of variants (length of first dimension)."""
        return self.shape[0]

    # noinspection PyAttributeOutsideInit
    def set_index(self, index):
        """Set or reset the index.

        Parameters
        ----------
        index : string or pair of strings, optional
            Names of columns to use for positional index, e.g., 'POS' if table
            contains a 'POS' column and records from a single
            chromosome/contig, or ('CHROM', 'POS') if table contains records
            from multiple chromosomes/contigs.

        """
        if index is None:
            pass
        elif isinstance(index, str):
            index = SortedIndex(self[index], copy=False)
        elif isinstance(index, (tuple, list)) and len(index) == 2:
            index = SortedMultiIndex(self[index[0]], self[index[1]],
                                     copy=False)
        else:
            raise ValueError('invalid index argument, expected string or '
                             'pair of strings, found %s' % repr(index))
        self.index = index

    def query_position(self, chrom=None, position=None):
        """Query the table, returning row or rows matching the given genomic
        position.

        Parameters
        ----------
        chrom : string, optional
            Chromosome/contig.
        position : int, optional
            Position (1-based).

        Returns
        -------
        result : row or VariantTable

        """

        if self.index is None:
            raise ValueError('no index has been set')
        if isinstance(self.index, SortedIndex):
            # ignore chrom
            loc = self.index.locate_key(position)
        else:
            loc = self.index.locate_key(chrom, position)
        return self[loc]

    def query_region(self, chrom=None, start=None, stop=None):
        """Query the table, returning row or rows within the given genomic
        region.

        Parameters
        ----------
        chrom : string, optional
            Chromosome/contig.
        start : int, optional
            Region start position (1-based).
        stop : int, optional
            Region stop position (1-based).

        Returns
        -------
        result : VariantTable

        """
        if self.index is None:
            raise ValueError('no index has been set')
        if isinstance(self.index, SortedIndex):
            # ignore chrom
            loc = self.index.locate_range(start, stop)
        else:
            loc = self.index.locate_range(chrom, start, stop)
        return self[loc]

    def to_vcf(self, path, rename=None, number=None, description=None,
               fill=None, write_header=True):
        r"""Write to a variant call format (VCF) file.

        Parameters
        ----------
        path : string
            File path.
        rename : dict, optional
            Rename these columns in the VCF.
        number : dict, optional
            Override the number specified in INFO headers.
        description : dict, optional
            Descriptions for the INFO and FILTER headers.
        fill : dict, optional
            Fill values used for missing data in the table.
        write_header : bool, optional
            If True write VCF header.

        Examples
        --------
        Setup a variant table to write out::

            >>> import allel
            >>> chrom = [b'chr1', b'chr1', b'chr2', b'chr2', b'chr3']
            >>> pos = [2, 6, 3, 8, 1]
            >>> ids = ['a', 'b', 'c', 'd', 'e']
            >>> ref = [b'A', b'C', b'T', b'G', b'N']
            >>> alt = [(b'T', b'.'),
            ...        (b'G', b'.'),
            ...        (b'A', b'C'),
            ...        (b'C', b'A'),
            ...        (b'X', b'.')]
            >>> qual = [1.2, 2.3, 3.4, 4.5, 5.6]
            >>> filter_qd = [True, True, True, False, False]
            >>> filter_dp = [True, False, True, False, False]
            >>> dp = [12, 23, 34, 45, 56]
            >>> qd = [12.3, 23.4, 34.5, 45.6, 56.7]
            >>> flg = [True, False, True, False, True]
            >>> ac = [(1, -1), (3, -1), (5, 6), (7, 8), (9, -1)]
            >>> xx = [(1.2, 2.3), (3.4, 4.5), (5.6, 6.7), (7.8, 8.9),
            ...       (9.0, 9.9)]
            >>> columns = [chrom, pos, ids, ref, alt, qual, filter_dp,
            ...            filter_qd, dp, qd, flg, ac, xx]
            >>> records = list(zip(*columns))
            >>> dtype = [('CHROM', 'S4'),
            ...          ('POS', 'u4'),
            ...          ('ID', 'S1'),
            ...          ('REF', 'S1'),
            ...          ('ALT', ('S1', 2)),
            ...          ('qual', 'f4'),
            ...          ('filter_dp', bool),
            ...          ('filter_qd', bool),
            ...          ('dp', int),
            ...          ('qd', float),
            ...          ('flg', bool),
            ...          ('ac', (int, 2)),
            ...          ('xx', (float, 2))]
            >>> vt = allel.VariantTable(records, dtype=dtype)

        Now write out to VCF and inspect the result::

            >>> rename = {'dp': 'DP', 'qd': 'QD', 'filter_qd': 'QD'}
            >>> fill = {'ALT': b'.', 'ac': -1}
            >>> number = {'ac': 'A'}
            >>> description = {'ac': 'Allele counts', 'filter_dp': 'Low depth'}
            >>> vt.to_vcf('example.vcf', rename=rename, fill=fill,
            ...           number=number, description=description)
            >>> print(open('example.vcf').read())
            ##fileformat=VCFv4.1
            ##fileDate=...
            ##source=...
            ##INFO=<ID=DP,Number=1,Type=Integer,Description="">
            ##INFO=<ID=QD,Number=1,Type=Float,Description="">
            ##INFO=<ID=ac,Number=A,Type=Integer,Description="Allele counts">
            ##INFO=<ID=flg,Number=0,Type=Flag,Description="">
            ##INFO=<ID=xx,Number=2,Type=Float,Description="">
            ##FILTER=<ID=QD,Description="">
            ##FILTER=<ID=dp,Description="Low depth">
            #CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO
            chr1	2	a	A	T	1.2	QD;dp	DP=12;QD=12.3;ac=1;flg;xx=...
            chr1	6	b	C	G	2.3	QD	DP=23;QD=23.4;ac=3;xx=3.4,4.5
            chr2	3	c	T	A,C	3.4	QD;dp	DP=34;QD=34.5;ac=5,6;flg;x...
            chr2	8	d	G	C,A	4.5	PASS	DP=45;QD=45.6;ac=7,8;xx=7...
            chr3	1	e	N	X	5.6	PASS	DP=56;QD=56.7;ac=9;flg;xx=...

        """

        write_vcf(path, callset=self, rename=rename, number=number,
                  description=description, fill=fill,
                  write_header=write_header)


class FeatureTable(NumpyRecArrayWrapper):
    """Table of genomic features (e.g., genes, exons, etc.).

    Parameters
    ----------
    data : array_like, structured, shape (n_variants,)
        Variant records.
    copy : bool, optional
        If True, make a copy of `data`.
    **kwargs : keyword arguments, optional
        Further keyword arguments are passed through to
        :func:`numpy.rec.array`.

    """

    def __init__(self, data, copy=False, **kwargs):
        super(FeatureTable, self).__init__(data, copy=copy, **kwargs)

    @property
    def n_features(self):
        """Number of features (length of first dimension)."""
        return self.shape[0]

    def to_mask(self, size, start_name='start', stop_name='end'):
        """Construct a mask array where elements are True if the fall within
        features in the table.

        Parameters
        ----------

        size : int
            Size of chromosome/contig.
        start_name : string, optional
            Name of column with start coordinates.
        stop_name : string, optional
            Name of column with stop coordinates.

        Returns
        -------

        mask : ndarray, bool

        """
        m = np.zeros(size, dtype=bool)
        for start, stop in self[[start_name, stop_name]]:
            m[start-1:stop] = True
        return m

    @staticmethod
    def from_gff3(path, attributes=None, region=None, score_fill=-1, phase_fill=-1,
                  attributes_fill='.', dtype=None):
        """Read a feature table from a GFF3 format file.

        Parameters
        ----------
        path : string
            File path.
        attributes : list of strings, optional
            List of columns to extract from the "attributes" field.
        region : string, optional
            Genome region to extract. If given, file must be position
            sorted, bgzipped and tabix indexed. Tabix must also be installed
            and on the system path.
        score_fill : int, optional
            Value to use where score field has a missing value.
        phase_fill : int, optional
            Value to use where phase field has a missing value.
        attributes_fill : object or list of objects, optional
            Value(s) to use where attribute field(s) have a missing value.
        dtype : numpy dtype, optional
            Manually specify a dtype.

        Returns
        -------
        ft : FeatureTable

        """

        a = gff3_to_recarray(path, attributes=attributes, region=region,
                             score_fill=score_fill, phase_fill=phase_fill,
                             attributes_fill=attributes_fill, dtype=dtype)
        if a is None:
            return None
        else:
            return FeatureTable(a, copy=False)
