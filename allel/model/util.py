# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


# third-party imports
import numpy as np


# internal imports
from allel.util import asarray_ndim, check_dim0_aligned, ensure_dim1_aligned


__all__ = ['create_allele_mapping', 'locate_private_alleles', 'locate_fixed_differences',
           'sample_to_haplotype_selection']


def create_allele_mapping(ref, alt, alleles, dtype='i1'):
    """Create an array mapping variant alleles into a different allele index
    system.

    Parameters
    ----------
    ref : array_like, S1, shape (n_variants,)
        Reference alleles.
    alt : array_like, S1, shape (n_variants, n_alt_alleles)
        Alternate alleles.
    alleles : array_like, S1, shape (n_variants, n_alleles)
        Alleles defining the new allele indexing.
    dtype : dtype, optional
        Output dtype.

    Returns
    -------
    mapping : ndarray, int8, shape (n_variants, n_alt_alleles + 1)

    Examples
    --------
    Example with biallelic variants::

        >>> import allel
        >>> ref = [b'A', b'C', b'T', b'G']
        >>> alt = [b'T', b'G', b'C', b'A']
        >>> alleles = [[b'A', b'T'],  # no transformation
        ...            [b'G', b'C'],  # swap
        ...            [b'T', b'A'],  # 1 missing
        ...            [b'A', b'C']]  # 1 missing
        >>> mapping = allel.create_allele_mapping(ref, alt, alleles)
        >>> mapping
        array([[ 0,  1],
               [ 1,  0],
               [ 0, -1],
               [-1,  0]], dtype=int8)

    Example with multiallelic variants::

        >>> ref = [b'A', b'C', b'T']
        >>> alt = [[b'T', b'G'],
        ...        [b'A', b'T'],
        ...        [b'G', b'.']]
        >>> alleles = [[b'A', b'T'],
        ...            [b'C', b'T'],
        ...            [b'G', b'A']]
        >>> mapping = create_allele_mapping(ref, alt, alleles)
        >>> mapping
        array([[ 0,  1, -1],
               [ 0, -1,  1],
               [-1,  0, -1]], dtype=int8)

    See Also
    --------
    GenotypeArray.map_alleles, HaplotypeArray.map_alleles, AlleleCountsArray.map_alleles

    """

    ref = asarray_ndim(ref, 1)
    alt = asarray_ndim(alt, 1, 2)
    alleles = asarray_ndim(alleles, 1, 2)
    check_dim0_aligned(ref, alt, alleles)

    # reshape for convenience
    ref = ref[:, None]
    if alt.ndim == 1:
        alt = alt[:, None]
    if alleles.ndim == 1:
        alleles = alleles[:, None]
    source_alleles = np.append(ref, alt, axis=1)

    # setup output array
    out = np.empty(source_alleles.shape, dtype=dtype)
    out.fill(-1)

    # find matches
    for ai in range(source_alleles.shape[1]):
        match = source_alleles[:, ai, None] == alleles
        match_i, match_j = match.nonzero()
        out[match_i, ai] = match_j

    return out


def locate_fixed_differences(ac1, ac2):
    """Locate variants with no shared alleles between two populations.

    Parameters
    ----------
    ac1 : array_like, int, shape (n_variants, n_alleles)
        Allele counts array from the first population.
    ac2 : array_like, int, shape (n_variants, n_alleles)
        Allele counts array from the second population.

    Returns
    -------
    loc : ndarray, bool, shape (n_variants,)

    See Also
    --------
    allel.stats.diversity.windowed_df

    Examples
    --------

    >>> import allel
    >>> g = allel.GenotypeArray([[[0, 0], [0, 0], [1, 1], [1, 1]],
    ...                          [[0, 1], [0, 1], [0, 1], [0, 1]],
    ...                          [[0, 1], [0, 1], [1, 1], [1, 1]],
    ...                          [[0, 0], [0, 0], [1, 1], [2, 2]],
    ...                          [[0, 0], [-1, -1], [1, 1], [-1, -1]]])
    >>> ac1 = g.count_alleles(subpop=[0, 1])
    >>> ac2 = g.count_alleles(subpop=[2, 3])
    >>> loc_df = allel.locate_fixed_differences(ac1, ac2)
    >>> loc_df
    array([ True, False, False,  True,  True], dtype=bool)

    """

    # check inputs
    ac1 = asarray_ndim(ac1, 2)
    ac2 = asarray_ndim(ac2, 2)
    check_dim0_aligned(ac1, ac2)
    ac1, ac2 = ensure_dim1_aligned(ac1, ac2)

    # stack allele counts for convenience
    pac = np.dstack([ac1, ac2])

    # count numbers of alleles called in each population
    pan = np.sum(pac, axis=1)

    # count the numbers of populations with each allele
    npa = np.sum(pac > 0, axis=2)

    # locate variants with allele calls in both populations
    non_missing = np.all(pan > 0, axis=1)

    # locate variants where all alleles are only found in a single population
    no_shared_alleles = np.all(npa <= 1, axis=1)

    return non_missing & no_shared_alleles


def locate_private_alleles(*acs):
    """Locate alleles that are found only in a single population.

    Parameters
    ----------
    *acs : array_like, int, shape (n_variants, n_alleles)
        Allele counts arrays from each population.

    Returns
    -------
    loc : ndarray, bool, shape (n_variants, n_alleles)
        Boolean array where elements are True if allele is private to a
        single population.

    Examples
    --------

    >>> import allel
    >>> g = allel.GenotypeArray([[[0, 0], [0, 0], [1, 1], [1, 1]],
    ...                          [[0, 1], [0, 1], [0, 1], [0, 1]],
    ...                          [[0, 1], [0, 1], [1, 1], [1, 1]],
    ...                          [[0, 0], [0, 0], [1, 1], [2, 2]],
    ...                          [[0, 0], [-1, -1], [1, 1], [-1, -1]]])
    >>> ac1 = g.count_alleles(subpop=[0, 1])
    >>> ac2 = g.count_alleles(subpop=[2])
    >>> ac3 = g.count_alleles(subpop=[3])
    >>> loc_private_alleles = allel.locate_private_alleles(ac1, ac2, ac3)
    >>> loc_private_alleles
    array([[ True, False, False],
           [False, False, False],
           [ True, False, False],
           [ True,  True,  True],
           [ True,  True, False]], dtype=bool)
    >>> loc_private_variants = np.any(loc_private_alleles, axis=1)
    >>> loc_private_variants
    array([ True, False,  True,  True,  True], dtype=bool)

    """

    # check inputs
    acs = [asarray_ndim(ac, 2) for ac in acs]
    check_dim0_aligned(*acs)
    acs = ensure_dim1_aligned(*acs)

    # stack allele counts for convenience
    pac = np.dstack(acs)

    # count the numbers of populations with each allele
    npa = np.sum(pac > 0, axis=2)

    # locate alleles found only in a single population
    loc_pa = npa == 1

    return loc_pa


def sample_to_haplotype_selection(indices, ploidy):
    return [(i * ploidy) + n for i in indices for n in range(ploidy)]
