# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np


from allel.model.ndarray import GenotypeArray, HaplotypeArray
from allel.util import check_ploidy, check_min_samples, check_type, check_dtype


def mendel_errors(parent_genotypes, progeny_genotypes):
    """Locate genotype calls not consistent with Mendelian transmission of
    alleles.

    Parameters
    ----------
    parent_genotypes : array_like, int, shape (n_variants, 2, 2)
        Genotype calls for the two parents.
    progeny_genotypes : array_like, int, shape (n_variants, n_progeny, 2)
        Genotype calls for the progeny.

    Returns
    -------
    me : ndarray, int, shape (n_variants, n_progeny)
        Count of Mendel errors for each progeny genotype call.

    Examples
    --------
    The following are all consistent with Mendelian transmission. Note that a
    value of 0 is returned for missing calls::

        >>> import allel
        >>> import numpy as np
        >>> genotypes = np.array([
        ...     # aa x aa -> aa
        ...     [[0, 0], [0, 0], [0, 0], [-1, -1], [-1, -1], [-1, -1]],
        ...     [[1, 1], [1, 1], [1, 1], [-1, -1], [-1, -1], [-1, -1]],
        ...     [[2, 2], [2, 2], [2, 2], [-1, -1], [-1, -1], [-1, -1]],
        ...     # aa x ab -> aa or ab
        ...     [[0, 0], [0, 1], [0, 0], [0, 1], [-1, -1], [-1, -1]],
        ...     [[0, 0], [0, 2], [0, 0], [0, 2], [-1, -1], [-1, -1]],
        ...     [[1, 1], [0, 1], [1, 1], [0, 1], [-1, -1], [-1, -1]],
        ...     # aa x bb -> ab
        ...     [[0, 0], [1, 1], [0, 1], [-1, -1], [-1, -1], [-1, -1]],
        ...     [[0, 0], [2, 2], [0, 2], [-1, -1], [-1, -1], [-1, -1]],
        ...     [[1, 1], [2, 2], [1, 2], [-1, -1], [-1, -1], [-1, -1]],
        ...     # aa x bc -> ab or ac
        ...     [[0, 0], [1, 2], [0, 1], [0, 2], [-1, -1], [-1, -1]],
        ...     [[1, 1], [0, 2], [0, 1], [1, 2], [-1, -1], [-1, -1]],
        ...     # ab x ab -> aa or ab or bb
        ...     [[0, 1], [0, 1], [0, 0], [0, 1], [1, 1], [-1, -1]],
        ...     [[1, 2], [1, 2], [1, 1], [1, 2], [2, 2], [-1, -1]],
        ...     [[0, 2], [0, 2], [0, 0], [0, 2], [2, 2], [-1, -1]],
        ...     # ab x bc -> ab or ac or bb or bc
        ...     [[0, 1], [1, 2], [0, 1], [0, 2], [1, 1], [1, 2]],
        ...     [[0, 1], [0, 2], [0, 0], [0, 1], [0, 1], [1, 2]],
        ...     # ab x cd -> ac or ad or bc or bd
        ...     [[0, 1], [2, 3], [0, 2], [0, 3], [1, 2], [1, 3]],
        ... ])
        >>> me = allel.stats.mendel_errors(genotypes[:, :2], genotypes[:, 2:])
        >>> me
        array([[0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]])

    The following are cases of 'non-parental' inheritance where one or two
    alleles are found in the progeny that are not present in either parent.
    Note that the number of errors may be 1 or 2 depending on the number of
    non-parental alleles::

        >>> genotypes = np.array([
        ...     # aa x aa -> ab or ac or bb or cc
        ...     [[0, 0], [0, 0], [0, 1], [0, 2], [1, 1], [2, 2]],
        ...     [[1, 1], [1, 1], [0, 1], [1, 2], [0, 0], [2, 2]],
        ...     [[2, 2], [2, 2], [0, 2], [1, 2], [0, 0], [1, 1]],
        ...     # aa x ab -> ac or bc or cc
        ...     [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2], [2, 2]],
        ...     [[0, 0], [0, 2], [0, 1], [1, 2], [1, 1], [1, 1]],
        ...     [[1, 1], [0, 1], [1, 2], [0, 2], [2, 2], [2, 2]],
        ...     # aa x bb -> ac or bc or cc
        ...     [[0, 0], [1, 1], [0, 2], [1, 2], [2, 2], [2, 2]],
        ...     [[0, 0], [2, 2], [0, 1], [1, 2], [1, 1], [1, 1]],
        ...     [[1, 1], [2, 2], [0, 1], [0, 2], [0, 0], [0, 0]],
        ...     # ab x ab -> ac or bc or cc
        ...     [[0, 1], [0, 1], [0, 2], [1, 2], [2, 2], [2, 2]],
        ...     [[0, 2], [0, 2], [0, 1], [1, 2], [1, 1], [1, 1]],
        ...     [[1, 2], [1, 2], [0, 1], [0, 2], [0, 0], [0, 0]],
        ...     # ab x bc -> ad or bd or cd or dd
        ...     [[0, 1], [1, 2], [0, 3], [1, 3], [2, 3], [3, 3]],
        ...     [[0, 1], [0, 2], [0, 3], [1, 3], [2, 3], [3, 3]],
        ...     [[0, 2], [1, 2], [0, 3], [1, 3], [2, 3], [3, 3]],
        ...     # ab x cd -> ae or be or ce or de
        ...     [[0, 1], [2, 3], [0, 4], [1, 4], [2, 4], [3, 4]],
        ... ])
        >>> me = allel.stats.mendel_errors(genotypes[:, :2], genotypes[:, 2:])
        >>> me
        array([[1, 1, 2, 2],
               [1, 1, 2, 2],
               [1, 1, 2, 2],
               [1, 1, 2, 2],
               [1, 1, 2, 2],
               [1, 1, 2, 2],
               [1, 1, 2, 2],
               [1, 1, 2, 2],
               [1, 1, 2, 2],
               [1, 1, 2, 2],
               [1, 1, 2, 2],
               [1, 1, 2, 2],
               [1, 1, 1, 2],
               [1, 1, 1, 2],
               [1, 1, 1, 2],
               [1, 1, 1, 1]])

    The following are cases of 'hemi-parental' inheritance, where progeny
    appear to have inherited two copies of an allele found only once in one of
    the parents::

        >>> genotypes = np.array([
        ...     # aa x ab -> bb
        ...     [[0, 0], [0, 1], [1, 1], [-1, -1]],
        ...     [[0, 0], [0, 2], [2, 2], [-1, -1]],
        ...     [[1, 1], [0, 1], [0, 0], [-1, -1]],
        ...     # ab x bc -> aa or cc
        ...     [[0, 1], [1, 2], [0, 0], [2, 2]],
        ...     [[0, 1], [0, 2], [1, 1], [2, 2]],
        ...     [[0, 2], [1, 2], [0, 0], [1, 1]],
        ...     # ab x cd -> aa or bb or cc or dd
        ...     [[0, 1], [2, 3], [0, 0], [1, 1]],
        ...     [[0, 1], [2, 3], [2, 2], [3, 3]],
        ... ])
        >>> me = allel.stats.mendel_errors(genotypes[:, :2], genotypes[:, 2:])
        >>> me
        array([[1, 0],
               [1, 0],
               [1, 0],
               [1, 1],
               [1, 1],
               [1, 1],
               [1, 1],
               [1, 1]])

    The following are cases of 'uni-parental' inheritance, where progeny
    appear to have inherited both alleles from a single parent::

        >>> genotypes = np.array([
        ...     # aa x bb -> aa or bb
        ...     [[0, 0], [1, 1], [0, 0], [1, 1]],
        ...     [[0, 0], [2, 2], [0, 0], [2, 2]],
        ...     [[1, 1], [2, 2], [1, 1], [2, 2]],
        ...     # aa x bc -> aa or bc
        ...     [[0, 0], [1, 2], [0, 0], [1, 2]],
        ...     [[1, 1], [0, 2], [1, 1], [0, 2]],
        ...     # ab x cd -> ab or cd
        ...     [[0, 1], [2, 3], [0, 1], [2, 3]],
        ... ])
        >>> me = allel.stats.mendel_errors(genotypes[:, :2], genotypes[:, 2:])
        >>> me
        array([[1, 1],
               [1, 1],
               [1, 1],
               [1, 1],
               [1, 1],
               [1, 1]])

    """

    # setup
    parent_genotypes = GenotypeArray(parent_genotypes)
    progeny_genotypes = GenotypeArray(progeny_genotypes)
    check_ploidy(parent_genotypes.ploidy, 2)
    check_ploidy(progeny_genotypes.ploidy, 2)

    # transform into per-call allele counts
    max_allele = max(parent_genotypes.max(), progeny_genotypes.max())
    parent_gc = parent_genotypes.to_allele_counts(max_allele=max_allele, dtype='i1')
    progeny_gc = progeny_genotypes.to_allele_counts(max_allele=max_allele, dtype='i1')

    # detect nonparental and hemiparental inheritance by comparing allele
    # counts between parents and progeny
    max_progeny_gc = parent_gc.clip(max=1).sum(axis=1)
    max_progeny_gc = max_progeny_gc[:, np.newaxis, :]
    me = (progeny_gc - max_progeny_gc).clip(min=0).sum(axis=2)

    # detect uniparental inheritance by finding cases where no alleles are
    # shared between parents, then comparing progeny allele counts to each
    # parent
    p1_gc = parent_gc[:, 0, np.newaxis, :]
    p2_gc = parent_gc[:, 1, np.newaxis, :]
    # find variants where parents don't share any alleles
    is_shared_allele = (p1_gc > 0) & (p2_gc > 0)
    no_shared_alleles = ~np.any(is_shared_allele, axis=2)
    # find calls where progeny genotype is identical to one or the other parent
    me[no_shared_alleles &
       (np.all(progeny_gc == p1_gc, axis=2) |
        np.all(progeny_gc == p2_gc, axis=2))] = 1

    # retrofit where either or both parent has a missing call
    me[np.any(parent_genotypes.is_missing(), axis=1)] = 0

    return me


# constants to represent inheritance states
INHERIT_UNDETERMINED = 0
INHERIT_PARENT1 = 1
INHERIT_PARENT2 = 2
INHERIT_NONSEG_REF = 3
INHERIT_NONSEG_ALT = 4
INHERIT_NONPARENTAL = 5
INHERIT_PARENT_MISSING = 6
INHERIT_MISSING = 7


def paint_transmission(parent_haplotypes, progeny_haplotypes):
    """Paint haplotypes inherited from a single diploid parent according to
    their allelic inheritance.

    Parameters
    ----------
    parent_haplotypes : array_like, int, shape (n_variants, 2)
        Both haplotypes from a single diploid parent.
    progeny_haplotypes : array_like, int, shape (n_variants, n_progeny)
        Haplotypes found in progeny of the given parent, inherited from the
        given parent. I.e., haplotypes from gametes of the given parent.

    Returns
    -------
    painting : ndarray, uint8, shape (n_variants, n_progeny)
        An array of integers coded as follows: 1 = allele inherited from
        first parental haplotype; 2 = allele inherited from second parental
        haplotype; 3 = reference allele, also carried by both parental
        haplotypes; 4 = non-reference allele, also carried by both parental
        haplotypes; 5 = non-parental allele; 6 = either or both parental
        alleles missing; 7 = missing allele; 0 = undetermined.

    Examples
    --------
    >>> import allel
    >>> haplotypes = allel.HaplotypeArray([
    ...     [0, 0, 0, 1, 2, -1],
    ...     [0, 1, 0, 1, 2, -1],
    ...     [1, 0, 0, 1, 2, -1],
    ...     [1, 1, 0, 1, 2, -1],
    ...     [0, 2, 0, 1, 2, -1],
    ...     [0, -1, 0, 1, 2, -1],
    ...     [-1, 1, 0, 1, 2, -1],
    ...     [-1, -1, 0, 1, 2, -1],
    ... ], dtype='i1')
    >>> painting = allel.stats.paint_transmission(haplotypes[:, :2],
    ...                                           haplotypes[:, 2:])
    >>> painting
    array([[3, 5, 5, 7],
           [1, 2, 5, 7],
           [2, 1, 5, 7],
           [5, 4, 5, 7],
           [1, 5, 2, 7],
           [6, 6, 6, 7],
           [6, 6, 6, 7],
           [6, 6, 6, 7]], dtype=uint8)

    """

    # check inputs
    parent_haplotypes = HaplotypeArray(parent_haplotypes)
    progeny_haplotypes = HaplotypeArray(progeny_haplotypes)
    if parent_haplotypes.n_haplotypes != 2:
        raise ValueError('exactly two parental haplotypes should be provided')

    # convenience variables
    parent1 = parent_haplotypes[:, 0, np.newaxis]
    parent2 = parent_haplotypes[:, 1, np.newaxis]
    progeny_is_missing = progeny_haplotypes < 0
    parent_is_missing = np.any(parent_haplotypes < 0, axis=1)
    # need this for broadcasting, but also need to retain original for later
    parent_is_missing_bc = parent_is_missing[:, np.newaxis]
    parent_diplotype = GenotypeArray(parent_haplotypes[:, np.newaxis, :])
    parent_is_hom_ref = parent_diplotype.is_hom_ref()
    parent_is_het = parent_diplotype.is_het()
    parent_is_hom_alt = parent_diplotype.is_hom_alt()

    # identify allele calls where inheritance can be determined
    is_callable = ~progeny_is_missing & ~parent_is_missing_bc
    is_callable_seg = is_callable & parent_is_het

    # main inheritance states
    inherit_parent1 = is_callable_seg & (progeny_haplotypes == parent1)
    inherit_parent2 = is_callable_seg & (progeny_haplotypes == parent2)
    nonseg_ref = (is_callable & parent_is_hom_ref & (progeny_haplotypes == parent1))
    nonseg_alt = (is_callable & parent_is_hom_alt & (progeny_haplotypes == parent1))
    nonparental = (
        is_callable & (progeny_haplotypes != parent1) & (progeny_haplotypes != parent2)
    )

    # record inheritance states
    # N.B., order in which these are set matters
    painting = np.zeros(progeny_haplotypes.shape, dtype='u1')
    painting[inherit_parent1] = INHERIT_PARENT1
    painting[inherit_parent2] = INHERIT_PARENT2
    painting[nonseg_ref] = INHERIT_NONSEG_REF
    painting[nonseg_alt] = INHERIT_NONSEG_ALT
    painting[nonparental] = INHERIT_NONPARENTAL
    painting[parent_is_missing] = INHERIT_PARENT_MISSING
    painting[progeny_is_missing] = INHERIT_MISSING

    return painting


def phase_progeny_by_transmission(g):
    """Phase progeny genotypes from a trio or cross using Mendelian
    transmission.

    Parameters
    ----------
    g : array_like, int, shape (n_variants, n_samples, 2)
        Genotype array, with parents as first two columns and progeny as
        remaining columns.

    Returns
    -------
    g : ndarray, int8, shape (n_variants, n_samples, 2)
        Genotype array with progeny phased where possible.

    Examples
    --------
    >>> import allel
    >>> g = allel.GenotypeArray([
    ...     [[0, 0], [0, 0], [0, 0]],
    ...     [[1, 1], [1, 1], [1, 1]],
    ...     [[0, 0], [1, 1], [0, 1]],
    ...     [[1, 1], [0, 0], [0, 1]],
    ...     [[0, 0], [0, 1], [0, 0]],
    ...     [[0, 0], [0, 1], [0, 1]],
    ...     [[0, 1], [0, 0], [0, 1]],
    ...     [[0, 1], [0, 1], [0, 1]],
    ...     [[0, 1], [1, 2], [0, 1]],
    ...     [[1, 2], [0, 1], [1, 2]],
    ...     [[0, 1], [2, 3], [0, 2]],
    ...     [[2, 3], [0, 1], [1, 3]],
    ...     [[0, 0], [0, 0], [-1, -1]],
    ...     [[0, 0], [0, 0], [1, 1]],
    ... ], dtype='i1')
    >>> g = allel.stats.phase_progeny_by_transmission(g)
    >>> print(g.to_str(row_threshold=None))
    0/0 0/0 0|0
    1/1 1/1 1|1
    0/0 1/1 0|1
    1/1 0/0 1|0
    0/0 0/1 0|0
    0/0 0/1 0|1
    0/1 0/0 1|0
    0/1 0/1 0/1
    0/1 1/2 0|1
    1/2 0/1 2|1
    0/1 2/3 0|2
    2/3 0/1 3|1
    0/0 0/0 ./.
    0/0 0/0 1/1
    >>> g.is_phased
    array([[False, False,  True],
           [False, False,  True],
           [False, False,  True],
           [False, False,  True],
           [False, False,  True],
           [False, False,  True],
           [False, False,  True],
           [False, False, False],
           [False, False,  True],
           [False, False,  True],
           [False, False,  True],
           [False, False,  True],
           [False, False, False],
           [False, False, False]], dtype=bool)

    """

    # setup
    g = GenotypeArray(g, dtype='i1', copy=True)
    check_ploidy(g.ploidy, 2)
    check_min_samples(g.n_samples, 3)

    # run the phasing
    from allel.opt.stats import phase_progeny_by_transmission_int8
    is_phased = phase_progeny_by_transmission_int8(g.values)
    g.is_phased = np.asarray(is_phased).view(bool)

    # outputs
    return g


def phase_parents_by_transmission(g, window_size):
    """Phase parent genotypes from a trio or cross, given progeny genotypes
    already phased by Mendelian transmission.

    Parameters
    ----------
    g : GenotypeArray
        Genotype array, with parents as first two columns and progeny as
        remaining columns, where progeny genotypes are already phased.
    window_size : int
        Number of previous heterozygous sites to include when phasing each
        parent. A number somewhere between 10 and 100 may be appropriate,
        depending on levels of heterozygosity and quality of data.

    Returns
    -------
    g : GenotypeArray
        Genotype array with parents phased where possible.

    """

    # setup
    check_type(g, GenotypeArray)
    check_dtype(g.values, 'i1')
    check_ploidy(g.ploidy, 2)
    if g.is_phased is None:
        raise ValueError('genotype array must first have progeny phased by transmission')
    check_min_samples(g.n_samples, 3)

    # run the phasing
    from allel.opt.stats import phase_parents_by_transmission_int8
    is_phased = g.is_phased.view('u1')
    phase_parents_by_transmission_int8(g.values, is_phased, window_size)

    # outputs
    return g


def phase_by_transmission(g, window_size, copy=True):
    """Phase genotypes in a trio or cross where possible using Mendelian
    transmission.

    Parameters
    ----------
    g : array_like, int, shape (n_variants, n_samples, 2)
        Genotype array, with parents as first two columns and progeny as
        remaining columns.
    window_size : int
        Number of previous heterozygous sites to include when phasing each
        parent. A number somewhere between 10 and 100 may be appropriate,
        depending on levels of heterozygosity and quality of data.
    copy : bool, optional
        If False, attempt to phase genotypes in-place. Note that this is
        only possible if the input array has int8 dtype, otherwise a copy is
        always made regardless of this parameter.

    Returns
    -------
    g : GenotypeArray
        Genotype array with progeny phased where possible.

    """

    # setup
    g = GenotypeArray(g, dtype='i1', copy=copy)
    check_ploidy(g.ploidy, 2)
    check_min_samples(g.n_samples, 3)

    # phase the progeny
    from allel.opt.stats import phase_progeny_by_transmission_int8
    is_phased = phase_progeny_by_transmission_int8(g.values)
    g.is_phased = np.asarray(is_phased).view(bool)

    # phase the parents
    from allel.opt.stats import phase_parents_by_transmission_int8
    phase_parents_by_transmission_int8(g.values, is_phased, window_size)

    return g
