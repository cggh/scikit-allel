# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np


from allel.model.ndarray import GenotypeArray, HaplotypeArray


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
        Mendel errors for each progeny genotype call.

    Examples
    --------
    TODO

    """

    # setup
    parent_genotypes = GenotypeArray(parent_genotypes)
    progeny_genotypes = GenotypeArray(progeny_genotypes)
    if parent_genotypes.ploidy != 2 or progeny_genotypes.ploidy != 2:
        raise ValueError('only diploid calls are supported')

    # transform into per-call allele counts
    max_allele = max(parent_genotypes.max(), progeny_genotypes.max())
    alleles = list(range(max_allele + 1))
    parent_gc = parent_genotypes.to_allele_counts(alleles=alleles, dtype='i1')
    progeny_gc = progeny_genotypes.to_allele_counts(alleles=alleles,
                                                    dtype='i1')

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
    """TODO

    """

    # check inputs
    parent_haplotypes = HaplotypeArray(parent_haplotypes)
    progeny_haplotypes = HaplotypeArray(progeny_haplotypes)
    if parent_haplotypes.n_haplotypes != 2:
        raise ValueError('exactly two parental haplotypes should be provided')

    # convenience variables
    parent1 = parent_haplotypes[:, 0, np.newaxis]
    parent2 = parent_haplotypes[:, 1, np.newaxis]
    gamete_is_missing = progeny_haplotypes < 0
    parent_is_missing = np.any(parent_haplotypes < 0, axis=1)
    # need this for broadcasting, but also need to retain original for later
    parent_is_missing_bc = parent_is_missing[:, np.newaxis]
    parent_diplotype = GenotypeArray(parent_haplotypes[:, np.newaxis, :])
    parent_is_hom_ref = parent_diplotype.is_hom_ref()
    parent_is_het = parent_diplotype.is_het()
    parent_is_hom_alt = parent_diplotype.is_hom_alt()

    # identify allele calls where inheritance can be determined
    is_callable = ~gamete_is_missing & ~parent_is_missing_bc
    is_callable_seg = is_callable & parent_is_het

    # main inheritance states
    inherit_parent1 = is_callable_seg & (progeny_haplotypes == parent1)
    inherit_parent2 = is_callable_seg & (progeny_haplotypes == parent2)
    nonseg_ref = (
        is_callable &
        parent_is_hom_ref &
        (progeny_haplotypes == parent1)
    )
    nonseg_alt = (
        is_callable &
        parent_is_hom_alt &
        (progeny_haplotypes == parent1)
    )
    nonparental = (
        is_callable &
        (progeny_haplotypes != parent1) &
        (progeny_haplotypes != parent2)
    )

    # record inheritance states
    # N.B., order in which these are set matters
    inheritance = np.zeros_like(progeny_haplotypes, dtype='u1')
    print(inherit_parent1.shape)
    inheritance[inherit_parent1] = INHERIT_PARENT1
    inheritance[inherit_parent2] = INHERIT_PARENT2
    inheritance[nonseg_ref] = INHERIT_NONSEG_REF
    inheritance[nonseg_alt] = INHERIT_NONSEG_ALT
    inheritance[nonparental] = INHERIT_NONPARENTAL
    inheritance[parent_is_missing] = INHERIT_PARENT_MISSING
    inheritance[gamete_is_missing] = INHERIT_MISSING

    return inheritance
