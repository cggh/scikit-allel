# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np


from allel.model.ndarray import GenotypeArray


def mendel_errors(parent_genotypes, progeny_genotypes):
    """Locate genotype calls not consistent with Mendelian segregation of
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


# TODO inheritance
