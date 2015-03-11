# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import logging
logger = logging.getLogger(__name__)
debug = logger.debug


import numpy as np


import allel.model


# Weir & Cockheram Fst (theta hat)
##################################

# r : number of populations sampled

# n_i : number of individuals sampled from population i

# ac_i : count of allele A in sample size of n_i from population i

# p_i : frequency of allele A in sample size of n_i from population i
#     ac_i / n_i

# h_i : observed proportion of individuals heterozygous for allele A

# n_bar : average sample size
#     sum_i[ n_i / r ]

# C : coefficient of variation of sample sizes

# n_C :
#     n_bar * (1 - C**2 / r)

# p_bar : average sample frequency of allele A
#     sum_i[ (n_i * p_i) / (r * n_bar) ]
# =   sum_i[ ac_i / (r * n_bar) ]

# ss : the sample variance of allele A frequencies over populations
#     sum_i[ (n_i * (p_i - p_bar)**2) / ((r - 1) * n_bar) ]

# h_bar : the average heterozygote frequency for allele A
#     sum_i[ (n_i * h_i) / (r * n_bar) ]


def weir_cockerham_f_statistics(g, subpops, allele=1):

    if not isinstance(g, allel.model.GenotypeArray):
        g = allel.model.GenotypeArray(g, copy=False)

    # number of populations sampled
    r = len(subpops)
    debug('r: %r', r)

    # allele counts within each population
    # shape (n_variants, n_alleles, n_populations)
    max_allele = g.max()
    acs = tuple(g.count_alleles(subpop=s, max_allele=max_allele)
                for s in subpops)
    acs = np.dstack(acs)
    debug('acs: %s, %r', acs.shape, acs)

    # number of chromosomes sampled from each population
    # shape (n_variants, n_populations)
    an = np.sum(acs, axis=1)
    debug('an: %s, %r', an.shape, an)

    # number of individuals sampled from each population
    # shape (n_variants, n_populations)
    n = an // 2
    debug('n: %s, %r', n.shape, n)

    # total numbers of individuals sampled
    n_total = np.sum(n, axis=1)
    debug('n_total: %s, %r', n_total.shape, n_total)

    # allele count within each population
    # shape (n_variants, n_populations)
    ac = acs[:, allele, :]
    debug('ac: %s, %r', ac.shape, ac)

    # allele frequency within each population
    # shape (n_variants, n_populations)
    # TODO what happens when an is zero?
    p = ac / an
    debug('p: %s, %r', p.shape, p)

    # average sample size
    # shape (n_variants,)
    n_bar = np.mean(n, axis=1)
    debug('n_bar: %s, %r', n_bar.shape, n_bar)

    # coefficient of variation of sample sizes
    # shape (n_variants,)
    # noinspection PyPep8Naming
    # C = np.std(n, axis=1, ddof=1) / n_bar
    # debug('C: %s, %r', C.shape, C)

    # n sub C
    # shape (n_variants,)
    # noinspection PyPep8Naming
    # n_C = n_bar * (1 - (C**2 / r))
    # debug('n_C: %s, %r', n_C.shape, n_C)
    n_C = (n_total - (np.sum(n**2, axis=1) / n_total)) / (r - 1)
    debug('n_C (alternative): %s, %r', n_C.shape, n_C)

    # average sample frequency of the allele
    # shape: (n_variants,)
    ac_total = np.sum(ac, axis=1)
    an_total = np.sum(an, axis=1)
    p_bar = ac_total / an_total
    debug('p_bar: %s, %r', p_bar.shape, p_bar)
    p_bar = np.sum(n * p, axis=1) / n_total
    debug('p_bar (alternative): %s, %r', p_bar.shape, p_bar)

    # sample variance of allele frequencies over populations
    # shape (n_variants,)
    s_squared = (
        np.sum(n * ((p - p_bar[:, None])**2), axis=1)
        / (n_bar * (r - 1))
    )
    debug('s_squared: %s, %r', s_squared.shape, s_squared)

    # average heterozygote frequency
    # shape (n_variants,)
    h_bar = g.count_het(allele=allele, axis=1) / n_total
    debug('h_bar: %s, %r', h_bar.shape, h_bar)

    # a
    # shape (n_variants,)
    a = ((n_bar / n_C)
         * (s_squared -
            ((1 / (n_bar - 1))
             * ((p_bar * (1 - p_bar))
                - ((r - 1) * s_squared / r)
                - (h_bar / 4)))))
    debug('a: %s, %r', a.shape, a)

    # b
    # shape (n_variants,)
    b = ((n_bar / (n_bar - 1))
         * ((p_bar * (1 - p_bar))
            - ((r - 1) * s_squared / r)
            - (((2 * n_bar) - 1) * h_bar / (4 * n_bar))))
    debug('b: %s, %r', b.shape, b)

    # c
    # shape (n_variants,)
    c = h_bar / 2
    debug('c: %s, %r', c.shape, c)

    return a, b, c


def naive_fst(g, subpops, allele=1):

    if not isinstance(g, allel.model.GenotypeArray):
        g = allel.model.GenotypeArray(g, copy=False)

    # number of populations sampled
    r = len(subpops)
    debug('r: %r', r)

    # allele counts within each population
    # shape (n_variants, n_alleles, n_populations)
    max_allele = g.max()
    acs = tuple(g.count_alleles(subpop=s, max_allele=max_allele)
                for s in subpops)
    acs = np.dstack(acs)
    debug('acs: %s, %r', acs.shape, acs)

    # number of chromosomes sampled from each population
    # shape (n_variants, n_populations)
    an = np.sum(acs, axis=1)
    debug('an: %s, %r', an.shape, an)

    # number of individuals sampled from each population
    # shape (n_variants, n_populations)
    # n = an // 2
    n = an
    debug('n: %s, %r', n.shape, n)

    # allele count within each population
    # shape (n_variants, n_populations)
    ac = acs[:, allele, :]

    # allele frequency within each population
    # shape (n_variants, n_populations)
    # TODO what happens when an is zero?
    p = ac / an
    debug('p: %s, %r', p.shape, p)

    # average sample frequency of the allele
    # shape: (n_variants,)
    p_bar = np.mean(p, axis=1)
    debug('p_bar: %s, %r', p_bar.shape, p_bar)

    # sample variance of allele frequencies over populations
    # shape (n_variants,)
    s_squared = np.var(p, axis=1, ddof=0)
    debug('s_squared: %s, %r', s_squared.shape, s_squared)
    # s_squared = (1 / (r - 1)) * np.sum((p - p_bar[:, None]) ** 2, axis=1)
    # debug('s_squared: %s, %r', s_squared.shape, s_squared)

    fst = s_squared / (p_bar * (1 - p_bar))

    return fst
