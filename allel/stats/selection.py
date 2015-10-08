# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np


from allel.model.ndarray import HaplotypeArray


def ehh_decay(h, truncate=False):
    """Compute the decay of extended haplotype homozygosity (EHH)
    moving away from the first variant.

    Parameters
    ----------
    h : array_like, int, shape (n_variants, n_haplotypes)
        Haplotype array.
    truncate : bool, optional
        If True, the return array will exclude trailing zeros.

    Returns
    -------
    ehh : ndarray, float, shape (n_variants, )
        EHH at successive variants from the first variant.

    """

    from allel.opt.stats import pairwise_shared_prefix_lengths_int8

    # check inputs
    # N.B., ensure int8 so we can use cython optimisation
    h = HaplotypeArray(np.asarray(h, dtype='i1'), copy=False)
    if h.max() > 1:
        raise NotImplementedError('only biallelic variants are supported')
    if h.min() < 0:
        raise NotImplementedError('missing calls are not supported')

    # initialise
    n_variants = h.n_variants  # number of rows, i.e., variants
    n_haplotypes = h.n_haplotypes  # number of columns, i.e., haplotypes
    n_pairs = (n_haplotypes * (n_haplotypes - 1)) // 2

    # compute the shared prefix length between all pairs of haplotypes
    spl = pairwise_shared_prefix_lengths_int8(h)

    # compute EHH by counting the number of shared prefixes extending beyond
    # each variant
    minlength = None if truncate else n_variants + 1
    b = np.bincount(spl, minlength=minlength)
    c = np.cumsum(b[::-1])[:-1]
    ehh = (c / n_pairs)[::-1]

    return ehh


def voight_painting(h):
    """Paint haplotypes, assigning a unique integer to each shared haplotype
    prefix.

    Parameters
    ----------
    h : array_like, int, shape (n_variants, n_haplotypes)
        Haplotype array.

    Returns
    -------
    painting : ndarray, int, shape (n_variants, n_haplotypes)
        Painting array.

    """

    from allel.opt.stats import paint_shared_prefixes_int8

    # check inputs
    # N.B., ensure int8 so we can use cython optimisation
    h = HaplotypeArray(np.asarray(h, dtype='i1'), copy=False)
    if h.max() > 1:
        raise NotImplementedError('only biallelic variants are supported')
    if h.min() < 0:
        raise NotImplementedError('missing calls are not supported')

    return paint_shared_prefixes_int8(h)


def plot_voight_painting(painting, palette='colorblind', flank='right',
                         ax=None, height_factor=0.01):
    """Plot a painting of shared haplotype prefixes.

    Parameters
    ----------
    painting : array_like, int, shape (n_variants, n_haplotypes)
        Painting array.
    ax : axes, optional
        The axes on which to draw. If not provided, a new figure will be
        created.
    palette : string, optional
        A Seaborn palette name.
    flank : {'right', 'left'}, optional
        If left, painting will be reversed along first axis.
    height_factor : float, optional
        If no axes provided, determine height of figure by multiplying
        height of painting array by this number.

    """

    import seaborn as sns
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    if flank == 'left':
        painting = painting[::-1]

    n_colors = painting.max()
    palette = sns.color_palette(palette, n_colors)
    # use white for singleton haplotypes
    cmap = mpl.colors.ListedColormap(['white'] + palette)

    # setup axes
    if ax is None:
        x = plt.rcParams['figure.figsize'][0]
        y = height_factor*painting.shape[1]
        fig, ax = plt.subplots(figsize=(x, y))
        sns.despine(ax=ax, bottom=True, left=True)

    ax.pcolormesh(painting.T, cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, painting.shape[0])
    ax.set_ylim(0, painting.shape[1])


def xpehh(h1, h2, pos, min_ehh=0):
    """Compute the unstandardized cross-population extended haplotype
    homozygosity score (XPEHH) for each variant.

    Parameters
    ----------
    h1 : array_like, int, shape (n_variants, n_haplotypes)
        Haplotype array for the first population.
    h2 : array_like, int, shape (n_variants, n_haplotypes)
        Haplotype array for the second population.
    pos : array_like, int, shape (n_variants,)
        Variant positions on physical or genetic map.
    min_ehh: float, optional
        Minimum EHH beyond which to truncate integrated haplotype
        homozygosity calculation.

    Returns
    -------
    score : ndarray, float, shape (n_variants,)
        Unstandardized XPEHH scores.

    Notes
    -----

    This function will calculate XPEHH for all variants. To exclude variants
    below a given minor allele frequency, filter the input haplotype arrays
    before passing to this function.

    This function does nothing about XPEHH calculations where haplotype
    homozygosity extends up to the first or last variant. There will be edge
    effects.

    This function currently does nothing to account for large gaps between
    variants. There will be edge effects near any large gaps.

    Note that the unstandardized score is returned. Usually these scores are
    then normalised in different allele frequency bins.

    Haplotype arrays from the two populations may have different numbers of
    haplotypes.

    """

    from allel.opt.stats import ihh_scan_int8

    # scan forward
    ihh1_fwd = ihh_scan_int8(h1, pos, min_ehh=min_ehh)
    ihh2_fwd = ihh_scan_int8(h2, pos, min_ehh=min_ehh)

    # scan backward
    ihh1_rev = ihh_scan_int8(h1[::-1], pos[::-1], min_ehh=min_ehh)[::-1]
    ihh2_rev = ihh_scan_int8(h2[::-1], pos[::-1], min_ehh=min_ehh)[::-1]

    # compute unstandardized score
    ihh1 = ihh1_fwd + ihh1_rev
    ihh2 = ihh2_fwd + ihh2_rev
    score = np.log(ihh1 / ihh2)

    return score


def ihs(h, pos, min_ehh=0):
    """Compute the unstandardized integrated haplotype score (IHS) for each
    variant, comparing integrated haplotype homozygosity between the
    reference and alternate alleles.

    Parameters
    ----------
    h : array_like, int, shape (n_variants, n_haplotypes)
        Haplotype array.
    pos : array_like, int, shape (n_variants,)
        Variant positions on physical or genetic map.
    min_ehh: float, optional
        Minimum EHH beyond which to truncate integrated haplotype
        homozygosity calculation.

    Returns
    -------
    score : ndarray, float, shape (n_variants,)
        Unstandardized IHS scores.

    Notes
    -----

    This function will calculate IHS for all variants. To exclude variants
    below a given minor allele frequency, filter the input haplotype array
    before passing to this function.

    This function computes IHS comparing the reference and alternate alleles.
    These can be polarised by switching the sign for any variant where the
    reference allele is derived.

    This function does nothing about IHS calculations where haplotype
    homozygosity extends up to the first or last variant. There will be edge
    effects.

    This function currently does nothing to account for large gaps between
    variants. There will be edge effects near any large gaps.

    Note that the unstandardized score is returned. Usually these scores are
    then normalised in different allele frequency bins.

    """

    from allel.opt.stats import ihh01_scan_int8

    # scan forward
    ihh0_fwd, ihh1_fwd = ihh01_scan_int8(h, pos, min_ehh=min_ehh)

    # scan backward
    ihh0_rev, ihh1_rev = ihh01_scan_int8(h[::-1], pos[::-1], min_ehh=min_ehh)
    ihh0_rev = ihh0_rev[::-1]
    ihh1_rev = ihh1_rev[::-1]

    # compute unstandardized score
    ihh0 = ihh0_fwd + ihh0_rev
    ihh1 = ihh1_fwd + ihh1_rev
    score = np.log(ihh1 / ihh0)

    return score
