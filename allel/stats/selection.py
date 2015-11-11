# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np


from allel.util import asarray_ndim
from allel.model.ndarray import HaplotypeArray
from allel.stats.window import moving_statistic


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
    indices : ndarray, int, shape (n_hapotypes,)
        Haplotype indices after sorting by prefix.

    """

    from allel.opt.stats import paint_shared_prefixes_int8

    # check inputs
    # N.B., ensure int8 so we can use cython optimisation
    h = HaplotypeArray(np.asarray(h, dtype='i1'), copy=False)
    if h.max() > 1:
        raise NotImplementedError('only biallelic variants are supported')
    if h.min() < 0:
        raise NotImplementedError('missing calls are not supported')

    # sort by prefix
    indices = h.prefix_argsort()
    h = np.take(h, indices, axis=1)

    # paint
    painting = paint_shared_prefixes_int8(h)

    return painting, indices


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

    Returns
    -------
    ax : axes

    """

    import seaborn as sns
    from matplotlib.colors import ListedColormap
    import matplotlib.pyplot as plt

    if flank == 'left':
        painting = painting[::-1]

    n_colors = painting.max()
    palette = sns.color_palette(palette, n_colors)
    # use white for singleton haplotypes
    cmap = ListedColormap(['white'] + palette)

    # setup axes
    if ax is None:
        w = plt.rcParams['figure.figsize'][0]
        h = height_factor*painting.shape[1]
        fig, ax = plt.subplots(figsize=(w, h))
        sns.despine(ax=ax, bottom=True, left=True)

    ax.pcolormesh(painting.T, cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, painting.shape[0])
    ax.set_ylim(0, painting.shape[1])


def fig_voight_painting(h, index=None, palette='colorblind',
                        height_factor=0.01, fig=None):
    """Make a figure of shared haplotype prefixes for both left and right
    flanks, centred on some variant of choice.

    Parameters
    ----------
    h : array_like, int, shape (n_variants, n_haplotypes)
        Haplotype array.
    index : int, optional
        Index of the variant within the haplotype array to centre on. If not
        provided, the middle variant will be used.
    palette : string, optional
        A Seaborn palette name.
    height_factor : float, optional
        If no axes provided, determine height of figure by multiplying
        height of painting array by this number.
    fig : figure
        The figure on which to draw. If not provided, a new figure will be
        created.

    Returns
    -------
    fig : figure

    Notes
    -----
    N.B., the ordering of haplotypes on the left and right flanks will be
    different. This means that haplotypes on the right flank **will not**
    correspond to haplotypes on the left flank at the same vertical position.

    """

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import seaborn as sns

    # check inputs
    h = asarray_ndim(h, 2)
    if index is None:
        # use midpoint
        index = h.shape[0] // 2

    # divide data into two flanks
    hl = h[:index+1][::-1]
    hr = h[index:]

    # paint both flanks
    pl, il = voight_painting(hl)
    pr, ir = voight_painting(hr)

    # compute ehh decay for both flanks
    el = ehh_decay(hl, truncate=False)
    er = ehh_decay(hr, truncate=False)

    # setup figure
    # fixed height for EHH decay subplot
    h_ehh = plt.rcParams['figure.figsize'][1] // 3
    # add height for paintings
    h_painting = height_factor*h.shape[1]
    if fig is None:
        w = plt.rcParams['figure.figsize'][0]
        h = h_ehh + h_painting
        fig = plt.figure(figsize=(w, h))

    # setup gridspec
    gs = GridSpec(2, 2,
                  width_ratios=[hl.shape[0], hr.shape[0]],
                  height_ratios=[h_painting, h_ehh])

    # plot paintings
    ax = fig.add_subplot(gs[0, 0])
    sns.despine(ax=ax, left=True, bottom=True)
    plot_voight_painting(pl, palette=palette, flank='left', ax=ax)
    ax = fig.add_subplot(gs[0, 1])
    sns.despine(ax=ax, left=True, bottom=True)
    plot_voight_painting(pr, palette=palette, flank='right', ax=ax)

    # plot ehh
    ax = fig.add_subplot(gs[1, 0])
    sns.despine(ax=ax, offset=3)
    x = np.arange(el.shape[0])
    y = el
    ax.fill_between(x, 0, y)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 1])
    ax.set_ylabel('EHH')
    ax.invert_xaxis()
    ax = fig.add_subplot(gs[1, 1])
    sns.despine(ax=ax, left=True, right=False, offset=3)
    ax.yaxis.tick_right()
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 1])
    x = np.arange(er.shape[0])
    y = er
    ax.fill_between(x, 0, y)

    # tidy up
    fig.tight_layout()

    return fig


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


def ihs(h, pos, min_ehh=0.05):
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

    assert min_ehh > 1.0/h.shape[1], \
        "There are insufficient haplotypes (n={nhap}) to decay to an EHH of " \
        "{minehh}".format(nhap=h.shape[0], minehh=min_ehh)

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


def plot_haplotype_frequencies(h, palette='Set1', singleton_color='#dddddd',
                               ax=None):
    """Plot haplotype frequencies.

    Parameters
    ----------
    h : array_like, int, shape (n_variants, n_haplotypes)
        Haplotype array.
    palette : string, optional
        A Seaborn palette name.
    ax : axes, optional
        The axes on which to draw. If not provided, a new figure will be
        created.

    Returns
    -------
    ax : axes

    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    # check inputs
    h = HaplotypeArray(h, copy=False)

    # setup figure
    if ax is None:
        width = plt.rcParams['figure.figsize'][0]
        height = width / 10
        fig, ax = plt.subplots(figsize=(width, height))
        sns.despine(ax=ax, left=True)

    # count distinct haplotypes
    hc = h.distinct_counts()

    # setup palette
    n_colors = np.count_nonzero(hc > 1)
    palette = sns.color_palette(palette, n_colors)

    # paint frequencies
    x1 = 0
    for i, c in enumerate(hc):
        x2 = x1 + c
        if c > 1:
            color = palette[i]
        else:
            color = singleton_color
        ax.axvspan(x1, x2, color=color)
        x1 = x2

    # tidy up
    ax.set_xlim(0, h.shape[1])
    ax.set_yticks([])

    return ax


def garud_h(h):
    """Compute the H1, H12, H123 and H2/H1 statistics for detecting signatures
    of soft sweeps, as defined in Garud et al. (2015).

    Parameters
    ----------
    h : array_like, int, shape (n_variants, n_haplotypes)
        Haplotype array.

    Returns
    -------
    h1 : float
        H1 statistic (sum of squares of haplotype frequencies).
    h12 : float
        H12 statistic (sum of squares of haplotype frequencies, combining
        the two most common haplotypes into a single frequency).
    h123 : float
        H123 statistic (sum of squares of haplotype frequencies, combining
        the three most common haplotypes into a single frequency).
    h2_h1 : float
        H2/H1 statistic, indicating the "softness" of a sweep.

    """

    # check inputs
    h = HaplotypeArray(h, copy=False)

    # compute haplotype frequencies
    f = h.distinct_frequencies()

    # compute H1
    h1 = np.sum(f**2)

    # compute H12
    h12 = np.sum(f[:2])**2 + np.sum(f[2:]**2)

    # compute H123
    h123 = np.sum(f[:3])**2 + np.sum(f[3:]**2)

    # compute H2/H1
    h2 = h1 - f[0]**2
    h2_h1 = h2 / h1

    return h1, h12, h123, h2_h1


def moving_garud_h(h, size, start=0, stop=None, step=None):
    """Compute the H1, H12, H123 and H2/H1 statistics for detecting signatures
    of soft sweeps, as defined in Garud et al. (2015), in moving windows,

    Parameters
    ----------
    h : array_like, int, shape (n_variants, n_haplotypes)
        Haplotype array.
    size : int
        The window size (number of variants).
    start : int, optional
        The index at which to start.
    stop : int, optional
        The index at which to stop.
    step : int, optional
        The number of variants between start positions of windows. If not
        given, defaults to the window size, i.e., non-overlapping windows.

    Returns
    -------
    h1 : ndarray, float, shape (n_windows,)
        H1 statistics (sum of squares of haplotype frequencies).
    h12 : ndarray, float, shape (n_windows,)
        H12 statistics (sum of squares of haplotype frequencies, combining
        the two most common haplotypes into a single frequency).
    h123 : ndarray, float, shape (n_windows,)
        H123 statistics (sum of squares of haplotype frequencies, combining
        the three most common haplotypes into a single frequency).
    h2_h1 : ndarray, float, shape (n_windows,)
        H2/H1 statistics, indicating the "softness" of a sweep.

    """

    h = moving_statistic(values=h, statistic=garud_h, size=size, start=start,
                         stop=stop, step=step)

    h1 = h[:, 0]
    h12 = h[:, 1]
    h123 = h[:, 2]
    h2_h1 = h[:, 3]

    return h1, h12, h123, h2_h1
