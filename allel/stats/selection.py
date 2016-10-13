# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import multiprocessing
from multiprocessing.pool import ThreadPool


import numpy as np


from allel.util import asarray_ndim, check_dim0_aligned, check_ndim
from allel.model.ndarray import HaplotypeArray
from allel.stats.window import moving_statistic, index_windows
from allel.stats.diversity import moving_tajima_d


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

    return ax


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


def compute_ihh_gaps(pos, map_pos, gap_scale, max_gap, is_accessible):
    """Compute spacing between variants for integrating haplotype
    homozygosity.

    Parameters
    ----------
    pos : array_like, int, shape (n_variants,)
        Variant positions (physical distance).
    map_pos : array_like, float, shape (n_variants,)
        Variant positions (genetic map distance).
    gap_scale : int, optional
        Rescale distance between variants if gap is larger than this value.
    max_gap : int, optional
        Do not report scores if EHH spans a gap larger than this number of
        base pairs.
    is_accessible : array_like, bool, optional
        Genome accessibility array. If provided, distance between variants
        will be computed as the number of accessible bases between them.

    Returns
    -------
    gaps : ndarray, float, shape (n_variants - 1,)

    """

    # check inputs
    if map_pos is None:
        # integrate over physical distance
        map_pos = pos
    else:
        map_pos = asarray_ndim(map_pos, 1)
        check_dim0_aligned(pos, map_pos)

    # compute physical gaps
    physical_gaps = np.diff(pos)

    # compute genetic gaps
    gaps = np.diff(map_pos).astype('f8')

    if is_accessible is not None:

        # compute accessible gaps
        is_accessible = asarray_ndim(is_accessible, 1)
        assert is_accessible.shape[0] > pos[-1], \
            'accessibility array too short'
        accessible_gaps = np.zeros_like(physical_gaps)
        for i in range(1, len(pos)):
            # N.B., expect pos is 1-based
            n_access = np.count_nonzero(is_accessible[pos[i-1]-1:pos[i]-1])
            accessible_gaps[i-1] = n_access

        # adjust using accessibility
        scaling = accessible_gaps / physical_gaps
        gaps = gaps * scaling

    elif gap_scale is not None and gap_scale > 0:

        scaling = np.ones(gaps.shape, dtype='f8')
        loc_scale = physical_gaps > gap_scale
        scaling[loc_scale] = gap_scale / physical_gaps[loc_scale]
        gaps = gaps * scaling

    if max_gap is not None and max_gap > 0:

        # deal with very large gaps
        gaps[physical_gaps > max_gap] = -1

    return gaps


def ihs(h, pos, map_pos=None, min_ehh=0.05, min_maf=0.05, include_edges=False,
        gap_scale=20000, max_gap=200000, is_accessible=None, use_threads=True):
    """Compute the unstandardized integrated haplotype score (IHS) for each
    variant, comparing integrated haplotype homozygosity between the
    reference (0) and alternate (1) alleles.

    Parameters
    ----------
    h : array_like, int, shape (n_variants, n_haplotypes)
        Haplotype array.
    pos : array_like, int, shape (n_variants,)
        Variant positions (physical distance).
    map_pos : array_like, float, shape (n_variants,)
        Variant positions (genetic map distance).
    min_ehh: float, optional
        Minimum EHH beyond which to truncate integrated haplotype
        homozygosity calculation.
    min_maf : float, optional
        Do not compute integrated haplotype homozogysity for variants with
        minor allele frequency below this value.
    include_edges : bool, optional
        If True, report scores even if EHH does not decay below `min_ehh`
        before reaching the edge of the data.
    gap_scale : int, optional
        Rescale distance between variants if gap is larger than this value.
    max_gap : int, optional
        Do not report scores if EHH spans a gap larger than this number of
        base pairs.
    is_accessible : array_like, bool, optional
        Genome accessibility array. If provided, distance between variants
        will be computed as the number of accessible bases between them.
    use_threads : bool, optional
        If True use multiple threads to compute.

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

    This function returns NaN for any IHS calculations where haplotype
    homozygosity does not decay below `min_ehh` before reaching the first or
    last variant. To disable this behaviour, set `include_edges` to True.

    Note that the unstandardized score is returned. Usually these scores are
    then standardized in different allele frequency bins.

    See Also
    --------
    standardize_by_allele_count

    """

    from allel.opt.stats import ihh01_scan_int8

    # check inputs
    h = np.asarray(h, dtype='i1')
    check_ndim(h, 2)
    pos = asarray_ndim(pos, 1)
    check_dim0_aligned(h, pos)

    # compute gaps between variants for integration
    gaps = compute_ihh_gaps(pos, map_pos, gap_scale, max_gap, is_accessible)

    # setup kwargs
    kwargs = dict(min_ehh=min_ehh, min_maf=min_maf, include_edges=include_edges)

    if use_threads and multiprocessing.cpu_count() > 1:
        # run with threads

        # create pool
        pool = ThreadPool(2)

        # scan forward
        result_fwd = pool.apply_async(ihh01_scan_int8, (h, gaps), kwargs)

        # scan backward
        result_rev = pool.apply_async(ihh01_scan_int8, (h[::-1], gaps[::-1]),
                                      kwargs)

        # wait for both to finish
        pool.close()
        pool.join()

        # obtain results
        ihh0_fwd, ihh1_fwd = result_fwd.get()
        ihh0_rev, ihh1_rev = result_rev.get()

        # cleanup
        pool.terminate()

    else:
        # run without threads

        # scan forward
        ihh0_fwd, ihh1_fwd = ihh01_scan_int8(h, gaps, **kwargs)

        # scan backward
        ihh0_rev, ihh1_rev = ihh01_scan_int8(h[::-1], gaps[::-1], **kwargs)

    # handle reverse scan
    ihh0_rev = ihh0_rev[::-1]
    ihh1_rev = ihh1_rev[::-1]

    # compute unstandardized score
    ihh0 = ihh0_fwd + ihh0_rev
    ihh1 = ihh1_fwd + ihh1_rev
    score = np.log(ihh1 / ihh0)

    return score


def xpehh(h1, h2, pos, map_pos=None, min_ehh=0.05, include_edges=False,
          gap_scale=20000, max_gap=200000, is_accessible=None,
          use_threads=True):
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
    map_pos : array_like, float, shape (n_variants,)
        Variant positions (genetic map distance).
    min_ehh: float, optional
        Minimum EHH beyond which to truncate integrated haplotype
        homozygosity calculation.
    include_edges : bool, optional
        If True, report scores even if EHH does not decay below `min_ehh`
        before reaching the edge of the data.
    gap_scale : int, optional
        Rescale distance between variants if gap is larger than this value.
    max_gap : int, optional
        Do not report scores if EHH spans a gap larger than this number of
        base pairs.
    is_accessible : array_like, bool, optional
        Genome accessibility array. If provided, distance between variants
        will be computed as the number of accessible bases between them.
    use_threads : bool, optional
        If True use multiple threads to compute.

    Returns
    -------
    score : ndarray, float, shape (n_variants,)
        Unstandardized XPEHH scores.

    Notes
    -----

    This function will calculate XPEHH for all variants. To exclude variants
    below a given minor allele frequency, filter the input haplotype arrays
    before passing to this function.

    This function returns NaN for any EHH calculations where haplotype
    homozygosity does not decay below `min_ehh` before reaching the first or
    last variant. To disable this behaviour, set `include_edges` to True.

    Note that the unstandardized score is returned. Usually these scores are
    then standardized genome-wide.

    Haplotype arrays from the two populations may have different numbers of
    haplotypes.

    See Also
    --------
    standardize

    """

    from allel.opt.stats import ihh_scan_int8

    # check inputs
    h1 = np.asarray(h1, dtype='i1')
    h2 = np.asarray(h2, dtype='i1')
    check_ndim(h1, 2)
    check_ndim(h2, 2)
    pos = asarray_ndim(pos, 1)
    check_dim0_aligned(h1, h2, pos)

    # compute gaps between variants for integration
    gaps = compute_ihh_gaps(pos, map_pos, gap_scale, max_gap, is_accessible)

    # setup kwargs
    kwargs = dict(min_ehh=min_ehh, include_edges=include_edges)

    if use_threads and multiprocessing.cpu_count() > 1:
        # use multiple threads

        # setup threadpool
        pool = ThreadPool(min(4, multiprocessing.cpu_count()))

        # scan forward
        res1_fwd = pool.apply_async(ihh_scan_int8, (h1, gaps), kwargs)
        res2_fwd = pool.apply_async(ihh_scan_int8, (h2, gaps), kwargs)

        # scan backward
        res1_rev = pool.apply_async(ihh_scan_int8, (h1[::-1], gaps[::-1]), kwargs)
        res2_rev = pool.apply_async(ihh_scan_int8, (h2[::-1], gaps[::-1]), kwargs)

        # wait for both to finish
        pool.close()
        pool.join()

        # obtain results
        ihh1_fwd = res1_fwd.get()
        ihh2_fwd = res2_fwd.get()
        ihh1_rev = res1_rev.get()
        ihh2_rev = res2_rev.get()

        # cleanup
        pool.terminate()

    else:
        # compute without threads

        # scan forward
        ihh1_fwd = ihh_scan_int8(h1, gaps, **kwargs)
        ihh2_fwd = ihh_scan_int8(h2, gaps, **kwargs)

        # scan backward
        ihh1_rev = ihh_scan_int8(h1[::-1], gaps[::-1], **kwargs)
        ihh2_rev = ihh_scan_int8(h2[::-1], gaps[::-1], **kwargs)

    # handle reverse scans
    ihh1_rev = ihh1_rev[::-1]
    ihh2_rev = ihh2_rev[::-1]

    # compute unstandardized score
    ihh1 = ihh1_fwd + ihh1_rev
    ihh2 = ihh2_fwd + ihh2_rev
    score = np.log(ihh1 / ihh2)

    return score


def nsl(h, use_threads=True):
    """Compute the unstandardized number of segregating sites by length (nSl)
    for each variant, comparing the reference and alternate alleles,
    after Ferrer-Admetlla et al. (2014).

    Parameters
    ----------
    h : array_like, int, shape (n_variants, n_haplotypes)
        Haplotype array.
    use_threads : bool, optional
        If True use multiple threads to compute.

    Returns
    -------
    score : ndarray, float, shape (n_variants,)

    Notes
    -----
    This function will calculate nSl for all variants. To exclude variants
    below a given minor allele frequency, filter the input haplotype array
    before passing to this function.

    This function computes nSl by comparing the reference and alternate
    alleles. These can be polarised by switching the sign for any variant where
    the reference allele is derived.

    This function does nothing about nSl calculations where haplotype
    homozygosity extends up to the first or last variant. There may be edge
    effects.

    Note that the unstandardized score is returned. Usually these scores are
    then standardized in different allele frequency bins.

    See Also
    --------
    standardize_by_allele_count

    """

    from allel.opt.stats import nsl01_scan_int8

    # check inputs
    h = np.asarray(h, dtype='i1')
    check_ndim(h, 2)

    # # check there are no invariant sites
    # ac = h.count_alleles()
    # assert np.all(ac.is_segregating()), 'please remove non-segregating sites'

    if use_threads and multiprocessing.cpu_count() > 1:

        # create pool
        pool = ThreadPool(2)

        # scan forward
        result_fwd = pool.apply_async(nsl01_scan_int8, args=(h,))

        # scan backward
        result_rev = pool.apply_async(nsl01_scan_int8, args=(h[::-1],))

        # wait for both to finish
        pool.close()
        pool.join()

        # obtain results
        nsl0_fwd, nsl1_fwd = result_fwd.get()
        nsl0_rev, nsl1_rev = result_rev.get()

    else:

        # scan forward
        nsl0_fwd, nsl1_fwd = nsl01_scan_int8(h)

        # scan backward
        nsl0_rev, nsl1_rev = nsl01_scan_int8(h[::-1])

    # handle backwards
    nsl0_rev = nsl0_rev[::-1]
    nsl1_rev = nsl1_rev[::-1]

    # compute unstandardized score
    nsl0 = nsl0_fwd + nsl0_rev
    nsl1 = nsl1_fwd + nsl1_rev
    score = np.log(nsl1 / nsl0)

    return score


def xpnsl(h1, h2, use_threads=True):
    """Cross-population version of the NSL statistic.

    Parameters
    ----------
    h1 : array_like, int, shape (n_variants, n_haplotypes)
        Haplotype array for the first population.
    h2 : array_like, int, shape (n_variants, n_haplotypes)
        Haplotype array for the second population.
    use_threads : bool, optional
        If True use multiple threads to compute.

    Returns
    -------
    score : ndarray, float, shape (n_variants,)
        Unstandardized XPNSL scores.

    """
    from allel.opt.stats import nsl_scan_int8

    # check inputs
    h1 = np.asarray(h1, dtype='i1')
    h2 = np.asarray(h2, dtype='i1')
    check_ndim(h1, 2)
    check_ndim(h2, 2)
    check_dim0_aligned(h1, h2)

    if use_threads and multiprocessing.cpu_count() > 1:
        # use multiple threads

        # setup threadpool
        pool = ThreadPool(min(4, multiprocessing.cpu_count()))

        # scan forward
        res1_fwd = pool.apply_async(nsl_scan_int8, args=(h1,))
        res2_fwd = pool.apply_async(nsl_scan_int8, args=(h2,))

        # scan backward
        res1_rev = pool.apply_async(nsl_scan_int8, args=(h1[::-1],))
        res2_rev = pool.apply_async(nsl_scan_int8, args=(h2[::-1],))

        # wait for both to finish
        pool.close()
        pool.join()

        # obtain results
        nsl1_fwd = res1_fwd.get()
        nsl2_fwd = res2_fwd.get()
        nsl1_rev = res1_rev.get()
        nsl2_rev = res2_rev.get()

        # cleanup
        pool.terminate()

    else:
        # compute without threads

        # scan forward
        nsl1_fwd = nsl_scan_int8(h1)
        nsl2_fwd = nsl_scan_int8(h2)

        # scan backward
        nsl1_rev = nsl_scan_int8(h1[::-1])
        nsl2_rev = nsl_scan_int8(h2[::-1])

    # handle reverse scans
    nsl1_rev = nsl1_rev[::-1]
    nsl2_rev = nsl2_rev[::-1]

    # compute unstandardized score
    nsl1 = nsl1_fwd + nsl1_rev
    nsl2 = nsl2_fwd + nsl2_rev
    score = np.log(nsl1 / nsl2)

    return score


def haplotype_diversity(h):
    """Estimate haplotype diversity.

    Parameters
    ----------
    h : array_like, int, shape (n_variants, n_haplotypes)
        Haplotype array.

    Returns
    -------
    hd : float
        Haplotype diversity.

    """

    # check inputs
    h = HaplotypeArray(h, copy=False)

    # number of haplotypes
    n = h.n_haplotypes

    # compute haplotype frequencies
    f = h.distinct_frequencies()

    # estimate haplotype diversity
    hd = (1 - np.sum(f**2)) * n / (n - 1)

    return hd


def moving_haplotype_diversity(h, size, start=0, stop=None, step=None):
    """Estimate haplotype diversity in moving windows.

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
    hd : ndarray, float, shape (n_windows,)
        Haplotype diversity.

    """

    hd = moving_statistic(values=h, statistic=haplotype_diversity, size=size,
                          start=start, stop=stop, step=step)
    return hd


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

    gh = moving_statistic(values=h, statistic=garud_h, size=size, start=start,
                          stop=stop, step=step)

    h1 = gh[:, 0]
    h12 = gh[:, 1]
    h123 = gh[:, 2]
    h2_h1 = gh[:, 3]

    return h1, h12, h123, h2_h1


def plot_haplotype_frequencies(h, palette='Paired', singleton_color='w',
                               ax=None):
    """Plot haplotype frequencies.

    Parameters
    ----------
    h : array_like, int, shape (n_variants, n_haplotypes)
        Haplotype array.
    palette : string, optional
        A Seaborn palette name.
    singleton_color : string, optional
        Color to paint singleton haplotypes.
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


def moving_hfs_rank(h, size, start=0, stop=None):
    """Helper function for plotting haplotype frequencies in moving windows.

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

    Returns
    -------
    hr : ndarray, int, shape (n_windows, n_haplotypes)
        Haplotype rank array.

    """

    # determine windows
    windows = np.asarray(list(index_windows(h, size=size, start=start,
                                            stop=stop, step=None)))

    # setup output
    hr = np.zeros((windows.shape[0], h.shape[1]), dtype='i4')

    # iterate over windows
    for i, (window_start, window_stop) in enumerate(windows):

        # extract haplotypes for the current window
        hw = h[window_start:window_stop]

        # count haplotypes
        hc = hw.distinct_counts()

        # ensure sorted descending
        hc.sort()
        hc = hc[::-1]

        # compute ranks for non-singleton haplotypes
        cp = 0
        for j, c in enumerate(hc):
            if c > 1:
                hr[i, cp:cp+c] = j+1
            cp += c

    return hr


def plot_moving_haplotype_frequencies(pos, h, size, start=0, stop=None, n=None,
                                      palette='Paired', singleton_color='w',
                                      ax=None):
    """Plot haplotype frequencies in moving windows over the genome.

    Parameters
    ----------
    pos : array_like, int, shape (n_items,)
        Variant positions, using 1-based coordinates, in ascending order.
    h : array_like, int, shape (n_variants, n_haplotypes)
        Haplotype array.
    size : int
        The window size (number of variants).
    start : int, optional
        The index at which to start.
    stop : int, optional
        The index at which to stop.
    n : int, optional
        Color only the `n` most frequent haplotypes (by default, all
        non-singleton haplotypes are colored).
    palette : string, optional
        A Seaborn palette name.
    singleton_color : string, optional
        Color to paint singleton haplotypes.
    ax : axes, optional
        The axes on which to draw. If not provided, a new figure will be
        created.

    Returns
    -------
    ax : axes

    """

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns

    # setup figure
    if ax is None:
        fig, ax = plt.subplots()

    # compute haplotype frequencies
    # N.B., here we use a haplotype rank data structure to enable the use of
    # pcolormesh() which is a lot faster than any other type of plotting
    # function
    hr = moving_hfs_rank(h, size=size, start=start, stop=stop)

    # truncate to n most common haplotypes
    if n:
        hr[hr > n] = 0

    # compute window start and stop positions
    windows = moving_statistic(pos, statistic=lambda v: (v[0], v[-1]),
                               size=size, start=start, stop=stop)

    # create color map
    colors = [singleton_color] + sns.color_palette(palette, n_colors=hr.max())
    cmap = mpl.colors.ListedColormap(colors)

    # draw colors
    x = np.append(windows[:, 0], windows[-1, -1])
    y = np.arange(h.shape[1]+1)
    ax.pcolormesh(x, y, hr.T, cmap=cmap)

    # tidy up
    ax.set_xlim(windows[0, 0], windows[-1, -1])
    ax.set_ylim(0, h.shape[1])
    ax.set_ylabel('haplotype count')
    ax.set_xlabel('position (bp)')

    return ax


def moving_delta_tajima_d(ac1, ac2, size, start=0, stop=None, step=None):
    """Compute the difference in Tajima's D between two populations in
    moving windows.

    Parameters
    ----------
    ac1 : array_like, int, shape (n_variants, n_alleles)
        Allele counts array for the first population.
    ac2 : array_like, int, shape (n_variants, n_alleles)
        Allele counts array for the second population.
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
    delta_d : ndarray, float, shape (n_windows,)
        Standardized delta Tajima's D.

    See Also
    --------
    allel.stats.diversity.moving_tajima_d

    """

    d1 = moving_tajima_d(ac1, size=size, start=start, stop=stop, step=step)
    d2 = moving_tajima_d(ac2, size=size, start=start, stop=stop, step=step)
    delta = d1 - d2
    delta_z = (delta - np.mean(delta)) / np.std(delta)
    return delta_z


def make_similar_sized_bins(x, n):
    """Utility function to create a set of bins over the range of values in `x`
    such that each bin contains roughly the same number of values.

    Parameters
    ----------
    x : array_like
        The values to be binned.
    n : int
        The number of bins to create.

    Returns
    -------
    bins : ndarray
        An array of bin edges.

    Notes
    -----
    The actual number of bins returned may be less than `n` if `x` contains
    integer values and any single value is represented more than len(x)//n
    times.

    """
    # copy and sort the array
    y = np.array(x).flatten()
    y.sort()

    # setup bins
    bins = [y[0]]

    # determine step size
    step = len(y) // n

    # add bin edges
    for i in range(step, len(y), step):

        # get value at this index
        v = y[i]

        # only add bin edge if larger than previous
        if v > bins[-1]:
            bins.append(v)

    # fix last bin edge
    bins[-1] = y[-1]

    return np.array(bins)


def standardize(score):
    """Centre and scale to unit variance."""
    score = asarray_ndim(score, 1)
    return (score - np.nanmean(score)) / np.nanstd(score)


def standardize_by_allele_count(score, aac, bins=None, n_bins=None,
                                diagnostics=True):
    """Standardize `score` within allele frequency bins.

    Parameters
    ----------
    score : array_like, float
        The score to be standardized, e.g., IHS or NSL.
    aac : array_like, int
        An array of alternate allele counts.
    bins : array_like, int, optional
        Allele count bins, overrides `n_bins`.
    n_bins : int, optional
        Number of allele count bins to use.
    diagnostics : bool, optional
        If True, plot some diagnostic information about the standardization.

    Returns
    -------
    score_standardized : ndarray, float
        Standardized scores.
    bins : ndarray, int
        Allele count bins used for standardization.

    """

    from scipy.stats import binned_statistic

    # check inputs
    score = asarray_ndim(score, 1)
    aac = asarray_ndim(aac, 1)
    check_dim0_aligned(score, aac)

    # remove nans
    nonan = ~np.isnan(score)
    score_nonan = score[nonan]
    aac_nonan = aac[nonan]

    if bins is None:
        # make our own similar sized bins

        # how many bins to make?
        if n_bins is None:
            # something vaguely reasonable
            n_bins = np.max(aac) // 2

        # make bins
        bins = make_similar_sized_bins(aac_nonan, n_bins)

    else:
        # user-provided bins
        bins = asarray_ndim(bins, 1)

    mean_score, _, _ = binned_statistic(aac_nonan, score_nonan,
                                        statistic=np.mean,
                                        bins=bins)
    std_score, _, _ = binned_statistic(aac_nonan, score_nonan,
                                       statistic=np.std,
                                       bins=bins)

    if diagnostics:
        import matplotlib.pyplot as plt
        x = (bins[:-1] + bins[1:]) / 2
        plt.figure()
        plt.fill_between(x,
                         mean_score - std_score,
                         mean_score + std_score,
                         alpha=.5,
                         label='std')
        plt.plot(x, mean_score, marker='o', label='mean')
        plt.grid(axis='y')
        plt.xlabel('Alternate allele count')
        plt.ylabel('Unstandardized score')
        plt.title('Standardization diagnostics')
        plt.legend()

    # apply standardization
    score_standardized = np.empty_like(score)
    for i in range(len(bins) - 1):
        x1 = bins[i]
        x2 = bins[i + 1]
        if i == 0:
            # first bin
            loc = (aac < x2)
        elif i == len(bins) - 2:
            # last bin
            loc = (aac >= x1)
        else:
            # middle bins
            loc = (aac >= x1) & (aac < x2)
        m = mean_score[i]
        s = std_score[i]
        score_standardized[loc] = (score[loc] - m) / s

    return score_standardized, bins
