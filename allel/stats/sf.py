# -*- coding: utf-8 -*-
import numpy as np


from allel.util import asarray_ndim, check_integer_dtype


def _check_dac_n(dac, n):
    dac = asarray_ndim(dac, 1)
    check_integer_dtype(dac)
    mx = np.max(dac)
    if n is None:
        n = mx
    elif n < mx:
        raise ValueError('number of chromosomes too small; expected {}, found {}'
                         .format(n, mx))
    return dac, int(n)


def _check_ac_n(ac, n):
    ac = asarray_ndim(ac, 2)
    if ac.shape[1] != 2:
        raise ValueError('only biallelic variants are supported')
    check_integer_dtype(ac)
    mx = np.max(np.sum(ac, axis=1))
    if n is None:
        n = mx
    elif n < mx:
        raise ValueError('number of chromosomes too small; expected {}, found {}'
                         .format(n, mx))
    return ac, int(n)


def sfs(dac, n=None):
    """Compute the site frequency spectrum given derived allele counts at
    a set of biallelic variants.

    Parameters
    ----------
    dac : array_like, int, shape (n_variants,)
        Array of derived allele counts.
    n : int, optional
        The total number of chromosomes called.

    Returns
    -------
    sfs : ndarray, int, shape (n_chromosomes,)
        Array where the kth element is the number of variant sites with k
        derived alleles.

    """

    # check input
    dac, n = _check_dac_n(dac, n)

    # need platform integer for bincount
    dac = dac.astype(int, copy=False)

    # compute site frequency spectrum
    x = n + 1
    s = np.bincount(dac, minlength=x)

    return s


def sfs_folded(ac, n=None):
    """Compute the folded site frequency spectrum given reference and
    alternate allele counts at a set of biallelic variants.

    Parameters
    ----------
    ac : array_like, int, shape (n_variants, 2)
        Allele counts array.
    n : int, optional
        The total number of chromosomes called.

    Returns
    -------
    sfs_folded : ndarray, int, shape (n_chromosomes//2,)
        Array where the kth element is the number of variant sites with a
        minor allele count of k.

    """

    # check input
    ac, n = _check_ac_n(ac, n)

    # compute minor allele counts
    mac = np.amin(ac, axis=1)

    # need platform integer for bincount
    mac = mac.astype(int, copy=False)

    # compute folded site frequency spectrum
    x = n//2 + 1
    s = np.bincount(mac, minlength=x)

    return s


def sfs_scaled(dac, n=None):
    """Compute the site frequency spectrum scaled such that a constant value is
    expected across the spectrum for neutral variation and constant
    population size.

    Parameters
    ----------
    dac : array_like, int, shape (n_variants,)
        Array of derived allele counts.
    n : int, optional
        The total number of chromosomes called.

    Returns
    -------
    sfs_scaled : ndarray, int, shape (n_chromosomes,)
        An array where the value of the kth element is the number of variants
        with k derived alleles, multiplied by k.

    """

    # compute site frequency spectrum
    s = sfs(dac, n=n)

    # apply scaling
    s = scale_sfs(s)

    return s


def scale_sfs(s):
    """Scale a site frequency spectrum.

    Parameters
    ----------
    s : array_like, int, shape (n_chromosomes,)
        Site frequency spectrum.

    Returns
    -------
    sfs_scaled : ndarray, int, shape (n_chromosomes,)
        Scaled site frequency spectrum.

    """
    k = np.arange(s.size)
    out = s * k
    return out


def sfs_folded_scaled(ac, n=None):
    """Compute the folded site frequency spectrum scaled such that a constant
    value is expected across the spectrum for neutral variation and constant
    population size.

    Parameters
    ----------
    ac : array_like, int, shape (n_variants, 2)
        Allele counts array.
    n : int, optional
        The total number of chromosomes called.

    Returns
    -------
    sfs_folded_scaled : ndarray, int, shape (n_chromosomes//2,)
        An array where the value of the kth element is the number of variants
        with minor allele count k, multiplied by the scaling factor
        (k * (n - k) / n).

    """

    # check input
    ac, n = _check_ac_n(ac, n)

    # compute the site frequency spectrum
    s = sfs_folded(ac, n=n)

    # apply scaling
    s = scale_sfs_folded(s, n)

    return s


def scale_sfs_folded(s, n):
    """Scale a folded site frequency spectrum.

    Parameters
    ----------
    s : array_like, int, shape (n_chromosomes//2,)
        Folded site frequency spectrum.
    n : int
        Number of chromosomes called.

    Returns
    -------
    sfs_folded_scaled : ndarray, int, shape (n_chromosomes//2,)
        Scaled folded site frequency spectrum.

    """
    k = np.arange(s.shape[0])
    out = s * k * (n - k) / n
    return out


def joint_sfs(dac1, dac2, n1=None, n2=None):
    """Compute the joint site frequency spectrum between two populations.

    Parameters
    ----------
    dac1 : array_like, int, shape (n_variants,)
        Derived allele counts for the first population.
    dac2 : array_like, int, shape (n_variants,)
        Derived allele counts for the second population.
    n1, n2 : int, optional
        The total number of chromosomes called in each population.

    Returns
    -------
    joint_sfs : ndarray, int, shape (m_chromosomes, n_chromosomes)
        Array where the (i, j)th element is the number of variant sites with i
        derived alleles in the first population and j derived alleles in the
        second population.

    """

    # check inputs
    dac1, n1 = _check_dac_n(dac1, n1)
    dac2, n2 = _check_dac_n(dac2, n2)

    # compute site frequency spectrum
    x = n1 + 1
    y = n2 + 1
    # need platform integer for bincount
    tmp = (dac1 * y + dac2).astype(int, copy=False)
    s = np.bincount(tmp)
    s.resize(x, y)
    return s


def joint_sfs_folded(ac1, ac2, n1=None, n2=None):
    """Compute the joint folded site frequency spectrum between two
    populations.

    Parameters
    ----------
    ac1 : array_like, int, shape (n_variants, 2)
        Allele counts for the first population.
    ac2 : array_like, int, shape (n_variants, 2)
        Allele counts for the second population.
    n1, n2 : int, optional
        The total number of chromosomes called in each population.

    Returns
    -------
    joint_sfs_folded : ndarray, int, shape (n1//2 + 1, n2//2 + 1)
        Array where the (i, j)th element is the number of variant sites with a
        minor allele count of i in the first population and j in the second
        population.

    """

    # check inputs
    ac1, n1 = _check_ac_n(ac1, n1)
    ac2, n2 = _check_ac_n(ac2, n2)

    # compute minor allele counts
    mac1 = np.amin(ac1, axis=1)
    mac2 = np.amin(ac2, axis=1)

    # compute site frequency spectrum
    x = n1//2 + 1
    y = n2//2 + 1
    tmp = (mac1 * y + mac2).astype(int, copy=False)
    s = np.bincount(tmp)
    s.resize(x, y)
    return s


def joint_sfs_scaled(dac1, dac2, n1=None, n2=None):
    """Compute the joint site frequency spectrum between two populations,
    scaled such that a constant value is expected across the spectrum for
    neutral variation, constant population size and unrelated populations.

    Parameters
    ----------
    dac1 : array_like, int, shape (n_variants,)
        Derived allele counts for the first population.
    dac2 : array_like, int, shape (n_variants,)
        Derived allele counts for the second population.
    n1, n2 : int, optional
        The total number of chromosomes called in each population.

    Returns
    -------
    joint_sfs_scaled : ndarray, int, shape (n1 + 1, n2 + 1)
        Array where the (i, j)th element is the scaled frequency of variant
        sites with i derived alleles in the first population and j derived
        alleles in the second population.

    """

    # compute site frequency spectrum
    s = joint_sfs(dac1, dac2, n1=n1, n2=n2)

    # apply scaling
    s = scale_joint_sfs(s)

    return s


def scale_joint_sfs(s):
    """Scale a joint site frequency spectrum.

    Parameters
    ----------
    s : array_like, int, shape (n1, n2)
        Joint site frequency spectrum.

    Returns
    -------
    joint_sfs_scaled : ndarray, int, shape (n1, n2)
        Scaled joint site frequency spectrum.

    """

    i = np.arange(s.shape[0])[:, None]
    j = np.arange(s.shape[1])[None, :]
    out = (s * i) * j
    return out


def joint_sfs_folded_scaled(ac1, ac2, n1=None, n2=None):
    """Compute the joint folded site frequency spectrum between two
    populations, scaled such that a constant value is expected across the
    spectrum for neutral variation, constant population size and unrelated
    populations.

    Parameters
    ----------
    ac1 : array_like, int, shape (n_variants, 2)
        Allele counts for the first population.
    ac2 : array_like, int, shape (n_variants, 2)
        Allele counts for the second population.
    n1, n2 : int, optional
        The total number of chromosomes called in each population.

    Returns
    -------
    joint_sfs_folded_scaled : ndarray, int, shape (n1//2 + 1, n2//2 + 1)
        Array where the (i, j)th element is the scaled frequency of variant
        sites with a minor allele count of i in the first population and j
        in the second population.

    """  # noqa

    # check inputs
    ac1, n1 = _check_ac_n(ac1, n1)
    ac2, n2 = _check_ac_n(ac2, n2)

    # compute site frequency spectrum
    s = joint_sfs_folded(ac1, ac2, n1=n1, n2=n2)

    # apply scaling
    s = scale_joint_sfs_folded(s, n1, n2)

    return s


def scale_joint_sfs_folded(s, n1, n2):
    """Scale a folded joint site frequency spectrum.

    Parameters
    ----------
    s : array_like, int, shape (m_chromosomes//2, n_chromosomes//2)
        Folded joint site frequency spectrum.
    n1, n2 : int, optional
        The total number of chromosomes called in each population.

    Returns
    -------
    joint_sfs_folded_scaled : ndarray, int, shape (m_chromosomes//2, n_chromosomes//2)
        Scaled folded joint site frequency spectrum.

    """  # noqa
    out = np.empty_like(s)
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            out[i, j] = s[i, j] * i * j * (n1 - i) * (n2 - j)
    return out


def fold_sfs(s, n):
    """Fold a site frequency spectrum.

    Parameters
    ----------
    s : array_like, int, shape (n_chromosomes,)
        Site frequency spectrum
    n : int
        Total number of chromosomes called.

    Returns
    -------
    sfs_folded : ndarray, int
        Folded site frequency spectrum

    """

    # check inputs
    s = asarray_ndim(s, 1)
    assert s.shape[0] <= n + 1, 'invalid number of chromosomes'

    # need to check s has all entries up to n
    if s.shape[0] < n + 1:
        sn = np.zeros(n + 1, dtype=s.dtype)
        sn[:s.shape[0]] = s
        s = sn

    # fold
    nf = (n + 1) // 2
    n = nf * 2
    o = s[:nf] + s[nf:n][::-1]

    return o


def fold_joint_sfs(s, n1, n2):
    """Fold a joint site frequency spectrum.

    Parameters
    ----------
    s : array_like, int, shape (m_chromosomes, n_chromosomes)
        Joint site frequency spectrum.
    n1, n2 : int, optional
        The total number of chromosomes called in each population.

    Returns
    -------
    joint_sfs_folded : ndarray, int
        Folded joint site frequency spectrum.

    """

    # check inputs
    s = asarray_ndim(s, 2)
    assert s.shape[0] <= n1 + 1, 'invalid number of chromosomes'
    assert s.shape[1] <= n2 + 1, 'invalid number of chromosomes'

    # need to check s has all entries up to m
    if s.shape[0] < n1 + 1:
        sm = np.zeros((n1 + 1, s.shape[1]), dtype=s.dtype)
        sm[:s.shape[0]] = s
        s = sm

    # need to check s has all entries up to n
    if s.shape[1] < n2 + 1:
        sn = np.zeros((s.shape[0], n2 + 1), dtype=s.dtype)
        sn[:, :s.shape[1]] = s
        s = sn

    # fold
    mf = (n1 + 1) // 2
    nf = (n2 + 1) // 2
    n1 = mf * 2
    n2 = nf * 2
    o = (s[:mf, :nf] +  # top left
         s[mf:n1, :nf][::-1] +  # top right
         s[:mf, nf:n2][:, ::-1] +  # bottom left
         s[mf:n1, nf:n2][::-1, ::-1])  # bottom right

    return o


def plot_sfs(s, yscale='log', bins=None, n=None,
             clip_endpoints=True, label=None, plot_kwargs=None,
             ax=None):
    """Plot a site frequency spectrum.

    Parameters
    ----------
    s : array_like, int, shape (n_chromosomes,)
        Site frequency spectrum.
    yscale : string, optional
        Y axis scale.
    bins : int or array_like, int, optional
        Allele count bins.
    n : int, optional
        Number of chromosomes sampled. If provided, X axis will be plotted
        as allele frequency, otherwise as allele count.
    clip_endpoints : bool, optional
        If True, do not plot first and last values from frequency spectrum.
    label : string, optional
        Label for data series in plot.
    plot_kwargs : dict-like
        Additional keyword arguments, passed through to ax.plot().
    ax : axes, optional
        Axes on which to draw. If not provided, a new figure will be created.

    Returns
    -------
    ax : axes
        The axes on which the plot was drawn.

    """

    import matplotlib.pyplot as plt
    import scipy

    # check inputs
    s = asarray_ndim(s, 1)

    # setup axes
    if ax is None:
        fig, ax = plt.subplots()

    # setup data
    if bins is None:
        if clip_endpoints:
            x = np.arange(1, s.shape[0]-1)
            y = s[1:-1]
        else:
            x = np.arange(s.shape[0])
            y = s
    else:
        if clip_endpoints:
            y, b, _ = scipy.stats.binned_statistic(
                np.arange(1, s.shape[0]-1),
                values=s[1:-1],
                bins=bins,
                statistic='sum')
        else:
            y, b, _ = scipy.stats.binned_statistic(
                np.arange(s.shape[0]),
                values=s,
                bins=bins,
                statistic='sum')
        # use bin midpoints for plotting
        x = (b[:-1] + b[1:]) / 2

    if n:
        # convert allele counts to allele frequencies
        x = x / n
        ax.set_xlabel('derived allele frequency')
    else:
        ax.set_xlabel('derived allele count')

    # do plotting
    if plot_kwargs is None:
        plot_kwargs = dict()
    ax.plot(x, y, label=label, **plot_kwargs)

    # tidy
    ax.set_yscale(yscale)
    ax.set_ylabel('site frequency')
    ax.autoscale(axis='x', tight=True)

    return ax


# noinspection PyIncorrectDocstring
def plot_sfs_folded(*args, **kwargs):
    """Plot a folded site frequency spectrum.

    Parameters
    ----------
    s : array_like, int, shape (n_chromosomes/2,)
        Site frequency spectrum.
    yscale : string, optional
        Y axis scale.
    bins : int or array_like, int, optional
        Allele count bins.
    n : int, optional
        Number of chromosomes sampled. If provided, X axis will be plotted
        as allele frequency, otherwise as allele count.
    clip_endpoints : bool, optional
        If True, do not plot first and last values from frequency spectrum.
    label : string, optional
        Label for data series in plot.
    plot_kwargs : dict-like
        Additional keyword arguments, passed through to ax.plot().
    ax : axes, optional
        Axes on which to draw. If not provided, a new figure will be created.

    Returns
    -------
    ax : axes
        The axes on which the plot was drawn.

    """

    ax = plot_sfs(*args, **kwargs)
    n = kwargs.get('n', None)
    if n:
        ax.set_xlabel('minor allele frequency')
    else:
        ax.set_xlabel('minor allele count')
    return ax


# noinspection PyIncorrectDocstring
def plot_sfs_scaled(*args, **kwargs):
    """Plot a scaled site frequency spectrum.

    Parameters
    ----------
    s : array_like, int, shape (n_chromosomes,)
        Site frequency spectrum.
    yscale : string, optional
        Y axis scale.
    bins : int or array_like, int, optional
        Allele count bins.
    n : int, optional
        Number of chromosomes sampled. If provided, X axis will be plotted
        as allele frequency, otherwise as allele count.
    clip_endpoints : bool, optional
        If True, do not plot first and last values from frequency spectrum.
    label : string, optional
        Label for data series in plot.
    plot_kwargs : dict-like
        Additional keyword arguments, passed through to ax.plot().
    ax : axes, optional
        Axes on which to draw. If not provided, a new figure will be created.

    Returns
    -------
    ax : axes
        The axes on which the plot was drawn.

    """
    kwargs.setdefault('yscale', 'linear')
    ax = plot_sfs(*args, **kwargs)
    ax.set_ylabel('scaled site frequency')
    return ax


# noinspection PyIncorrectDocstring
def plot_sfs_folded_scaled(*args, **kwargs):
    """Plot a folded scaled site frequency spectrum.

    Parameters
    ----------
    s : array_like, int, shape (n_chromosomes/2,)
        Site frequency spectrum.
    yscale : string, optional
        Y axis scale.
    bins : int or array_like, int, optional
        Allele count bins.
    n : int, optional
        Number of chromosomes sampled. If provided, X axis will be plotted
        as allele frequency, otherwise as allele count.
    clip_endpoints : bool, optional
        If True, do not plot first and last values from frequency spectrum.
    label : string, optional
        Label for data series in plot.
    plot_kwargs : dict-like
        Additional keyword arguments, passed through to ax.plot().
    ax : axes, optional
        Axes on which to draw. If not provided, a new figure will be created.

    Returns
    -------
    ax : axes
        The axes on which the plot was drawn.

    """
    kwargs.setdefault('yscale', 'linear')
    ax = plot_sfs_folded(*args, **kwargs)
    ax.set_ylabel('scaled site frequency')
    n = kwargs.get('n', None)
    if n:
        ax.set_xlabel('minor allele frequency')
    else:
        ax.set_xlabel('minor allele count')
    return ax


def plot_joint_sfs(s, ax=None, imshow_kwargs=None):
    """Plot a joint site frequency spectrum.

    Parameters
    ----------
    s : array_like, int, shape (n_chromosomes_pop1, n_chromosomes_pop2)
        Joint site frequency spectrum.
    ax : axes, optional
        Axes on which to draw. If not provided, a new figure will be created.
    imshow_kwargs : dict-like
        Additional keyword arguments, passed through to ax.imshow().

    Returns
    -------
    ax : axes
        The axes on which the plot was drawn.

    """

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    # check inputs
    s = asarray_ndim(s, 2)

    # setup axes
    if ax is None:
        w = plt.rcParams['figure.figsize'][0]
        fig, ax = plt.subplots(figsize=(w, w))

    # set plotting defaults
    if imshow_kwargs is None:
        imshow_kwargs = dict()
    imshow_kwargs.setdefault('cmap', 'jet')
    imshow_kwargs.setdefault('interpolation', 'none')
    imshow_kwargs.setdefault('aspect', 'auto')
    imshow_kwargs.setdefault('norm', LogNorm())

    # plot data
    ax.imshow(s.T, **imshow_kwargs)

    # tidy
    ax.invert_yaxis()
    ax.set_xlabel('derived allele count (population 1)')
    ax.set_ylabel('derived allele count (population 2)')

    return ax


# noinspection PyIncorrectDocstring
def plot_joint_sfs_folded(*args, **kwargs):
    """Plot a joint site frequency spectrum.

    Parameters
    ----------
    s : array_like, int, shape (n_chromosomes_pop1/2, n_chromosomes_pop2/2)
        Joint site frequency spectrum.
    ax : axes, optional
        Axes on which to draw. If not provided, a new figure will be created.
    imshow_kwargs : dict-like
        Additional keyword arguments, passed through to ax.imshow().

    Returns
    -------
    ax : axes
        The axes on which the plot was drawn.

    """
    ax = plot_joint_sfs(*args, **kwargs)
    ax.set_xlabel('minor allele count (population 1)')
    ax.set_ylabel('minor allele count (population 2)')
    return ax


# noinspection PyIncorrectDocstring
def plot_joint_sfs_scaled(*args, **kwargs):
    """Plot a scaled joint site frequency spectrum.

    Parameters
    ----------
    s : array_like, int, shape (n_chromosomes_pop1, n_chromosomes_pop2)
        Joint site frequency spectrum.
    ax : axes, optional
        Axes on which to draw. If not provided, a new figure will be created.
    imshow_kwargs : dict-like
        Additional keyword arguments, passed through to ax.imshow().

    Returns
    -------
    ax : axes
        The axes on which the plot was drawn.

    """
    imshow_kwargs = kwargs.get('imshow_kwargs', dict())
    imshow_kwargs.setdefault('norm', None)
    kwargs['imshow_kwargs'] = imshow_kwargs
    ax = plot_joint_sfs(*args, **kwargs)
    return ax


# noinspection PyIncorrectDocstring
def plot_joint_sfs_folded_scaled(*args, **kwargs):
    """Plot a scaled folded joint site frequency spectrum.

    Parameters
    ----------
    s : array_like, int, shape (n_chromosomes_pop1/2, n_chromosomes_pop2/2)
        Joint site frequency spectrum.
    ax : axes, optional
        Axes on which to draw. If not provided, a new figure will be created.
    imshow_kwargs : dict-like
        Additional keyword arguments, passed through to ax.imshow().

    Returns
    -------
    ax : axes
        The axes on which the plot was drawn.

    """
    imshow_kwargs = kwargs.get('imshow_kwargs', dict())
    imshow_kwargs.setdefault('norm', None)
    kwargs['imshow_kwargs'] = imshow_kwargs
    ax = plot_joint_sfs_folded(*args, **kwargs)
    ax.set_xlabel('minor allele count (population 1)')
    ax.set_ylabel('minor allele count (population 2)')
    return ax
