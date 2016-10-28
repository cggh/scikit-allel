# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import itertools


import numpy as np


from allel.model.ndarray import SortedIndex
from allel.util import asarray_ndim, ensure_square
from allel.stats.diversity import sequence_divergence
from allel.chunked import get_blen_array


def pairwise_distance(x, metric, chunked=False, blen=None):
    """Compute pairwise distance between individuals (e.g., samples or
    haplotypes).

    Parameters
    ----------
    x : array_like, shape (n, m, ...)
        Array of m observations (e.g., samples or haplotypes) in a space
        with n dimensions (e.g., variants). Note that the order of the first
        two dimensions is **swapped** compared to what is expected by
        scipy.spatial.distance.pdist.
    metric : string or function
        Distance metric. See documentation for the function
        :func:`scipy.spatial.distance.pdist` for a list of built-in
        distance metrics.
    chunked : bool, optional
        If True, use a block-wise implementation to avoid loading the entire
        input array into memory. This means that a distance matrix will be
        calculated for each block of the input array, and the results will
        be summed to produce the final output. For some distance metrics
        this will return a different result from the standard implementation.
    blen : int, optional
        Block length to use for chunked implementation.

    Returns
    -------
    dist : ndarray, shape (m * (m - 1) / 2,)
        Distance matrix in condensed form.

    Examples
    --------

    >>> import allel
    >>> g = allel.GenotypeArray([[[0, 0], [0, 1], [1, 1]],
    ...                          [[0, 1], [1, 1], [1, 2]],
    ...                          [[0, 2], [2, 2], [-1, -1]]])
    >>> d = allel.stats.pairwise_distance(g.to_n_alt(), metric='cityblock')
    >>> d
    array([ 3.,  4.,  3.])
    >>> import scipy.spatial
    >>> scipy.spatial.distance.squareform(d)
    array([[ 0.,  3.,  4.],
           [ 3.,  0.,  3.],
           [ 4.,  3.,  0.]])

    """

    import scipy.spatial

    # check inputs
    if not hasattr(x, 'ndim'):
        x = np.asarray(x)
    if x.ndim < 2:
        raise ValueError('array with at least 2 dimensions expected')

    if x.ndim == 2:
        # use scipy to calculate distance, it's most efficient

        def f(b):

            # transpose as pdist expects (m, n) for m observations in an
            # n-dimensional space
            t = b.T

            # compute the distance matrix
            return scipy.spatial.distance.pdist(t, metric=metric)

    else:
        # use our own implementation, it handles multidimensional observations

        def f(b):
            return pdist(b, metric=metric)

    if chunked:
        # use block-wise implementation
        blen = get_blen_array(x, blen)
        dist = None
        for i in range(0, x.shape[0], blen):
            j = min(x.shape[0], i+blen)
            block = x[i:j]
            if dist is None:
                dist = f(block)
            else:
                dist += f(block)

    else:
        # standard implementation
        dist = f(x)

    return dist


def pdist(x, metric):
    """Alternative implementation of :func:`scipy.spatial.distance.pdist`
    which is slower but more flexible in that arrays with >2 dimensions can be
    passed, allowing for multidimensional observations, e.g., diploid
    genotype calls or allele counts.

    Parameters
    ----------
    x : array_like, shape (n, m, ...)
        Array of m observations (e.g., samples or haplotypes) in a space
        with n dimensions (e.g., variants). Note that the order of the first
        two dimensions is **swapped** compared to what is expected by
        scipy.spatial.distance.pdist.
    metric : string or function
        Distance metric. See documentation for the function
        :func:`scipy.spatial.distance.pdist` for a list of built-in
        distance metrics.

    Returns
    -------
    dist : ndarray
        Distance matrix in condensed form.

    """

    if isinstance(metric, str):
        import scipy.spatial
        if hasattr(scipy.spatial.distance, metric):
            metric = getattr(scipy.spatial.distance, metric)
        else:
            raise ValueError('metric name not found')

    m = x.shape[1]
    dist = list()
    for i, j in itertools.combinations(range(m), 2):
        a = x[:, i, ...]
        b = x[:, j, ...]
        d = metric(a, b)
        dist.append(d)
    return np.array(dist)


def pairwise_dxy(pos, gac, start=None, stop=None, is_accessible=None):
    """Convenience function to calculate a pairwise distance matrix using
    nucleotide divergence (a.k.a. Dxy) as the distance metric.

    Parameters
    ----------
    pos : array_like, int, shape (n_variants,)
        Variant positions.
    gac : array_like, int, shape (n_variants, n_samples, n_alleles)
        Per-genotype allele counts.
    start : int, optional
        Start position of region to use.
    stop : int, optional
        Stop position of region to use.
    is_accessible : array_like, bool, shape (len(contig),), optional
        Boolean array indicating accessibility status for all positions in the
        chromosome/contig.

    Returns
    -------
    dist : ndarray
        Distance matrix in condensed form.

    See Also
    --------
    allel.model.ndarray.GenotypeArray.to_allele_counts

    """

    if not isinstance(pos, SortedIndex):
        pos = SortedIndex(pos, copy=False)
    gac = asarray_ndim(gac, 3)
    # compute this once here, to avoid repeated evaluation within the loop
    gan = np.sum(gac, axis=2)
    m = gac.shape[1]
    dist = list()
    for i, j in itertools.combinations(range(m), 2):
        ac1 = gac[:, i, ...]
        an1 = gan[:, i]
        ac2 = gac[:, j, ...]
        an2 = gan[:, j]
        d = sequence_divergence(pos, ac1, ac2, an1=an1, an2=an2,
                                start=start, stop=stop,
                                is_accessible=is_accessible)
        dist.append(d)
    return np.array(dist)


def pcoa(dist):
    """Perform principal coordinate analysis of a distance matrix, a.k.a.
    classical multi-dimensional scaling.

    Parameters
    ----------
    dist : array_like
        Distance matrix in condensed form.

    Returns
    -------
    coords : ndarray, shape (n_samples, n_dimensions)
        Transformed coordinates for the samples.
    explained_ratio : ndarray, shape (n_dimensions)
        Variance explained by each dimension.

    """
    import scipy.linalg

    # This implementation is based on the skbio.math.stats.ordination.PCoA
    # implementation, with some minor adjustments.

    # check inputs
    dist = ensure_square(dist)

    # perform scaling
    e_matrix = (dist ** 2) / -2
    row_means = np.mean(e_matrix, axis=1, keepdims=True)
    col_means = np.mean(e_matrix, axis=0, keepdims=True)
    matrix_mean = np.mean(e_matrix)
    f_matrix = e_matrix - row_means - col_means + matrix_mean
    eigvals, eigvecs = scipy.linalg.eigh(f_matrix)

    # deal with eigvals close to zero
    close_to_zero = np.isclose(eigvals, 0)
    eigvals[close_to_zero] = 0

    # sort descending
    idxs = eigvals.argsort()[::-1]
    eigvals = eigvals[idxs]
    eigvecs = eigvecs[:, idxs]

    # keep only positive eigenvalues
    keep = eigvals >= 0
    eigvecs = eigvecs[:, keep]
    eigvals = eigvals[keep]

    # compute coordinates
    coords = eigvecs * np.sqrt(eigvals)

    # compute ratio explained
    explained_ratio = eigvals / eigvals.sum()

    return coords, explained_ratio


def condensed_coords(i, j, n):
    """Transform square distance matrix coordinates to the corresponding
    index into a condensed, 1D form of the matrix.

    Parameters
    ----------
    i : int
        Row index.
    j : int
        Column index.
    n : int
        Size of the square matrix (length of first or second dimension).

    Returns
    -------
    ix : int

    """

    # guard conditions
    if i == j or i >= n or j >= n or i < 0 or j < 0:
        raise ValueError('invalid coordinates: %s, %s' % (i, j))

    # normalise order
    i, j = sorted([i, j])

    # calculate number of items in rows before this one (sum of arithmetic
    # progression)
    x = i * ((2 * n) - i - 1) / 2

    # add on previous items in current row
    ix = x + j - i - 1

    return int(ix)


def condensed_coords_within(pop, n):
    """Return indices into a condensed distance matrix for all
    pairwise comparisons within the given population.

    Parameters
    ----------
    pop : array_like, int
        Indices of samples or haplotypes within the population.
    n : int
        Size of the square matrix (length of first or second dimension).

    Returns
    -------
    indices : ndarray, int

    """

    return [condensed_coords(i, j, n)
            for i, j in itertools.combinations(sorted(pop), 2)]


def condensed_coords_between(pop1, pop2, n):
    """Return indices into a condensed distance matrix for all pairwise
    comparisons between two populations.

    Parameters
    ----------
    pop1 : array_like, int
        Indices of samples or haplotypes within the first population.
    pop2 : array_like, int
        Indices of samples or haplotypes within the second population.
    n : int
        Size of the square matrix (length of first or second dimension).

    Returns
    -------
    indices : ndarray, int

    """

    return [condensed_coords(i, j, n)
            for i, j in itertools.product(sorted(pop1), sorted(pop2))]


def plot_pairwise_distance(dist, labels=None, colorbar=True, ax=None,
                           imshow_kwargs=None):
    """Plot a pairwise distance matrix.

    Parameters
    ----------
    dist : array_like
        The distance matrix in condensed form.
    labels : sequence of strings, optional
        Sample labels for the axes.
    colorbar : bool, optional
        If True, add a colorbar to the current figure.
    ax : axes, optional
        The axes on which to draw. If not provided, a new figure will be
        created.
    imshow_kwargs : dict-like, optional
        Additional keyword arguments passed through to
        :func:`matplotlib.pyplot.imshow`.

    Returns
    -------
    ax : axes
        The axes on which the plot was drawn

    """

    import matplotlib.pyplot as plt

    # check inputs
    dist_square = ensure_square(dist)

    # set up axes
    if ax is None:
        # make a square figure
        x = plt.rcParams['figure.figsize'][0]
        fig, ax = plt.subplots(figsize=(x, x))
        fig.tight_layout()

    # setup imshow arguments
    if imshow_kwargs is None:
        imshow_kwargs = dict()
    imshow_kwargs.setdefault('interpolation', 'none')
    imshow_kwargs.setdefault('cmap', 'jet')
    imshow_kwargs.setdefault('vmin', np.min(dist))
    imshow_kwargs.setdefault('vmax', np.max(dist))

    # plot as image
    im = ax.imshow(dist_square, **imshow_kwargs)

    # tidy up
    if labels:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels, rotation=0)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    if colorbar:
        plt.gcf().colorbar(im, shrink=.5)

    return ax
