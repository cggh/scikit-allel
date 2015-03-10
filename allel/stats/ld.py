# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


from allel.util import asarray_ndim


def rogers_huff_r(gn):
    """Estimate the linkage disequilibrium parameter *r* for each pair of
    variants using the method of Rogers and Huff (2008).

    Parameters
    ----------

    gn : array_like, int8, shape (n_variants, n_samples)
        Diploid genotypes at biallelic variants, coded as the number of
        alternate alleles per call (i.e., 0 = hom ref, 1 = het, 2 = hom alt).

    Returns
    -------

    r : ndarray, float, shape (n_variants * (n_variants - 1) // 2,)
        Matrix in condensed form.

    Examples
    --------

    >>> import allel
    >>> g = allel.model.GenotypeArray([[[0, 0], [1, 1], [0, 0]],
    ...                                [[0, 0], [1, 1], [0, 0]],
    ...                                [[1, 1], [0, 0], [1, 1]],
    ...                                [[0, 0], [0, 1], [-1, -1]]], dtype='i1')
    >>> gn = g.to_n_alt(fill=-1)
    >>> gn
    array([[ 0,  2,  0],
           [ 0,  2,  0],
           [ 2,  0,  2],
           [ 0,  1, -1]], dtype=int8)
    >>> r = allel.stats.rogers_huff_r(gn)
    >>> r
    array([ 1.        , -1.00000012,  1.        , -1.00000012,  1.        , -1.        ], dtype=float32)
    >>> r ** 2
    array([ 1.        ,  1.00000024,  1.        ,  1.00000024,  1.        ,  1.        ], dtype=float32)
    >>> from scipy.spatial.distance import squareform
    >>> squareform(r ** 2)
    array([[ 0.        ,  1.        ,  1.00000024,  1.        ],
           [ 1.        ,  0.        ,  1.00000024,  1.        ],
           [ 1.00000024,  1.00000024,  0.        ,  1.        ],
           [ 1.        ,  1.        ,  1.        ,  0.        ]])

    """  # flake8: noqa

    # check inputs
    gn = asarray_ndim(gn, 2, dtype='i1')

    # compute correlation coefficients
    from allel.opt.stats import gn_pairwise_corrcoef_int8
    r = gn_pairwise_corrcoef_int8(gn)

    # convenience for singletons
    if r.size == 1:
        r = r[0]

    return r


def locate_unlinked(gn, size=100, step=20, threshold=.1):
    """Locate variants in approximate linkage equilibrium, where r**2 is
    below the given `threshold`.

    Parameters
    ----------

    gn : array_like, int8, shape (n_variants, n_samples)
        Diploid genotypes at biallelic variants, coded as the number of
        alternate alleles per call (i.e., 0 = hom ref, 1 = het, 2 = hom alt).
    size : int
        Window size (number of variants).
    step : int
        Number of variants to advance to the next window.
    threshold : float
        Maximum value of r**2 to include variants.

    Returns
    -------

    loc : ndarray, bool, shape (n_variants)
        Boolean array where True items locate variants in approximate
        linkage equilibrium.

    """

    # check inputs
    gn = asarray_ndim(gn, 2, dtype='i1')

    from allel.opt.stats import gn_locate_unlinked_int8
    loc = gn_locate_unlinked_int8(gn, size, step, threshold)

    return loc
