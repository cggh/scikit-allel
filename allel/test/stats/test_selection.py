# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np
import pytest


from allel.test.tools import assert_array_equal, assert_array_almost_equal
from allel import ihs, xpehh, nsl, xpnsl, ehh_decay, voight_painting, pbs
from allel.opt.stats import ssl01_scan, nsl01_scan, ihh01_scan,\
    ssl2ihh, ihh_scan


# noinspection PyUnusedLocal
def sum_ssl(ssl, *args, **kwargs):
    return np.sum(ssl)


def test_ssl01_scan_a():

    # 2 haplotypes, identical
    h = np.array([[0, 0],
                  [0, 0],
                  [0, 0]])
    expect0 = [1, 2, 3]
    expect1 = [0, 0, 0]
    actual0, actual1 = ssl01_scan(h, sum_ssl)
    assert_array_equal(expect0, actual0)
    assert_array_equal(expect1, actual1)


def test_ssl01_scan_b():

    # 2 haplotypes, identical
    h = np.array([[1, 1],
                  [1, 1],
                  [1, 1]])
    expect0 = [0, 0, 0]
    expect1 = [1, 2, 3]
    actual0, actual1 = ssl01_scan(h, sum_ssl)
    assert_array_equal(expect0, actual0)
    assert_array_equal(expect1, actual1)


def test_ssl01_scan_c():

    # 2 haplotypes, identical
    h = np.array([[0, 0],
                  [0, 0],
                  [1, 1],
                  [1, 1]])
    expect0 = [1, 2, 0, 0]
    expect1 = [0, 0, 3, 4]
    actual0, actual1 = ssl01_scan(h, sum_ssl)
    assert_array_equal(expect0, actual0)
    assert_array_equal(expect1, actual1)


def test_ssl01_scan_d():

    # 2 haplotypes, different
    h = np.array([[0, 1],
                  [0, 1],
                  [1, 0],
                  [1, 0]])
    expect0 = [0, 0, 0, 0]
    expect1 = [0, 0, 0, 0]
    actual0, actual1 = ssl01_scan(h, sum_ssl)
    assert_array_equal(expect0, actual0)
    assert_array_equal(expect1, actual1)


def test_ssl01_scan_e():

    # 3 haplotypes, 3 pairs, identical
    h = np.array([[0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0]])
    expect0 = [3, 6, 9]
    expect1 = [0, 0, 0]
    actual0, actual1 = ssl01_scan(h, sum_ssl)
    assert_array_equal(expect0, actual0)
    assert_array_equal(expect1, actual1)


def test_ssl01_scan_f():

    # 4 haplotypes,
    h = np.array([[0, 0, 1, 1],
                  [0, 0, 1, 1],
                  [0, 0, 1, 1]])
    expect0 = [1, 2, 3]
    expect1 = [1, 2, 3]
    actual0, actual1 = ssl01_scan(h, sum_ssl)
    assert_array_equal(expect0, actual0)
    assert_array_equal(expect1, actual1)


def test_nsl01_scan_a():

    h = np.array([[0, 0, 0, 1, 1, 1],
                  [0, 0, 0, 1, 1, 1],
                  [0, 0, 0, 1, 1, 1],
                  [0, 0, 0, 1, 1, 1]])
    nsl0, nsl1 = nsl01_scan(h)
    expect_nsl0 = [1, 2, 3, 4]
    assert_array_almost_equal(expect_nsl0, nsl0)
    expect_nsl1 = [1, 2, 3, 4]
    assert_array_almost_equal(expect_nsl1, nsl1)


def test_nsl01_scan_b():

    h = np.array([[0, 0, 0, 1],
                  [0, 0, 1, 0],
                  [0, 1, 0, 0],
                  [1, 0, 0, 0]])
    nsl0, nsl1 = nsl01_scan(h)
    expect_nsl0 = [1, 4 / 3, 4 / 3, 4 / 3]
    assert_array_almost_equal(expect_nsl0, nsl0)
    expect_nsl1 = [np.nan, np.nan, np.nan, np.nan]
    assert_array_almost_equal(expect_nsl1, nsl1)


def test_nsl01_scan_c():

    h = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 1, 0],
                  [1, 0, 0]])
    nsl0, nsl1 = nsl01_scan(h)
    expect_nsl0 = [1, np.nan, np.nan, 1]
    assert_array_almost_equal(expect_nsl0, nsl0)
    expect_nsl1 = [np.nan, 1, 1, np.nan]
    assert_array_almost_equal(expect_nsl1, nsl1)


def test_ihh_scan_a():
    # simple case: 1 haplotype pair, haplotype homozygosity over all variants
    gaps = np.array([10, 10], dtype='f8')
    h = np.array([[0, 0],
                  [0, 0],
                  [0, 0]])

    # do not include edges
    expect = [np.nan, np.nan, np.nan]
    actual = ihh_scan(h, gaps, min_ehh=0, include_edges=False)
    assert_array_almost_equal(expect, actual)

    # include edges
    expect = [0, 10, 20]
    actual = ihh_scan(h, gaps, min_ehh=0, include_edges=True)
    assert_array_almost_equal(expect, actual)


def test_ihh_scan_b():
    # 1 haplotype pair, haplotype homozygosity over all variants
    # handling of large gap (encoded as -1)
    gaps = np.array([10, -1], dtype='f8')
    h = np.array([[0, 0],
                  [0, 0],
                  [0, 0]])

    # do not include edges
    expect = [np.nan, np.nan, np.nan]
    actual = ihh_scan(h, gaps, min_ehh=0, include_edges=False)
    assert_array_almost_equal(expect, actual)

    # include edges
    expect = [0, 10, np.nan]
    actual = ihh_scan(h, gaps, min_ehh=0, include_edges=True)
    assert_array_almost_equal(expect, actual)


def test_ihh_scan_c():
    # simple case: 1 haplotype pair, haplotype homozygosity decays
    gaps = np.array([10, 10], dtype='f8')
    h = np.array([[0, 1],
                  [0, 0],
                  [0, 0]])

    # do not include edges
    expect = [0, 5, 15]
    actual = ihh_scan(h, gaps, min_ehh=0, include_edges=False)
    assert_array_almost_equal(expect, actual)

    # include edges
    expect = [0, 5, 15]
    actual = ihh_scan(h, gaps, min_ehh=0, include_edges=True)
    assert_array_almost_equal(expect, actual)


def test_ihh_scan_d():
    # edge case: start from 0 haplotype homozygosity
    gaps = np.array([10], dtype='f8')
    h = np.array([[0, 1],
                  [1, 0]])

    expect = [0, 0]
    actual = ihh_scan(h, gaps, min_ehh=0, include_edges=False)
    assert_array_almost_equal(expect, actual)

    expect = [0, 0]
    actual = ihh_scan(h, gaps, min_ehh=0, include_edges=True)
    assert_array_almost_equal(expect, actual)


def test_ihh_scan_e():
    # edge case: start from haplotype homozygosity below min_ehh
    gaps = np.array([10], dtype='f8')
    h = np.array([[0, 0, 1],
                  [0, 1, 0]])

    expect = [np.nan, 10/6]
    actual = ihh_scan(h, gaps, min_ehh=0, include_edges=False)
    assert_array_almost_equal(expect, actual)

    expect = [0, 10/6]
    actual = ihh_scan(h, gaps, min_ehh=0, include_edges=True)
    assert_array_almost_equal(expect, actual)

    expect = [0, 0]
    actual = ihh_scan(h, gaps, min_ehh=0.5, include_edges=False)
    assert_array_almost_equal(expect, actual)

    expect = [0, 0]
    actual = ihh_scan(h, gaps, min_ehh=0.5, include_edges=True)
    assert_array_almost_equal(expect, actual)


def test_ihh01_scan_a():
    gaps = np.array([10, 10, 10], dtype='f8')
    h = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 1, 0],
                  [1, 0, 0]])

    ihh0, ihh1 = ihh01_scan(h, gaps, min_ehh=0.05, include_edges=False)
    expect_ihh0 = [np.nan, np.nan, np.nan, 5]
    assert_array_almost_equal(expect_ihh0, ihh0)
    expect_ihh1 = [np.nan, 5, 5, np.nan]
    assert_array_almost_equal(expect_ihh1, ihh1)

    ihh0, ihh1 = ihh01_scan(h, gaps, min_ehh=0, include_edges=True)
    expect_ihh0 = [0, np.nan, np.nan, 5]
    assert_array_almost_equal(expect_ihh0, ihh0)
    expect_ihh1 = [np.nan, 5, 5, np.nan]
    assert_array_almost_equal(expect_ihh1, ihh1)


def test_ihh01_scan_b():
    gaps = np.array([10, 10, 10], dtype='f8')
    h = np.array([[0, 0, 0, 1],
                  [0, 0, 1, 0],
                  [0, 1, 0, 0],
                  [1, 0, 0, 0]])

    ihh0, ihh1 = ihh01_scan(h, gaps, min_ehh=0.05, include_edges=False)
    x = (10 * (1 + 1 / 3) / 2) + (10 * (1 / 3 + 0) / 2)
    expect_ihh0 = [np.nan, np.nan, x, x]
    assert_array_almost_equal(expect_ihh0, ihh0)
    expect_ihh1 = [np.nan, np.nan, np.nan, np.nan]
    assert_array_almost_equal(expect_ihh1, ihh1)

    ihh0, ihh1 = ihh01_scan(h, gaps, min_ehh=0, include_edges=False)
    expect_ihh0 = [np.nan, np.nan, x, x]
    assert_array_almost_equal(expect_ihh0, ihh0)
    expect_ihh1 = [np.nan, np.nan, np.nan, np.nan]
    assert_array_almost_equal(expect_ihh1, ihh1)

    ihh0, ihh1 = ihh01_scan(h, gaps, min_ehh=0, include_edges=True)
    expect_ihh0 = [0, 10 * (1 + 1 / 3) / 2, x, x]
    assert_array_almost_equal(expect_ihh0, ihh0)
    expect_ihh1 = [np.nan, np.nan, np.nan, np.nan]
    assert_array_almost_equal(expect_ihh1, ihh1)


def test_ihh01_scan_c():
    gaps = np.array([10, 10, 10], dtype='f8')
    h = np.array([[0, 0, 0, 1, 1, 1],
                  [0, 0, 0, 1, 1, 1],
                  [0, 0, 0, 1, 1, 1],
                  [0, 0, 0, 1, 1, 1]])

    ihh0, ihh1 = ihh01_scan(h, gaps, min_ehh=0.05)
    expect_ihh0 = [np.nan, np.nan, np.nan, np.nan]
    assert_array_almost_equal(expect_ihh0, ihh0)
    expect_ihh1 = [np.nan, np.nan, np.nan, np.nan]
    assert_array_almost_equal(expect_ihh1, ihh1)

    ihh0, ihh1 = ihh01_scan(h, gaps, min_ehh=0, include_edges=True)
    expect_ihh0 = [0, 10, 20, 30]
    assert_array_almost_equal(expect_ihh0, ihh0)
    expect_ihh1 = [0, 10, 20, 30]
    assert_array_almost_equal(expect_ihh1, ihh1)


def test_ihh01_scan_d():
    gaps = np.array([10, 10, 10], dtype='f8')
    h = np.array([[0, 0, 1, 1, 1, 0],
                  [0, 1, 0, 1, 0, 1],
                  [1, 0, 0, 0, 1, 1],
                  [0, 0, 0, 1, 1, 1]])

    ihh0, ihh1 = ihh01_scan(h, gaps, min_ehh=0.05)
    x = (10 * (1 + 1 / 3) / 2) + (10 * (1 / 3 + 0) / 2)
    expect_ihh0 = [np.nan, np.nan, x, x]
    assert_array_almost_equal(expect_ihh0, ihh0)
    expect_ihh1 = [np.nan, np.nan, x, x]
    assert_array_almost_equal(expect_ihh1, ihh1)

    ihh0, ihh1 = ihh01_scan(h, gaps, min_ehh=0)
    expect_ihh0 = [np.nan, np.nan, x, x]
    assert_array_almost_equal(expect_ihh0, ihh0)
    expect_ihh1 = [np.nan, np.nan, x, x]
    assert_array_almost_equal(expect_ihh1, ihh1)

    ihh0, ihh1 = ihh01_scan(h, gaps, min_ehh=0, include_edges=True)
    expect_ihh0 = [0, 10 * 2 / 3, x, x]
    assert_array_almost_equal(expect_ihh0, ihh0)
    expect_ihh1 = [0, 10 * 2 / 3, x, x]
    assert_array_almost_equal(expect_ihh1, ihh1)


def test_ihh01_scan_e():
    # min_maf
    gaps = np.array([10, 10], dtype='f8')
    h = np.array([[0, 0, 1],
                  [0, 0, 1],
                  [0, 0, 1]])

    expect_ihh0 = [0, 10, 20]
    expect_ihh1 = [np.nan, np.nan, np.nan]
    ihh0, ihh1 = ihh01_scan(h, gaps, min_ehh=0, min_maf=0, include_edges=True)
    assert_array_almost_equal(expect_ihh0, ihh0)
    assert_array_almost_equal(expect_ihh1, ihh1)

    expect_ihh0 = [np.nan, np.nan, np.nan]
    expect_ihh1 = [np.nan, np.nan, np.nan]
    ihh0, ihh1 = ihh01_scan(h, gaps, min_ehh=0, min_maf=0.4, include_edges=True)
    assert_array_almost_equal(expect_ihh0, ihh0)
    assert_array_almost_equal(expect_ihh1, ihh1)


def test_ssl2ihh_a():

    # 2 haplotypes, 1 pair
    ssl = np.array([3], dtype='i4')
    gaps = np.array([1, 2, 4], dtype='f8')
    vidx = 3
    min_ehh = 0
    expect = (4 * (1 + 1) / 2) + (2 * (1 + 1) / 2) + (1 * (1 + 0) / 2)
    actual = ssl2ihh(ssl, max(ssl), vidx, gaps, min_ehh)
    assert expect == actual


def test_ssl2ihh_b():

    # 3 haplotypes, 3 pairs
    ssl = np.array([3, 0, 0], dtype='i4')
    gaps = np.array([1, 2, 4], dtype='f8')
    vidx = 3
    min_ehh = 0
    expect = (4 * (1/3 + 1/3) / 2) + (2 * (1/3 + 1/3) / 2) + \
             (1 * (1/3 + 0) / 2)
    actual = ssl2ihh(ssl, max(ssl), vidx, gaps, min_ehh)
    assert expect == actual


def test_ssl2ihh_c():

    # 3 haplotypes, 3 pairs
    ssl = np.array([3, 2, 1], dtype='i4')
    gaps = np.array([1, 2, 4], dtype='f8')
    vidx = 3
    min_ehh = 0
    expect = (4 * (1 + 2/3) / 2) + (2 * (2/3 + 1/3) / 2) + \
             (1 * (1/3 + 0) / 2)
    actual = ssl2ihh(ssl, max(ssl), vidx, gaps, min_ehh)
    assert expect == actual


def test_ssl2ihh_d():

    # 3 haplotypes, 3 pairs
    ssl = np.array([0, 1, 3], dtype='i4')
    gaps = np.array([1, 2, 4], dtype='f8')
    vidx = 3
    min_ehh = 0
    expect = (4 * (2/3 + 1/3) / 2) + (2 * (1/3 + 1/3) / 2) + \
             (1 * (1/3 + 0) / 2)
    actual = ssl2ihh(ssl, max(ssl), vidx, gaps, min_ehh)
    assert expect == actual


def test_ssl2ihh_e():

    # 2 haplotypes, 1 pair
    # no matches beyond current variant
    ssl = np.array([1], dtype='i4')
    gaps = np.array([1, 2, 4], dtype='f8')
    vidx = 3
    min_ehh = 0
    expect = 4 * (1 + 0) / 2
    actual = ssl2ihh(ssl, max(ssl), vidx, gaps, min_ehh)
    assert expect == actual


def test_ssl2ihh_f():

    # 2 haplotypes, 1 pair
    # never falls to min_ehh
    ssl = np.array([4], dtype='i4')
    gaps = np.array([1, 2, 4], dtype='f8')
    vidx = 3
    min_ehh = 0
    actual = ssl2ihh(ssl, max(ssl), vidx, gaps, min_ehh)
    assert np.isnan(actual), actual


def test_ssl2ihh_g():

    # 2 haplotypes, 1 pair
    # never falls to min_ehh, but include_edges
    ssl = np.array([4], dtype='i4')
    gaps = np.array([1, 2, 4], dtype='f8')
    vidx = 3
    min_ehh = 0
    expect = (4 + 2 + 1)
    actual = ssl2ihh(ssl, max(ssl), vidx, gaps, min_ehh, include_edges=True)
    assert expect == actual


def test_ssl2ihh_h():

    # 3 haplotypes, 3 pairs
    ssl = np.array([0, 1, 3], dtype='i4')
    gaps = np.array([1, 2, 4], dtype='f8')
    vidx = 3
    min_ehh = 0.5
    expect = 4 * (2/3 + 1/3) / 2
    actual = ssl2ihh(ssl, max(ssl), vidx, gaps, min_ehh)
    assert expect == actual


def test_ihs():
    n_variants = 1000
    n_haplotypes = 20
    h = np.random.randint(0, 2, size=(n_variants, n_haplotypes)).astype('i1')
    pos = np.arange(0, n_variants * 10, 10)

    for use_threads in True, False:
        for min_ehh in 0, 0.05, 0.5:
            for include_edges in True, False:
                score = ihs(h, pos, min_ehh=min_ehh,
                            include_edges=include_edges,
                            use_threads=use_threads)
                assert isinstance(score, np.ndarray)
                assert (n_variants,) == score.shape
                assert np.dtype('f8') == score.dtype

    with pytest.raises(ValueError):
        ihs(h, pos[1:])

    with pytest.raises(ValueError):
        ihs(h, pos, map_pos=pos[1:])


hap1 = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0],
    [1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],  # core variants
    [1, 1, 1, 1, 0],
    [1, 1, 1, 0, 0],
    [1, 1, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
])


hap2 = np.array([
    [0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0],
    [1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1],  # core variant
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0],
    [1, 1, 1, 0, 0],
    [1, 1, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
])


def test_ihs_data():
    h = np.hstack([hap1, hap2])
    pos = np.arange(1, h.shape[0] + 1)
    expect = np.log(5.5/1.5)

    for use_threads in True, False:
        for include_edges in True, False:
            score = ihs(h, pos, include_edges=include_edges,
                        use_threads=use_threads)
            actual = score[9]
            assert expect == actual


def test_xpehh():
    n_variants = 1000
    n_haplotypes = 20
    h1 = np.random.randint(0, 2, size=(n_variants, n_haplotypes)).astype('i1')
    h2 = np.random.randint(0, 2, size=(n_variants, n_haplotypes)).astype('i1')
    pos = np.arange(0, n_variants * 10, 10)

    for use_threads in True, False:
        for min_ehh in 0, 0.05, 0.5:
            for include_edges in True, False:
                score = xpehh(h1, h2, pos, min_ehh=min_ehh,
                              include_edges=include_edges,
                              use_threads=use_threads)
                assert isinstance(score, np.ndarray)
                assert (n_variants,) == score.shape
                assert np.dtype('f8') == score.dtype

    with pytest.raises(ValueError):
        xpehh(h1, h2[1:], pos)

    with pytest.raises(ValueError):
        xpehh(h1[1:], h2, pos)

    with pytest.raises(ValueError):
        xpehh(h1, h2, pos[1:])

    with pytest.raises(ValueError):
        xpehh(h1, h2, pos, map_pos=pos[1:])


def test_xpehh_data():
    pos = np.arange(1, hap1.shape[0] + 1)
    expect = -np.log(5.5/1.5)

    for use_threads in True, False:
        for include_edges in True, False:
            score = xpehh(hap1, hap2, pos, include_edges=include_edges,
                          use_threads=use_threads)
            actual = score[9]
            assert expect == actual


def test_nsl():
    n_variants = 1000
    n_haplotypes = 20
    h = np.random.randint(0, 2, size=(n_variants, n_haplotypes)).astype('i1')

    for use_threads in True, False:
        score = nsl(h, use_threads=use_threads)
        assert isinstance(score, np.ndarray)
        assert (n_variants,) == score.shape
        assert np.dtype('f8') == score.dtype


def test_xpnsl():
    n_variants = 1000
    n_haplotypes = 20
    h1 = np.random.randint(0, 2, size=(n_variants, n_haplotypes)).astype('i1')
    h2 = np.random.randint(0, 2, size=(n_variants, n_haplotypes)).astype('i1')

    for use_threads in True, False:
        score = xpnsl(h1, h2, use_threads=use_threads)
        assert isinstance(score, np.ndarray)
        assert (n_variants,) == score.shape
        assert np.dtype('f8') == score.dtype


def test_ehh_decay():
    h = [[0, 0, 1, 1],
         [0, 0, 1, 1],
         [0, 0, 0, 1],
         [0, 0, 0, 0],
         [0, 1, 0, 0]]
    e = [2/6, 2/6, 1/6, 1/6, 0]
    a = ehh_decay(h)
    assert_array_equal(e, a)


def test_voight_painting():
    h = [[0, 0, 1, 1],
         [0, 0, 1, 1],
         [0, 0, 0, 1],
         [0, 0, 0, 0],
         [0, 1, 0, 0]]
    e = [[1, 1, 2, 2],
         [1, 1, 2, 2],
         [1, 1, 0, 0],
         [1, 1, 0, 0],
         [0, 0, 0, 0]]
    a, _ = voight_painting(h)
    assert_array_equal(e, a)


def test_pbs():

    # minimal input data, sanity check for output existence and type
    ac1 = [[2, 0], [0, 2], [1, 1], [2, 0], [0, 2]]
    ac2 = [[1, 1], [2, 0], [0, 2], [2, 0], [0, 2]]
    ac3 = [[0, 2], [1, 1], [2, 0], [2, 0], [0, 2]]
    ret = pbs(ac1, ac2, ac3, window_size=2, window_step=1)
    assert isinstance(ret, np.ndarray)
    assert 1 == ret.ndim
    assert 4 == ret.shape[0]
    assert 'f' == ret.dtype.kind
    # regression check
    expect = [0.52349464,  0., -0.85199356, np.nan]
    assert_array_almost_equal(expect, ret)
    # final value is nan because variants in final window are non-segregating
    assert np.isnan(ret[3])
