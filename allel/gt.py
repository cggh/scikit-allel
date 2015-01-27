# -*- coding: utf-8 -*-
"""
This module provides functions for manipulating arrays of discrete genotype
calls.

Conventions
-----------

By convention, an array of discrete genotype calls for a set of
samples at a set of variants is canonically represented as a
3-dimensional numpy array of integers.  The first dimension
corresponds to the variants genotyped (i.e., the length of the first
dimension is equal to the number of variants), the second dimension
corresponds to the samples genotyped, and the third dimension
corresponds to the ploidy of the samples.

For example, the array `g` below is a discrete genotype array storing
genotype calls for 3 variants in 2 samples with ploidy 2 (i.e.,
diploid)::

    >>> import numpy as np
    >>> g = np.array([[[0, 0], [0, 1]],
    ...               [[0, 1], [1, 1]],
    ...               [[0, 2], [-1, -1]]], dtype='i1')
    >>> g.dtype
    dtype('int8')
    >>> g.ndim
    3
    >>> n_variants, n_samples, ploidy = g.shape
    >>> n_variants
    3
    >>> n_samples
    2
    >>> ploidy
    2

An array of genotype calls for a single variant at all samples can be
obtained by indexing the first dimension, e.g.::

    >>> # view genotype calls at the second variant in all samples
    ... g[1]
    array([[0, 1],
           [1, 1]], dtype=int8)

An array of genotype calls for a single sample at all variants can be
obtained by indexing the second dimension, e.g.::

    >>> # view genotype calls at all variants in the second sample
    ... g[:, 1]
    array([[ 0,  1],
           [ 1,  1],
           [-1, -1]], dtype=int8)

A genotype call for a specific sample at a specific variant can be obtained
by indexing the first and second dimensions, e.g.::

    >>> # genotype call at the first variant, first sample is homozygous
    ... # reference
    ... g[0, 0]
    array([0, 0], dtype=int8)
    >>> # genotype call at the first variant, second sample is heterozygous
    ... g[0, 1]
    array([0, 1], dtype=int8)
    >>> # genotype call at the second variant, second sample is homozygous for
    ... # the first alternate allele
    ... g[1, 1]
    array([1, 1], dtype=int8)
    >>> # genotype call at the third variants, second sample is missing
    ... g[2, 1]
    array([-1, -1], dtype=int8)

Allelism
~~~~~~~~

Each integer within the array corresponds to an allele call, where 0
is the reference allele, 1 is the first alternate allele, 2 is the
second alternate allele, ... and -1 (or any other negative integer) is
a missing call. The actual alleles (i.e., the alternate nucleotide
sequences) and the physical positions of the variants within the
genome of an organism are stored in separate arrays, discussed
elsewhere.

A single byte integer dtype (int8) can represent up to 127 distinct
alleles, which is usually sufficient for most applications. Larger
integer dtypes can be used if more alleles are required. 

In many applications the number of distinct alleles for each variant
is small, e.g., less than 10, or even 2 (all variants are
biallelic). In these cases the canonical genotype array is not the
most compact way of storing genotype data in memory. This module
defines functions for bit-packing diploid genotype calls into single
bytes, and for transforming genotype arrays into sparse matrices,
which can assist in cases where memory usage needs to be minimised. Note
however that these more compact representations do not allow the same
flexibility in terms of using numpy universal functions to access and
manipulate data.

Phased and unphased calls
~~~~~~~~~~~~~~~~~~~~~~~~~

The canonical genotype array can store either phased or unphased
genotype calls. If the genotypes are phased (i.e., haplotypes have
been resolved) then individual haplotypes can be extracted by indexing
the second and third dimensions, e.g.::

    >>> # view the maternal haplotype for the second sample
    ... g[:, 1, 0]
    array([ 0,  1, -1], dtype=int8)

If the genotype calls are unphased then slices along the third
dimension have no meaning, and the ordering of alleles along the third
dimension is arbitrary. N.B, this means that an unphased diploid
heterozygous call could be stored as (0, 1) or equivalently as (1, 0).

"""


from __future__ import absolute_import, print_function, division


def is_called(g):
    """Find non-missing genotype calls.

    Parameters
    ----------

    g : array_like, shape (n_variants, n_samples, ploidy)
        Genotype array.

    Returns
    -------

    is_called : ndarray, bool
        Array of shape (n_variants, n_samples) where elements are True if
        the genotype call is non-missing.

    Examples
    --------

    TODO

    """
    pass


def is_missing(g):
    """TODO

    """
    pass


def is_hom(g):
    """TODO

    """
    pass


def is_het(g):
    """TODO

    """
    pass


def is_hom_ref(g):
    """TODO

    """
    pass


def is_hom_alt(g):
    """TODO

    """
    pass


def count_called(g, axis=None):
    """TODO

    """
    pass


def count_missing(g, axis=None):
    """TODO

    """
    pass


def count_hom(g, axis=None):
    """TODO

    """
    pass


def count_het(g, axis=None):
    """TODO

    """
    pass


def count_hom_ref(g, axis=None):
    """TODO

    """
    pass


def count_hom_alt(g, axis=None):
    """TODO

    """
    pass


def as_haplotypes(g):
    """TODO

    """
    pass


def as_n_alt(g):
    """TODO

    """
    pass


def as_012(g):
    """TODO

    """
    pass


def as_allele_counts(g):
    """TODO

    """
    pass


def pack_diploid(g):
    """TODO

    """
    pass


def unpack_diploid(g):
    """TODO

    """
    pass


def max_allele(g, axis=None):
    """TODO

    """
    pass


def allelism(g):
    """TODO

    """
    pass


def allele_number(g):
    """TODO

    """
    pass


def allele_count(g, allele=1):
    """TODO

    """
    pass


def allele_frequency(g, allele=1):
    """TODO

    """
    pass


def allele_counts(g, alleles=None):
    """TODO

    """
    pass


def allele_frequencies(g, alleles=None):
    """TODO

    """
    pass


def is_variant(g):
    """TODO

    """
    pass


def is_non_variant(g):
    """TODO

    """
    pass


def is_segregating(g):
    """TODO

    """
    pass


def is_non_segregating(g):
    """TODO

    """
    pass


def is_singleton(g):
    """TODO

    """
    pass


def is_doubleton(g):
    """TODO

    """
    pass


def count_variant(g):
    """TODO

    """
    pass


def count_non_variant(g):
    """TODO

    """
    pass


def count_segregating(g):
    """TODO

    """
    pass


def count_non_segregating(g):
    """TODO

    """
    pass


def count_singleton(g):
    """TODO

    """
    pass


def count_doubleton(g):
    """TODO

    """
    pass


def windowed_genotype_counts():
    """TODO

    """
    pass


def windowed_genotype_density():
    """TODO

    """
    pass


def windowed_genotype_rate():
    """TODO

    """
    pass


def plot_discrete_calldata():
    """TODO

    """
    pass


def plot_continuous_calldata():
    """TODO

    """
    pass


def plot_diploid_genotypes():
    """TODO

    """
    pass


def plot_genotype_counts_by_sample():
    """TODO

    """
    pass


def plot_genotype_counts_by_variant():
    """TODO

    """
    pass


def plot_continuous_calldata_by_sample():
    """TODO

    """
    pass


def plot_windowed_genotype_counts():
    """TODO

    """
    pass


def plot_windowed_genotype_density():
    """TODO

    """
    pass


def plot_windowed_genotype_rate():
    """TODO

    """
    pass
