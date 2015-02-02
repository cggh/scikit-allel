# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np
g = np.array([[[0, 0], [0, 1]],
              [[0, 1], [1, 1]],
              [[0, 2], [-1, -1]]], dtype='i1')
g.dtype
g.ndim
n_variants, n_samples, ploidy = g.shape
n_variants
n_samples
ploidy
# view genotype calls at the second variant in all samples
g[1]
# view genotype calls at all variants in the second sample
g[:, 1]
# genotype call at the first variant, first sample is homozygous reference
g[0, 0]
# genotype call at the first variant, second sample is heterozygous
g[0, 1]
# genotype call at the second variant, second sample is homozygous for the
# first alternate allele
g[1, 1]
# genotype call at the third variants, second sample is missing
g[2, 1]


# is_called()
#############

import allel
import numpy as np
g = np.array([[[0, 0], [0, 1]],
              [[0, 1], [1, 1]],
              [[0, 2], [-1, -1]]], dtype='i1')
allel.gt.is_called(g)


# is_missing()
##############

import allel
import numpy as np
g = np.array([[[0, 0], [0, 1]],
              [[0, 1], [1, 1]],
              [[0, 2], [-1, -1]]], dtype='i1')
allel.gt.is_missing(g)


# is_hom()
##########

import allel
import numpy as np
g = np.array([[[0, 0], [0, 1]],
              [[0, 1], [1, 1]],
              [[0, 2], [-1, -1]]], dtype='i1')
allel.gt.is_hom(g)


# to_haplotypes()
#################

import allel
import numpy as np
g = np.array([[[0, 0], [0, 1]],
              [[0, 1], [1, 1]],
              [[0, 2], [-1, -1]]], dtype='i1')
allel.gt.to_haplotypes(g)


# from_haplotypes()
###################

import allel
import numpy as np
h = np.array([[0, 0, 0, 1],
              [0, 1, 1, 1],
              [0, 2, -1, -1]], dtype='i1')
allel.gt.from_haplotypes(h, ploidy=2)


# to_n_alt()
############

import allel
import numpy as np
g = np.array([[[0, 0], [0, 1]],
              [[0, 2], [1, 1]],
              [[2, 2], [-1, -1]]], dtype='i1')
allel.gt.to_n_alt(g)


# to_allele_counts()
####################

import allel
import numpy as np
g = np.array([[[0, 0], [0, 1]],
              [[0, 2], [1, 1]],
              [[2, 2], [-1, -1]]], dtype='i1')
allel.gt.to_allele_counts(g)
allel.gt.to_allele_counts(g, alleles=(0, 1))


# to_packed()
#############

import allel
import numpy as np
g = np.array([[[0, 0], [0, 1]],
              [[0, 2], [1, 1]],
              [[2, 2], [-1, -1]]], dtype='i1')
allel.gt.to_packed(g)


# from_packed()
###############

import allel
import numpy as np
packed = np.array([[0, 1],
                   [2, 17],
                   [34, 239]], dtype='u1')
allel.gt.from_packed(packed)


# max_allele()
##############

import allel
import numpy as np
g = np.array([[[0, 0], [0, 1]],
              [[0, 2], [1, 1]],
              [[2, 2], [-1, -1]]], dtype='i1')
allel.gt.max_allele(g)
allel.gt.max_allele(g, axis=(0, 2))
allel.gt.max_allele(g, axis=(1, 2))


# allelism()
############

import allel
import numpy as np
g = np.array([[[0, 0], [0, 1]],
              [[0, 2], [1, 1]],
              [[2, 2], [-1, -1]]], dtype='i1')
allel.gt.allelism(g)


# allele_number()
#################

import allel
import numpy as np
g = np.array([[[0, 0], [0, 1]],
              [[0, 2], [1, 1]],
              [[2, 2], [-1, -1]]], dtype='i1')
allel.gt.allele_number(g)


# allele_count()
################

import allel
import numpy as np
g = np.array([[[0, 0], [0, 1]],
              [[0, 2], [1, 1]],
              [[2, 2], [-1, -1]]], dtype='i1')
allel.gt.allele_count(g, allele=1)
allel.gt.allele_count(g, allele=2)


# allele_frequency()
####################

import allel
import numpy as np
g = np.array([[[0, 0], [0, 1]],
              [[0, 2], [1, 1]],
              [[2, 2], [-1, -1]]], dtype='i1')
af, ac, an = allel.gt.allele_frequency(g, allele=1)
af
af, ac, an = allel.gt.allele_frequency(g, allele=2)
af


# allele_counts()
#################

import allel
import numpy as np
g = np.array([[[0, 0], [0, 1]],
              [[0, 2], [1, 1]],
              [[2, 2], [-1, -1]]], dtype='i1')
allel.gt.allele_counts(g)
allel.gt.allele_counts(g, alleles=(1, 2))


# allele_frequencies()
######################

import allel
import numpy as np
g = np.array([[[0, 0], [0, 1]],
              [[0, 2], [1, 1]],
              [[2, 2], [-1, -1]]], dtype='i1')
af, ac, an = allel.gt.allele_frequencies(g)
af
af, ac, an = allel.gt.allele_frequencies(g, alleles=(1, 2))
af


# is_variant()
##############

import allel
import numpy as np
g = np.array([[[0, 0], [0, 0]],
              [[0, 0], [0, 1]],
              [[0, 2], [1, 1]],
              [[2, 2], [-1, -1]]], dtype='i1')
allel.gt.is_variant(g)


# is_non_variant()
##################

import allel
import numpy as np
g = np.array([[[0, 0], [0, 0]],
              [[0, 0], [0, 1]],
              [[0, 2], [1, 1]],
              [[2, 2], [-1, -1]]], dtype='i1')
allel.gt.is_non_variant(g)


# is_segregating()
##################

import allel
import numpy as np
g = np.array([[[0, 0], [0, 0]],
              [[0, 0], [0, 1]],
              [[1, 1], [1, 2]],
              [[2, 2], [-1, -1]]], dtype='i1')
allel.gt.is_segregating(g)


# is_non_segregating()
######################

import allel
import numpy as np
g = np.array([[[0, 0], [0, 0]],
              [[0, 0], [0, 1]],
              [[1, 1], [1, 2]],
              [[2, 2], [-1, -1]]], dtype='i1')
allel.gt.is_non_segregating(g)


# is_singleton()
################

import allel
import numpy as np
g = np.array([[[0, 0], [0, 0]],
              [[0, 0], [0, 1]],
              [[1, 1], [1, 2]],
              [[2, 2], [-1, -1]]], dtype='i1')
allel.gt.is_singleton(g, allele=1)
allel.gt.is_singleton(g, allele=2)


# is_doubleton()
################

import allel
import numpy as np
g = np.array([[[0, 0], [0, 0]],
              [[0, 0], [1, 1]],
              [[1, 1], [1, 2]],
              [[2, 2], [-1, -1]]], dtype='i1')
allel.gt.is_doubleton(g, allele=1)
allel.gt.is_doubleton(g, allele=2)


# count()
#########

import allel
import numpy as np
g = np.array([[[0, 0], [0, 0]],
              [[0, 0], [1, 1]],
              [[1, 1], [1, 2]],
              [[2, 2], [-1, -1]]], dtype='i1')
b = allel.gt.is_called(g)
allel.gt.count(b)
allel.gt.count(b, axis='variants')
allel.gt.count(b, axis='samples')
b = allel.gt.is_variant(g)
allel.gt.count(b)


# windowed_count()
##################

import allel
import numpy as np
g = np.array([[[0, 0], [0, 0]],
              [[0, 1], [0, 1]],
              [[1, 1], [1, 2]],
              [[2, 2], [-1, -1]]], dtype='i1')
pos = np.array([2, 14, 15, 27])
b = allel.gt.is_variant(g)
counts, bin_edges = allel.gt.windowed_count(pos, b, window=10)
bin_edges
counts
counts, bin_edges = allel.gt.windowed_count(pos, b, window=10,
                                            start=1,
                                            stop=27)
bin_edges
counts
b = allel.gt.is_het(g)
counts, bin_edges = allel.gt.windowed_count(pos, b, window=10)
bin_edges
counts


# windowed_density()
####################

import allel
import numpy as np
g = np.array([[[0, 0], [0, 0]],
              [[0, 1], [0, 1]],
              [[1, 1], [1, 2]],
              [[2, 2], [-1, -1]]], dtype='i1')
pos = np.array([1, 14, 15, 27])
b = allel.gt.is_variant(g)
densities, counts, bin_edges = allel.gt.windowed_density(pos, b,
                                                         window=10)
bin_edges
counts
densities
is_accessible = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          1, 1, 1, 1, 0, 0, 1, 1, 0, 0,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=bool)
densities, counts, bin_edges = allel.gt.windowed_density(
    pos, b, window=10, is_accessible=is_accessible, fill=np.nan
)
bin_edges
counts
densities


# to_sparse()
#############

import allel
import numpy as np
g = np.array([[[0, 0], [0, 0]],
              [[0, 1], [0, 1]],
              [[1, 1], [0, 0]],
              [[0, 0], [-1, -1]]], dtype='i1')
m = allel.gt.to_sparse(g, format='csr')
m
m.data
m.indices
m.indptr


# from_sparse()
###############

import allel
import numpy as np
import scipy.sparse
data = np.array([ 1,  1,  1,  1, -1, -1], dtype=np.int8)
indices = np.array([1, 3, 0, 1, 2, 3], dtype=np.int32)
indptr = np.array([0, 0, 2, 4, 6], dtype=np.int32)
m = scipy.sparse.csr_matrix((data, indices, indptr))
g = allel.gt.from_sparse(m, ploidy=2)
g

