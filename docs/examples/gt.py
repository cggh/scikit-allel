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


# as_haplotypes()
#################

import allel
import numpy as np
g = np.array([[[0, 0], [0, 1]],
              [[0, 1], [1, 1]],
              [[0, 2], [-1, -1]]], dtype='i1')
allel.gt.as_haplotypes(g)


# as_n_alt()
#################

import allel
import numpy as np
g = np.array([[[0, 0], [0, 1]],
              [[0, 2], [1, 1]],
              [[2, 2], [-1, -1]]], dtype='i1')
allel.gt.as_n_alt(g)


# as_allele_counts()
####################

import allel
import numpy as np
g = np.array([[[0, 0], [0, 1]],
              [[0, 2], [1, 1]],
              [[2, 2], [-1, -1]]], dtype='i1')
allel.gt.as_allele_counts(g)
allel.gt.as_allele_counts(g, alleles=(0, 1))


# pack_diploid()
################

import allel
import numpy as np
g = np.array([[[0, 0], [0, 1]],
              [[0, 2], [1, 1]],
              [[2, 2], [-1, -1]]], dtype='i1')
allel.gt.pack_diploid(g)


# unpack_diploid()
##################

import allel
import numpy as np
packed = np.array([[0, 1],
                   [2, 17],
                   [34, 239]], dtype='u1')
allel.gt.unpack_diploid(packed)


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
