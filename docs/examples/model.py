# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


# GenotypeArray
###############

import allel
import numpy as np
data = np.array([[[0, 0], [0, 1]],
                 [[0, 1], [1, 1]],
                 [[0, 2], [-1, -1]]], dtype='i1')
g = allel.GenotypeArray(data)
g.dtype
g.ndim
g.shape
g.n_variants
g.n_samples
g.ploidy

g[1]

g[:, 1]

g[1, 0]

data = np.array([[[0, 0, 0], [0, 0, 1]],
                 [[0, 1, 1], [1, 1, 1]],
                 [[0, 1, 2], [-1, -1, -1]]], dtype='i1')
g_triploid = allel.GenotypeArray(data)
g_triploid.ploidy