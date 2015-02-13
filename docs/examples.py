# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import allel
import numpy as np
np.random.seed(42)
g = allel.GenotypeArray([[[0, 0], [0, 1]],
                         [[0, 2], [1, 1]],
                         [[1, 2], [2, 1]],
                         [[2, 2], [-1, -1]]])
g.haploidify_samples()
g = allel.GenotypeArray([[[0, 0, 0], [0, 0, 1]],
                         [[0, 1, 1], [1, 1, 1]],
                         [[0, 1, 2], [-1, -1, -1]]])
g.haploidify_samples()


import allel
g = allel.GenotypeArray([[[0, 0], [0, 0], [0, 0]],
                         [[0, 0], [0, 1], [1, 1]],
                         [[0, 0], [1, 1], [2, 2]],
                         [[1, 1], [1, 2], [-1, -1]]])
g.heterozygosity_observed()
g.heterozygosity_expected()
g.inbreeding_coefficient()

import allel
g = allel.GenotypeArray([[[0, 0], [0, 0]],
                         [[0, 0], [0, 1]],
                         [[0, 0], [1, 1]],
                         [[0, 1], [1, 1]],
                         [[1, 1], [1, 1]],
                         [[0, 0], [1, 2]],
                         [[0, 1], [1, 2]],
                         [[0, 1], [-1, -1]]])
g.mean_pairwise_difference()

