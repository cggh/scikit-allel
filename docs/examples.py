# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import allel
import numpy as np
np.random.seed(42)
g = allel.GenotypeArray([[[0, 0], [0, 1]],
                         [[0, 2], [1, 1]],
                         [[1, 2], [2, 1]],
                         [[2, 2], [-1, -1]]])
g.haploidify()
g = allel.GenotypeArray([[[0, 0, 0], [0, 0, 1]],
                         [[0, 1, 1], [1, 1, 1]],
                         [[0, 1, 2], [-1, -1, -1]]])
g.haploidify()
