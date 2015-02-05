# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


# GenotypeArray
###############

import allel
g = allel.GenotypeArray([[[0, 0], [0, 1]],
                         [[0, 1], [1, 1]],
                         [[0, 2], [-1, -1]]], dtype='i1')
g.dtype
g.ndim
g.shape
g.n_variants
g.n_samples
g.ploidy

g[1]

g[:, 1]

g[1, 0]

g = allel.GenotypeArray([[[0, 0, 0], [0, 0, 1]],
                         [[0, 1, 1], [1, 1, 1]],
                         [[0, 1, 2], [-1, -1, -1]]], dtype='i1')
g.ploidy


# allele_count
##############

import allel
g = allel.GenotypeArray([[[0, 0], [0, 1]],
                         [[0, 1], [1, 1]],
                         [[2, 2], [-1, -1]]], dtype='i1')


# HaplotypeArray
################

import allel
h = allel.HaplotypeArray([[0, 0, 0, 1],
                          [0, 1, 1, 1],
                          [0, 2, -1, -1]], dtype='i1')
h.dtype
h.ndim
h.shape
h.n_variants
h.n_haplotypes

h[1]

h[:, 1]

h[1, 0]

h.view_genotypes(ploidy=2)


# PosArray
##########

import allel
pos = allel.PositionIndex([2, 5, 14, 15, 42, 42, 77], dtype='i4')
pos.dtype
pos.ndim
pos.shape
pos.n_variants


# locate_position
#################

import allel
pos = allel.PositionIndex([3, 6, 11])
pos.locate_key(6)
pos.locate_key(7) is None


# locate_positons
#################

import allel
pos1 = allel.PositionIndex([3, 6, 11, 20, 35])
pos2 = allel.PositionIndex([4, 6, 20, 39])
cond1, cond2 = pos1.locate_keys(pos2)
cond1
cond2
pos1[cond1]
pos2[cond2]


# intersect
###########

import allel
pos1 = allel.PositionIndex([3, 6, 11, 20, 35])
pos2 = allel.PositionIndex([4, 6, 20, 39])
pos1.intersect(pos2)


# locate_interval
#################

import allel
pos = allel.PositionIndex([3, 6, 11, 20, 35])
loc = pos.locate_range(4, 32)
loc
pos[loc]


# locate_intervals
##################

import allel
import numpy as np
pos = allel.PositionIndex([3, 6, 11, 20, 35])
intervals = np.array([[0, 2], [6, 17], [12, 15], [31, 35], [100, 120]])
starts = intervals[:, 0]
stops = intervals[:, 1]
cond1, cond2 = pos.locate_ranges(starts, stops)
cond1
cond2
pos[cond1]
intervals[cond2]

