# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division

import numpy as np
from hmmlearn import hmm
from allel.chunked import ChunkedArray


def call_roh(g, pos, ix, emission=(0.001, 0.0025, 0.01),
             transition=1e-6, min_roh=100000,
             is_accessible=None, contig_size=None):

    # this function computes the likely ROH using an HMM model.
    # define model:
    ### 2 state model, 3 observable
    # 0: ROH
    # 1: normal
    # 2: inaccessible
    emission_px = np.array(emission)
    emission_px.sort()

    # start probability
    start_prob = np.repeat(1/emission_px.size, emission_px.size)

    # transition between underlying states
    transitions = derive_transition_mx(transition, emission_px.size)

    # probability of inaccessible
    if is_accessible is None:
        assert contig_size is not None, \
            "If accessibility not provided must specify size of contig"
        # may be a mem problem?
        is_accessible = ChunkedArray(np.repeat(True, contig_size))

    p_accessible = is_accessible.mean()
    emission_mx = derive_emission_mx(emission_px, p_accessible)

    # initialize HMM
    roh_hmm = hmm.MultinomialHMM(n_components=emission_px.size)

    roh_hmm.n_symbols_ = 3
    roh_hmm.startprob_ = start_prob
    roh_hmm.transmat_ = transitions
    roh_hmm.emissionprob_ = emission_mx

    is_heterozygote = g.is_het()

    hz = is_heterozygote[:, ix]
    pred, obs = predict_roh_state(roh_hmm, hz, pos, is_accessible)
    homozygous_windows = get_state_windows(pred, state=0)

    # filter by roh size
    roh_sizes = np.diff(homozygous_windows, axis=1).flatten()
    roh = np.compress(roh_sizes >= min_roh, homozygous_windows, axis=0)

    # return roh and froh
    return roh, np.diff(roh, axis=1).sum() / is_accessible.size


def predict_roh_state(model, is_het, pos, accessible):

    # assume non-included pos are homozygous reference
    assert is_het.shape == pos.shape

    # declaration of this object may result in large mem usage
    observations = ChunkedArray(
        np.zeros((accessible.size, 1), dtype='int'))

    for i in np.compress(is_het, pos):
        observations[i - 1] = 1

    for i in np.where(accessible == 0)[0]:
        observations[i - 1] = 2

    predictions = model.predict(X=observations)
    return predictions, observations


def derive_emission_mx(prob, pa):
    # one row per p in prob
    # hom, het, unobserved
    mx = [[(1-p) * (1-pa), p * (1-pa), pa] for p in prob]
    mxe = np.array(mx)
    assert mxe.shape == (prob.size, 3)
    return mxe


def derive_transition_mx(pr, nstates):
    # this is a symmetric matrix
    mx = np.zeros((nstates, nstates))
    effective_tp = pr/(nstates-1)
    for i in range(nstates):
        for j in range(nstates):
            if i == j:
                mx[i, j] = 1 - pr
            else:
                mx[i, j] = effective_tp
    return mx


# This function translates the yes/no into a set of windows
def get_state_windows(predicted_state, state=0):

    assert isinstance(predicted_state, np.ndarray), \
        "get_state_windows expects an ndarray"
    wh = np.where(predicted_state == state)[0]
    if wh.size == 0:
        # then there are no things of interest
        return np.empty(0)
    elif wh.size == 1:
        # the whole thing is one big thing of interest
        return np.array([1, predicted_state.size])

    intervals = list()
    iv_start = wh[0]

    pos = None
    for i, pos in enumerate(wh[1:]):
        if (pos - wh[i]) > 1:
            intervals.append([iv_start, wh[i]])
            iv_start = pos
    intervals.append([iv_start, pos])

    roh = np.array(intervals)
    # correct for fact that pos 1 is in index 0.
    return roh + 1