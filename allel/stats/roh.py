# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np


from allel.model.ndarray import GenotypeVector
from allel.util import asarray_ndim, check_dim0_aligned


def roh_mhmm(gv, pos, phet_roh=0.001, phet_nonroh=(0.0025, 0.01), transition=1e-6,
             min_roh=0, is_accessible=None, contig_size=None):

    """Call ROH (runs of homozygosity) in a single individual given a genotype vector.

    This function computes the likely ROH using a Multinomial HMM model. There are 3
    observable states at each position in a chromosome/contig: 0 = Hom, 1 = Het,
    2 = inaccessible (i.e., unobserved).

    The model is provided with a probability of observing a het in a ROH (`phet_roh`) and one
    or more probabilities of observing a het in a non-ROH, as this probability may not be
    constant across the genome (`phet_nonroh`).

    Parameters
    ----------
    gv : array_like, int, shape (n_variants, ploidy)
        Genotype vector.
    pos: array_like, int, shape (n_variants,)
        Positions of variants, same 0th dimension as `g`.
    phet_roh: float, optional
        Probability of observing a heterozygote in a ROH. Appropriate values
        will depend on de novo mutation rate and genotype error rate.
    phet_nonroh: tuple of floats, optional
        One or more probabilites of observing a heterozygote outside of ROH.
        Appropriate values will depend primarily on nucleotide diversity within
        the population, but also on mutation rate and genotype error rate.
    transition: float, optional
        Probability of moving between states.
    min_roh: integer, optional
        Minimum size (bp) to condsider as a ROH. Will depend on contig size
        and recombination rate.
    is_accessible: array_like, bool, shape (`contig_size`,), optional
        Boolean array for each position in contig describing whether accessible
        or not.
    contig_size: int, optional
        If is_accessible not known/not provided, allows specification of
        total length of contig.

    Returns
    -------
    roh: ndarray, int, shape (n_roh, 2)
        Span windows of ROH.
    froh: float
        Proportion of genome in a ROH.

    Notes
    -----
    This function currently requires around 4GB for a contig size of ~50Mbp.

    """

    from hmmlearn import hmm

    # setup inputs
    if isinstance(phet_nonroh, float):
        phet_nonroh = phet_nonroh,
    gv = GenotypeVector(gv)
    pos = asarray_ndim(pos, 1)
    check_dim0_aligned(gv, pos)
    is_accessible = asarray_ndim(is_accessible, 1, dtype=bool)

    # heterozygote probabilities
    het_px = np.concatenate([(phet_roh,), phet_nonroh])

    # start probabilities (all equal)
    start_prob = np.repeat(1/het_px.size, het_px.size)

    # transition between underlying states
    transition_mx = _mhmm_derive_transition_mx(transition, het_px.size)

    # probability of inaccessible
    if is_accessible is None:
        assert contig_size is not None, \
            "If accessibility not provided must specify size of contig"
        p_accessible = 1.0
    else:
        p_accessible = is_accessible.mean()
        assert contig_size is None, "Contig size only specified when " \
                                    "is_accessible is not provided"
        contig_size = is_accessible.size

    emission_mx = _mhmm_derive_emission_mx(het_px, p_accessible)

    # initialize HMM
    roh_hmm = hmm.MultinomialHMM(n_components=het_px.size)
    roh_hmm.n_symbols_ = 3
    roh_hmm.startprob_ = start_prob
    roh_hmm.transmat_ = transition_mx
    roh_hmm.emissionprob_ = emission_mx

    # locate heterozygous calls
    is_het = gv.is_het()

    # predict ROH state
    pred, obs = _mhmm_predict_roh_state(roh_hmm, is_het, pos, is_accessible, contig_size)

    # find ROH windows
    homozygous_windows = _mhmm_get_state_windows(pred, state=0)

    # filter by ROH size
    roh_sizes = np.diff(homozygous_windows, axis=1).flatten()
    roh = np.compress(roh_sizes >= min_roh, homozygous_windows, axis=0)

    # compute FROH
    froh = np.diff(roh, axis=1).sum() / contig_size

    return roh, froh


def _mhmm_predict_roh_state(model, is_het, pos, is_accessible, contig_size):

    # construct observations, one per position in contig
    observations = np.zeros((contig_size, 1), dtype='i1')

    # these are hets
    observations[np.compress(is_het, pos) - 1] = 1

    # these are unobserved
    if is_accessible is not None:
        observations[~is_accessible] = 2

    predictions = model.predict(X=observations)
    return predictions, observations


def _mhmm_derive_emission_mx(het_px, p_accessible):
    # one row per p in prob
    # hom, het, unobserved
    mx = [[(1 - p) * p_accessible, p * p_accessible, 1 - p_accessible] for p in het_px]
    mxe = np.array(mx)
    assert mxe.shape == (het_px.size, 3)
    return mxe


def _mhmm_derive_transition_mx(transition, nstates):
    # this is a symmetric matrix
    mx = np.zeros((nstates, nstates))
    effective_tp = transition / (nstates - 1)
    for i in range(nstates):
        for j in range(nstates):
            if i == j:
                mx[i, j] = 1 - transition
            else:
                mx[i, j] = effective_tp
    return mx


def _mhmm_get_state_windows(predicted_state, state=0):
    """Translates the yes/no into a set of windows."""

    wh = np.where(predicted_state == state)[0]
    if wh.size == 0:
        # then there are no things of interest
        return np.empty((0, 2))
    elif wh.size == predicted_state.size:
        # the whole thing is one big thing of interest
        return np.array([1, predicted_state.size])
    else:
        intervals = list()
        iv_start = wh[0]

        for i, pos in enumerate(wh[1:]):
            if (pos - wh[i]) > 1:
                intervals.append([iv_start, wh[i]])
                iv_start = pos
        # final interval
        intervals.append([iv_start, pos])
        roh = np.array(intervals)
        # correct for fact that pos 1 is in index 0.
        return roh + 1
