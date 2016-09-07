# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import numpy as np


def call_roh(g, pos, ix, phet_roh=0.001, phet_nonroh=(0.0025, 0.01),
             transition=1e-6, min_roh=100000,
             is_accessible=None, contig_size=None, plot=True,
             plotwinsize=50000):

    """
    Call ROH (runs of homozygosity) of a individual given a genotype array
    This function computes the likely ROH using an HMM model. There are 3
    observable states: 0 = Hom, 1 = Het, 2 = inaccessible (ie unobserved).

    The model is provided with a probability of observing a het in a ROH (
    phet_roh) and one or more probabilities of observing a het in a non-ROH,
    as this probability may not be constant across the genome (phet_nonroh).

    Parameters
    ----------
    g : array_like, int, shape (n_variants, n_samples, ploidy)
        Genotype array.
    pos: array_like, int, shape (n_variants)
        Positions of variants, same 0th dimension as g
    ix: integer
        index of the sample on which to call roh
    phet_roh: float
        Probability of observing a heterozygote in an ROH. Appropriate values
        will depend on de novo mutation rate, frequency of sequencing error &
        spurious het calls due to misalignment.
    phet_nonroh: tuple, array_like, float
        One or more probabilites of observing a heterozygote in an non ROH.
        Appropriate values will depend on population history, frequency of
        sequencing error, mutation rate, and spurious het calls due to
        misalignment.
    transition: float
        Probability of moving between states
    min_roh: integer
        Minimum size (bp) to condsider as a ROH. Will depend on contig size
        and recombination rate. In humans typically > 1Mbp
    is_accessible: array_like, bool, shape (npositions)
        Boolean array for each locus in contig describing whether accessible
        or not.
    contig_size: int
        If is_accessible not known/not provided, allows specification of
        total length of contig.
    plot: bool
        Whether to plot the ROH with heterzygosity and inaccessible regions
    plotwinsize: int
        Size of window in which to summarize heterozygosity for the plot


    Returns
    -------
    roh: ndarray, int, shape (n_roh, 2)
        Span windows of ROH
    froh: float
        Proportion of genome in ROH

    """

    from hmmlearn import hmm

    if isinstance(phet_nonroh, float):
        phet_nonroh = np.array((phet_nonroh,))

    emission_px = np.concatenate([(phet_roh, ), phet_nonroh])

    # start probability
    start_prob = np.repeat(1/emission_px.size, emission_px.size)

    # transition between underlying states
    transitions = _derive_transition_mx(transition, emission_px.size)

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

    emission_mx = _derive_emission_mx(emission_px, p_accessible)

    # initialize HMM
    roh_hmm = hmm.MultinomialHMM(n_components=emission_px.size)

    roh_hmm.n_symbols_ = 3
    roh_hmm.startprob_ = start_prob
    roh_hmm.transmat_ = transitions
    roh_hmm.emissionprob_ = emission_mx

    gsub = g.take([ix], axis=1)
    hz = gsub.count_het(axis=1).astype("bool")

    pred, obs = _predict_roh_state(roh_hmm, hz, pos, is_accessible, contig_size)
    homozygous_windows = _get_state_windows(pred, state=0)

    # filter by roh size
    roh_sizes = np.diff(homozygous_windows, axis=1).flatten()
    roh = np.compress(roh_sizes >= min_roh, homozygous_windows, axis=0)
    froh = np.diff(roh, axis=1).sum() / contig_size

    if plot:
        import matplotlib.pyplot as plt
        from allel.stats import windowed_statistic

        window = np.arange(0, is_accessible.size, plotwinsize)
        windows = np.vstack([window[:-1], window[1:]]).T

        count_hz, win, count = windowed_statistic(pos=pos, values=hz,
                                                  statistic=np.sum,
                                                  windows=windows)

        if is_accessible is not None:

            # remove the last val as not a complete window
            count_acc = np.add.reduceat(
                is_accessible, window)[:-1]

            # if accessible bases in a window < 1000, don't plot.
            # b/c Leads to spiky data
            count_acc[count_acc < 1000] = 0
            count_hz[count_acc < 1000] = 0
            heterozyg = count_hz/count_acc

            assert not np.any(count_hz > count_acc), \
                "heterozygosity exceeds 1.0. Check accessibility array input"
        else:
            heterozyg = count_hz/plotwinsize

        x = win.mean(axis=1) * 1e-6

        plt.figure(figsize=(6, 1.5))
        plt.plot(x, heterozyg, "b-", label="Heterozygosity")
        plt.ylabel('Heterozygosity')

        for low, upp in roh * 1e-6:
            plt.axvspan(low, upp, facecolor='g', alpha=0.5, edgecolor='none')

        if is_accessible is not None:
            inaccessible_windows = _get_state_windows(is_accessible)

            for low, upp in (inaccessible_windows * 1e-6):
                # don't plot inaccessible gaps < 5kb
                if (upp - low) > .005:
                    plt.axvspan(low, upp, facecolor='m', alpha=0.1,
                                edgecolor='none')

        plt.xlabel('Position (Mbp)')
        plt.title(" ".join(['Called runs of homozygosity.',
                            r'$F_{ROH}:$',
                            "{0:.4f}".format(froh)]))
        plt.tick_params(top="off", right="off")
        plt.legend()

    # return roh and froh
    return roh, froh


def _predict_roh_state(model, is_het, pos, is_accessible=None,
                       contig_size=None):

    # assume non-included pos are homozygous reference
    assert is_het.shape == pos.shape

    # declaration of this object may result in large mem usage
    if is_accessible is not None:
        contig_size = is_accessible.size

    observations = np.zeros((contig_size, 1), dtype='int')

    # these are hets
    observations[np.compress(is_het, pos) - 1] = 1

    # these are unobserved
    if is_accessible is not None:
        observations[~is_accessible] = 2

    predictions = model.predict(X=observations)
    return predictions, observations


def _derive_emission_mx(prob, pa):
    # one row per p in prob
    # hom, het, unobserved
    mx = [[(1-p)*pa, p*pa, 1-pa] for p in prob]
    mxe = np.array(mx)
    assert mxe.shape == (prob.size, 3)
    return mxe


def _derive_transition_mx(pr, nstates):
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
def _get_state_windows(predicted_state, state=0):

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
        intervals.append([iv_start, pos])
        roh = np.array(intervals)
        # correct for fact that pos 1 is in index 0.
        return roh + 1