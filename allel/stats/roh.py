# -*- coding: utf-8 -*-
import numpy as np

from allel.model.ndarray import GenotypeVector
from allel.util import asarray_ndim, check_dim0_aligned
from allel.stats.misc import tabulate_state_blocks
from allel.stats.window import equally_accessible_windows, windowed_statistic, position_windows


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
        Positions of variants, same 0th dimension as `gv`.
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
    df_roh: DataFrame
        Data frame where each row describes a run of homozygosity. Columns are 'start',
        'stop', 'length' and 'is_marginal'. Start and stop are 1-based, stop-inclusive.
    froh: float
        Proportion of genome in a ROH.

    Notes
    -----
    This function requires `hmmlearn <http://hmmlearn.readthedocs.io/en/latest/>`_ to be
    installed.

    This function currently requires around 4GB memory for a contig size of ~50Mbp.

    """

    from hmmlearn import hmm

    # setup inputs
    if isinstance(phet_nonroh, float):
        phet_nonroh = phet_nonroh,
    gv = GenotypeVector(gv)
    pos = asarray_ndim(pos, 1)
    check_dim0_aligned(gv, pos)
    is_accessible = asarray_ndim(is_accessible, 1, dtype=bool, allow_none=True)

    # heterozygote probabilities
    het_px = np.concatenate([(phet_roh,), phet_nonroh])

    # start probabilities (all equal)
    start_prob = np.repeat(1/het_px.size, het_px.size)

    # transition between underlying states
    transition_mx = _hmm_derive_transition_matrix(transition, het_px.size)

    # probability of inaccessible
    if is_accessible is None:
        if contig_size is None:
            raise ValueError(
                "If is_accessible argument is not provided, you must provide `contig_size`")
        p_accessible = 1.0
    else:
        p_accessible = is_accessible.mean()
        contig_size = is_accessible.size

    emission_mx = _mhmm_derive_emission_matrix(het_px, p_accessible)

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
    df_blocks = tabulate_state_blocks(pred, states=list(range(len(het_px))))
    df_roh = df_blocks[(df_blocks.state == 0)].reset_index(drop=True)
    # adapt the dataframe for ROH
    for col in 'state', 'support', 'start_lidx', 'stop_ridx', 'size_max':
        del df_roh[col]
    df_roh.rename(columns={'start_ridx': 'start',
                           'stop_lidx': 'stop',
                           'size_min': 'length'},
                  inplace=True)
    # make coordinates 1-based
    df_roh['start'] = df_roh['start'] + 1
    df_roh['stop'] = df_roh['stop'] + 1

    # filter by ROH size
    if min_roh > 0:
        df_roh = df_roh[df_roh.length >= min_roh]

    # compute FROH
    froh = df_roh.length.sum() / contig_size

    return df_roh, froh


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


def roh_poissonhmm(gv, pos, phet_roh=0.001, phet_nonroh=(0.0025, 0.01), transition=1e-3,
                   window_size=1000, min_roh=0, is_accessible=None, contig_size=None):

    """Call ROH (runs of homozygosity) in a single individual given a genotype vector.

    This function computes the likely ROH using a Poisson HMM model. The chromosome is divided into
    equally accessible windows of specified size, then the number of hets observed in each is used
    to fit a Poisson HMM. Note this is much faster than `roh_mhmm`, but at the cost of some
    resolution.

    The model is provided with a probability of observing a het in a ROH (`phet_roh`) and one
    or more probabilities of observing a het in a non-ROH, as this probability may not be
    constant across the genome (`phet_nonroh`).

    Parameters
    ----------
    gv : array_like, int, shape (n_variants, ploidy)
        Genotype vector.
    pos: array_like, int, shape (n_variants,)
        Positions of variants, same 0th dimension as `gv`.
    phet_roh: float, optional
        Probability of observing a heterozygote in a ROH. Appropriate values
        will depend on de novo mutation rate and genotype error rate.
    phet_nonroh: tuple of floats, optional
        One or more probabilites of observing a heterozygote outside of ROH.
        Appropriate values will depend primarily on nucleotide diversity within
        the population, but also on mutation rate and genotype error rate.
    transition: float, optional
        Probability of moving between states. This is based on windows, so a larger window size may
        call for a larger transitional probability
    window_size: integer, optional
        Window size (equally accessible bases) to consider as a potential ROH. Setting this window
        too small may result in spurious ROH calls, while too large will result in a lack of
        resolution.
    min_roh: integer, optional
        Minimum size (bp) to condsider as a ROH. Will depend on contig size and recombination rate.
    is_accessible: array_like, bool, shape (`contig_size`,), optional
        Boolean array for each position in contig describing whether accessible
        or not. Although optional, highly recommended so invariant sites are distinguishable from
        sites where variation is inaccessible
    contig_size: integer, optional
        If is_accessible is not available, use this to specify the size of the contig, and assume
        all sites are accessible.


    Returns
    -------
    df_roh: DataFrame
        Data frame where each row describes a run of homozygosity. Columns are 'start',
        'stop', 'length' and 'is_marginal'. Start and stop are 1-based, stop-inclusive.
    froh: float
        Proportion of genome in a ROH.

    Notes
    -----
    This function requires `pomegranate` (>= 0.9.0) to be installed.

    """

    from pomegranate import HiddenMarkovModel, PoissonDistribution

    is_accessible = asarray_ndim(is_accessible, 1, dtype=bool, allow_none=True)

    # equally accessbile windows
    if is_accessible is None:
        if contig_size is None:
            raise ValueError(
                "If is_accessible argument is not provided, you must provide `contig_size`")

        # given no accessibility provided use the standard window calculation
        roh_windows = position_windows(
            pos=pos, size=window_size, step=window_size, start=1, stop=contig_size)
    else:
        contig_size = is_accessible.size
        roh_windows = equally_accessible_windows(is_accessible, window_size)

    ishet = GenotypeVector(gv).is_het()
    counts, wins, records = windowed_statistic(pos, ishet, np.sum, windows=roh_windows)

    # heterozygote probabilities
    het_px = np.concatenate([(phet_roh,), phet_nonroh])

    # start probabilities (all equal)
    start_prob = np.repeat(1/het_px.size, het_px.size)

    # transition between underlying states
    transition_mx = _hmm_derive_transition_matrix(transition, het_px.size)

    dists = [PoissonDistribution(x * window_size) for x in het_px]

    model = HiddenMarkovModel.from_matrix(transition_probabilities=transition_mx,
                                          distributions=dists,
                                          starts=start_prob)

    prediction = np.array(model.predict(counts[:, None]))

    df_blocks = tabulate_state_blocks(prediction, states=list(range(len(het_px))))
    df_roh = df_blocks[(df_blocks.state == 0)].reset_index(drop=True)

    # adapt the dataframe for ROH
    df_roh["start"] = df_roh.start_ridx.apply(lambda y: roh_windows[y, 0])
    df_roh["stop"] = df_roh.stop_lidx.apply(lambda y: roh_windows[y, 1])
    df_roh["length"] = df_roh.stop - df_roh.start

    # filter by ROH size
    if min_roh > 0:
        df_roh = df_roh[df_roh.length >= min_roh]

    # compute FROH
    froh = df_roh.length.sum() / contig_size

    return df_roh[["start", "stop", "length", "is_marginal"]], froh


def _mhmm_derive_emission_matrix(het_px, p_accessible):
    # one row per p in prob
    # hom, het, unobserved
    mx = [[(1 - p) * p_accessible, p * p_accessible, 1 - p_accessible] for p in het_px]
    mx = np.array(mx)
    assert mx.shape == (het_px.size, 3)
    return mx


def _hmm_derive_transition_matrix(transition, nstates):
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
