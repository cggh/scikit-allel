# -*- coding: utf-8 -*-
from collections import OrderedDict


import numpy as np


from allel.compat import memoryview_safe
from allel.model.ndarray import SortedIndex
from allel.util import asarray_ndim, check_dim0_aligned, check_integer_dtype
from allel.opt.stats import state_transitions


def jackknife(values, statistic):
    """Estimate standard error for `statistic` computed over `values` using
    the jackknife.

    Parameters
    ----------
    values : array_like or tuple of array_like
        Input array, or tuple of input arrays.
    statistic : function
        The statistic to compute.

    Returns
    -------
    m : float
        Mean of jackknife values.
    se : float
        Estimate of standard error.
    vj : ndarray
        Statistic values computed for each jackknife iteration.

    """

    if isinstance(values, tuple):
        # multiple input arrays
        n = len(values[0])
        masked_values = [np.ma.asarray(v) for v in values]
        for m in masked_values:
            assert m.ndim == 1, 'only 1D arrays supported'
            assert m.shape[0] == n, 'input arrays not of equal length'
            m.mask = np.zeros(m.shape, dtype=bool)

    else:
        n = len(values)
        masked_values = np.ma.asarray(values)
        assert masked_values.ndim == 1, 'only 1D arrays supported'
        masked_values.mask = np.zeros(masked_values.shape, dtype=bool)

    # values of the statistic calculated in each jackknife iteration
    vj = list()

    for i in range(n):

        if isinstance(values, tuple):
            # multiple input arrays
            for m in masked_values:
                m.mask[i] = True
            x = statistic(*masked_values)
            for m in masked_values:
                m.mask[i] = False

        else:
            masked_values.mask[i] = True
            x = statistic(masked_values)
            masked_values.mask[i] = False

        vj.append(x)

    # convert to array for convenience
    vj = np.array(vj)

    # compute mean of jackknife values
    m = vj.mean()

    # compute standard error
    sv = ((n - 1) / n) * np.sum((vj - m) ** 2)
    se = np.sqrt(sv)

    return m, se, vj


def plot_variant_locator(pos, step=None, ax=None, start=None,
                         stop=None, flip=False,
                         line_kwargs=None):
    """
    Plot lines indicating the physical genome location of variants from a
    single chromosome/contig. By default the top x axis is in variant index
    space, and the bottom x axis is in genome position space.

    Parameters
    ----------

    pos : array_like
        A sorted 1-dimensional array of genomic positions from a single
        chromosome/contig.
    step : int, optional
        Plot a line for every `step` variants.
    ax : axes, optional
        The axes on which to draw. If not provided, a new figure will be
        created.
    start : int, optional
        The start position for the region to draw.
    stop : int, optional
        The stop position for the region to draw.
    flip : bool, optional
        Flip the plot upside down.
    line_kwargs : dict-like
        Additional keyword arguments passed through to `plt.Line2D`.

    Returns
    -------

    ax : axes
        The axes on which the plot was drawn

    """

    import matplotlib.pyplot as plt

    # check inputs
    pos = SortedIndex(pos, copy=False)

    # set up axes
    if ax is None:
        x = plt.rcParams['figure.figsize'][0]
        y = x / 7
        fig, ax = plt.subplots(figsize=(x, y))
        fig.tight_layout()

    # determine x axis limits
    if start is None:
        start = np.min(pos)
    if stop is None:
        stop = np.max(pos)
    loc = pos.locate_range(start, stop)
    pos = pos[loc]
    if step is None:
        step = len(pos) // 100
    ax.set_xlim(start, stop)

    # plot the lines
    if line_kwargs is None:
        line_kwargs = dict()
    # line_kwargs.setdefault('linewidth', .5)
    n_variants = len(pos)
    for i, p in enumerate(pos[::step]):
        xfrom = p
        xto = (
            start +
            ((i * step / n_variants) * (stop-start))
        )
        line = plt.Line2D([xfrom, xto], [0, 1], **line_kwargs)
        ax.add_line(line)

    # invert?
    if flip:
        ax.invert_yaxis()
        ax.xaxis.tick_top()
    else:
        ax.xaxis.tick_bottom()

    # tidy up
    ax.set_yticks([])
    ax.xaxis.set_tick_params(direction='out')
    for spine in 'left', 'right':
        ax.spines[spine].set_visible(False)

    return ax


def tabulate_state_transitions(x, states, pos=None):
    """Construct a dataframe where each row provides information about a state transition.

    Parameters
    ----------
    x : array_like, int
        1-dimensional array of state values.
    states : set
        Set of states of interest. Any state value not in this set will be ignored.
    pos : array_like, int, optional
        Array of positions corresponding to values in `x`.

    Returns
    -------
    df : DataFrame

    Notes
    -----
    The resulting dataframe includes one row at the start representing the first state
    observation and one row at the end representing the last state observation.

    Examples
    --------
    >>> import allel
    >>> x = [1, 1, 0, 1, 1, 2, 2, 0, 2, 1, 1]
    >>> df = allel.tabulate_state_transitions(x, states={1, 2})
    >>> df
       lstate  rstate  lidx  ridx
    0      -1       1    -1     0
    1       1       2     4     5
    2       2       1     8     9
    3       1      -1    10    -1
    >>> pos = [2, 4, 7, 8, 10, 14, 19, 23, 28, 30, 31]
    >>> df = allel.tabulate_state_transitions(x, states={1, 2}, pos=pos)
    >>> df
       lstate  rstate  lidx  ridx  lpos  rpos
    0      -1       1    -1     0    -1     2
    1       1       2     4     5    10    14
    2       2       1     8     9    28    30
    3       1      -1    10    -1    31    -1

    """

    # check inputs
    x = asarray_ndim(x, 1)
    check_integer_dtype(x)
    x = memoryview_safe(x)

    # find state transitions
    switch_points, transitions, _ = state_transitions(x, states)

    # start to build a dataframe
    items = [('lstate', transitions[:, 0]),
             ('rstate', transitions[:, 1]),
             ('lidx', switch_points[:, 0]),
             ('ridx', switch_points[:, 1])]

    # deal with optional positions
    if pos is not None:
        pos = asarray_ndim(pos, 1)
        check_dim0_aligned(x, pos)
        check_integer_dtype(pos)

        # find switch positions
        switch_positions = np.take(pos, switch_points)
        # deal with boundary transitions
        switch_positions[0, 0] = -1
        switch_positions[-1, 1] = -1

        # add columns into dataframe
        items += [('lpos', switch_positions[:, 0]),
                  ('rpos', switch_positions[:, 1])]

    import pandas
    return pandas.DataFrame.from_dict(OrderedDict(items))


def tabulate_state_blocks(x, states, pos=None):
    """Construct a dataframe where each row provides information about continuous state blocks.

    Parameters
    ----------
    x : array_like, int
        1-dimensional array of state values.
    states : set
        Set of states of interest. Any state value not in this set will be ignored.
    pos : array_like, int, optional
        Array of positions corresponding to values in `x`.

    Returns
    -------
    df : DataFrame

    Examples
    --------
    >>> import allel
    >>> x = [1, 1, 0, 1, 1, 2, 2, 0, 2, 1, 1]
    >>> df = allel.tabulate_state_blocks(x, states={1, 2})
    >>> df
       state  support  start_lidx     ...       size_min  size_max  is_marginal
    0      1        4          -1     ...              5        -1         True
    1      2        3           4     ...              4         4        False
    2      1        2           8     ...              2        -1         True
    [3 rows x 9 columns]
    >>> pos = [2, 4, 7, 8, 10, 14, 19, 23, 28, 30, 31]
    >>> df = allel.tabulate_state_blocks(x, states={1, 2}, pos=pos)
    >>> df
       state  support  start_lidx     ...      stop_rpos  length_min  length_max
    0      1        4          -1     ...             14           9          -1
    1      2        3           4     ...             30          15          19
    2      1        2           8     ...             -1           2          -1
    [3 rows x 15 columns]

    """

    # check inputs
    x = asarray_ndim(x, 1)
    check_integer_dtype(x)
    x = memoryview_safe(x)

    # find state transitions
    switch_points, transitions, observations = state_transitions(x, states)

    # setup some helpers
    t = transitions[1:, 0]
    o = observations[1:]
    s1 = switch_points[:-1]
    s2 = switch_points[1:]
    is_marginal = (s1[:, 0] < 0) | (s2[:, 1] < 0)
    size_min = s2[:, 0] - s1[:, 1] + 1
    size_max = s2[:, 1] - s1[:, 0] - 1
    size_max[is_marginal] = -1

    # start to build a dataframe
    items = [
        ('state', t),
        ('support', o),
        ('start_lidx', s1[:, 0]),
        ('start_ridx', s1[:, 1]),
        ('stop_lidx', s2[:, 0]),
        ('stop_ridx', s2[:, 1]),
        ('size_min', size_min),
        ('size_max', size_max),
        ('is_marginal', is_marginal)
    ]

    # deal with optional positions
    if pos is not None:
        pos = asarray_ndim(pos, 1)
        check_dim0_aligned(x, pos)
        check_integer_dtype(pos)

        # obtain switch positions
        switch_positions = np.take(pos, switch_points)
        # deal with boundary transitions
        switch_positions[0, 0] = -1
        switch_positions[-1, 1] = -1

        # setup helpers
        p1 = switch_positions[:-1]
        p2 = switch_positions[1:]
        length_min = p2[:, 0] - p1[:, 1] + 1
        length_max = p2[:, 1] - p1[:, 0] - 1
        length_max[is_marginal] = -1

        items += [
            ('start_lpos', p1[:, 0]),
            ('start_rpos', p1[:, 1]),
            ('stop_lpos', p2[:, 0]),
            ('stop_rpos', p2[:, 1]),
            ('length_min', length_min),
            ('length_max', length_max),
        ]

    import pandas
    return pandas.DataFrame.from_dict(OrderedDict(items))
