"""
Implements sample weighting techniques for financial machine learning,
focusing on uniqueness (concurrency) and time decay.

Reference:
    De Prado, M. (2018) Advances in financial machine learning.
    John Wiley & Sons, Chapter 4.
"""

import numpy as np
import pandas as pd


def expand_label_for_meta_labeling(
    close_index: pd.Index,
    timestamp: pd.Series,
    molecule: pd.Index,
) -> pd.Series:
    """
    Compute the number of concurrent events for each timestamp.

    This function expands the event start/end times (`timestamp`)
    to a full series indicating how many events are active
    at each point in time.

    Reference:
        Based on Snippet 4.1, Page 60 (label concurrency).

    Parameters
    ----------
    close_index : pd.Index
        The master index of all timestamps (e.g., from price bars).
    timestamp : pd.Series
        Series where index is event start time, value is event end time.
    molecule : pd.Index
        The subset of event start times to process. All overlapping events
        in `timestamp` contribute to concurrency, including events outside
        this subset.

    Returns
    -------
    pd.Series
        A Series on `close_index` from the first start in `molecule` through
        its latest event end, containing the count of all active events.
    """
    # Filter events that are relevant to this molecule
    ts = timestamp.fillna(close_index[-1])
    molecule_end = ts.loc[molecule].max()
    ts = ts[(ts >= molecule[0]) & (ts.index <= molecule_end)]

    if ts.empty:
        # Return an empty series; align in the caller will handle it
        return pd.Series(dtype=float)

    # Find min/max index locations
    iloc_min = close_index.searchsorted(ts.index[0])
    iloc_max = close_index.searchsorted(molecule_end)

    # Create a count series over the relevant time span
    count = pd.Series(0, index=close_index[iloc_min : iloc_max + 1])

    for t_in, t_out in ts.items():
        # Ensure t_out is within the count index
        # (it might be larger than iloc_max if fillna was used)
        t_out = min(t_out, count.index[-1])
        count.loc[t_in:t_out] += 1

    return count.loc[molecule[0] : molecule_end]


def calculate_average_uniqueness(
    index_matrix: pd.DataFrame,
) -> pd.Series:
    """
    Calculate the average uniqueness of each event.

    Uniqueness is calculated as 1/c_t, averaged over the
    duration of the event.

    Reference:
        Snippet 4.2, Page 62.

    Parameters
    ----------
    index_matrix : pd.DataFrame
        An indicator matrix (T x N) where T is timestamps
        and N is the number of events.

    Returns
    -------
    pd.Series
        A Series of average uniqueness for each event (column).
    """
    # c_t: Concurrency at each timestamp
    concurrency = index_matrix.sum(axis=1)

    # 1/c_t: Uniqueness at each timestamp
    # This is a (T x N) DataFrame, 0 where event is not active
    uniqueness = index_matrix.div(concurrency, axis=0).fillna(0)

    # Sum of 1/c_t for each event
    total_uniqueness = uniqueness.sum(axis=0)

    # Number of active periods for each event
    event_duration = (index_matrix > 0).sum(axis=0)

    # Average uniqueness: sum(1/c_t) / sum(I)
    average_uniqueness = total_uniqueness / event_duration

    # Handle events that never occurred (duration 0)
    average_uniqueness = average_uniqueness.fillna(0)

    return average_uniqueness


def sequential_bootstrap(
    indicator_matrix: pd.DataFrame | np.ndarray,
    sample_length: int | None = None,
    random_state: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Draw event positions with replacement using average added uniqueness.

    Parameters
    ----------
    indicator_matrix : pd.DataFrame or np.ndarray
        Binary matrix with observation intervals in rows and events in columns.
        Each event must contain at least one active interval. Row ordering and
        DataFrame labels do not affect the sampling probabilities.
    sample_length : int or None, default=None
        Number of draws. Defaults to the number of event columns. Must be a
        nonnegative integer. A matrix with no columns permits only zero draws.
    random_state : int, np.random.Generator or None, default=None
        Nonnegative seed or generator. A supplied generator advances in place.
        None creates a new generator. The legacy global random state is unused.

    Returns
    -------
    np.ndarray
        Integer column positions in draw order, including repeated selections.
        For DataFrame column labels, use ``indicator_matrix.columns.take`` on
        the returned positions. A zero-length draw consumes no random numbers.

    Raises
    ------
    ValueError
        If the matrix is not two-dimensional, real numeric or boolean, finite,
        binary, or has an event with no active intervals; if the requested
        length is invalid; or if random_state is unsupported.

    Notes
    -----
    For candidate j, average 1 / (c_t + 1) over its active intervals, where c_t
    counts previous selections, including repetitions. Normalize these scores
    to obtain the next draw probabilities. The first draw is uniform. This is
    the Chapter 4 sampling method described in the module reference, not a
    guarantee of improved model performance for every sample.

    Examples
    --------
    >>> sequential_bootstrap(np.ones((2, 1)), sample_length=3, random_state=0)
    array([0, 0, 0])
    """
    values = np.asarray(indicator_matrix)
    if values.ndim != 2 or values.dtype.kind not in "biuf":
        raise ValueError("indicator_matrix must be a two-dimensional binary matrix")
    if not np.all(np.isfinite(values)) or not np.all((values == 0) | (values == 1)):
        raise ValueError("indicator_matrix must contain only finite zeros and ones")
    matrix = values.astype(float, copy=False)
    event_count = matrix.shape[1]
    durations = matrix.sum(axis=0)
    if np.any(durations == 0):
        raise ValueError("each event must contain at least one active interval")
    if sample_length is None:
        sample_length = event_count
    if (
        isinstance(sample_length, (bool, np.bool_))
        or not isinstance(sample_length, (int, np.integer))
        or sample_length < 0
    ):
        raise ValueError("sample_length must be a nonnegative integer")
    if event_count == 0 and sample_length > 0:
        raise ValueError("cannot draw from an indicator matrix with no events")
    if random_state is not None and not isinstance(random_state, np.random.Generator):
        if (
            isinstance(random_state, (bool, np.bool_))
            or not isinstance(random_state, (int, np.integer))
            or random_state < 0
        ):
            raise ValueError("random_state must be a nonnegative seed or Generator")
    generator = np.random.default_rng(random_state)
    selected = np.empty(sample_length, dtype=np.intp)
    concurrency = np.zeros(matrix.shape[0], dtype=float)
    for draw in range(sample_length):
        scores = (matrix.T @ (1.0 / (concurrency + 1.0))) / durations
        selected[draw] = generator.choice(event_count, p=scores / scores.sum())
        concurrency += matrix[:, selected[draw]]
    return selected


def sample_weight_absolute_return_meta_labeling(
    timestamp: pd.Series, price: pd.Series, molecule: pd.Index
) -> pd.Series:
    """
    Calculate sample weights based on absolute log-return attribution.

    The weight of a sample is proportional to the sum of the absolute
    log-returns that occurred during the event, divided by the
    concurrency of those returns.

    w_i = sum_{t=t_i,0}^{t_i,1} [ |r_t| / c_t ]

    Weights are then normalized to sum to the number of samples.

    Reference:
        Based on Snippet 4.10, Page 70.

    Parameters
    ----------
    timestamp : pd.Series
        Series where index is event start time, value is event end time.
    price : pd.Series
        Series of prices.
    molecule : pd.Index
        The subset of event start times to process.

    Returns
    -------
    pd.Series
        A Series of sample weights, normalized to sum to N.
    """
    # 1. Compute concurrency
    concurrency_events = expand_label_for_meta_labeling(
        price.index, timestamp, molecule
    )

    # 2. Compute absolute log returns
    log_return = np.log(price).diff().abs()

    # Align returns and concurrency
    # Use 'left' join to keep the log_return (price) index
    log_return, concurrency_events = log_return.align(
        concurrency_events, join="left", fill_value=0
    )

    weight = pd.Series(index=molecule, dtype=float)

    # 3. Calculate weighted returns
    for t_in, t_out in timestamp.loc[weight.index].items():
        if t_out not in log_return.index:
            # Find the closest preceding index
            t_out = log_return.index[log_return.index.searchsorted(t_out) - 1]

        # r_t / c_t
        # Filter concurrency > 0 to avoid division by zero
        relevant_concurrency = concurrency_events.loc[t_in:t_out]
        relevant_log_return = log_return.loc[t_in:t_out]

        active_periods = relevant_concurrency > 0
        if active_periods.any():
            weighted_return = (
                relevant_log_return[active_periods]
                / relevant_concurrency[active_periods]
            )
            weight.loc[t_in] = weighted_return.sum()
        else:
            weight.loc[t_in] = 0.0

    weight = weight.abs()

    # 4. Normalize
    if weight.sum() == 0:
        # Avoid division by zero if all weights are 0
        return pd.Series(1.0, index=molecule)

    weight *= len(weight) / weight.sum()
    return weight


def calculate_time_decay(weight: pd.Series, clf_last_weight: float = 1.0) -> pd.Series:
    """
    Apply a time-decay factor to sample weights.

    This function applies a linearly decaying weight, where the
    most recent observation has weight 1.0 and the oldest
    has weight `clf_last_weight`.

    Reference:
        Snippet 4.11, Page 71.

    Parameters
    ----------
    weight : pd.Series
        The original sample weights (e.g., from uniqueness).
    clf_last_weight : float, default=1.0
        The weight to assign to the oldest observation.
        If 1.0, all weights are 1 (no decay).
        If 0.0, weights decay linearly to 0.

    Returns
    -------
    pd.Series
        The new weights with time decay applied.
    """
    clf_weight = weight.sort_index().cumsum()

    if clf_last_weight < 0 or clf_last_weight > 1:
        raise ValueError("clf_last_weight must be between 0 and 1")

    # Calculate slope
    if clf_last_weight == 1.0 or clf_weight.empty or clf_weight.iloc[-1] == 0:
        slope = 0.0
        const = 1.0
    else:
        slope = (1.0 - clf_last_weight) / clf_weight.iloc[-1]
        const = 1.0 - slope * clf_weight.iloc[-1]

    clf_weight = const + slope * clf_weight
    clf_weight[clf_weight < 0] = 0.0  # Should not happen if clf_last_weight >= 0

    return clf_weight
