"""
Functions for calculating bet size based on model probabilities and
other strategy parameters.

Includes implementations from de Prado (2018).
"""

import numpy as np
import pandas as pd
from numba import jit
from scipy.stats import norm

from RiskLabAI._deprecation import deprecated_alias
from RiskLabAI.hpc import mp_pandas_obj


def probability_bet_size(probabilities: np.ndarray, sides: np.ndarray) -> np.ndarray:
    r"""
    Calculate the bet size based on probabilities and side.

    The bet size is derived from the probability of a correct bet,
    transformed from [0, 1] to [-1, 1].

    .. math::
       \text{bet_size} = \text{side} \times (2 \times \Phi(p) - 1)

    Where \(\Phi\) is the CDF of a standard normal distribution and
    \(p\) is the probability from the meta-model.

    Parameters
    ----------
    probabilities : np.ndarray
        Array of probabilities (e.g., from a meta-model).
    sides : np.ndarray
        Array indicating the side of the bet (e.g., 1 for long, -1 for short).

    Returns
    -------
    np.ndarray
        Array of bet sizes, ranging from -1 to 1.
    """
    return sides * (2 * norm.cdf(probabilities) - 1)


@jit(nopython=True)
def average_bet_sizes(
    price_dates: np.ndarray,
    start_dates: np.ndarray,
    end_dates: np.ndarray,
    bet_sizes: np.ndarray,
) -> np.ndarray:
    """
    Compute average bet sizes for each date (Numba-optimized).

    This function calculates the concurrent average bet size for each
    timestamp in `price_dates`. A bet is active at `price_dates[i]` if
    `start_dates[j] <= price_dates[i] <= end_dates[j]`.

    Parameters
    ----------
    price_dates : np.ndarray
        Array of timestamps for which to calculate the average bet size.
        Must be numerical (e.g., int64 from .to_numpy()).
    start_dates : np.ndarray
        Array of bet start timestamps (numerical).
    end_dates : np.ndarray
        Array of bet end timestamps (numerical).
    bet_sizes : np.ndarray
        Array of bet sizes corresponding to each (start, end) pair.

    Returns
    -------
    np.ndarray
        Array of average bet sizes for each date in `price_dates`.
    """
    num_dates = len(price_dates)
    avg_bet_sizes = np.zeros(num_dates)

    for i in range(num_dates):
        total = 0.0
        count = 0
        for j in range(len(start_dates)):
            if start_dates[j] <= price_dates[i] <= end_dates[j]:
                total += bet_sizes[j]
                count += 1
        if count > 0:
            avg_bet_sizes[i] = total / count

    return avg_bet_sizes


def strategy_bet_sizing(
    price_timestamps: pd.Index,
    times: pd.Series,
    sides: pd.Series,
    probabilities: pd.Series,
) -> pd.Series:
    """
    Calculate the average bet size for a trading strategy.

    This function orchestrates the calculation of bet sizes from
    probabilities and then computes the concurrent average bet size
    over the strategy's history.

    Parameters
    ----------
    price_timestamps : pd.Index
        The Index of the master price series, defining all timestamps
        for the output.
    times : pd.Series
        Series with bet start times as indices and end times as values.
    sides : pd.Series
        Series indicating the side of the position (1 or -1).
    probabilities : pd.Series
        Series of probabilities (from meta-model) for each bet.

    Returns
    -------
    pd.Series
        Series of average bet sizes, indexed by `price_timestamps`.
    """
    # Align inputs
    common_index = times.index.intersection(sides.index).intersection(
        probabilities.index
    )
    if len(common_index) == 0:
        return pd.Series(0.0, index=price_timestamps)

    _times = times.loc[common_index]
    _sides = sides.loc[common_index]
    _probabilities = probabilities.loc[common_index]

    # 1. Calculate individual bet sizes
    bet_sizes_arr = probability_bet_size(_probabilities.to_numpy(), _sides.to_numpy())

    # 2. Calculate concurrent average
    avg_bet_sizes_arr = average_bet_sizes(
        price_timestamps.to_numpy(),
        _times.index.to_numpy(),
        _times.values,
        bet_sizes_arr,
    )

    return pd.Series(avg_bet_sizes_arr, index=price_timestamps)


# --- Bet-sizing functions from de Prado (2018), AFML Chapter 10. ---


def avg_active_signals(signals: pd.DataFrame, n_threads: int) -> pd.DataFrame:
    """
    Calculate the average signal among active signals using parallel processing.

    Reference: De Prado, M. (2018) Advances in financial machine learning.
    Methodology: SNIPPET 10.2

    Parameters
    ----------
    signals : pd.DataFrame
        DataFrame with signal start times as index, and columns 't1' (end time)
        and 'signal' (signal value).
    n_threads : int
        Number of threads to use for parallel execution via `mp_pandas_obj`.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the average active signal at each time point.
    """
    # 1) time points where signals change (either one starts or one ends)
    time_points = set(signals["t1"].dropna().values)
    time_points = time_points.union(signals.index.values)
    time_points = list(time_points)
    time_points.sort()

    # 2) call parallel function
    out = mp_pandas_obj(
        mp_avg_active_signals,
        ("molecule", time_points),
        n_threads,
        signals=signals,
    )
    return out


def mp_avg_active_signals(signals: pd.DataFrame, molecule: list) -> pd.Series:
    """
    Worker function for `avg_active_signals`.

    At time `loc`, average signal among those still active.
    Signal is active if:
    a) issued before or at `loc` AND
    b) `loc` before signal's endtime, or endtime is still unknown (NaT).

    Reference: De Prado, M. (2018) Advances in financial machine learning.
    Methodology: SNIPPET 10.2

    Parameters
    ----------
    signals : pd.DataFrame
        DataFrame of signals (see `avgActiveSignals`).
    molecule : list
        The list of timestamps (timePoints) assigned to this worker.

    Returns
    -------
    pd.Series
        Series indexed by `loc` (timestamp) with the average signal value.
    """
    # A signal i is active at `loc` iff start_i <= loc < t1_i. The average of
    # active signals therefore equals (sum of `signal` over signals started by
    # `loc`) minus (sum over signals ended by `loc`), divided by the analogous
    # active count. Prefix sums + searchsorted compute this for every `loc` in
    # O((n + m) log n) instead of the O(n * m) double scan, with identical
    # values (verified in test/test_performance.py).
    molecule_list = list(molecule)
    if len(signals) == 0 or len(molecule_list) == 0:
        return pd.Series(0.0, index=molecule_list)

    molecule_array = np.asarray(molecule_list)
    signal_values = signals["signal"].to_numpy(dtype=float)

    starts = signals.index.values
    start_order = np.argsort(starts, kind="mergesort")
    sorted_starts = starts[start_order]
    cum_signal_start = np.concatenate(([0.0], np.cumsum(signal_values[start_order])))
    cum_count_start = np.arange(len(sorted_starts) + 1)

    finite = ~pd.isnull(signals["t1"]).to_numpy()
    end_times = signals["t1"].to_numpy()[finite]
    end_order = np.argsort(end_times, kind="mergesort")
    sorted_ends = end_times[end_order]
    cum_signal_end = np.concatenate(
        ([0.0], np.cumsum(signal_values[finite][end_order]))
    )
    cum_count_end = np.arange(len(sorted_ends) + 1)

    started = np.searchsorted(sorted_starts, molecule_array, side="right")
    ended = np.searchsorted(sorted_ends, molecule_array, side="right")
    active_signal = cum_signal_start[started] - cum_signal_end[ended]
    active_count = cum_count_start[started] - cum_count_end[ended]

    with np.errstate(invalid="ignore", divide="ignore"):
        averages = np.where(active_count > 0, active_signal / active_count, 0.0)

    return pd.Series(averages, index=molecule_list)


def discrete_signal(signal: pd.Series, step_size: float) -> pd.Series:
    """
    Discretize a signal to a specific step size, capping at +/- 1.

    Reference: De Prado, M. (2018) Advances in financial machine learning.
    Methodology: SNIPPET 10.3

    Parameters
    ----------
    signal : pd.Series
        The continuous signal values (e.g., from -1 to 1).
    step_size : float
        The step size for discretization (e.g., 0.1).

    Returns
    -------
    pd.Series
        The discretized signal.
    """
    discretized = (signal / step_size).round() * step_size
    discretized[discretized > 1] = 1.0
    discretized[discretized < -1] = -1.0
    return discretized


def generate_signal(
    events: pd.DataFrame,
    step_size: float,
    probability: pd.Series,
    prediction: pd.Series,
    n_classes: int,
    n_threads: int,
) -> pd.Series:
    """
    Generate a discretized, averaged signal from model predictions.

    Reference: De Prado, M. (2018) Advances in financial machine learning.
    Methodology: SNIPPET 10.1

    Parameters
    ----------
    events : pd.DataFrame
        DataFrame of events, must include 't1' (end times) and
        optionally 'side' (for meta-labeling).
    step_size : float
        The step size for final signal discretization.
    probability : pd.Series
        Probability of class 1 (e.g., from `predict_proba`).
    prediction : pd.Series
        The predicted class (e.g., 1 or -1).
    n_classes : int
        Number of classes in the prediction.
    n_threads : int
        Number of threads for `avg_active_signals`.

    Returns
    -------
    pd.Series
        The final, averaged, and discretized signal.
    """
    if probability.shape[0] == 0:
        return pd.Series()

    # 1) generate signals from multinomial classification (one-vs-rest, OvR)
    # t-value of OvR
    t_value = (probability - 1.0 / n_classes) / (
        (probability * (1.0 - probability)) ** 0.5
    )
    signal = prediction * (2 * norm.cdf(t_value) - 1)  # signal = side * size

    if "side" in events:
        signal *= events.loc[signal.index, "side"]  # meta-labeling

    # 2) compute average signal among those concurrently open
    signal_df = signal.to_frame("signal").join(events[["t1"]], how="left")
    avg_signal = avg_active_signals(signal_df, n_threads)

    # 3) discretize signal
    discretized_signal = discrete_signal(signal=avg_signal, step_size=step_size)
    return discretized_signal


def bet_size_sigmoid(w: float, x: float) -> float:
    """
    Calculate bet size using a sigmoid function.

    Reference: De Prado, M. (2018) Advances in financial machine learning.
    Methodology: p.145 SNIPPET 10.4

    Parameters
    ----------
    w : float
        Coefficient regulating the sigmoid width.
    x : float
        The divergence (e.g., forecast_price - market_price).

    Returns
    -------
    float
        The bet size, bounded between -1 and 1.
    """
    return x / np.sqrt(w + x**2)


def target_position(
    w: float, f: float, actual_price: float, maximum_position_size: int
) -> int:
    """
    Calculate the target position size.

    Reference: De Prado, M. (2018) Advances in financial machine learning.
    Methodology: p.145 SNIPPET 10.4

    Parameters
    ----------
    w : float
        Coefficient regulating the sigmoid width.
    f : float
        Forecasted price.
    actual_price : float
        Actual (current) market price.
    maximum_position_size : int
        Maximum absolute position size.

    Returns
    -------
    int
        The target position size (integer).
    """
    return int(bet_size_sigmoid(w, f - actual_price) * maximum_position_size)


def inverse_price(f: float, w: float, m: float) -> float:
    """
    Calculates the inverse price given a bet size.

    Reference: De Prado, M. (2018) Advances in financial machine learning.
    Methodology: p.145 SNIPPET 10.4

    Parameters
    ----------
    f : float
        Forecasted price.
    w : float
        Coefficient regulating the sigmoid width.
    m : float
        Bet size (as a fraction of max size, -1 to 1).

    Returns
    -------
    float
        The implied price for bet size `m`.
    """
    if m == 1.0 or m == -1.0:
        return f  # Avoid division by zero
    return f - m * np.sqrt(w / (1 - m**2))


def limit_price(
    target_position_size: int,
    current_position: int,
    f: float,
    w: float,
    maximum_position_size: int,
) -> float:
    """
    Calculate the limit price for adjusting position.

    Reference: De Prado, M. (2018) Advances in financial machine learning.
    Methodology: p.145 SNIPPET 10.4

    Parameters
    ----------
    target_position_size : int
        The target position size.
    current_position : int
        The current position size.
    f : float
        Forecasted price.
    w : float
        Coefficient regulating the sigmoid width.
    maximum_position_size : int
        Maximum absolute position size.

    Returns
    -------
    float
        The average limit price.
    """
    if target_position_size == current_position:
        return f  # No change

    sgn = np.sign(target_position_size - current_position)
    limit = 0.0

    # Average price from current to target position
    for i in range(abs(current_position + sgn), abs(target_position_size + sgn)):
        limit += inverse_price(f, w, i / float(maximum_position_size))

    limit /= abs(target_position_size - current_position)
    return limit


def compute_sigmoid_width(x: float, m: float) -> float:
    """
    Get the 'w' coefficient implied by a divergence and bet size.

    Reference: De Prado, M. (2018) Advances in financial machine learning.
    Methodology: p.145 SNIPPET 10.4

    Parameters
    ----------
    x : float
        Divergence (forecast - market).
    m : float
        Bet size (fraction, -1 to 1, not 0, 1, or -1).

    Returns
    -------
    float
        The implied 'w' coefficient.
    """
    if m == 0.0 or m == 1.0 or m == -1.0:
        return np.inf  # w is undefined
    return x**2 * ((1 / m**2) - 1)


# --------------------------------------------------------------------------- #
# Deprecated camelCase aliases (the historical AFML-style names). Each keeps
# working and emits a DeprecationWarning; scheduled for removal in 2.1.0.
# See NAMING_CANON_2.0.0.md.
# --------------------------------------------------------------------------- #
avgActiveSignals = deprecated_alias(
    avg_active_signals, "avgActiveSignals", removed_in="2.1.0"
)
mpAvgActiveSignals = deprecated_alias(
    mp_avg_active_signals, "mpAvgActiveSignals", removed_in="2.1.0"
)
discreteSignal = deprecated_alias(discrete_signal, "discreteSignal", removed_in="2.1.0")
Signal = deprecated_alias(generate_signal, "Signal", removed_in="2.1.0")
betSize = deprecated_alias(bet_size_sigmoid, "betSize", removed_in="2.1.0")
TPos = deprecated_alias(target_position, "TPos", removed_in="2.1.0")
inversePrice = deprecated_alias(inverse_price, "inversePrice", removed_in="2.1.0")
limitPrice = deprecated_alias(limit_price, "limitPrice", removed_in="2.1.0")
getW = deprecated_alias(compute_sigmoid_width, "getW", removed_in="2.1.0")
