"""
Functions for calculating bet size based on model probabilities and
other strategy parameters.

Includes implementations from de Prado (2018).

## TODO:
- [ ] **HPC Dependency:** The `mpPandasObj` placeholder in
      `strategy_bet_sizing` and `avgActiveSignals` should be
      hardened. If `RiskLabAI.hpc` is a core dependency,
      consider raising an `ImportError` instead of using a
      placeholder that returns an empty DataFrame, which could
      fail silently.
"""

from typing import Optional, Any
import numpy as np
import pandas as pd
from numba import jit
from scipy.stats import norm

# Assuming RiskLabAI.hpc provides mpPandasObj
# Since it's not provided, we'll create a placeholder for type hinting
try:
    from RiskLabAI.hpc import mpPandasObj
except ImportError:
    # Placeholder for type hinting if hpc module is not available
    def mpPandasObj(*args: Any, **kwargs: Any) -> pd.DataFrame:
        print("Warning: RiskLabAI.hpc.mpPandasObj not found. Using placeholder.")
        return pd.DataFrame()


def probability_bet_size(
    probabilities: np.ndarray, sides: np.ndarray
) -> np.ndarray:
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
    bet_sizes_arr = probability_bet_size(
        _probabilities.to_numpy(), _sides.to_numpy()
    )

    # 2. Calculate concurrent average
    avg_bet_sizes_arr = average_bet_sizes(
        price_timestamps.to_numpy(),
        _times.index.to_numpy(),
        _times.values,
        bet_sizes_arr,
    )

    return pd.Series(avg_bet_sizes_arr, index=price_timestamps)


# --- The following functions appear to be from de Prado (2018) ---
# --- Naming convention (camelCase) is preserved for reference. ---

def avgActiveSignals(
    signals: pd.DataFrame, nThreads: int
) -> pd.DataFrame:
    """
    Calculate the average signal among active signals using parallel processing.
    
    Reference: De Prado, M. (2018) Advances in financial machine learning.
    Methodology: SNIPPET 10.2

    Parameters
    ----------
    signals : pd.DataFrame
        DataFrame with signal start times as index, and columns 't1' (end time)
        and 'signal' (signal value).
    nThreads : int
        Number of threads to use for parallel execution via `mpPandasObj`.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the average active signal at each time point.
    """
    # 1) time points where signals change (either one starts or one ends)
    timePoints = set(signals["t1"].dropna().values)
    timePoints = timePoints.union(signals.index.values)
    timePoints = list(timePoints)
    timePoints.sort()
    
    # 2) call parallel function
    out = mpPandasObj(
        mpAvgActiveSignals,
        ("molecule", timePoints),
        nThreads,
        signals=signals,
    )
    return out


def mpAvgActiveSignals(
    signals: pd.DataFrame, molecule: list
) -> pd.Series:
    """
    Worker function for `avgActiveSignals`.
    
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
    out = pd.Series()
    for loc in molecule:
        # Keep signal that contain loc
        signal_ = (signals.index.values <= loc) & (
            (loc < signals["t1"]) | pd.isnull(signals["t1"])
        )
        act = signals[signal_].index
        if len(act) > 0:
            out[loc] = signals.loc[act, "signal"].mean()
        else:
            out[loc] = 0  # no signals active at this time
    return out


def discreteSignal(signal: pd.Series, stepSize: float) -> pd.Series:
    """
    Discretize a signal to a specific step size, capping at +/- 1.

    Reference: De Prado, M. (2018) Advances in financial machine learning.
    Methodology: SNIPPET 10.3

    Parameters
    ----------
    signal : pd.Series
        The continuous signal values (e.g., from -1 to 1).
    stepSize : float
        The step size for discretization (e.g., 0.1).

    Returns
    -------
    pd.Series
        The discretized signal.
    """
    discretized = (signal / stepSize).round() * stepSize
    discretized[discretized > 1] = 1.0
    discretized[discretized < -1] = -1.0
    return discretized


def Signal(
    events: pd.DataFrame,
    stepSize: float,
    probability: pd.Series,
    prediction: pd.Series,
    nClasses: int,
    nThreads: int,
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
    stepSize : float
        The step size for final signal discretization.
    probability : pd.Series
        Probability of class 1 (e.g., from `predict_proba`).
    prediction : pd.Series
        The predicted class (e.g., 1 or -1).
    nClasses : int
        Number of classes in the prediction.
    nThreads : int
        Number of threads for `avgActiveSignals`.

    Returns
    -------
    pd.Series
        The final, averaged, and discretized signal.
    """
    if probability.shape[0] == 0:
        return pd.Series()

    # 1) generate signals from multinomial classification (one-vs-rest, OvR)
    # t-value of OvR
    t_value = (probability - 1.0 / nClasses) / (
        (probability * (1.0 - probability)) ** 0.5
    )
    signal = prediction * (2 * norm.cdf(t_value) - 1)  # signal = side * size

    if "side" in events:
        signal *= events.loc[signal.index, "side"]  # meta-labeling

    # 2) compute average signal among those concurrently open
    signal_df = signal.to_frame("signal").join(
        events[["t1"]], how="left"
    )
    avg_signal = avgActiveSignals(signal_df, nThreads)
    
    # 3) discretize signal
    discretized_signal = discreteSignal(signal=avg_signal, stepSize=stepSize)
    return discretized_signal


def betSize(w: float, x: float) -> float:
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


def TPos(
    w: float, f: float, acctualPrice: float, maximumPositionSize: int
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
    acctualPrice : float
        Actual (current) market price.
    maximumPositionSize : int
        Maximum absolute position size.

    Returns
    -------
    int
        The target position size (integer).
    """
    return int(
        betSize(w, f - acctualPrice) * maximumPositionSize
    )


def inversePrice(f: float, w: float, m: float) -> float:
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
        return f # Avoid division by zero
    return f - m * np.sqrt(w / (1 - m**2))


def limitPrice(
    targetPositionSize: int,
    cPosition: int,
    f: float,
    w: float,
    maximumPositionSize: int,
) -> float:
    """
    Calculate the limit price for adjusting position.

    Reference: De Prado, M. (2018) Advances in financial machine learning.
    Methodology: p.145 SNIPPET 10.4

    Parameters
    ----------
    targetPositionSize : int
        The target position size.
    cPosition : int
        The current position size.
    f : float
        Forecasted price.
    w : float
        Coefficient regulating the sigmoid width.
    maximumPositionSize : int
        Maximum absolute position size.

    Returns
    -------
    float
        The average limit price.
    """
    if targetPositionSize == cPosition:
        return f # No change
        
    sgn = np.sign(targetPositionSize - cPosition)
    lP = 0.0
    
    # Average price from current to target position
    for i in range(
        abs(cPosition + sgn), abs(targetPositionSize + sgn)
    ):
        lP += inversePrice(f, w, i / float(maximumPositionSize))
        
    lP /= abs(targetPositionSize - cPosition)
    return lP


def getW(x: float, m: float) -> float:
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
        return np.inf # w is undefined
    return x**2 * ( (1 / m**2) - 1 )