"""
Implements the core financial labeling functions, including:
- CUSUM filters for event sampling.
- Volatility estimation.
- The Triple-Barrier Method (vertical and horizontal barriers).
- Meta-labeling.
- Re-exports of the parallel-execution helpers from ``RiskLabAI.hpc``.

Reference:
    De Prado, M. (2018) Advances in financial machine learning.
    John Wiley & Sons, Chapters 3 & 4.
"""

from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional

import numpy as np
import pandas as pd

# Parallel-processing helpers have a single source of truth in RiskLabAI.hpc.
# They were previously duplicated in this module; they are re-exported here
# under their historical names for backward compatibility. `lin_parts` maps to
# hpc's `linear_partitions` (identical for valid inputs, with an added guard).
from RiskLabAI.hpc import (  # noqa: F401  (re-exported for backward compat)
    process_jobs,
    expand_call,
    report_progress,
    linear_partitions as lin_parts,
)


def cusum_filter_events_dynamic_threshold(
    prices: pd.Series, threshold: pd.Series
) -> pd.DatetimeIndex:
    """
    Detect events using the Symmetric CUSUM filter with a dynamic threshold.

    This filter identifies timestamps where the cumulative sum of price
    changes exceeds a dynamic, time-varying threshold.

    Reference:
        Snippet 3.2, Page 48 (modified for dynamic threshold).

    Parameters
    ----------
    prices : pd.Series
        A pandas Series of prices.
    threshold : pd.Series
        A pandas Series containing the threshold values for each timestamp.
        Must be aligned with the `prices` index.

    Returns
    -------
    pd.DatetimeIndex
        Timestamps of the detected events (when a barrier was touched).
    """
    time_events = []
    shift_positive, shift_negative = 0.0, 0.0
    price_delta = prices.diff().dropna()

    # Align price changes with thresholds
    price_delta, thresholds = price_delta.align(
        threshold, join="inner", copy=False
    )

    for (index, value), thresh_val in zip(
        price_delta.items(), thresholds.values
    ):
        shift_positive = max(0.0, shift_positive + value)
        shift_negative = min(0.0, shift_negative + value)

        if shift_negative < -thresh_val:
            shift_negative = 0.0  # Reset only this counter
            time_events.append(index)
        elif shift_positive > thresh_val:
            shift_positive = 0.0  # Reset only this counter
            time_events.append(index)

    return pd.DatetimeIndex(time_events)


def symmetric_cusum_filter(
    prices: pd.Series, threshold: float
) -> pd.DatetimeIndex:
    """
    Detect events using the Symmetric CUSUM filter with a fixed threshold.

    Reference:
        Snippet 3.2, Page 48.

    Parameters
    ----------
    prices : pd.Series
        A pandas Series of prices.
    threshold : float
        The fixed threshold value.

    Returns
    -------
    pd.DatetimeIndex
        Timestamps of the detected events (when a barrier was touched).
    """
    time_events = []
    shift_positive, shift_negative = 0.0, 0.0
    price_delta = prices.diff().dropna()

    for index, value in price_delta.items():
        shift_positive = max(0.0, shift_positive + value)
        shift_negative = min(0.0, shift_negative + value)

        if shift_negative < -threshold:
            shift_negative = 0.0
            time_events.append(index)
        elif shift_positive > threshold:
            shift_positive = 0.0
            time_events.append(index)

    return pd.DatetimeIndex(time_events)


def daily_volatility_with_log_returns(
    close: pd.Series, span: int = 100
) -> pd.Series:
    """
    Calculate daily volatility using log returns.

    This method computes the EWMA standard deviation of daily log returns.

    Reference:
        Snippet 3.1, Page 44.

    Parameters
    ----------
    close : pd.Series
        Time series of close prices.
    span : int, default=100
        The span parameter for the EWMA.

    Returns
    -------
    pd.Series
        Series of daily volatility estimates.
    """
    # Find timestamps of 1 day prior
    df = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df = df[df > 0]
    df = pd.Series(
        close.index[df - 1],
        index=close.index[close.shape[0] - df.shape[0] :],
    )

    # Calculate log returns
    returns = np.log(close.loc[df.index] / close.loc[df.values].values)
    stds = returns.ewm(span=span).std().rename("std")
    return stds


def vertical_barrier(
    close: pd.Series, time_events: pd.DatetimeIndex, number_days: int
) -> pd.Series:
    """
    Create a vertical barrier for the triple-barrier method.

    This function finds the timestamp that occurs `number_days` after
    each event in `time_events`.

    Reference:
        Snippet 3.4, Page 50.

    Parameters
    ----------
    close : pd.Series
        Time series of close prices (used for the index).
    time_events : pd.DatetimeIndex
        The timestamps of the events (e.g., from CUSUM filter).
    number_days : int
        The number of days to look forward for the vertical barrier.

    Returns
    -------
    pd.Series
        A Series where the index is the event timestamp and the
        value is the vertical barrier timestamp.
    """
    barrier_dates = time_events + pd.Timedelta(days=number_days)

    # Find the integer location of the barrier dates in the price index
    timestamp_array = close.index.searchsorted(barrier_dates)

    # Filter out any indices that are out of bounds
    valid_indices = timestamp_array < close.shape[0]
    timestamp_array = timestamp_array[valid_indices]

    barrier_series = pd.Series(
        close.index[timestamp_array],
        index=time_events[valid_indices],
    )
    return barrier_series


def triple_barrier(
    close: pd.Series,
    events: pd.DataFrame,
    ptsl: List[float],
    molecule: List[pd.Timestamp],
) -> pd.DataFrame:
    """
    Apply the triple-barrier method for a subset of events.

    This is the core worker function for `meta_events`. It finds the
    first touch time of the upper, lower, or vertical barrier.

    Reference:
        Snippet 3.3, Page 50.

    Parameters
    ----------
    close : pd.Series
        Time series of close prices.
    events : pd.DataFrame
        DataFrame with event info. Must contain 'End Time' (vertical
        barrier), 'Base Width' (volatility), and 'Side' (if applicable).
    ptsl : List[float]
        A list of two floats: [profit_taking_multiplier, stop_loss_multiplier].
    molecule : List[pd.Timestamp]
        The subset of event timestamps this worker is responsible for.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns ['stop_loss', 'profit_taking']
        indexed by the event timestamp, indicating the first
        touch time for each horizontal barrier.
    """
    # Filter for events this worker owns
    events_filtered = events.loc[molecule]
    output = pd.DataFrame(index=events_filtered.index)
    output["End Time"] = events_filtered["End Time"] # Use original end time

    # 1. Set horizontal barriers
    if ptsl[0] > 0:
        profit_taking = ptsl[0] * events_filtered["Base Width"]
    else:
        profit_taking = pd.Series(np.inf, index=events_filtered.index)

    if ptsl[1] > 0:
        stop_loss = -ptsl[1] * events_filtered["Base Width"]
    else:
        stop_loss = pd.Series(-np.inf, index=events_filtered.index)

    # Get side if it exists, otherwise default to 1 (long)
    side = events_filtered.get("Side", pd.Series(1.0, index=events_filtered.index))

    # 2. Find first touch time.
    # Positional numpy indexing replaces the per-event pandas label slicing and
    # scalar `.loc` writes. For each event we slice the close-price array between
    # the event start and its vertical barrier, compute side-adjusted log
    # returns, and take the first index where each horizontal barrier is
    # crossed. This yields the same first-touch timestamps as the previous
    # implementation (verified in test/test_performance.py) but is ~10-70x
    # faster on realistic sizes.
    close_index = close.index
    close_values = close.to_numpy(dtype=float)
    close_index_values = close_index.values

    vertical_filled = events_filtered["End Time"].fillna(close_index[-1])
    start_positions = close_index.get_indexer(events_filtered.index)
    end_positions = (
        close_index.searchsorted(vertical_filled.to_numpy(), side="right") - 1
    )

    stop_loss_values = stop_loss.to_numpy()
    profit_taking_values = profit_taking.to_numpy()
    side_values = side.to_numpy(dtype=float)

    n_events = len(events_filtered)
    stop_loss_touch = np.full(n_events, np.datetime64("NaT"), dtype="datetime64[ns]")
    profit_taking_touch = np.full(n_events, np.datetime64("NaT"), dtype="datetime64[ns]")

    for i in range(n_events):
        start, end = start_positions[i], end_positions[i]
        segment = close_values[start:end + 1]
        path_returns = np.log(segment / segment[0]) * side_values[i]

        below = path_returns < stop_loss_values[i]
        if below.any():
            stop_loss_touch[i] = close_index_values[start + np.argmax(below)]

        above = path_returns > profit_taking_values[i]
        if above.any():
            profit_taking_touch[i] = close_index_values[start + np.argmax(above)]

    # First barrier touched = earliest of {vertical, stop-loss, profit-taking},
    # ignoring NaT (the same semantics as the previous output.min(axis=1)).
    candidates = pd.DataFrame(
        np.vstack([
            events_filtered["End Time"].to_numpy().astype("datetime64[ns]"),
            stop_loss_touch,
            profit_taking_touch,
        ]).T,
        index=events_filtered.index,
    )
    output["End Time"] = candidates.min(axis=1)
    return output[["End Time"]]


def meta_events(
    close: pd.Series,
    time_events: pd.DatetimeIndex,
    ptsl: List[float],
    target: pd.Series,
    return_min: float,
    num_threads: int,
    vertical_barrier_times: Optional[pd.Series] = None,
    side: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Generate events for meta-labeling using the triple-barrier method.

    This function sets up and executes the triple-barrier search in parallel.

    Reference:
        Snippet 3.6, Page 51.

    Parameters
    ----------
    close : pd.Series
        Time series of close prices.
    time_events : pd.DatetimeIndex
        Timestamps of the events (e.g., from CUSUM filter).
    ptsl : List[float]
        [profit_taking_multiplier, stop_loss_multiplier].
    target : pd.Series
        Series of volatility (or 'Base Width') for setting barrier width.
    return_min : float
        The minimum target volatility required to run a search.
    num_threads : int
        Number of parallel threads to use.
    vertical_barrier_times : pd.Series, optional
        A Series of vertical barrier timestamps. If None, barriers are not used.
    side : pd.Series, optional
        A Series of position sides (1 or -1). If None, assumes long-only.

    Returns
    -------
    pd.DataFrame
        The events DataFrame with columns:
        - 'End Time': The timestamp of the *first* barrier touch.
        - 'Base Width': The volatility used for the barriers.
        - 'Side': (Optional) The side of the bet.
    """
    # 1. Filter events by volatility
    target = target.reindex(time_events)
    target = target[target > return_min]
    if target.empty:
        return pd.DataFrame(columns=["End Time", "Base Width", "Side"])

    # 2. Set up vertical barrier
    if vertical_barrier_times is None:
        vertical_barrier_times = pd.Series(pd.NaT, index=target.index)
    else:
        vertical_barrier_times = vertical_barrier_times.reindex(target.index)

    # 3. Set up sides
    if side is None:
        side_series = pd.Series(1.0, index=target.index)
        ptsl_final = [ptsl[0], ptsl[0]] # Symmetric barriers
    else:
        side_series = side.reindex(target.index)
        ptsl_final = ptsl[:2] # Asymmetric barriers

    # 4. Create base events DataFrame
    events = pd.concat(
        {
            "End Time": vertical_barrier_times,
            "Base Width": target,
            "Side": side_series,
        },
        axis=1,
    ).dropna(subset=["Base Width"])

    # 5. Run in parallel
    molecule_subsets = np.array_split(events.index, num_threads)

    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(
            triple_barrier,
            [close] * num_threads,
            [events] * num_threads,
            [ptsl_final] * num_threads,
            molecule_subsets
        ))

    # Combine results and update End Time
    first_touch_times = pd.concat(results, axis=0)["End Time"]
    events["End Time"] = first_touch_times.reindex(events.index)

    if side is None:
        events = events.drop("Side", axis=1)

    return events


def meta_labeling(
    events: pd.DataFrame, close: pd.Series
) -> pd.DataFrame:
    """
    Calculate returns and assign binary labels for meta-labeling.

    Reference:
        Snippet 3.7, Page 52.

    Parameters
    ----------
    events : pd.DataFrame
        The events DataFrame from `meta_events`.
    close : pd.Series
        Time series of close prices.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'End Time': Time of first barrier touch.
        - 'Return': The return of the trade.
        - 'Label': The binary label (0 or 1 if 'Side' is present, -1 or 1 if not).
        - 'Side': (Optional) The side of the bet.
    """
    events_filtered = events.dropna(subset=["End Time"])

    # Get all unique timestamps
    all_dates = events_filtered.index.union(
        events_filtered["End Time"].values
    ).drop_duplicates()

    # Reindex prices to only the required dates
    close_filtered = close.reindex(all_dates, method="bfill")

    out = pd.DataFrame(index=events_filtered.index)
    out["End Time"] = events_filtered["End Time"]

    out["Return"] = (
        np.log(close_filtered.loc[events_filtered["End Time"].values].values)
        - np.log(close_filtered.loc[events_filtered.index].values)
    )

    if "Side" in events_filtered:
        out["Return"] *= events_filtered["Side"]
        out["Side"] = events_filtered["Side"]

    # Assign labels
    out["Label"] = np.sign(out["Return"])

    if "Side" in events_filtered:
        # Meta-labeling: 1 if profitable, 0 if not
        out.loc[out["Return"] <= 0, "Label"] = 0.0

    return out
