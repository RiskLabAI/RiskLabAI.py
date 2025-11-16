"""
Implements the Trend-Scanning Labeling method.

This method labels events based on the t-value of the linear regression
slope over various forward-looking windows, selecting the window
with the most significant trend.

Reference:
    De Prado, M. (2018) Advances in financial machine learning.
    John Wiley & Sons, Chapter 4.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Tuple, Optional

def calculate_t_value_linear_regression(prices: pd.Series) -> float:
    """
    Calculate the t-value of the slope of a linear regression.

    Parameters
    ----------
    prices : pd.Series
        Time series of prices.

    Returns
    -------
    float
        The t-value of the regression slope. Returns np.nan if
        regression fails (e.g., constant series or < 2 data points).
    """
    # --- SUGGESTION: Explicitly handle insufficient data ---
    if prices.shape[0] < 2:
        return np.nan
        
    x = np.arange(prices.shape[0])
    try:
        ols = stats.linregress(x, prices.values)
    except ValueError:
        return np.nan
        
    if ols.stderr == 0:
        # Handle perfect fit (vertical line, or constant)
        return np.sign(ols.slope) * np.inf if ols.slope != 0 else 0.0

    return ols.slope / ols.stderr


def find_trend_using_trend_scanning(
    molecule: pd.Index, close: pd.Series, span: Tuple[int, int]
) -> pd.DataFrame:
    """
    Implement the trend-scanning method.

    For each event in `molecule`, this function scans forward over
    various window lengths (defined by `span`). It computes the t-value
    of the OLS slope for each window and identifies the window
    that maximizes `|t_value|`.

    Reference:
        Snippet 4.1, Page 69.

    Parameters
    ----------
    molecule : pd.Index
        The timestamps of the events to be labeled.
    close : pd.Series
        Time series of close prices.
    span : Tuple[int, int]
        A tuple of (min_span, max_span) defining the range of
        window lengths to scan. `min_span` must be >= 2.
        `max_span` is exclusive, as in `range()`.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by event, with columns:
        - 'End Time': The timestamp of the *end* of the vertical barrier
                      (i.e., end of the max span).
        - 't-Value': The t-value of the most significant trend found.
        - 'Trend': The sign of the trend (-1, 0, or 1).
    """
    outputs = pd.DataFrame(
        index=molecule, columns=["End Time", "t-Value", "Trend"]
    )

    # --- SUGGESTION: Add robustness check for span ---
    # Ensure min_span < max_span and min_span is at least 2 for OLS.
    if span[0] >= span[1] or span[0] < 2:
        outputs["Trend"] = pd.to_numeric(outputs["Trend"], downcast="signed")
        return outputs.dropna(subset=["Trend"])

    spans = range(*span)
    # Use span[1] - 1 directly. It's safer than max(spans)
    # which fails on an empty range.
    max_span_val = span[1] - 1 

    for index in molecule:
        t_values = pd.Series(dtype="float64")
        
        try:
            location = close.index.get_loc(index)
        except KeyError:
            continue # Event timestamp not in close index

        # Ensure we don't scan past the end of the series
        if location + max_span_val >= close.shape[0]:
            continue

        # Get timestamp for the vertical barrier (end of max span)
        vertical_barrier_time = close.index[location + max_span_val]

        for span_val in spans:
            # End of this specific window
            tail_time = close.index[location + span_val - 1]
            window_prices = close.loc[index:tail_time]
            
            t_values.loc[tail_time] = calculate_t_value_linear_regression(
                window_prices
            )
        
        if t_values.empty:
            continue

        # Find the window end that maximized |t_value|
        # Use idxmax on the absolute values, but get the original t-value
        best_t_value_idx = t_values.replace([-np.inf, np.inf, np.nan], 0).abs().idxmax()
        best_t_value = t_values[best_t_value_idx]
        
        outputs.loc[index] = [
            vertical_barrier_time,
            best_t_value,
            np.sign(best_t_value),
        ]

    outputs["End Time"] = pd.to_datetime(outputs["End Time"])
    outputs["Trend"] = pd.to_numeric(outputs["Trend"], downcast="signed")

    return outputs.dropna(subset=["Trend"])