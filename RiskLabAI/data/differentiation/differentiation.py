"""
Implements fractional differentiation of time series, including both
expanding-window (standard) and fixed-width-window (FFD) methods.

Includes functions to find the minimum differentiation factor 'd' that
results in a stationary series.

Reference:
    De Prado, M. (2018) Advances in financial machine learning.
    John Wiley & Sons, Chapter 5.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from typing import Tuple, Optional

def calculate_weights_std(degree: float, size: int) -> np.ndarray:
    """
    Compute weights for standard (expanding window) fractional differentiation.

    Reference:
        Snippet 5.2, Page 82.

    Parameters
    ----------
    degree : float
        The degree of differentiation (d).
    size : int
        The number of weights to generate (e.g., the series length).

    Returns
    -------
    np.ndarray
        An array of weights, ordered for dot product (w_0 at the end).
    """
    weights = [1.0]
    for k in range(1, size):
        weight = -weights[-1] / k * (degree - k + 1)
        weights.append(weight)
    
    # Reverse for dot product: [w_k, ..., w_1, w_0]
    return np.array(weights[::-1]).reshape(-1, 1)

def calculate_weights_ffd(degree: float, threshold: float = 1e-5) -> np.ndarray:
    """
    Compute weights for Fixed-Width Window Fractional Differentiation (FFD).

    The weights are generated until they fall below the `threshold`.

    Reference:
        Snippet 5.3, Page 83.

    Parameters
    ----------
    degree : float
        The degree of differentiation (d).
    threshold : float, default=1e-5
        The minimum absolute weight to include.

    Returns
    -------
    np.ndarray
        An array of weights, ordered for dot product (w_0 at the end).
    """
    weights = [1.0]
    k = 1
    while True:
        weight = -weights[-1] / k * (degree - k + 1)
        if abs(weight) < threshold:
            break
        weights.append(weight)
        k += 1

    # Reverse for dot product: [w_k, ..., w_1, w_0]
    return np.array(weights[::-1]).reshape(-1, 1)

def fractional_difference_std(
    series: pd.DataFrame,
    degree: float,
    threshold: float = 0.01
) -> pd.DataFrame:
    """
    Compute the standard (expanding window) fractionally differentiated series.

    This method uses all available past data for each calculation.

    Reference:
        Snippet 5.2, Page 82.

    Parameters
    ----------
    series : pd.DataFrame
        DataFrame of time series (one column per series).
    degree : float
        The degree of differentiation (d).
    threshold : float, default=0.01
        The weight-loss threshold to determine the minimum number
        of observations required (warm-up period).

    Returns
    -------
    pd.DataFrame
        DataFrame of fractionally differentiated series.
    """
    # 1. Compute weights for the full series
    weights = calculate_weights_std(degree, series.shape[0])
    
    # 2. Determine warm-up period
    weights_cumsum_abs = np.cumsum(np.abs(weights))
    weights_cumsum_abs /= weights_cumsum_abs[-1]
    skip = np.searchsorted(weights_cumsum_abs, threshold)
    
    result_df = pd.DataFrame(index=series.index, columns=series.columns)

    for name in series.columns:
        # Use .ffill() - fillna(method=) is deprecated
        series_ffill = series[[name]].ffill().dropna()
        if series_ffill.empty or series_ffill.shape[0] < skip:
            continue
            
        series_np = series_ffill.to_numpy()

        for iloc in range(skip, series_np.shape[0]):
            # Get the relevant window of data and weights
            window_data = series_np[:iloc + 1]
            window_weights = weights[-(iloc + 1):]
            
            # Dot product
            result_df.loc[series_ffill.index[iloc], name] = np.dot(
                window_weights.T, window_data
            )[0, 0]

    return result_df.dropna(how='all')


def fractional_difference_fixed(
    series: pd.DataFrame,
    degree: float,
    threshold: float = 1e-5
) -> pd.DataFrame:
    """
    Compute the Fixed-Width Window (FFD) fractionally differentiated series.

    This method iterates over columns and applies a fast, convolved
    FFD calculation to each one.

    Reference:
        Snippet 5.3, Page 83.

    Parameters
    ----------
    series : pd.DataFrame
        DataFrame of time series (one column per series).
    degree : float
        The degree of differentiation (d).
    threshold : float, default=1e-5
        Threshold to determine the fixed window width.

    Returns
    -------
    pd.DataFrame
        DataFrame of fractionally differentiated series.
    """
    result_df = pd.DataFrame(index=series.index)
    
    for name in series.columns:
        result_df[name] = fractional_difference_fixed_single(
            series[name], degree, threshold
        )
            
    return result_df.dropna(how='all')


def fractional_difference_fixed_single(
    series: pd.Series,
    degree: float,
    threshold: float = 1e-5
) -> pd.Series:
    """
    Compute the FFD series for a single `pd.Series` using np.convolve.

    Parameters
    ----------
    series : pd.Series
        Time series.
    degree : float
        The degree of differentiation (d).
    threshold : float, default=1e-5
        Threshold to determine the fixed window width.

    Returns
    -------
    pd.Series
        The fractionally differentiated series.
    """
    
    # 1. Compute weights
    # Reverse weights: calculate_weights_ffd returns [w_k, ..., w_0]
    weights = calculate_weights_ffd(degree, threshold).flatten()[::-1]
    width = len(weights)
    
    # 2. Prepare data (drop leading NaNs)
    series_ffill = series.ffill().dropna()
    
    if series_ffill.empty or series_ffill.shape[0] < width:
        # Not enough data to convolve
        return pd.Series(index=series.index, dtype="float64")
        
    series_np = series_ffill.to_numpy()
    
    # 3. Apply convolution
    # 'valid' mode computes the dot product only where the
    # window fully overlaps the series.
    convolved_vals = np.convolve(series_np, weights, mode='valid')
    
    # 4. Create result series on the valid index
    # The result aligns with the *end* of the window
    result_index = series_ffill.index[width - 1:]
    valid_results = pd.Series(convolved_vals, index=result_index)
    
    # 5. Reindex to original full index
    # This correctly places NaNs at the start (warm-up)
    # and in any gaps that were in the original series.
    return valid_results.reindex(series.index)


def plot_weights(
    degree_range: Tuple[float, float],
    number_degrees: int,
    size: int,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot the weights of fractionally differentiated series for various degrees.

    Parameters
    ----------
    degree_range : Tuple[float, float]
        (min_degree, max_degree) to plot.
    number_degrees : int
        Number of 'd' values to plot in the range.
    size : int
        Length of the time series (number of weights).
    ax : plt.Axes, optional
        Matplotlib axes to plot on. If None, a new figure/axes is created.

    Returns
    -------
    plt.Axes
        The axes object with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        
    weights_df = pd.DataFrame()
    for degree in np.linspace(degree_range[0], degree_range[1], number_degrees):
        degree = round(degree, 2)
        weights = calculate_weights_std(degree, size)
        weights_df[degree] = pd.Series(weights.flatten(), index=range(size - 1, -1, -1))
    
    weights_df.plot(ax=ax)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Weight")
    ax.set_title("Fractional Differentiation Weights")
    ax.legend(title="Degree (d)")
    return ax


def find_optimal_ffd_simple(
    input_series: pd.DataFrame,
    p_value_threshold: float = 0.05
) -> pd.DataFrame:
    """
    Find the minimum 'd' that passes the ADF test, for a range of d.

    This function tests `d` in 11 steps from 0 to 1.

    Reference:
        Snippet 5.4, Page 85.

    Parameters
    ----------
    input_series : pd.DataFrame
        DataFrame containing a 'close' column.
    p_value_threshold : float, default=0.05
        The ADF test p-value to pass.

    Returns
    -------
    pd.DataFrame
        DataFrame of ADF test results for each 'd'.
    """
    results_list = []
    
    # Resample to daily to ensure consistent lags
    series_daily = np.log(input_series[['close']]).resample('1D').last().dropna()
    
    for d in np.linspace(0, 1, 11):
        differentiated = fractional_difference_fixed(
            series_daily, d, threshold=0.01
        ).dropna()
        
        if differentiated.empty:
            continue
            
        corr = np.corrcoef(
            series_daily.loc[differentiated.index, 'close'],
            differentiated['close']
        )[0, 1]
        
        try:
            adf_result = adfuller(
                differentiated['close'], maxlag=1, regression='c', autolag=None
            )
            
            results_list.append(
                {
                    'd': d,
                    'adfStat': adf_result[0],
                    'pVal': adf_result[1],
                    'lags': adf_result[2],
                    'nObs': adf_result[3],
                    '95% conf': adf_result[4]['5%'],
                    'corr': corr
                }
            )
        except Exception:
            # Handle cases where ADF test fails (e.g., insufficient data)
            continue
    
    if not results_list:
        return pd.DataFrame()
        
    return pd.DataFrame(results_list).set_index('d')


def fractionally_differentiated_log_price(
    input_series: pd.Series,
    threshold: float = 1e-5,
    step: float = 0.01,
    p_value_threshold: float = 0.05
) -> pd.Series:
    """
    Find the minimum 'd' that makes a log-price series stationary.

    This function iteratively increases 'd' by 'step' until the ADF test
    p-value falls below `p_value_threshold`.

    Parameters
    ----------
    input_series : pd.Series
        Time series of prices.
    threshold : float, default=1e-5
        The weight threshold for FFD.
    step : float, default=0.01
        The increment for testing 'd'.
    p_value_threshold : float, default=0.05
        The significance level for the ADF test.

    Returns
    -------
    pd.Series
        The fractionally differentiated series with the minimum
        stationary-passing 'd'.
    """
    log_price = np.log(input_series)
    degree = 0.0
    p_value = 1.0
    
    differentiated_series = None

    while p_value > p_value_threshold:
        degree += step
        if degree > 2.0: # Safety break
             raise ValueError("Failed to find stationary 'd' < 2.0")
             
        differentiated = fractional_difference_fixed_single(
            log_price, degree, threshold=threshold
        ).dropna()
        
        if differentiated.empty:
            continue # Not enough data for this 'd'
        
        try:
            adf_test = adfuller(
                differentiated, maxlag=1, regression='c', autolag=None
            )
            p_value = adf_test[1]
        except Exception:
            p_value = 1.0 # Failed test, keep going
        
        if differentiated_series is None:
            differentiated_series = differentiated # Store first valid series
    
    # Return the last computed series that passed
    if differentiated_series is None:
        raise ValueError("Could not generate any differentiated series.")
        
    return differentiated