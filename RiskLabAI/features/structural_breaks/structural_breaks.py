"""
Implements the core logic for the (G)SADF (Generalized) Supreme
Augmented Dickey-Fuller test for structural breaks.

Reference:
    De Prado, M. (2018) Advances in financial machine learning.
    John Wiley & Sons, Chapter 17.
"""

import numpy as np
import pandas as pd
from typing import List, Union, Tuple, Dict, Any

def lag_dataframe(
    market_data: pd.DataFrame, lags: Union[int, List[int]]
) -> pd.DataFrame:
    """
    Apply lags to a DataFrame.

    Reference:
        Snippet 17.3, Page 253.

    Parameters
    ----------
    market_data : pd.DataFrame
        DataFrame of price or log price.
    lags : Union[int, List[int]]
        An integer number of lags (e.g., 3 creates lags 0, 1, 2, 3)
        or a specific list of lags to create.

    Returns
    -------
    pd.DataFrame
        A DataFrame with lagged columns, e.g., 'price_0', 'price_1', ...
    """
    lagged_parts = []
    
    if isinstance(lags, int):
        lags_list = range(lags + 1)
    else:
        lags_list = [int(lag) for lag in lags]

    for lag in lags_list:
        lagged_data = market_data.shift(lag)
        lagged_data.columns = [f"{col}_{lag}" for col in market_data.columns]
        lagged_parts.append(lagged_data)

    return pd.concat(lagged_parts, axis=1)


def prepare_data(
    log_price_series: pd.Series, constant: str, lags: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare the y and X matrices for ADF regression.

    Reference:
        Snippet 17.3, Page 253.

    Parameters
    ----------
    log_price_series : pd.Series  # <-- CHANGED: Accept Series
        Series of log prices.
    constant : str
        Type of regression constant ('nc', 'c', 'ct', 'ctt').
    lags : int
        Number of lags to include.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - y_df: The dependent variable (delta log price).
        - x_df: The independent variables (lagged level, lagged deltas, constants).
    """
    # <-- ADDED: Convert univariate series to frame for internal processing
    log_price = log_price_series.to_frame()
    
    price_diff = log_price.diff().dropna()
    y_df = price_diff

    # 1. Lagged level
    x_df = log_price.shift(1).copy()
    x_df.columns = ["level_l1"]

    # 2. Lagged deltas
    if lags > 0:
        lagged_deltas = price_diff.shift(1)
        lagged_deltas.columns = ["delta_l1"]
        
        if lags > 1:
            for i in range(2, lags + 1):
                lagged_deltas[f'delta_l{i}'] = price_diff.shift(i)

        x_df = x_df.join(lagged_deltas, how='outer')

    # 3. Add constants
    if constant == "c":
        x_df["constant"] = 1
    elif constant == "ct":
        x_df["constant"] = 1
        x_df["trend"] = np.arange(1, len(x_df) + 1)
    elif constant == "ctt":
        x_df["constant"] = 1
        x_df["trend"] = np.arange(1, len(x_df) + 1)
        x_df["trend_sq"] = x_df["trend"] ** 2
    
    # Align y and X by dropping NaNs created by lagging
    combined = y_df.join(x_df, how='inner').dropna()
    
    y_df = combined.iloc[:, [0]]
    x_df = combined.iloc[:, 1:]
    
    return y_df, x_df


def compute_beta(
    y_window: np.ndarray, x_window: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute OLS beta coefficients and their variance.

    Reference:
        Snippet 17.2, Page 251.

    Parameters
    ----------
    y_window : np.ndarray
        Window of the dependent variable.
    x_window : np.ndarray
        Window of the independent variables.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - beta_mean: The OLS coefficients.
        - beta_variance: The variance-covariance matrix of the coefficients.
    """
    try:
        xt_x_inv = np.linalg.inv(x_window.T @ x_window)
        xt_y = x_window.T @ y_window
        
        beta_mean = xt_x_inv @ xt_y
        
        error = y_window - (x_window @ beta_mean)
        variance_e = (error.T @ error) / (x_window.shape[0] - x_window.shape[1])
        beta_variance = variance_e * xt_x_inv
        
        return beta_mean, beta_variance
        
    except np.linalg.LinAlgError:
        # Handle singular matrix
        return np.full((x_window.shape[1], 1), np.nan), \
               np.full((x_window.shape[1], x_window.shape[1]), np.nan)


def get_expanding_window_adf(
    log_price: pd.Series,
    min_sample_length: int,
    constant: str,
    lags: int,
) -> pd.Series:
    """
    Compute the ADF t-statistic over an expanding window.

    This is useful for plotting the evolution of the test statistic over time.

    Reference:
        Based on Snippet 17.2, Page 251.

    Parameters
    ----------
    log_price : pd.Series
        Series of log prices.
    min_sample_length : int
        The minimum number of samples to start the expanding window.
    constant : str
        Type of regression constant ('nc', 'c', 'ct', 'ctt').
    lags : int
        Number of lags to include in the regression.

    Returns
    -------
    pd.Series
        A Series of ADF t-statistics, indexed by timestamp.
    """
    # <-- CHANGED: Pass Series directly to prepare_data
    y_df, x_df = prepare_data(log_price, constant=constant, lags=lags)
    
    adf_stats = []
    timestamps = []
    
    for i in range(min_sample_length, y_df.shape[0] + 1):
        y_window = y_df.iloc[:i].values
        x_window = x_df.iloc[:i].values
        
        beta_mean, beta_variance = compute_beta(y_window, x_window)
        
        if np.isnan(beta_variance[0, 0]):
            t_stat = np.nan
        else:
            beta_std_level = beta_variance[0, 0] ** 0.5
            if beta_std_level == 0:
                t_stat = -np.inf if beta_mean[0, 0] < 0 else np.inf
            else:
                t_stat = beta_mean[0, 0] / beta_std_level
                
        adf_stats.append(t_stat)
        timestamps.append(y_df.index[i - 1])

    return pd.Series(adf_stats, index=timestamps)


def get_bsadf_statistic(
    log_price: pd.Series,  # <-- CHANGED: Accept Series
    min_sample_length: int, 
    constant: str, 
    lags: int
) -> Dict[str, Any]:
    """
    Compute the Backward Supremum ADF (BSADF) statistic.

    This test runs an expanding ADF test starting from every possible
    point in the series and finds the supremum (highest) t-statistic.
    This is used to detect the *origination* of a bubble.

    Reference:
        Snippet 17.4, Page 253. (Renamed from `adf` for clarity).

    Parameters
    ----------
    log_price : pd.Series  # <-- CHANGED
        Series of log prices.
    min_sample_length : int
        Minimum sample length for each ADF test.
    constant : str
        Type of regression constant ('nc', 'c', 'ct', 'ctt').
    lags : int
        Number of lags to include.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - 'Time': The timestamp of the end of the series.
        - 'bsadf': The BSADF statistic (the supremum ADF).
    """
    # 1. Prepare the full X, y matrices
    # <-- CHANGED: Pass Series directly
    y, x = prepare_data(log_price, constant=constant, lags=lags)
    
    # 2. Define all possible start points
    # <-- BUG FIX: Removed '+ lags' from the range
    start_points = range(0, y.shape[0] - min_sample_length + 1)
    bsadf = -np.inf  # Supremum ADF
    
    y_np, x_np = y.values, x.values

    # 3. Loop over all expanding windows
    for start in start_points:
        y_window, x_window = y_np[start:], x_np[start:]
        
        # 4. Compute ADF regression for this window
        beta_mean, beta_variance = compute_beta(y_window, x_window)
        
        if np.isnan(beta_variance[0, 0]):
            continue

        # 5. Get t-statistic for the first coefficient (the level)
        beta_mean_level = beta_mean[0, 0]
        beta_std_level = beta_variance[0, 0] ** 0.5
        
        if beta_std_level == 0:
            t_stat = -np.inf if beta_mean_level < 0 else np.inf
        else:
            t_stat = beta_mean_level / beta_std_level
        
        if t_stat > bsadf:
            bsadf = t_stat

    return {"Time": log_price.index[-1], "bsadf": bsadf}