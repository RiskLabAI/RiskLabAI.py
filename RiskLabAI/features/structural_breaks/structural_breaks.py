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
        temp_data = market_data.shift(lag).copy()
        temp_data.columns = [f"{col}_{lag}" for col in temp_data.columns]
        lagged_parts.append(temp_data)

    return pd.concat(lagged_parts, axis=1)


def prepare_data(
    series: pd.DataFrame, constant: str, lags: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare the y and X matrices for ADF regression.

    Reference:
        Snippet 17.2, Page 252.

    Parameters
    ----------
    series : pd.DataFrame
        A single-column DataFrame of price or log price.
    constant : str
        The type of trend to include:
        - 'nc': No constant or trend.
        - 'c': Constant only.
        - 'ct': Constant and linear trend.
        - 'ctt': Constant, linear, and quadratic trend.
    lags : int
        The number of lags to include in the regression.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - y: The dependent variable (differenced series).
        - X: The matrix of independent variables (lagged level,
             lagged diffs, and trend components).
    """
    # 1. Create lagged differenced series
    series_diff = series.diff().dropna()
    x = lag_dataframe(series_diff, lags).dropna()
    
    # 2. Replace lag 0 of diffs with lag 1 of levels
    x.iloc[:, 0] = series.values[-x.shape[0] - 1 : -1, 0]
    y = series_diff.iloc[-x.shape[0] :].values

    x_arr = x.to_numpy()

    # 3. Add trend components
    if constant != "nc":
        # Add constant
        x_arr = np.append(x_arr, np.ones((x_arr.shape[0], 1)), axis=1)

    if constant[:2] == "ct":
        # Add linear trend
        trend = np.arange(x_arr.shape[0]).reshape(-1, 1)
        x_arr = np.append(x_arr, trend, axis=1)

    if constant == "ctt":
        # Add quadratic trend
        x_arr = np.append(x_arr, trend**2, axis=1)

    return y, x_arr


def compute_beta(
    y: np.ndarray, x: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit the ADF specification using OLS.

    Reference:
        Snippet 17.4, Page 253.

    Parameters
    ----------
    y : np.ndarray
        Dependent variable.
    x : np.ndarray
        Matrix of independent variables.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - beta_mean: The OLS coefficient estimates.
        - beta_variance: The OLS covariance matrix.
    """
    # (X'X)^-1
    x_inv = np.linalg.inv(x.T @ x)
    
    # beta = (X'X)^-1 @ (X'y)
    beta_mean = x_inv @ (x.T @ y)
    
    # epsilon = y - X @ beta
    epsilon = y - (x @ beta_mean)
    
    # V[beta] = (e'e / (T-K)) * (X'X)^-1
    beta_variance = (
        (epsilon.T @ epsilon) / (x.shape[0] - x.shape[1])
    ) * x_inv

    return beta_mean, beta_variance


def adf(
    log_price: pd.DataFrame,
    min_sample_length: int,
    constant: str,
    lags: int,
) -> Dict[str, Any]:
    """
    Run the ADF test over expanding windows (SADF's inner loop).

    This function computes the ADF statistic for all possible
    start dates, finding the maximum (Supremum) ADF statistic.

    Reference:
        Snippet 17.1, Page 251.

    Parameters
    ----------
    log_price : pd.DataFrame
        DataFrame of log prices.
    min_sample_length : int
        Minimum sample length for the regression.
    constant : str
        Trend component ('nc', 'c', 'ct', 'ctt').
    lags : int
        Number of lags to include.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - 'Time': The timestamp of the end of the series.
        - 'gsadf': The BSADF statistic (the supremum ADF).
    """
    # 1. Prepare the full X, y matrices
    y, x = prepare_data(log_price, constant=constant, lags=lags)
    
    # 2. Define all possible start points
    start_points = range(0, y.shape[0] + lags - min_sample_length + 1)
    bsadf = -np.inf  # Supremum ADF
    all_adf = []

    # 3. Loop over all expanding windows
    for start in start_points:
        y_window, x_window = y[start:], x[start:]
        
        # 4. Compute ADF regression for this window
        beta_mean, beta_variance = compute_beta(y_window, x_window)
        
        # 5. Get t-statistic for the first coefficient (the level)
        beta_mean_level = beta_mean[0, 0]
        beta_std_level = beta_variance[0, 0] ** 0.5
        
        if beta_std_level == 0:
            t_stat = -np.inf # Avoid division by zero
        else:
            t_stat = beta_mean_level / beta_std_level
            
        all_adf.append(t_stat)

        if t_stat > bsadf:
            bsadf = t_stat # Update supremum

    out = {"Time": log_price.index[-1], "gsadf": bsadf}
    return out