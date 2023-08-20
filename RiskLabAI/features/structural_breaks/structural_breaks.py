import numpy as np
import pandas as pd

def lag_dataframe(
    market_data: pd.DataFrame,
    lags: int
) -> pd.DataFrame:
    """
    Apply lags to DataFrame.

    Reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    Methodology: Snippet 17.3

    :param market_data: Data of price or log price.
    :param lags: Arrays of lag or integer that shows number of lags.
    :return: DataFrame with lagged data.
    """
    lagged_data = pd.DataFrame()

    if isinstance(lags, int):
        lags = range(lags + 1)
    else:
        lags = [int(lag) for lag in lags]

    for lag in lags:
        temp_data = pd.DataFrame(market_data.shift(lag).copy(deep=True))
        temp_data.columns = [f'{i}_{lag}' for i in temp_data.columns]
        lagged_data = lagged_data.join(temp_data, how='outer')

    return lagged_data


def prepare_data(
    series: pd.DataFrame,
    constant: str,
    lags: int
) -> (np.ndarray, np.ndarray):
    """
    Prepare the datasets.

    Reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    Methodology: Snippet 17.2

    :param series: Data of price or log price.
    :param constant: String that must be "nc" or "ct" or "ctt".
    :param lags: Arrays of lag or integer that shows number of lags.
    :return: Tuple of y and x arrays.
    """
    series_ = series.diff().dropna()
    x = lag_dataframe(series_, lags).dropna()
    x.iloc[:, 0] = series.values[-x.shape[0]-1:-1, 0]
    y = series_.iloc[-x.shape[0]:].values

    if constant != 'nc':
        x = np.append(x, np.ones((x.shape[0], 1)), axis=1)

    if constant[:2] == 'ct':
        trend = np.arange(x.shape[0]).reshape(-1, 1)
        x = np.append(x, trend, axis=1)

    if constant == 'ctt':
        x = np.append(x, trend**2, axis=1)

    return y, x


def compute_beta(
    y: np.ndarray,
    x: np.ndarray
) -> (np.ndarray, np.ndarray):
    """
    Fit the ADF specification.

    Reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    Methodology: Snippet 17.4

    :param y: Dependent variable.
    :param x: Matrix of independent variable.
    :return: Tuple of beta_mean and beta_variance.
    """
    x_inverse = np.linalg.inv(np.dot(x.T, x))
    beta_mean = x_inverse * np.dot(x.T, y)
    epsilon = y - np.dot(x, beta_mean)
    beta_variance = np.dot(epsilon.T, epsilon) / (x.shape[0] - x.shape[1]) * x_inverse

    return beta_mean, beta_variance


def adf(
    log_price: pd.DataFrame,
    min_sample_length: int,
    constant: str,
    lags: int
) -> dict:
    """
    SADF's inner loop.

    Reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    Methodology: Snippet 17.1

    :param log_price: Pandas DataFrame of log price.
    :param min_sample_length: Minimum sample length.
    :param constant: String that must be "nc" or "ct" or "ctt".
    :param lags: Arrays of lag or integer that shows number of lags.
    :return: Dictionary with Time and gsadf values.
    """
    y, x = prepare_data(log_price, constant=constant, lags=lags)
    start_points, bsadf, all_adf = range(0, y.shape[0] + lags - min_sample_length + 1), -np.inf, []

    for start in start_points:
        y_, x_ = y[start:], x[start:]
        beta_mean_, beta_std_ = compute_beta(y_, x_)
        beta_mean_, beta_std_ = beta_mean_[0, 0], beta_std_[0, 0] ** 0.5
        all_adf.append(beta_mean_ / beta_std_)

        if all_adf[-1] > bsadf:
            bsadf = all_adf[-1]

    out = {'Time': log_price.index[-1], 'gsadf': bsadf}
    return out
