import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

def calculate_weights(
        degree: float,
        size: int
) -> np.ndarray:
    """
    Compute the weights for fractionally differentiated series.

    :param degree: Degree of the binomial series.
    :param size: Length of the time series.
    :return: Array of weights.

    Formula:
        .. math::
            w(k) = -w(k-1) / k * (degree - k + 1)
    """
    weights = [1.]
    for k in range(1, size):
        weight = -weights[-1] / k * (degree - k + 1)
        weights.append(weight)
    return np.array(weights[::-1]).reshape(-1, 1)

def plot_weights(
        degree_range: tuple[float, float],
        number_degrees: int,
        size: int
):
    """
    Plot the weights of fractionally differentiated series.

    :param degree_range: Tuple containing the minimum and maximum degree values.
    :param number_degrees: Number of degrees to plot.
    :param size: Length of the time series.
    """
    weights_df = pd.DataFrame()
    for degree in np.linspace(degree_range[0], degree_range[1], number_degrees):
        degree = round(degree, 2)
        weight = calculate_weights(degree, size)
        weight_df = pd.DataFrame(weight, index=range(weight.shape[0])[::-1], columns=[degree])
        weights_df = weights_df.join(weight_df, how='outer')

    ax = weights_df.plot()
    ax.show()

def fractional_difference(
        series: pd.DataFrame,
        degree: float,
        threshold: float = 0.01
) -> pd.DataFrame:
    """
    Compute the standard fractionally differentiated series.

    :param series: Dataframe of dates and prices.
    :param degree: Degree of the binomial series.
    :param threshold: Threshold for weight-loss.
    :return: Dataframe of fractionally differentiated series.

    Methodology reference:
        De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons, p. 82.
    """
    weights = calculate_weights(degree, series.shape[0])
    weights_cumsum = np.cumsum(abs(weights))
    weights_cumsum /= weights_cumsum[-1]
    skip = weights_cumsum[weights_cumsum > threshold].shape[0]
    result_dict = {}
    for name in series.columns:
        series_filtered = series[[name]].fillna(method='ffill').dropna()
        result_series = pd.Series(dtype="float64")
        for iloc in range(skip, series_filtered.shape[0]):
            date = series_filtered.index[iloc]
            price = series.loc[date, name]
            if isinstance(price, (pd.Series, pd.DataFrame)):
                price = price.resample('1m').mean()
            if not np.isfinite(price).any():
                continue
            try:
                result_series.loc[date] = np.dot(weights[-(iloc + 1):, :].T, series_filtered.loc[:date])[0, 0]
            except:
                continue
        result_dict[name] = result_series.copy(deep=True)
    return pd.concat(result_dict, axis=1)

def calculate_weights_ffd(
        degree: float,
        threshold: float
) -> np.ndarray:
    """
    Compute the weights for fixed-width window fractionally differentiated method.

    :param degree: Degree of the binomial series.
    :param threshold: Threshold for weight-loss.
    :return: Array of weights.

    Methodology reference:
        De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons, p. 83.
    """
    weights = [1.]
    k = 1
    while abs(weights[-1]) >= threshold:
        weight = -weights[-1] / k * (degree - k + 1)
        weights.append(weight)
        k += 1
    return np.array(weights[::-1]).reshape(-1, 1)[1:]

def fractional_difference_fixed(
        series: pd.DataFrame,
        degree: float,
        threshold: float = 1e-5
) -> pd.DataFrame:
    """
    Compute the fixed-width window fractionally differentiated series.

    :param series: Dataframe of dates and prices.
    :param degree: Degree of the binomial series.
    :param threshold: Threshold for weight-loss.
    :return: Dataframe of fractionally differentiated series.

    Methodology reference:
        De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons, p. 83.
    """
    weights = calculate_weights_ffd(degree, threshold)
    width = len(weights) - 1
    result_dict = {}
    for name in series.columns:
        series_filtered = series[[name]].fillna(method='ffill').dropna()
        result_series = pd.Series(dtype="float64")
        for iloc in range(width, series_filtered.shape[0]):
            day1 = series_filtered.index[iloc - width]
            day2 = series_filtered.index[iloc]
            if not np.isfinite(series.loc[day2, name]):
                continue
            result_series[day2] = np.dot(weights.T, series_filtered.loc[day1:day2])[0, 0]
        result_dict[name] = result_series.copy(deep=True)
    return pd.concat(result_dict, axis=1)

def fractional_difference_fixed_single(
        series: pd.Series,
        degree: float,
        threshold: float = 1e-5
) -> pd.DataFrame:
    """
    Compute the fixed-width window fractionally differentiated series.

    :param series: Series of dates and prices.
    :param degree: Degree of the binomial series.
    :param threshold: Threshold for weight-loss.
    :return: Fractionally differentiated series.

    Methodology reference:
        De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons, p. 83.
    """
    weights = calculate_weights_ffd(degree, threshold)
    width = len(weights) - 1
    series_filtered = series.fillna(method='ffill').dropna()
    result_series = pd.Series(dtype="float64", index=series.index)
    for iloc in range(width, series_filtered.shape[0]):
        day1 = series_filtered.index[iloc - width]
        day2 = series_filtered.index[iloc]
        if not np.isfinite(series.loc[day2]):
            continue
        result_series[day2] = np.dot(weights.T, series_filtered.loc[day1:day2])[0]

    return result_series

def minimum_ffd(
        input_series: pd.DataFrame
) -> pd.DataFrame:
    """
    Find the minimum degree value that passes the ADF test.

    :param input_series: Dataframe of input data.
    :return: Dataframe of ADF test results.

    Methodology reference:
        De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons, p. 85.
    """
    results = pd.DataFrame(columns=['adfStat', 'pVal', 'lags', 'nObs', '95% conf', 'corr'])
    for d in np.linspace(0, 1, 11):
        dataframe = np.log(input_series[['close']]).resample('1D').last()
        differentiated = fractional_difference_fixed(dataframe, d, threshold=0.01)
        corr = np.corrcoef(dataframe.loc[differentiated.index, 'close'], differentiated['close'])[0, 1]
        differentiated = adfuller(differentiated['close'], maxlag=1, regression='c', autolag=None)
        results.loc[d] = list(differentiated[:4]) + [differentiated[4]['5%']] + [corr]
    return results

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

def get_weights(
    degree: float,
    length: int
) -> np.ndarray:
    """
    Calculate the weights for the fractional differentiation method.

    :param degree: Degree of binomial series.
    :param length: Length of the series.
    :return: Array of calculated weights.

    Related mathematical formula:
    .. math::
        w_i = -w_{i-1}/i*(degree - i + 1)
    """
    weights = [1.]
    k = 1
    while True:
        weight = -weights[-1] / k * (degree - k + 1)
        if abs(weight) < 1e-5:
            break
        weights.append(weight)
        k += 1
    return np.array(weights[::-1]).reshape(-1, 1)[1:]

def fractional_difference(
    series: pd.DataFrame,
    degree: float,
    threshold: float = 0.01
) -> pd.DataFrame:
    """
    Calculate the fractionally differentiated series using the fixed-width window method.

    :param series: DataFrame of dates and prices.
    :param degree: Degree of binomial series.
    :param threshold: Threshold for weight-loss.
    :return: DataFrame of fractionally differentiated series.

    Related mathematical formula:
    .. math::
        F_t^{(d)} = \sum_{i=0}^{t} w_i F_{t-i}
    """
    weights = get_weights(degree, series.shape[0])
    skip = weights.shape[0]
    output = {}
    for name in series.columns:
        series_filtered = series[[name]].fillna(method='ffill').dropna()
        series_result = pd.Series(dtype='float64')
        for iloc in range(skip, series_filtered.shape[0]):
            date = series_filtered.index[iloc]
            price = series.loc[date, name]
            if not np.isfinite(price):
                continue
            try:
                series_result.loc[date] = np.dot(weights.T, series_filtered.loc[:date])[0, 0]
            except:
                continue
        output[name] = series_result.copy(deep=True)
    return pd.concat(output, axis=1)

def minimum_adf_degree(
    input_series: pd.DataFrame
) -> pd.DataFrame:
    """
    Find the minimum degree value that passes the ADF test.

    :param input_series: DataFrame of input series.
    :return: DataFrame of output results with ADF statistics.

    Related mathematical formula:
    .. math::
        F_t^{(d)} = \sum_{i=0}^{t} w_i F_{t-i}
    """
    output = pd.DataFrame(columns=['adfStat', 'pVal', 'lags', 'nObs', '95% conf', 'corr'])
    for d in np.linspace(0, 1, 11):
        dataframe = np.log(input_series[['close']]).resample('1D').last()
        differentiated = fractional_difference(dataframe, d, threshold=0.01)
        corr = np.corrcoef(dataframe.loc[differentiated.index, 'close'], differentiated['close'])[0, 1]
        adf_result = adfuller(differentiated['close'], maxlag=1, regression='c', autolag=None)
        output.loc[d] = list(adf_result[:4]) + [adf_result[4]['5%']] + [corr]
    return output

def fractionally_differentiated_log_price(
    input_series: pd.Series,
    threshold=0.01,
    step=0.1,
    base_p_value=0.05
) -> float:
    """
    Calculate the fractionally differentiated log price with the minimum degree differentiation
    that passes the Augmented Dickey-Fuller (ADF) test.

    :param input_series: Time series of input data.
    :param threshold: The threshold for fractionally differentiating the log price.
    :param step: The increment step for adjusting the differentiation degree.
    :param base_p_value: The significance level for the ADF test.
    :return: Fractionally differentiated log price series.

    Methodology reference:
        De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons, p. 85.
    """
    log_price = np.log(input_series)
    degree = -step
    p_value = 1

    while p_value > base_p_value:
        degree += step
        differentiated = fractional_difference_fixed_single(log_price, degree, threshold=threshold)
        adf_test = adfuller(differentiated.dropna(), maxlag=1, regression='c', autolag=None)
        p_value = adf_test[1]

    return differentiated
