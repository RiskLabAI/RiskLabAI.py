import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Tuple


def calculate_t_value_linear_regression(
        price: pd.Series
) -> float:
    """
    Calculate the t-value of a linear trend.

    This function computes the t-value of a linear trend in a time series of prices. 
    The t-value is calculated as the ratio of the slope to the standard error of the regression.

    :param price: Time series of prices as a Pandas Series.
    :return: Calculated t-value as a float.
    """
    x = np.arange(price.shape[0])
    ols = stats.linregress(x, price.values)
    t_value = ols.slope / ols.stderr
    
    return t_value


def find_trend_using_trend_scanning(
        molecule: pd.Index,
        close: pd.Series,
        span: Tuple[int, int]
) -> pd.DataFrame:
    """
    Implement the trend scanning method to find trends.

    This function identifies trends in a time series of prices using the trend scanning method. 
    It calculates the t-value for linear regression of price over a range of spans and 
    identifies the span with the maximum absolute t-value as the trend. The sign of the t-value 
    indicates the direction of the trend.

    :param molecule: Index of observations to label as a Pandas Index.
    :param close: Time series of prices as a Pandas Series.
    :param span: Range of span lengths to evaluate for the maximum absolute t-value as a tuple.
    :return: DataFrame containing trend information with columns ['End Time', 't-Value', 'Trend'].
    """
    outputs = pd.DataFrame(index=molecule, columns=['End Time', 't-Value', 'Trend'])
    spans = range(*span)

    for index in molecule:
        t_values = pd.Series(dtype='float64')
        location = close.index.get_loc(index)

        if location + max(spans) > close.shape[0]:
            continue

        for span in spans:
            tail = close.index[location + span - 1]
            window_prices = close.loc[index:tail]
            t_values.loc[tail] = calculate_t_value_linear_regression(window_prices)

        tail = t_values.replace([-np.inf, np.inf, np.nan], 0).abs().idxmax()
        outputs.loc[index, ['End Time', 't-Value', 'Trend']] = t_values.index[-1], t_values[tail], np.sign(t_values[tail])

    outputs['End Time'] = pd.to_datetime(outputs['End Time'])
    outputs['Trend'] = pd.to_numeric(outputs['Trend'], downcast='signed')

    return outputs.dropna(subset=['Trend'])
