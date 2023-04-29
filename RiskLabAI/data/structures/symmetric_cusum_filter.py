import pandas as pd
from typing import List

def symmetric_cusum_filter(
    input_data: pd.DataFrame,
    threshold: float
) -> pd.DatetimeIndex:
    """
    Implementation of the symmetric CUSUM filter.

    :param input_data: DataFrame of prices and dates
    :type input_data: pd.DataFrame
    :param threshold: Threshold value for CUSUM filter
    :type threshold: float
    :return: DatetimeIndex of events based on the symmetric CUSUM filter
    :rtype: pd.DatetimeIndex

    Reference:
        De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
        Methodology 39.
    """
    time_events, shift_positive, shift_negative = [], 0, 0
    price_delta = input_data.diff()

    for i in price_delta.index[1:]:
        shift_positive = max(0, shift_positive + price_delta.loc[i])
        shift_negative = min(0, shift_negative + price_delta.loc[i])

        if shift_negative < -threshold:
            shift_negative = 0
            time_events.append(i)
        elif shift_positive > threshold:
            shift_positive = 0
            time_events.append(i)

    return pd.DatetimeIndex(time_events)
