# import pandas as pd
# from typing import List

# """
#     Implementation of the symmetric CUSUM filter.

#     :param input_data: DataFrame of prices and dates
#     :type input_data: pd.DataFrame
#     :param threshold: Threshold value for CUSUM filter
#     :type threshold: float
#     :return: DatetimeIndex of events based on the symmetric CUSUM filter
#     :rtype: pd.DatetimeIndex

#     Reference:
#         De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
#         Methodology 39.
# """
# def symmetric_cusum_filter(
#     input_data: pd.DataFrame,
#     threshold: float
# ) -> pd.DatetimeIndex:
#     time_events, shift_positive, shift_negative = [], 0, 0
#     price_delta = input_data.diff()

#     for i in price_delta.index[1:]:
#         shift_positive = max(0, shift_positive + price_delta.loc[i])
#         shift_negative = min(0, shift_negative + price_delta.loc[i])

#         if shift_negative < -threshold:
#             shift_negative = 0
#             time_events.append(i)
#         elif shift_positive > threshold:
#             shift_positive = 0
#             time_events.append(i)

#     return pd.DatetimeIndex(time_events)



import pandas as pd
from typing import List

def symmetric_cusum_filter(
    input_data: pd.DataFrame,
    threshold: float
) -> pd.DatetimeIndex:
    """
    Implementation of the symmetric CUSUM filter.

    This method is used to detect changes in a time series data.

    :param input_data: DataFrame containing price data.
    :type input_data: pd.DataFrame
    :param threshold: Threshold value for the CUSUM filter.
    :type threshold: float
    :return: Datetime index of events based on the symmetric CUSUM filter.
    :rtype: pd.DatetimeIndex

    .. note:: 
       Reference:
       De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
       Methodology 39.

    .. math::
       S_t^+ = max(0, S_{t-1}^+ + \Delta p_t)
       S_t^- = min(0, S_{t-1}^- + \Delta p_t)

       where:
       - :math:`S_t^+` is the positive CUSUM at time :math:`t`
       - :math:`S_t^-` is the negative CUSUM at time :math:`t`
       - :math:`\Delta p_t` is the price change at time :math:`t`
    """
    
    time_events = []
    cumulative_sum_positive = 0
    cumulative_sum_negative = 0
    price_change = input_data.diff()

    for time_index in price_change.index[1:]:
        cumulative_sum_positive = max(0, cumulative_sum_positive + price_change.loc[time_index])
        cumulative_sum_negative = min(0, cumulative_sum_negative + price_change.loc[time_index])

        if cumulative_sum_negative < -threshold:
            cumulative_sum_negative = 0
            time_events.append(time_index)
        elif cumulative_sum_positive > threshold:
            cumulative_sum_positive = 0
            time_events.append(time_index)

    return pd.DatetimeIndex(time_events)