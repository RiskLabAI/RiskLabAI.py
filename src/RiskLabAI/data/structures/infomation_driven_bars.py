from typing import Tuple
import numpy as np
import pandas as pd

from .utilities import *

def generate_information_driven_bars(
    tick_data: pd.DataFrame,
    bar_type: str = "volume",
    initial_expected_ticks: int = 2000
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Implements Information-Driven Bars.

    Reference: De Prado, M. (2018) Advances in Financial Machine Learning. John Wiley & Sons.

    This function computes the Information-Driven Bars based on tick data and the chosen bar type.
    The following formulas are used in this function:

    - The absolute value of the mean of the input data:

      .. math:: E_b = |\bar{x}|

    - The compute_thresholds function is called to compute times_delta, thetas_absolute, thresholds,
      times, thetas, and grouping_id.

    :param tick_data: DataFrame of tick data.
    :param bar_type: User can choose between "tick", "volume", or "dollar" imbalanced bars. Default is "volume".
    :param initial_expected_ticks: The initial value of expected ticks. Default is 2000.
    :return: Tuple containing the OHLCV DataFrame, thetas_absolute, and thresholds.
    """

    if bar_type == "volume":
        input_data = tick_data['volumelabeled']
    elif bar_type == "tick":
        input_data = tick_data['label']
    elif bar_type == "dollar":
        input_data = tick_data['dollarslabeled']
    else:
        raise ValueError("Invalid bar type. Choose 'tick', 'volume', or 'dollar'.")

    bar_expected_value = np.abs(input_data.mean())

    times_delta, thetas_absolute, thresholds, times, thetas, grouping_id = compute_thresholds(input_data, initial_expected_ticks, bar_expected_value)

    tick_grouped = tick_data.reset_index().assign(groupingID=grouping_id)
    dates = tick_grouped.groupby('groupingID', as_index=False).first()['dates']
    tick_data_grouped = tick_grouped.groupby('groupingID')

    ohlcv_dataframe = create_ohlcv_dataframe(tick_data_grouped)
    ohlcv_dataframe.set_index(dates, drop=True, inplace=True)

    return ohlcv_dataframe, thetas_absolute, thresholds