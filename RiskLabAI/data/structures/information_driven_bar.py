from typing import Tuple
import numpy as np
import pandas as pd

from RiskLabAI.data.structures.utilities_lopez import compute_thresholds, create_ohlcv_dataframe

def generate_information_driven_bars(
    tick_data: pd.DataFrame,
    bar_type: str = "volume",
    initial_expected_ticks: int = 2000
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Implements Information-Driven Bars.

    This function computes the Information-Driven Bars based on tick data and the chosen bar type.

    :param tick_data: DataFrame of tick data.
    :type tick_data: pd.DataFrame
    :param bar_type: Type of the bar. Can be "tick", "volume", or "dollar".
    :type bar_type: str, default "volume"
    :param initial_expected_ticks: The initial value of expected ticks.
    :type initial_expected_ticks: int, default 2000

    :return: Tuple containing the OHLCV DataFrame, absolute thetas, and thresholds.
    :rtype: Tuple[pd.DataFrame, np.ndarray, np.ndarray]

    .. note:: 
       Reference:
       De Prado, M. (2018) Advances in Financial Machine Learning. John Wiley & Sons.

    .. math::
       E_b = |\bar{x}|

       where:
       - :math:`E_b` is the expected value of the bars.
       - :math:`\bar{x}` is the mean of the input data.

    The compute_thresholds function is called to compute times_delta, thetas_absolute, thresholds,
    times, thetas, and grouping_id.
    """

    if bar_type == "volume":
        input_data = tick_data['volume_labeled']
    elif bar_type == "tick":
        input_data = tick_data['label']
    elif bar_type == "dollar":
        input_data = tick_data['dollar_labeled']
    else:
        raise ValueError("Invalid bar type. Choose 'tick', 'volume', or 'dollar'.")

    bar_expected_value = np.abs(input_data.mean())
    times_delta, thetas_absolute, thresholds, times, thetas, grouping_id = compute_thresholds(
        input_data, initial_expected_ticks, bar_expected_value)

    tick_grouped = tick_data.reset_index().assign(grouping_id=grouping_id)
    dates = tick_grouped.groupby('grouping_id', as_index=False).first()['dates']
    tick_data_grouped = tick_grouped.groupby('grouping_id')

    ohlcv_dataframe = create_ohlcv_dataframe(tick_data_grouped)
    ohlcv_dataframe.set_index(dates, drop=True, inplace=True)

    return ohlcv_dataframe, thetas_absolute, thresholds
