import pandas as pd
from typing import Optional
from .create_ohlcv_dataframe import *

def generate_tick_bar_dataframe(
    tick_data: pd.DataFrame,
    tick_per_bar: int = 10,
    number_bars: Optional[int] = None
) -> pd.DataFrame:
    """
    Takes a dataframe and generates a tick bar dataframe.

    :param tick_data: Dataframe of tick data
    :type tick_data: pd.DataFrame
    :param tick_per_bar: Number of ticks in each bar, defaults to 10
    :type tick_per_bar: int, optional
    :param number_bars: Number of bars, defaults to None
    :type number_bars: Optional[int], optional
    :return: A dataframe containing OHLCV data and other relevant information based on tick bars
    :rtype: pd.DataFrame
    """
    # If tick_per_bar is not mentioned, then calculate it with the number of all ticks divided by the number of bars
    if not tick_per_bar:
        tick_per_bar = int(tick_data.shape[0] / number_bars)

    tick_grouped = tick_data.reset_index().assign(grouping_id=lambda x: x.index // tick_per_bar)
    dates = tick_grouped.groupby('grouping_id', as_index=False).first()['dates']  # group data sets based on ticks per bar
    tick_data_grouped = tick_grouped.groupby('grouping_id')
    ohlcv_dataframe = create_ohlcv_dataframe(tick_data_grouped)  # create a dataframe based on tick bars
    ohlcv_dataframe.set_index(dates, drop=True, inplace=True)  # set dates column as index

    return ohlcv_dataframe
