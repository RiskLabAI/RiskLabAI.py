import pandas as pd
from typing import Optional

from .create_ohlcv_dataframe import *

def generate_volume_bar_dataframe(
    tick_data: pd.DataFrame,
    volume_per_bar: int = 10000,
    number_bars: Optional[int] = None
) -> pd.DataFrame:
    """
    Takes a dataframe and generates a volume bar dataframe.

    :param tick_data: Dataframe of tick data
    :type tick_data: pd.DataFrame
    :param volume_per_bar: Volumes in each bar, defaults to 10000
    :type volume_per_bar: int, optional
    :param number_bars: Number of bars, defaults to None
    :type number_bars: Optional[int], optional
    :return: A dataframe containing OHLCV data and other relevant information based on volume bars
    :rtype: pd.DataFrame
    """
    tick_data['volume_cumulated'] = tick_data['size'].cumsum()  # cumulative sum of size

    # If volume_per_bar is not mentioned, then calculate it with all volumes divided by the number of bars
    if not volume_per_bar:
        volume_total = tick_data['volume_cumulated'].values[-1]
        volume_per_bar = volume_total / number_bars
        volume_per_bar = round(volume_per_bar, -2)  # round to the nearest hundred

    tick_grouped = tick_data.reset_index().assign(grouping_id=lambda x: x.volume_cumulated // volume_per_bar)
    dates = tick_grouped.groupby('grouping_id', as_index=False).first()['dates']  # group date times based on volume
    tick_data_grouped = tick_grouped.groupby('grouping_id')
    ohlcv_dataframe = create_ohlcv_dataframe(tick_data_grouped)  # create a dataframe based on tick bars
    ohlcv_dataframe.set_index(dates, drop=True, inplace=True)  # set dates column as index

    return ohlcv_dataframe
