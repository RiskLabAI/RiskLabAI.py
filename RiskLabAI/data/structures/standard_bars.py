import pandas as pd
from typing import Optional

from .utilities import *
from ...utils import *

def generate_dollar_bar_dataframe(
    tick_data: pd.DataFrame,
    dollar_per_bar: int = 100000,
    number_bars: Optional[int] = None
) -> pd.DataFrame:
    """
    Takes a dataframe and generates a dollar bar dataframe.

    :param tick_data: Dataframe of tick data
    :type tick_data: pd.DataFrame
    :param dollar_per_bar: Dollars in each bar, defaults to 100000
    :type dollar_per_bar: int, optional
    :param number_bars: Number of bars, defaults to None
    :type number_bars: Optional[int], optional
    :return: A dataframe containing OHLCV data and other relevant information based on dollar bars
    :rtype: pd.DataFrame
    """
    tick_data['dollar'] = tick_data['price'] * tick_data['size']  # generate dollar column by multiplying price and size
    tick_data['dollars_cumulated'] = tick_data['dollar'].cumsum()  # cumulative sum of dollars

    # If dollar_per_bar is not mentioned, then calculate it with dollars divided by the number of bars
    if not dollar_per_bar:
        dollars_total = tick_data['dollars_cumulated'].values[-1]
        dollar_per_bar = dollars_total / number_bars
        dollar_per_bar = round(dollar_per_bar, -2)  # round to the nearest hundred

    tick_grouped = tick_data.reset_index().assign(grouping_id=lambda x: x.dollars_cumulated // dollar_per_bar)
    dates = tick_grouped.groupby('grouping_id', as_index=False).first()['dates']  # group date times based on dollars
    tick_data_grouped = tick_grouped.groupby('grouping_id')
    ohlcv_dataframe = create_ohlcv_dataframe(tick_data_grouped)  # create a dataframe based on tick bars
    ohlcv_dataframe.set_index(dates, drop=True, inplace=True)  # set dates column as index

    return ohlcv_dataframe

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

def generate_time_bar_dataframe(
    tick_data: pd.DataFrame,
    frequency: str = "5Min"
) -> pd.DataFrame:
    """
    Takes a dataframe and generates a time bar dataframe.

    :param tick_data: Dataframe of tick data
    :type tick_data: pd.DataFrame
    :param frequency: Frequency for rounding date time, defaults to "5Min"
    :type frequency: str, optional
    :return: A dataframe containing OHLCV data and other relevant information based on time bars with the specified frequency
    :rtype: pd.DataFrame
    """
    tick_data_grouped = tick_data.groupby(pd.Grouper(freq=frequency))  # group data sets based on time frequency
    ohlcv_dataframe = create_ohlcv_dataframe(tick_data_grouped)  # create a dataframe based on time bars with frequency freq
    return ohlcv_dataframe

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
