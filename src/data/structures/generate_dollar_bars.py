import pandas as pd
from typing import Optional

from .create_ohlcv_dataframe import *

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
