import pandas as pd
from .create_ohlcv_dataframe import *

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
