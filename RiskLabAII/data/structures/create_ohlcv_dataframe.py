def create_ohlcv_dataframe(
    tick_data_grouped: pd.core.groupby.DataFrameGroupBy
) -> pd.DataFrame:
    """
    Takes a grouped dataframe, combining and creating a new one with information about
    prices and volume.

    :param tick_data_grouped: Grouped dataframes based on some criteria (e.g., time)
    :type tick_data_grouped: pd.core.groupby.DataFrameGroupBy
    :return: A dataframe containing OHLCV data and other relevant information
    :rtype: pd.DataFrame
    """
    ohlc = tick_data_grouped['price'].ohlc()  # find price in each tick
    ohlc['volume'] = tick_data_grouped['size'].sum()  # find volume traded
    ohlc['value_of_trades'] = tick_data_grouped.apply(
        lambda x: (x['price'] * x['size']).sum() / x['size'].sum()
    )  # find value of trades
    ohlc['price_mean'] = tick_data_grouped['price'].mean()  # mean of price
    ohlc['tick_count'] = tick_data_grouped['price'].count()  # number of ticks
    ohlc['price_mean_log_return'] = (
        np.log(ohlc['price_mean']) - np.log(ohlc['price_mean'].shift(1))
    )  # find log return
    return ohlc
