import sys
import time
import numpy as np
import pandas as pd
from typing import Tuple
from RiskLabAI.data.structures.utilities_lopez import *

def progress_bar(
    value: int,
    end_value: int,
    start_time: float,
    bar_length: int = 20
) -> None:
    """
    Display a progress bar in the console.

    :param value: Current progress value.
    :param end_value: The end value indicating 100% progress.
    :param start_time: Time when the event started.
    :param bar_length: The length of the progress bar in characters. Default is 20.
    """
    percent = float(value) / end_value
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    remaining = int(((time.time() - start_time) / value) * (end_value - value) / 60)

    sys.stdout.write(
        "\rCompleted: [{0}] {1}% - {2} minutes remaining.".format(
            arrow + spaces, int(round(percent * 100)), remaining
        )
    )
    sys.stdout.flush()


def ewma(
    input_array: np.ndarray,
    window_length: int
) -> np.ndarray:
    """
    Computes the Exponentially Weighted Moving Average (EWMA).

    :param input_array: The input time series array.
    :param window_length: Window length for the EWMA.
    :return: The EWMA values.
    """
    N = input_array.shape[0]
    output_ewma = np.empty(N, dtype='float64')
    omega = 1
    alpha = 2 / float(window_length + 1)
    current_value = input_array[0]
    output_ewma[0] = current_value

    for i in range(1, N):
        omega += (1 - alpha) ** i
        current_value = current_value * (1 - alpha) + input_array[i]
        output_ewma[i] = current_value / omega

    return output_ewma


def compute_grouping(
    target_col: pd.Series,
    initial_expected_ticks: int,
    bar_size: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Group a DataFrame based on a feature and calculates thresholds.

    :param target_col: Target column of tick dataframe.
    :param initial_expected_ticks: Initial expected ticks.
    :param bar_size: Initial expected size in each tick.
    :return: Arrays of times_delta, thetas_absolute, thresholds, times, thetas, grouping_id.
    """
    N = target_col.shape[0]
    target_col = target_col.values.astype(np.float64)
    thetas_absolute, thresholds, thetas, grouping_id = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    thetas_absolute[0], current_theta = np.abs(target_col[0]), target_col[0]
    start_time = time.time()
    current_group_id, time_prev, expected_ticks, expected_bar_value = 0, 0, initial_expected_ticks, bar_size
    times_delta, times = [], []

    for i in range(1, N):
        current_theta += target_col[i]
        thetas[i] = current_theta
        theta_absolute = np.abs(current_theta)
        thetas_absolute[i] = theta_absolute
        threshold = expected_ticks * expected_bar_value
        thresholds[i] = threshold
        grouping_id[i] = current_group_id

        if theta_absolute >= threshold:
            current_group_id += 1
            current_theta = 0
            times_delta.append(np.float64(i - time_prev))
            times.append(i)
            time_prev = i
            expected_ticks = ewma(np.array(times_delta), window_length=np.int64(len(times_delta)))[-1]
            expected_bar_value = np.abs(ewma(target_col[:i], window_length=np.int64(initial_expected_ticks))[0])

        progress_bar(i, N, start_time)

    return times_delta, thetas_absolute, thresholds, times, thetas, grouping_id

import pandas as pd
import numpy as np
from typing import Tuple


def generate_information_driven_bars(
    tick_data: pd.DataFrame,
    bar_type: str = "volume",
    tick_expected_initial: int = 2000
) -> Tuple[pd.DataFrame, np.array, np.array]:
    """
    Implements Information-Driven Bars as per the methodology described in
    "Advances in financial machine learning" by De Prado (2018).

    :param tick_data: DataFrame of tick data.
    :param bar_type: Type of the bars, options: "tick", "volume", "dollar".
    :param tick_expected_initial: Initial expected ticks value.
    :return: A tuple containing the OHLCV DataFrame, thetas absolute array,
        and thresholds array.
    """
    if bar_type == "volume":
        input_data = tick_data['volume_labeled']
    elif bar_type == "tick":
        input_data = tick_data['label']
    elif bar_type == "dollar":
        input_data = tick_data['dollarslabeled']
    else:
        raise ValueError("Invalid bar_type provided. Choose among 'tick', 'volume', 'dollar'.")

    bar_expected_value = np.abs(input_data.mean())

    times_delta, thetas_absolute, thresholds, _, _, grouping_id = compute_grouping(
        input_data, tick_expected_initial, bar_expected_value)

    tick_grouped = tick_data.reset_index().assign(grouping_id=grouping_id)
    dates = tick_grouped.groupby('grouping_id', as_index=False).first()['dates']

    ohlcv_dataframe = ohlcv(tick_grouped.groupby('grouping_id'))
    ohlcv_dataframe.set_index(dates, drop=True, inplace=True)

    return ohlcv_dataframe, thetas_absolute, thresholds


def ohlcv(
    tick_data_grouped: pd.core.groupby.generic.DataFrameGroupBy
) -> pd.DataFrame:
    """
    Computes various statistics for the grouped tick data.

    Takes a grouped dataframe, combines the data, and creates a new one with
    information about prices, volume, and other statistics. This is typically
    used in the context of financial tick data to generate OHLCV data
    (Open, High, Low, Close, Volume).

    :param tick_data_grouped: Grouped DataFrame containing tick data.
    :return: A DataFrame containing OHLCV data and other derived statistics.
    """
    ohlc = tick_data_grouped['price'].ohlc()
    ohlc['volume'] = tick_data_grouped['size'].sum()
    ohlc['value_of_trades'] = tick_data_grouped.apply(
        lambda x: (x['price'] * x['size']).sum() / x['size'].sum()
    )
    ohlc['price_mean'] = tick_data_grouped['price'].mean()
    ohlc['tick_count'] = tick_data_grouped['price'].count()
    ohlc['price_mean_log_return'] = np.log(ohlc['price_mean']) - np.log(ohlc['price_mean'].shift(1))

    return ohlc


def generate_time_bar(
    tick_data: pd.DataFrame,
    frequency: str = "5Min"
) -> pd.DataFrame:
    """
    Generates time bars for tick data.

    This function groups tick data by a specified time frequency and then
    computes OHLCV (Open, High, Low, Close, Volume) statistics.

    :param tick_data: DataFrame containing tick data.
    :param frequency: Time frequency for rounding datetime.
    :return: A DataFrame containing OHLCV data grouped by time.
    """
    tick_data_grouped = tick_data.groupby(pd.Grouper(freq=frequency))
    ohlcv_dataframe = ohlcv(tick_data_grouped)
    return ohlcv_dataframe


def generate_tick_bar(
    tick_data: pd.DataFrame,
    ticks_per_bar: int = 10,
    number_bars: int = None
) -> pd.DataFrame:
    """
    Generates tick bars for tick data.

    This function groups tick data by a specified number of ticks and then
    computes OHLCV statistics.

    :param tick_data: DataFrame containing tick data.
    :param ticks_per_bar: Number of ticks in each bar.
    :param number_bars: Number of bars to generate.
    :return: A DataFrame containing OHLCV data grouped by tick count.
    """
    if not ticks_per_bar:
        ticks_per_bar = int(tick_data.shape[0] / number_bars)

    tick_grouped = tick_data.reset_index().assign(grouping_id=lambda x: x.index // ticks_per_bar)
    dates = tick_grouped.groupby('grouping_id', as_index=False).first()['dates']
    tick_data_grouped = tick_grouped.groupby('grouping_id')
    ohlcv_dataframe = ohlcv(tick_data_grouped)
    ohlcv_dataframe.set_index(dates, drop=True, inplace=True)

    return ohlcv_dataframe

def generate_volume_bar(
    tick_data: pd.DataFrame,
    volume_per_bar: int = 10000,
    number_bars: int = None
) -> pd.DataFrame:
    """
    Generates volume bars for tick data.

    This function groups tick data by a specified volume size and then computes OHLCV statistics.

    :param tick_data: DataFrame containing tick data.
    :param volume_per_bar: Volume size for each bar.
    :param number_bars: Number of bars to generate.

    :return: A DataFrame containing OHLCV data grouped by volume.
    """
    tick_data['volume_cumulated'] = tick_data['size'].cumsum()

    if not volume_per_bar:
        volume_total = tick_data['volume_cumulated'].values[-1]
        volume_per_bar = volume_total / number_bars
        volume_per_bar = round(volume_per_bar, -2)

    tick_grouped = tick_data.reset_index().assign(
        grouping_id=lambda x: x['volume_cumulated'] // volume_per_bar
    )
    dates = tick_grouped.groupby('grouping_id', as_index=False).first()['dates']
    tick_data_grouped = tick_grouped.groupby('grouping_id')
    ohlcv_dataframe = ohlcv(tick_data_grouped)
    ohlcv_dataframe.set_index(dates, drop=True, inplace=True)

    return ohlcv_dataframe


def generate_dollar_bar(
    tick_data: pd.DataFrame,
    dollar_per_bar: float = 100000,
    number_bars: int = None
) -> pd.DataFrame:
    """
    Generates dollar bars for tick data.

    This function groups tick data by a specified dollar amount and then computes OHLCV statistics.

    :param tick_data: DataFrame containing tick data.
    :param dollar_per_bar: Dollar amount for each bar.
    :param number_bars: Number of bars to generate.

    :return: A DataFrame containing OHLCV data grouped by dollar amount.
    """
    tick_data['dollar'] = tick_data['price'] * tick_data['size']
    tick_data['dollars_cumulated'] = tick_data['dollar'].cumsum()

    if not dollar_per_bar:
        dollars_total = tick_data['dollars_cumulated'].values[-1]
        dollar_per_bar = dollars_total / number_bars
        dollar_per_bar = round(dollar_per_bar, -2)

    tick_grouped = tick_data.reset_index().assign(
        grouping_id=lambda x: x['dollars_cumulated'] // dollar_per_bar
    )
    dates = tick_grouped.groupby('grouping_id', as_index=False).first()['dates']
    tick_data_grouped = tick_grouped.groupby('grouping_id')
    ohlcv_dataframe = ohlcv(tick_data_grouped)
    ohlcv_dataframe.set_index(dates, drop=True, inplace=True)

    return ohlcv_dataframe


def calculate_pca_weights(
    covariance_matrix: np.ndarray,
    risk_distribution: np.ndarray = None,
    risk_target: float = 1.0
) -> np.ndarray:
    """
    Calculates hedging weights using the covariance matrix, risk distribution, and risk target.

    :param covariance_matrix: Covariance matrix.
    :param risk_distribution: Risk distribution vector.
    :param risk_target: Risk target value.

    :return: Weights.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    indices = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[indices], eigenvectors[:, indices]

    if risk_distribution is None:
        risk_distribution = np.zeros(covariance_matrix.shape[0])
        risk_distribution[-1] = 1.0

    loads = risk_target * (risk_distribution / eigenvalues) ** 0.5
    weights = np.dot(eigenvectors, np.reshape(loads, (-1, 1)))

    return weights


def events(
    input_data: pd.DataFrame,
    threshold: float
) -> pd.DatetimeIndex:
    """
    Implementation of the symmetric CUSUM filter.

    This function computes time events when certain price change thresholds are met.

    :param input_data: DataFrame of prices and dates.
    :param threshold: Threshold for price change.

    :return: DatetimeIndex containing events.
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
