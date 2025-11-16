import numpy as np
import pandas as pd
import time
from typing import List, Tuple
from .ewma import ewma  # <-- SUGGESTION 1: Import the correct, jitted ewma
from .progress import progress_bar


def compute_thresholds(
    target_column: np.ndarray,
    initial_expected_ticks: int,
    initial_bar_size: float
) -> Tuple[List[float], np.ndarray, np.ndarray, List[int], np.ndarray, np.ndarray]:
    """
    Groups the target_column DataFrame based on a feature and calculates thresholds.

    This function groups the target_column DataFrame based on a feature
    and calculates the thresholds, which can be used in financial machine learning
    applications such as dynamic time warping.

    :param target_column: Target column of the DataFrame.
    :type target_column: np.ndarray
    :param initial_expected_ticks: Initial expected number of ticks.
    :type initial_expected_ticks: int
    :param initial_bar_size: Initial expected size of each tick.
    :type initial_bar_size: float
    :return: A tuple containing the time deltas, absolute theta values, thresholds,
        times, theta values, and grouping IDs.
    :rtype: Tuple[List[float], np.ndarray, np.ndarray, List[int], np.ndarray, np.ndarray]
    """
    num_values = target_column.shape[0]
    target_column_values = target_column.astype(np.float64)
    absolute_thetas = np.zeros(num_values)
    thresholds = np.zeros(num_values)
    thetas = np.zeros(num_values)
    grouping_ids = np.zeros(num_values)

    current_theta = target_column_values[0]
    thetas[0] = current_theta
    absolute_thetas[0] = np.abs(current_theta)
    current_grouping_id = 0
    grouping_ids[0] = current_grouping_id

    time_deltas = []
    times = []
    previous_time = 0
    expected_ticks = initial_expected_ticks
    expected_bar_value = initial_bar_size

    start_time = time.time()

    for i in range(1, num_values):
        current_theta += target_column_values[i]
        thetas[i] = current_theta
        absolute_theta = np.abs(current_theta)
        absolute_thetas[i] = absolute_theta

        threshold = expected_ticks * expected_bar_value
        thresholds[i] = threshold
        grouping_ids[i] = current_grouping_id

        if absolute_theta >= threshold:
            current_grouping_id += 1
            current_theta = 0
            time_delta = np.float64(i - previous_time)
            time_deltas.append(time_delta)
            times.append(i)
            previous_time = i
            
            # --- SUGGESTION 1 (Continued) ---
            # Call the correct `ewma` function and use the correct `window` parameter.
            # Note: The logic of recalculating the EWMA over the full history
            # inside the loop is O(N^2) but may be the intended design.
            # This change just fixes the function being called.
            expected_ticks = ewma(
                np.array(time_deltas), window=len(time_deltas)
            )[-1]
            
            expected_bar_value = np.abs(
                ewma(
                    target_column_values[:i], window=initial_expected_ticks
                )[-1]
            )

        progress_bar(i, num_values, start_time)

    return time_deltas, absolute_thetas, thresholds, times, thetas, grouping_ids


def create_ohlcv_dataframe(
    tick_data_grouped: pd.core.groupby.DataFrameGroupBy
) -> pd.DataFrame:
    """
    Takes a grouped DataFrame and creates a new one with OHLCV data and other relevant information.

    :param tick_data_grouped: Grouped DataFrame based on some criteria (e.g., time).
    :type tick_data_grouped: pd.core.groupby.DataFrameGroupBy
    :return: A DataFrame containing OHLCV data and other relevant information.
    :rtype: pd.DataFrame
    """
    
    # --- SUGGESTION 2: Vectorized Approach ---
    # This is much faster than using .apply()
    
    ohlc = tick_data_grouped['price'].ohlc()
    volume = tick_data_grouped['size'].sum()
    
    # Calculate VWAP (Value of Trades)
    value = (tick_data_grouped['price'] * tick_data_grouped['size']).sum()
    
    ohlc['volume'] = volume
    ohlc['value_of_trades'] = value / volume  # This is the VWAP
    ohlc['price_mean'] = tick_data_grouped['price'].mean()
    ohlc['tick_count'] = tick_data_grouped['price'].count()
    
    # Handle potential 0-volume bars to avoid NaN
    ohlc['value_of_trades'] = ohlc['value_of_truths'].fillna(0)
    
    ohlc['price_mean_log_return'] = np.log(ohlc['price_mean']) - np.log(ohlc['price_mean'].shift(1))

    return ohlc