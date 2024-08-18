import pandas as pd
import numpy as np 
import multiprocessing as mp
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
import datetime
import time
import sys
from typing import List


def cusum_filter_events_dynamic_threshold(
        prices: pd.Series,
        threshold: pd.Series
) -> pd.DatetimeIndex:
    """
    Detect events using the Symmetric Cumulative Sum (CUSUM) filter.

    The Symmetric CUSUM filter is a change-point detection algorithm used to identify events where the price difference
    exceeds a predefined threshold.

    :param prices: A pandas Series of prices.
    :param threshold: A pandas Series containing the predefined threshold values for event detection.
    :return: A pandas DatetimeIndex containing timestamps of detected events.

    References:
    - De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons. (Methodology: 39)
    """
    time_events, shift_positive, shift_negative = [], 0, 0
    price_delta = prices.diff().dropna()
    thresholds = threshold.copy()
    price_delta, thresholds = price_delta.align(thresholds, join="inner", copy=False)

    for (index, value), threshold_ in zip(price_delta.to_dict().items(), thresholds.to_dict().values()):
        shift_positive = max(0, shift_positive + value)
        shift_negative = min(0, shift_negative + value)

        if shift_negative < -threshold_:
            shift_negative = 0
            time_events.append(index)

        elif shift_positive > threshold_:
            shift_positive = 0
            time_events.append(index)

    return pd.DatetimeIndex(time_events)

def symmetric_cusum_filter(
        prices: pd.Series,
        threshold: float) -> pd.DatetimeIndex:
    """
    Implements the symmetric CUSUM filter.

    The symmetric CUSUM filter is a change-point detection algorithm used to identify events where the price difference exceeds a predefined threshold.

    :param prices: A pandas Series of prices.
    :param threshold: The predefined threshold for detecting events.
    :return: A pandas DatetimeIndex of event timestamps.

    References:
    - De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons. (Methodology: 39)
    """
    time_events, shift_positive, shift_negative = [], 0, 0
    price_delta = prices.diff()

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




def aggregate_ohlcv(tick_data_grouped) -> pd.DataFrame:
    """
    Aggregates tick data into OHLCV bars.

    :param tick_data_grouped: A pandas GroupBy object of tick data.
    :return: A pandas DataFrame with OHLCV bars.
    """
    ohlc = tick_data_grouped['price'].ohlc()
    ohlc['volume'] = tick_data_grouped['size'].sum()
    ohlc['value_of_trades'] = tick_data_grouped.apply(
        lambda x: (x['price'] * x['size']).sum() / x['size'].sum())
    ohlc['price_mean'] = tick_data_grouped['price'].mean()
    ohlc['tick_count'] = tick_data_grouped['price'].count()
    ohlc['price_mean_log_return'] = np.log(ohlc['price_mean']) - np.log(ohlc['price_mean'].shift(1))

    return ohlc


def generate_time_bars(
        tick_data: pd.DataFrame,
        frequency: str = "5Min") -> pd.DataFrame:
    """
    Generates time bars from tick data.

    :param tick_data: A pandas DataFrame of tick data.
    :param frequency: The frequency for time bar aggregation.
    :return: A pandas DataFrame with time bars.
    """
    tick_data_grouped = tick_data.groupby(pd.Grouper(freq=frequency))
    ohlcv_dataframe = aggregate_ohlcv(tick_data_grouped)

    return ohlcv_dataframe


def compute_daily_volatility(
        close: pd.Series,
        span: int = 63) -> pd.DataFrame:
    """
    Computes the daily volatility at intraday estimation points.

    :param close: A pandas Series of close prices.
    :param span: The span parameter for the EWMA.
    :return: A pandas DataFrame with returns and volatilities.

    References:
    - De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons. (Methodology: Page 44)
    """
    df = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df = df[df > 0]
    df = pd.Series(close.index[df - 1], index=close.index[close.shape[0] - df.shape[0]:])
    returns = (close.loc[df.index] / close.loc[df.values].values - 1).rename("rets")
    stds = returns.ewm(span=span).std().rename("std")

    return pd.concat([returns, stds], axis=1)

def daily_volatility_with_log_returns(
        close: pd.Series,
        span: int = 100
) -> pd.Series:
    """
    Calculate the daily volatility at intraday estimation points using Exponentially Weighted Moving Average (EWMA).

    :param close: A pandas Series of daily close prices.
    :param span: The span parameter for the Exponentially Weighted Moving Average (EWMA).
    :return: A pandas Series containing daily volatilities.

    References:
    - De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons. (Methodology: Page 44)
    """
    df = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df = df[df > 0]
    df = pd.Series(close.index[df - 1], index=close.index[close.shape[0] - df.shape[0]:])
    returns = np.log(close.loc[df.index] / close.loc[df.values].values)
    stds = returns.ewm(span=span).std().rename("std")

    return stds

def triple_barrier(
    close: pd.Series,
    events: pd.DataFrame,
    profit_taking_stop_loss: list[float, float],
    molecule: list
) -> pd.DataFrame:
    # Filter molecule to ensure all timestamps exist in events
    molecule = [m for m in molecule if m in events.index]

    # Continue with the existing logic
    events_filtered = events.loc[molecule]
    output = events_filtered[['End Time']].copy(deep=True)

    if profit_taking_stop_loss[0] > 0:
        profit_taking = profit_taking_stop_loss[0] * events_filtered['Base Width']
    else:
        profit_taking = pd.Series(index=events.index)

    if profit_taking_stop_loss[1] > 0:
        stop_loss = -profit_taking_stop_loss[1] * events_filtered['Base Width']
    else:
        stop_loss = pd.Series(index=events.index)

    for location, timestamp in events_filtered['End Time'].fillna(close.index[-1]).items():
        df = close[location:timestamp]
        df = np.log(df / close[location]) * events_filtered.at[location, 'Side']
        output.loc[location, 'stop_loss'] = df[df < stop_loss[location]].index.min()
        output.loc[location, 'profit_taking'] = df[df > profit_taking[location]].index.min()

    return output

def get_barrier_touch_time(close: pd.Series, 
                          time_events: pd.DatetimeIndex, 
                          ptsl: float, 
                          target: pd.Series, 
                          return_min: float, 
                          num_threads: int, 
                          timestamp: pd.Series = False) -> pd.DataFrame:
    """
    Finds the time of the first barrier touch.
    
    :param close: A dataframe of dates and close prices.
    :param time_events: A pandas time index containing the timestamps that will seed every triple barrier.
    :param ptsl: A non-negative float that sets the width of the two barriers.
    :param target: A pandas series of targets, expressed in terms of absolute returns.
    :param return_min: The minimum target return required for running a triple barrier search.
    :param num_threads: The number of threads.
    :param timestamp: A pandas series with the timestamps of the vertical barriers (False when disabled).
    :return: A dataframe with timestamp of the vertical barrier and unit width of the horizontal barriers.
    """
    target = target.loc[time_events]
    target = target[target > return_min]

    if timestamp is False:
        timestamp = pd.Series(pd.NaT, index=time_events)

    side_position = pd.Series(1., index=target.index)
    events = pd.concat({'timestamp': timestamp, 'target': target, 'side': side_position}, axis=1).dropna(subset=['target'])
    
    with ProcessPoolExecutor(num_threads) as executor:
        dataframe = list(executor.map(triple_barrier, [(close, events.loc[molecule], [ptsl, ptsl]) for molecule in np.array_split(events.index, num_threads)]))
    dataframe = pd.concat(dataframe, axis=0)
    
    events['timestamp'] = dataframe.dropna(how='all').min(axis=1)
    events = events.drop('side', axis=1)
    return events


def vertical_barrier(
    close: pd.Series,
    time_events: pd.DatetimeIndex,
    number_days: int
) -> pd.Series:
    """
    Shows one way to define a vertical barrier.

    :param close: A dataframe of prices and dates.
    :param time_events: A vector of timestamps.
    :param number_days: A number of days for the vertical barrier.
    :return: A pandas series with the timestamps of the vertical barriers.
    """
    timestamp_array = close.index.searchsorted(time_events + pd.Timedelta(days=number_days))
    timestamp_array = timestamp_array[timestamp_array < close.shape[0]]
    timestamp_array = pd.Series(close.index[timestamp_array], index=time_events[:timestamp_array.shape[0]])
    return timestamp_array


def get_labels(events: pd.DataFrame, 
               close: pd.Series) -> pd.DataFrame:
    """
    Label the observations.
    
    :param events: A dataframe with timestamp of the vertical barrier and unit width of the horizontal barriers.
    :param close: A dataframe of dates and close prices.
    :return: A dataframe with the return realized at the time of the first touched barrier and the label.
    """
    events_filtered = events.dropna(subset=['timestamp'])
    all_dates = events_filtered.index.union(events_filtered['timestamp'].values).drop_duplicates()
    close_filtered = close.reindex(all_dates, method='bfill')
    
    out = pd.DataFrame(index=events_filtered.index)
    out['ret'] = close_filtered.loc[events_filtered['timestamp'].values].values / close_filtered.loc[events_filtered.index] - 1
    out['bin'] = np.sign(out['ret'])
    return out

def meta_events(
    close: pd.Series,
    time_events: pd.DatetimeIndex,
    ptsl: List[float],
    target: pd.Series,
    return_min: float,
    num_threads: int,
    timestamp: pd.Series = False,
    side: pd.Series = None
) -> pd.DataFrame:
    # Filter target by time_events and return_min
    target = target.loc[time_events]
    target = target[target > return_min]

    # Ensure timestamp is correctly initialized
    if timestamp is False:
        timestamp = pd.Series(pd.NaT, index=time_events)
    else:
        timestamp = timestamp.loc[time_events]

    if side is None:
        side_position, profit_loss = pd.Series(1., index=target.index), [ptsl[0], ptsl[0]]
    else:
        side_position, profit_loss = side.loc[target.index], ptsl[:2]

    # Include 'target' and 'timestamp' in the events DataFrame
    events = pd.concat({'End Time': timestamp, 'Base Width': target, 'Side': side_position, 'target': target, 'timestamp': timestamp}, axis=1).dropna(subset=['Base Width'])

    if num_threads > 1:
        with ProcessPoolExecutor(max_workers=num_threads) as executor:
            df0 = list(executor.map(
                triple_barrier,
                [close] * num_threads,
                [events] * num_threads,
                [profit_loss] * num_threads,
                np.array_split(time_events, num_threads)
            ))
    else:
        df0 = list(map(
            triple_barrier,
            [close] * num_threads,
            [events] * num_threads,
            [profit_loss] * num_threads,
            np.array_split(time_events, num_threads)
        ))        
    df0 = pd.concat(df0, axis=0)

    events['End Time'] = df0.dropna(how='all').min(axis=1)

    if side is None:
        events = events.drop('Side', axis=1)
    
    # Return events including the 'target' and 'timestamp' columns
    return events

def meta_labeling(
    events: pd.DataFrame,
    close: pd.Series
) -> pd.DataFrame:
    """
    Expands label to incorporate meta-labeling.

    :param events: DataFrame with timestamp of vertical barrier and unit width of the horizontal barriers.
    :param close: Series of close prices with date indices.
    :return: DataFrame containing the return and binary labels for each event.

    Reference:
    De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    Methodology: 51
    """
    events_filtered = events.dropna(subset=['End Time'])
    all_dates = events_filtered.index.union(events_filtered['End Time'].values).drop_duplicates()
    close_filtered = close.reindex(all_dates, method='bfill')
    out = pd.DataFrame(index=events_filtered.index)
    out['End Time'] = events['End Time']
    out['Return'] = close_filtered.loc[events_filtered['End Time'].values].values / close_filtered.loc[events_filtered.index] - 1
    if 'Side' in events_filtered:
        out['Return'] *= events_filtered['Side']
    out['Label'] = np.sign(out['Return'])
    if 'Side' in events_filtered:
        out.loc[out['Return'] <= 0, 'Label'] = 0
        out['Side'] = events_filtered['Side']
    return out

def drop_label(
        events: pd.DataFrame,
        percent_min: float = .05) -> pd.DataFrame:
    """
    Presents a procedure that recursively drops observations associated with extremely rare labels.

    :param events: DataFrame with columns: Dates, ret, and bin.
    :param percent_min: Minimum percentage.
    :return: DataFrame with the updated events.

    Reference:
    De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    Methodology: 54
    """
    while True:
        dataframe = events['bin'].value_counts(normalize=True)
        if dataframe.min() > percent_min or dataframe.shape[0] < 3:
            break
        print('dropped label', dataframe.idxmin(), dataframe.min())
        events = events[events['bin'] != dataframe.idxmin()]
    return events


def lin_parts(num_atoms: int, num_threads: int) -> np.ndarray:
    """
    Partition of atoms with a single loop.

    :param num_atoms: Total number of atoms.
    :param num_threads: Number of threads for parallel processing.
    :return: Numpy array with partition indices.
    """
    parts = np.linspace(0, num_atoms, min(num_threads, num_atoms) + 1)
    parts = np.ceil(parts).astype(int)
    return parts


def nested_parts(num_atoms: int, num_threads: int, upper_triang: bool = False) -> np.ndarray:
    """
    Partition of atoms with an inner loop.

    :param num_atoms: Total number of atoms.
    :param num_threads: Number of threads for parallel processing.
    :param upper_triang: Whether the first rows are the heaviest.
    :return: Numpy array with partition indices.
    """
    parts, num_threads_ = [0], min(num_threads, num_atoms)
    for num in range(num_threads_):
        part = 1 + 4 * (parts[-1] ** 2 + parts[-1] + num_atoms * (num_atoms + 1.) / num_threads_)
        part = (-1 + part ** .5) / 2.
        parts.append(part)
    parts = np.round(parts).astype(int)
    if upper_triang:
        parts = np.cumsum(np.diff(parts)[::-1])
        parts = np.append(np.array([0]), parts)
    return parts


def mp_pandas_obj(
        func,
        pd_obj,
        num_threads: int = 24,
        mp_batches: int = 1,
        lin_mols: bool = True,
        **kargs) -> pd.DataFrame:
    """
    Parallelize jobs, return a DataFrame or Series.

    :param func: Function to be parallelized.
    :param pd_obj: Tuple with argument name for the molecule and list of atoms grouped into molecules.
    :param num_threads: Number of threads for parallel processing.
    :param mp_batches: Number of multi-processing batches.
    :param lin_mols: Whether to use linear molecule partitioning.
    :param kargs: Any other arguments needed by func.
    :return: DataFrame with the results of the parallelized function.

    Example:
    df1 = mp_pandas_obj(func, ('molecule', df0.index), 24, **kargs)
    """
    arg_list = list(kargs.values())
    if lin_mols:
        parts = lin_parts(len(arg_list[1]), num_threads * mp_batches)
    else:
        parts = nested_parts(len(arg_list[1]), num_threads * mp_batches)
    jobs = []
    for i in range(1, len(parts)):
        job = {pd_obj[0]: pd_obj[1][parts[i - 1]:parts[i]], 'func': func}
        job.update(kargs)
        jobs.append(job)
    if num_threads == 1:
        out = process_jobs_(jobs)
    else:
        out = process_jobs(jobs, num_threads=num_threads)
    if isinstance(out[0], pd.DataFrame):
        df0 = pd.DataFrame()
    elif isinstance(out[0], pd.Series):
        df0 = pd.Series()
    else:
        return out
    for i in out:
        df0 = df0.append(i)
    df0 = df0.sort_index()
    return df0


def process_jobs_(jobs: list) -> list:
    """
    Run jobs sequentially, for debugging.

    :param jobs: List of jobs to be processed.
    :return: List of job results.
    """
    out = []
    for job in jobs:
        out_ = expand_call(job)
        out.append(out_)
    return out


def report_progress(job_num: int, num_jobs: int, time0: float, task: str) -> None:
    """
    Report progress as asynchronous jobs are completed.

    :param job_num: Current job number.
    :param num_jobs: Total number of jobs.
    :param time0: Start time.
    :param task: Task name.
    :return: None
    """
    msg = [float(job_num) / num_jobs, (time.time() - time0) / 60.]
    msg.append(msg[1] * (1 / msg[0] - 1))
    timestamp = str(datetime.datetime.fromtimestamp(time.time()))
    msg = timestamp + ' ' + str(round(msg[0] * 100, 2)) + '% ' + task + ' done after ' + str(
        round(msg[1], 2)) + ' minutes. Remaining ' + str(round(msg[2], 2)) + ' minutes.'
    if job_num < num_jobs:
        sys.stderr.write(msg + '\r')
    else:
        sys.stderr.write(msg + '\n')
    return


def process_jobs(jobs: list, task: str = None, num_threads: int = 24) -> list:
    """
    Run jobs in parallel in multiple threads.

    :param jobs: List of jobs to be processed.
    :param task: Task name for progress reporting.
    :param num_threads: Number of threads for parallel processing.
    :return: List of job results.
    """
    if task is None:
        task = jobs[0]['func'].__name__
    pool = mp.Pool(processes=num_threads)
    outputs, out, time0 = pool.imap_unordered(expand_call, jobs), [], time.time()
    # Process asyn output, report progress
    for i, out_ in enumerate(outputs, 1):
        out.append(out_)
        report_progress(i, len(jobs), time0, task)
    pool.close()
    pool.join()  # this is needed to prevent memory leaks
    return out


def expand_call(kargs):
    """
    Expand the arguments of a callback function, kargs['func'].

    :param kargs: Dictionary with the function to call and the arguments to pass.
    :return: Result of the function call.
    """
    func = kargs['func']
    del kargs['func']
    out = func(**kargs)
    return out
