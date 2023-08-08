# import pandas as pd
# import numpy as np 
# import datetime
# import time
# import sys
# from scipy import stats
# from statsmodels.stats import stattools

# """
#   function: shows the progress bar
#   reference: n/a
#   methodology: n/a
# """
# def progressBar(value, # value of an event
#                 endValue, # length of that event
#                 startTime, # start time of the event
#                 barLength = 20): # length of bar

#     percent = float(value)/endValue # progress in percent
#     arrow = '-'*int(round(percent*barLength) - 1) + '>' # set the arrow
#     spaces = ' '*(barLength - len(arrow)) # show spaces
#     remaining = int(((time.time() - startTime)/value)*(endValue - value)/60) 
#     # calculating remaining time to finish
#     sys.stdout.write("\rCompleted: [{0}] {1}% - {2} minutes remaining.".format(
#                      arrow + spaces, int(round(percent*100)), remaining)) # print state of the progress
#     sys.stdout.flush() # release stdout

# """
#   function: computes the ewma, ewma var, and ewma stds
#   reference: https://stackoverflow.com/questions/40754262/pandas-ewm-std-calculation
#   methodology: n/a
# """
# def ewma(input, # input time series array 
#          windowLength): # window for exponential weighted moving average
#   N = input.shape[0] # length of array
#   outputEwma = np.empty(N, dtype = 'float64') # array for output 
#   omega = 1 # initial weight
#   ALPHA = 2/float(windowLength + 1) # tuning parameter for outputEwma
#   thisInputValue = input[0] # initialize first value of outputEwma
#   outputEwma[0] = thisInputValue
#   for i in range(1, N):
#       omega += (1 - ALPHA)**i # updating weight based on α and i
#       thisInputValue = thisInputValue*(1 - ALPHA) + input[i]  # update thisInputValue
#       outputEwma[i] = thisInputValue/omega # update outputEwma
#   return outputEwma

# """
#   function: grouping dataframe based on a feature and then calculates thresholds
#   reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
#   methodology: n/a
# """
# def grouping(targetCol, # target column of tick dataframe
#              tickExpectedInit, # initial expected ticks
#              barSize): # initial expected size in each tick
#   timesDelta, times = [], []
#   timePrev, tickExpected, barExpectedValue  = 0, tickExpectedInit, barSize
#   N = targetCol.shape[0] # number of dates in dataframe
#   targetCol = targetCol.values.astype(np.float64)
#   thetasAbsolute, thresholds, thetas, groupingID = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
#   thetasAbsolute[0], thetaCurrent = np.abs(targetCol[0]), targetCol[0]  # set initial value of θ and |θ|
#   t = time.time() # value of time from 1970 in ms
#   groupingIDcurrent = 0 # set first group_Id to 0
#   for i in range(1, N):
#     thetaCurrent += targetCol[i] # update current_θ by adding next value of target
#     thetas[i] = thetaCurrent # update θs
#     thetaAbsolute = np.abs(thetaCurrent) # absolute value of current_θ
#     thetasAbsolute[i] = thetaAbsolute  # update abs_θs
#     threshold = tickExpected*barExpectedValue # multiply expected ticks and expected value of target to calculating threshold
#     thresholds[i] = threshold   # update thresholds
#     groupingID[i] = groupingIDcurrent # update group_Id
#     # this stage is for going to next group_Id and resetting parameters
#     if thetaAbsolute >= threshold:
#       groupingIDcurrent += 1
#       thetaCurrent = 0
#       timesDelta.append(np.float64(i - timePrev)) # append the length of time values that took untill passing threshold
#       times.append(i) # append the number of time value that we passed threshold in it
#       timePrev = i
#       tickExpected = ewma(np.array(timesDelta), windowLength=np.int64(len(timesDelta)))[-1] # update expected ticks with ewma
#       barExpectedValue = np.abs(ewma(targetCol[:i], windowLength=np.int64(tickExpectedInit*1))[-1]) # update expected value of b with ewma
#     progressBar(i, N, t) # show progress bar
#   return timesDelta, thetasAbsolute, thresholds, times, thetas, groupingID

import pandas as pd
import numpy as np
import datetime
import time
import sys
from scipy import stats
from statsmodels.stats import stattools

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
    
    sys.stdout.write("\rCompleted: [{0}] {1}% - {2} minutes remaining.".format(
        arrow + spaces, int(round(percent * 100)), remaining))
    sys.stdout.flush()

def compute_ewma(
    input_array: np.ndarray,
    window_length: int
) -> np.ndarray:
    """
    Computes the Exponentially Weighted Moving Average (EWMA).

    :param input_array: The input time series array.
    :param window_length: Window length for the EWMA.
    :return: The EWMA values.

    The EWMA formula for a series \( x \) is given by:
    .. math::

        EWMA_i = \frac{x_i + (1 - \alpha) x_{i-1} + (1 - \alpha)^2 x_{i-2} + ...}{\omega}

    Where:
    \( \alpha \) is the decay factor given by \( \frac{2} {window\_length + 1} \)
    \( \omega \) is the weight and is computed iteratively using the formula:
    .. math::

        \omega = \omega + (1 - \alpha)^i
    """
    N = input_array.shape[0]
    output_ewma = np.empty(N, dtype='float64')
    omega = 1
    ALPHA = 2 / float(window_length + 1)
    current_value = input_array[0]
    output_ewma[0] = current_value
    
    for i in range(1, N):
        omega += (1 - ALPHA) ** i
        current_value = current_value * (1 - ALPHA) + input_array[i]
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
            expected_ticks = compute_ewma(np.array(times_delta), window_length=np.int64(len(times_delta)))[-1]
            expected_bar_value = np.abs(compute_ewma(target_col[:i], window_length=np.int64(initial_expected_ticks))[0])
            
        progress_bar(i, N, start_time)
        
    return times_delta, thetas_absolute, thresholds, times, thetas, grouping_id




def generate_information_driven_bars(
        tick_data: pd.DataFrame,
        bar_type: str = "volume",
        tick_expected_initial: int = 2000
) -> (pd.DataFrame, np.array, np.array):
    """
    Implements Information-Driven Bars as per the methodology described in 
    "Advances in financial machine learning" by De Prado (2018).

    :param tick_data: DataFrame of tick data.
    :param bar_type: Type of the bars, options: "tick", "volume", "dollar".
    :param tick_expected_initial: Initial expected ticks value.
    :return: A tuple containing the OHLCV DataFrame, thetas absolute array, and thresholds array.

    .. note:: 
        The function is based on methodology 29 from the mentioned reference.

    .. math:: 
        \text{bar_expected_value} = |\text{input_data.mean()}|
    """
    if bar_type == "volume":
        input_data = tick_data['volume_labeled']
    elif bar_type == "tick":
        input_data = tick_data['label']
    elif bar_type == "dollar":
        input_data = tick_data['dollars_labeled']
    else:
        raise ValueError("Invalid bar_type provided. Choose among 'tick', 'volume', 'dollar'.")

    bar_expected_value = np.abs(input_data.mean())
    
    # Here, I assume that you have defined the 'grouping' function elsewhere in your code.
    times_delta, thetas_absolute, thresholds, _, _, grouping_id = compute_grouping(input_data, tick_expected_initial, bar_expected_value)
    
    tick_grouped = tick_data.reset_index().assign(grouping_id=grouping_id)
    dates = tick_grouped.groupby('grouping_id', as_index=False).first()['dates']

    # Here, I assume that you have defined the 'ohlcv' function elsewhere in your code.
    ohlcv_dataframe = ohlcv(tick_grouped.groupby('grouping_id'))
    ohlcv_dataframe.set_index(dates, drop=True, inplace=True)

    return ohlcv_dataframe, thetas_absolute, thresholds

# """
#   function: implements Information-Driven Bars
#   reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
#   methodology: 29
# """
# def infoBar(tickData, # dataframe of tick data
#             type = "volume", # User can choose between "tick", "volume" or "dollar" imbalanced bars
#             tickExpectedInit = 2000): # The value of tickExpectedInit
#   if type == "volume":
#     inputData = tickData['volumelabeled'] # use volume column with sign of log return in same day
#   elif type == "tick":
#     inputData = tickData['label'] # use sign of log return column
#   elif type == "dollar":
#     inputData = tickData['dollarslabeled'] # use the value of price*volume with sign of log return
#   else:
#     print("Error")
#   barExpectedValue = np.abs(inputData.mean()) # expected value of b 
#   timesDelta, thetasAbsolute, thresholds, times, thetas, groupingID = grouping(inputData, tickExpectedInit, barExpectedValue) # calculate thresholds
#   tickGrouped = tickData.reset_index().assign(groupingID = groupingID) # group based on groupingID
#   dates = tickGrouped.groupby('groupingID', as_index = False).first()['dates']
#   tickDataGrouped = tickGrouped.groupby('groupingID') 
#   ohlcvDataframe = ohlcv(tickDataGrouped) # create a dataframe based on bars
#   ohlcvDataframe.set_index(dates, drop = True, inplace = True)  # set index
#   return ohlcvDataframe, thetasAbsolute, thresholds

# """
#   function: Takes grouped dataframe, combining and creating the new one with info. about prices and volume.
#   reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
#   methodology: n/a
# """
# def ohlcv(tickDataGrouped): #grouped dataframes
#   ohlc = tickDataGrouped['price'].ohlc() # find price in each tick
#   ohlc['volume'] = tickDataGrouped['size'].sum() # find volume traded
#   ohlc['ValueOfTrades'] = tickDataGrouped.apply(lambda x:(x['price']*x['size']).sum()/x['size'].sum()) # find value of trades
#   ohlc['PriceMean'] = tickDataGrouped['price'].mean() # mean of price
#   ohlc['TickCount'] = tickDataGrouped['price'].count()# number of ticks
#   ohlc['PriceMeanLogReturn'] = np.log(ohlc['PriceMean']) - np.log(ohlc['PriceMean'].shift(1)) # find log return
#   return ohlc


# import pandas as pd
# import numpy as np


def ohlcv(
        tick_data_grouped: pd.core.groupby.generic.DataFrameGroupBy
) -> pd.DataFrame:
    """
    Computes various statistics for the grouped tick data.

    Takes a grouped dataframe, combines the data, and creates a new one with information about prices, volume, and other
    statistics. This is typically used in the context of financial tick data to generate OHLCV data (Open, High, Low, Close, Volume).

    :param tick_data_grouped: Grouped DataFrame containing tick data.
    :return: A DataFrame containing OHLCV data and other derived statistics.

    .. note:: 
        The methodology is based on practices from "Advances in financial machine learning" by De Prado (2018).
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


# """
#   function: Takes dataframe and generating time bar dataframe
#   reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
#   methodology: n/a
# """
# def timeBar(tickData, # dataframe of tick data
#             frequency = "5Min"): # frequency for rounding date time
#   tickDataGrouped = tickData.groupby(pd.Grouper(freq = frequency)) # group data sets based on time freq
#   ohlcvDataframe = ohlcv(tickDataGrouped) # create a dataframe based on time bars with frequency freq
#   return ohlcvDataframe


# """
#   function: Takes dataframe and generating tick bar dataframe
#   reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
#   methodology: n/a
# """
# def tickBar(tickData, # dataframe of tick data
#             tickPerBar = 10,  # number of ticks in each bar
#             numberBars = None): # number of bars
#   # if tickPerBar is not mentioned, then calculate it with number of all ticks divided by number of bars
#   if not tickPerBar:
#     tickPerBar = int(tickData.shape[0]/numberBars)
#   tickGrouped = tickData.reset_index().assign(groupingID = lambda x: x.index//tickPerBar)
#   dates = tickGrouped.groupby('groupingID', as_index = False).first()['dates'] # group data sets based on ticks per bar
#   tickDataGrouped = tickGrouped.groupby('groupingID')
#   ohlcvDataframe = ohlcv(tickDataGrouped) # create a dataframe based on tick bars
#   ohlcvDataframe.set_index(dates, drop = True, inplace = True) # set dates column as index
#   return ohlcvDataframe

# """
#   function: Takes dataframe and generating volume bar dataframe
#   reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
#   methodology: n/a
# """
# def volumeBar(tickData, # dataframe of tick data
#               volumePerBar = 10000,   # volumes in each bar
#               numberBars = None): # number of bars
#   tickData['volumecumulated'] = tickData['size'].cumsum()  # cumulative sum of size
#   # if volumePerBar is not mentioned, then calculate it with all volumes divided by number of bars
#   if not volumePerBar:
#     volumeTotal = tickData['volumecumulated'].values[-1]
#     volumePerBar = volumeTotal/numberBars
#     volumePerBar = round(volumePerBar, -2) # round to the nearest hundred
#   tickGrouped = tickData.reset_index().assign(groupingID = lambda x: x.volumecumulated//volumePerBar)
#   dates = tickGrouped.groupby('groupingID', as_index = False).first()['dates']  # groupe date times based on volume
#   tickDataGrouped = tickGrouped.groupby('groupingID')
#   ohlcvDataframe = ohlcv(tickDataGrouped)  # create a dataframe based on tick bars
#   ohlcvDataframe.set_index(dates, drop = True, inplace = True)  # set dates column as index
#   return ohlcvDataframe


import pandas as pd

def generate_time_bar(
        tick_data: pd.DataFrame,
        frequency: str = "5Min"
) -> pd.DataFrame:
    """
    Generates time bars for tick data.

    This function groups tick data by a specified time frequency and then computes OHLCV (Open, High, Low, Close, Volume) statistics.

    :param tick_data: DataFrame containing tick data.
    :param frequency: Time frequency for rounding datetime.
    :return: A DataFrame containing OHLCV data grouped by time.

    .. note::
        The methodology is based on practices from "Advances in financial machine learning" by De Prado (2018).
    """
    tick_data_grouped = tick_data.groupby(pd.Grouper(freq=frequency))
    ohlcv_dataframe = generate_ohlcv_data(tick_data_grouped)
    return ohlcv_dataframe


def generate_tick_bar(
        tick_data: pd.DataFrame,
        ticks_per_bar: int = 10,
        number_bars: int = None
) -> pd.DataFrame:
    """
    Generates tick bars for tick data.

    This function groups tick data by a specified number of ticks and then computes OHLCV statistics.

    :param tick_data: DataFrame containing tick data.
    :param ticks_per_bar: Number of ticks in each bar.
    :param number_bars: Number of bars to generate.
    :return: A DataFrame containing OHLCV data grouped by tick count.

    .. note::
        The methodology is based on practices from "Advances in financial machine learning" by De Prado (2018).
    """
    if not ticks_per_bar:
        ticks_per_bar = int(tick_data.shape[0] / number_bars)

    tick_grouped = tick_data.reset_index().assign(grouping_id=lambda x: x.index // ticks_per_bar)
    dates = tick_grouped.groupby('grouping_id', as_index=False).first()['dates']
    tick_data_grouped = tick_grouped.groupby('grouping_id')
    ohlcv_dataframe = generate_ohlcv_data(tick_data_grouped)
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

    .. note::
        The methodology is based on practices from "Advances in financial machine learning" by De Prado (2018).
    """
    tick_data['volume_cumulated'] = tick_data['size'].cumsum()

    if not volume_per_bar:
        volume_total = tick_data['volume_cumulated'].values[-1]
        volume_per_bar = volume_total / number_bars
        volume_per_bar = round(volume_per_bar, -2)

    tick_grouped = tick_data.reset_index().assign(grouping_id=lambda x: x.volumecumulated // volume_per_bar)
    dates = tick_grouped.groupby('grouping_id', as_index=False).first()['dates']
    tick_data_grouped = tick_grouped.groupby('grouping_id')
    ohlcv_dataframe = generate_ohlcv_data(tick_data_grouped)
    ohlcv_dataframe.set_index(dates, drop=True, inplace=True)

    return ohlcv_dataframe


# """
#   function: Takes dataframe and generating volume bar dataframe
#   reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
#   methodology: n/a
# """
# def dollarBar(tickData, # dataframe of tick data
#               dollarPerBar = 100000,  # dollars in each bar
#               numberBars = None): # number of bars
#   tickData['dollar'] = tickData['price']*tickData['size'] # generate dollar column by multiplying price and size
#   tickData['DollarsCumulated'] = tickData['dollar'].cumsum()  # cumulative sum of dollars
#   # if volume_per_bar is not mentioned, then calculate it with dollars divided by number of bars
#   if not dollarPerBar:
#     dollarsTotal = tickData['DollarsCumulated'].values[-1]
#     dollarPerBar = dollarsTotal/numberBars
#     dollarPerBar = round(dollarPerBar, -2) # round to the nearest hundred
#   tickGrouped = tickData.reset_index().assign(groupingID=lambda x: x.DollarsCumulated//dollarPerBar)
#   dates = tickGrouped.groupby('groupingID', as_index = False).first()['dates'] # group date times based on dollars
#   tickDataGrouped = tickGrouped.groupby('groupingID')
#   ohlcvDataframe = ohlcv(tickDataGrouped) # create a dataframe based on tick bars
#   ohlcvDataframe.set_index(dates, drop = True, inplace = True)  # set dates column as index
#   return ohlcvDataframe


# """
#   function: Calculates hedging weights using cov, risk distribution(RiskDist) and σ
#   reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
#   methodology: 36
# """
# def PCAWeights(cov, #covariance matrix
#                riskDisturbution = None, # risk distribution
#                risktarget = 1.):  # risk target
#   eigenValues, eigenVectors = np.linalg.eigh(cov) # must be Hermitian
#   indices = eigenValues.argsort()[::-1] # arguments for sorting eigenValues descending
#   eigenValues, eigenVectors = eigenValues[indices], eigenVectors[:, indices] 
#   # if riskDisturbution is nothing, it will assume all risk must be allocated to the principal component with
#   # smallest eigenvalue, and the weights will be the last eigenvector re-scaled to match σ
#   if riskDisturbution is None:
#     riskDisturbution = np.zeros(cov.shape[0])
#     riskDisturbution[-1] = 1.
#   loads = risktarget*(riskDisturbution/eigenValues)**.5  # represent the allocation in the new (orthogonal) basis
#   weights = np.dot(eigenVectors, np.reshape(loads, (-1, 1))) # calculate weights
#   return weights


# """
#   function:  Implementation of the symmetric CUSUM filter
#   reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
#   methodology: 39
# """
# def events(input, # dataframe of prices and dates
#            threshold): # threshold
#   timeEvents, shiftPositive, shiftNegative = [], 0, 0
#   # dataframe with price differences
#   priceDelta = input.diff()
#   for i in priceDelta.index[1:]:
#     # compute shiftNegative/shiftPositive with min/max of 0 and delta prices in each day
#     shiftPositive = max(0, shiftPositive+priceDelta.loc[i]) # compare price diff with zero
#     shiftNegative = min(0, shiftNegative+priceDelta.loc[i]) # compare price diff with zero
#     if shiftNegative <- threshold:
#       shiftNegative = 0 # reset shiftNegative to 0
#       timeEvents.append(i) # append this time into timeEvents
#     elif shiftPositive > threshold:
#       shiftPositive = 0 # reset shiftPositive to 0
#       timeEvents.append(i) # append this time into timeEvents
#   return pd.DatetimeIndex(timeEvents)


# import numpy as np
# import pandas as pd

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

    .. note::
        The methodology is based on practices from "Advances in financial machine learning" by De Prado (2018).
    """
    tick_data['dollar'] = tick_data['price'] * tick_data['size']
    tick_data['dollars_cumulated'] = tick_data['dollar'].cumsum()

    if not dollar_per_bar:
        dollars_total = tick_data['dollars_cumulated'].values[-1]
        dollar_per_bar = dollars_total / number_bars
        dollar_per_bar = round(dollar_per_bar, -2)

    tick_grouped = tick_data.reset_index().assign(grouping_id=lambda x: x.dollars_cumulated // dollar_per_bar)
    dates = tick_grouped.groupby('grouping_id', as_index=False).first()['dates']
    tick_data_grouped = tick_grouped.groupby('grouping_id')
    ohlcv_dataframe = generate_ohlcv_data(tick_data_grouped)
    ohlcv_dataframe.set_index(dates, drop=True, inplace=True)

    return ohlcv_dataframe


def calculate_pca_weights(
        cov: np.ndarray,
        risk_distribution: np.ndarray = None,
        risk_target: float = 1.0
) -> np.ndarray:
    """
    Calculates hedging weights using covariance matrix, risk distribution, and risk target.

    :param cov: Covariance matrix.
    :param risk_distribution: Risk distribution vector.
    :param risk_target: Risk target value.
    :return: Weights.

    .. note::
        The methodology is based on practices from "Advances in financial machine learning" by De Prado (2018).
    """
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    indices = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[indices], eigenvectors[:, indices]

    if risk_distribution is None:
        risk_distribution = np.zeros(cov.shape[0])
        risk_distribution[-1] = 1.

    loads = risk_target * (risk_distribution / eigenvalues)**0.5
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

    .. note::
        The methodology is based on practices from "Advances in financial machine learning" by De Prado (2018).
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
