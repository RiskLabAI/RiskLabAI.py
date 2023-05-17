import pandas as pd
import numpy as np 
import datetime
import time
import sys
from scipy import stats
from statsmodels.stats import stattools

"""
function: shows the progress bar
reference: n/a
methodology: n/a
"""
def progressBar(value, # value of an event
                endValue, # length of that event
                startTime, # start time of the event
                barLength = 20): # length of bar

    percent = float(value)/endValue # progress in percent
    arrow = '-'*int(round(percent*barLength) - 1) + '>' # set the arrow
    spaces = ' '*(barLength - len(arrow)) # show spaces
    remaining = int(((time.time() - startTime)/value)*(endValue - value)/60) 
    # calculating remaining time to finish
    sys.stdout.write("\rCompleted: [{0}] {1}% - {2} minutes remaining.".format(
                     arrow + spaces, int(round(percent*100)), remaining)) # print state of the progress
    sys.stdout.flush() # release stdout

"""
function: computes the ewma, ewma var, and ewma stds
reference: https://stackoverflow.com/questions/40754262/pandas-ewm-std-calculation
methodology: n/a
"""
def ewma(input, # input time series array 
         windowLength): # window for exponential weighted moving average
  N = input.shape[0] # length of array
  outputEwma = np.empty(N, dtype = 'float64') # array for output 
  omega = 1 # initial weight
  ALPHA = 2/float(windowLength + 1) # tuning parameter for outputEwma
  thisInputValue = input[0] # initialize first value of outputEwma
  outputEwma[0] = thisInputValue
  for i in range(1, N):
      omega += (1 - ALPHA)**i # updating weight based on α and i
      thisInputValue = thisInputValue*(1 - ALPHA) + input[i]  # update thisInputValue
      outputEwma[i] = thisInputValue/omega # update outputEwma
  return outputEwma

"""
function: grouping dataframe based on a feature and then calculates thresholds
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: n/a
"""
def grouping(targetCol, # target column of tick dataframe
             tickExpectedInit, # initial expected ticks
             barSize): # initial expected size in each tick
  timesDelta, times = [], []
  timePrev, tickExpected, barExpectedValue  = 0, tickExpectedInit, barSize
  N = targetCol.shape[0] # number of dates in dataframe
  targetCol = targetCol.values.astype(np.float64)
  thetasAbsolute, thresholds, thetas, groupingID = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
  thetasAbsolute[0], thetaCurrent = np.abs(targetCol[0]), targetCol[0]  # set initial value of θ and |θ|
  t = time.time() # value of time from 1970 in ms
  groupingIDcurrent = 0 # set first group_Id to 0
  for i in range(1, N):
    thetaCurrent += targetCol[i] # update current_θ by adding next value of target
    thetas[i] = thetaCurrent # update θs
    thetaAbsolute = np.abs(thetaCurrent) # absolute value of current_θ
    thetasAbsolute[i] = thetaAbsolute  # update abs_θs
    threshold = tickExpected*barExpectedValue # multiply expected ticks and expected value of target to calculating threshold
    thresholds[i] = threshold   # update thresholds
    groupingID[i] = groupingIDcurrent # update group_Id
    # this stage is for going to next group_Id and resetting parameters
    if thetaAbsolute >= threshold:
      groupingIDcurrent += 1
      thetaCurrent = 0
      timesDelta.append(np.float64(i - timePrev)) # append the length of time values that took untill passing threshold
      times.append(i) # append the number of time value that we passed threshold in it
      timePrev = i
      tickExpected = ewma(np.array(timesDelta), windowLength=np.int64(len(timesDelta)))[-1] # update expected ticks with ewma
      barExpectedValue = np.abs(ewma(targetCol[:i], windowLength=np.int64(tickExpectedInit*1))[-1]) # update expected value of b with ewma
    progressBar(i, N, t) # show progress bar
  return timesDelta, thetasAbsolute, thresholds, times, thetas, groupingID

"""
function: implements Information-Driven Bars
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: 29
"""
def infoBar(tickData, # dataframe of tick data
            type = "volume", # User can choose between "tick", "volume" or "dollar" imbalanced bars
            tickExpectedInit = 2000): # The value of tickExpectedInit
  if type == "volume":
    inputData = tickData['volumelabeled'] # use volume column with sign of log return in same day
  elif type == "tick":
    inputData = tickData['label'] # use sign of log return column
  elif type == "dollar":
    inputData = tickData['dollarslabeled'] # use the value of price*volume with sign of log return
  else:
    print("Error")
  barExpectedValue = np.abs(inputData.mean()) # expected value of b 
  timesDelta, thetasAbsolute, thresholds, times, thetas, groupingID = grouping(inputData, tickExpectedInit, barExpectedValue) # calculate thresholds
  tickGrouped = tickData.reset_index().assign(groupingID = groupingID) # group based on groupingID
  dates = tickGrouped.groupby('groupingID', as_index = False).first()['dates']
  tickDataGrouped = tickGrouped.groupby('groupingID') 
  ohlcvDataframe = ohlcv(tickDataGrouped) # create a dataframe based on bars
  ohlcvDataframe.set_index(dates, drop = True, inplace = True)  # set index
  return ohlcvDataframe, thetasAbsolute, thresholds

"""
function: Takes grouped dataframe, combining and creating the new one with info. about prices and volume.
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: n/a
"""
def ohlcv(tickDataGrouped): #grouped dataframes
  ohlc = tickDataGrouped['price'].ohlc() # find price in each tick
  ohlc['volume'] = tickDataGrouped['size'].sum() # find volume traded
  ohlc['ValueOfTrades'] = tickDataGrouped.apply(lambda x:(x['price']*x['size']).sum()/x['size'].sum()) # find value of trades
  ohlc['PriceMean'] = tickDataGrouped['price'].mean() # mean of price
  ohlc['TickCount'] = tickDataGrouped['price'].count()# number of ticks
  ohlc['PriceMeanLogReturn'] = np.log(ohlc['PriceMean']) - np.log(ohlc['PriceMean'].shift(1)) # find log return
  return ohlc


"""
function: Takes dataframe and generating time bar dataframe
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: n/a
"""
def timeBar(tickData, # dataframe of tick data
            frequency = "5Min"): # frequency for rounding date time
  tickDataGrouped = tickData.groupby(pd.Grouper(freq = frequency)) # group data sets based on time freq
  ohlcvDataframe = ohlcv(tickDataGrouped) # create a dataframe based on time bars with frequency freq
  return ohlcvDataframe


"""
function: Takes dataframe and generating tick bar dataframe
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: n/a
"""
def tickBar(tickData, # dataframe of tick data
            tickPerBar = 10,  # number of ticks in each bar
            numberBars = None): # number of bars
  # if tickPerBar is not mentioned, then calculate it with number of all ticks divided by number of bars
  if not tickPerBar:
    tickPerBar = int(tickData.shape[0]/numberBars)
  tickGrouped = tickData.reset_index().assign(groupingID = lambda x: x.index//tickPerBar)
  dates = tickGrouped.groupby('groupingID', as_index = False).first()['dates'] # group data sets based on ticks per bar
  tickDataGrouped = tickGrouped.groupby('groupingID')
  ohlcvDataframe = ohlcv(tickDataGrouped) # create a dataframe based on tick bars
  ohlcvDataframe.set_index(dates, drop = True, inplace = True) # set dates column as index
  return ohlcvDataframe

"""
function: Takes dataframe and generating volume bar dataframe
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: n/a
"""
def volumeBar(tickData, # dataframe of tick data
              volumePerBar = 10000,   # volumes in each bar
              numberBars = None): # number of bars
  tickData['volumecumulated'] = tickData['size'].cumsum()  # cumulative sum of size
  # if volumePerBar is not mentioned, then calculate it with all volumes divided by number of bars
  if not volumePerBar:
    volumeTotal = tickData['volumecumulated'].values[-1]
    volumePerBar = volumeTotal/numberBars
    volumePerBar = round(volumePerBar, -2) # round to the nearest hundred
  tickGrouped = tickData.reset_index().assign(groupingID = lambda x: x.volumecumulated//volumePerBar)
  dates = tickGrouped.groupby('groupingID', as_index = False).first()['dates']  # groupe date times based on volume
  tickDataGrouped = tickGrouped.groupby('groupingID')
  ohlcvDataframe = ohlcv(tickDataGrouped)  # create a dataframe based on tick bars
  ohlcvDataframe.set_index(dates, drop = True, inplace = True)  # set dates column as index
  return ohlcvDataframe

"""
function: Takes dataframe and generating volume bar dataframe
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: n/a
"""
def dollarBar(tickData, # dataframe of tick data
              dollarPerBar = 100000,  # dollars in each bar
              numberBars = None): # number of bars
  tickData['dollar'] = tickData['price']*tickData['size'] # generate dollar column by multiplying price and size
  tickData['DollarsCumulated'] = tickData['dollar'].cumsum()  # cumulative sum of dollars
  # if volume_per_bar is not mentioned, then calculate it with dollars divided by number of bars
  if not dollarPerBar:
    dollarsTotal = tickData['DollarsCumulated'].values[-1]
    dollarPerBar = dollarsTotal/numberBars
    dollarPerBar = round(dollarPerBar, -2) # round to the nearest hundred
  tickGrouped = tickData.reset_index().assign(groupingID=lambda x: x.DollarsCumulated//dollarPerBar)
  dates = tickGrouped.groupby('groupingID', as_index = False).first()['dates'] # group date times based on dollars
  tickDataGrouped = tickGrouped.groupby('groupingID')
  ohlcvDataframe = ohlcv(tickDataGrouped) # create a dataframe based on tick bars
  ohlcvDataframe.set_index(dates, drop = True, inplace = True)  # set dates column as index
  return ohlcvDataframe


"""
function: Calculates hedging weights using cov, risk distribution(RiskDist) and σ
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: 36
"""
def PCAWeights(cov, #covariance matrix
               riskDisturbution = None, # risk distribution
               risktarget = 1.):  # risk target
  eigenValues, eigenVectors = np.linalg.eigh(cov) # must be Hermitian
  indices = eigenValues.argsort()[::-1] # arguments for sorting eigenValues descending
  eigenValues, eigenVectors = eigenValues[indices], eigenVectors[:, indices] 
  # if riskDisturbution is nothing, it will assume all risk must be allocated to the principal component with
  # smallest eigenvalue, and the weights will be the last eigenvector re-scaled to match σ
  if riskDisturbution is None:
    riskDisturbution = np.zeros(cov.shape[0])
    riskDisturbution[-1] = 1.
  loads = risktarget*(riskDisturbution/eigenValues)**.5  # represent the allocation in the new (orthogonal) basis
  weights = np.dot(eigenVectors, np.reshape(loads, (-1, 1))) # calculate weights
  return weights


"""
function:  Implementation of the symmetric CUSUM filter
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: 39
"""
def events(input, # dataframe of prices and dates
           threshold): # threshold
  timeEvents, shiftPositive, shiftNegative = [], 0, 0
  # dataframe with price differences
  priceDelta = input.diff()
  for i in priceDelta.index[1:]:
    # compute shiftNegative/shiftPositive with min/max of 0 and delta prices in each day
    shiftPositive = max(0, shiftPositive+priceDelta.loc[i]) # compare price diff with zero
    shiftNegative = min(0, shiftNegative+priceDelta.loc[i]) # compare price diff with zero
    if shiftNegative <- threshold:
      shiftNegative = 0 # reset shiftNegative to 0
      timeEvents.append(i) # append this time into timeEvents
    elif shiftPositive > threshold:
      shiftPositive = 0 # reset shiftPositive to 0
      timeEvents.append(i) # append this time into timeEvents
  return pd.DatetimeIndex(timeEvents)
