import pandas as pd
import numpy as np 
import datetime
import time
import sys
from scipy import stats
from statsmodels.stats import stattools
import multiprocessing as mp

"""----------------------------------------------------------------------
    function:  Implementation of the symmetric CUSUM filter
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: 39
----------------------------------------------------------------------"""
def eventscusum(input, # dataframe of prices and dates
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

"""----------------------------------------------------------------------
    function: Takes grouped dataframe, combining and creating the new one with info. about prices and volume.
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: n/a
----------------------------------------------------------------------"""
def ohlcv(tickDataGrouped): #grouped dataframes
  ohlc = tickDataGrouped['price'].ohlc() # find price in each tick
  ohlc['volume'] = tickDataGrouped['size'].sum() # find volume traded
  ohlc['ValueOfTrades'] = tickDataGrouped.apply(lambda x:(x['price']*x['size']).sum()/x['size'].sum()) # find value of trades
  ohlc['PriceMean'] = tickDataGrouped['price'].mean() # mean of price
  ohlc['TickCount'] = tickDataGrouped['price'].count()# number of ticks
  ohlc['PriceMeanLogReturn'] = np.log(ohlc['PriceMean']) - np.log(ohlc['PriceMean'].shift(1)) # find log return
  return ohlc

"""----------------------------------------------------------------------
    function: Takes dataframe and generating time bar dataframe
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: n/a
----------------------------------------------------------------------"""
def timeBar(tickData, # dataframe of tick data
            frequency = "5Min"): # frequency for rounding date time
  tickDataGrouped = tickData.groupby(pd.Grouper(freq = frequency)) # group data sets based on time freq
  ohlcvDataframe = ohlcv(tickDataGrouped) # create a dataframe based on time bars with frequency freq
  return ohlcvDataframe

"""----------------------------------------------------------------------
    function: computes the daily volatility at intraday estimation points
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: Page 44
----------------------------------------------------------------------"""
def dailyVol(close, # dataframe of dates and close price
             span = 63): # parameter for ewm
    dataframe = close.index.searchsorted(close.index - pd.Timedelta(days = 1)) # searchsort a lag of one day in dates column
    dataframe = dataframe[dataframe > 0] # drop indexes when it's lower than 1
    dataframe = pd.Series(close.index[dataframe - 1], index=close.index[close.shape[0] - dataframe.shape[0]:]) # dataframe of dates and a lag of them
    returns = (close.loc[dataframe.index]/close.loc[dataframe.values].values - 1).rename("rets") # dataframe of returns
    stds = returns.ewm(span = span).std().rename("std") # dataframe of ewma stds
    return returns, stds

"""----------------------------------------------------------------------
    function:  implements the triple-barrier method
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: Page 45
----------------------------------------------------------------------"""
def tripleBarrier(close, # dataframe of dates and close price
                  events, # dataframe with timestamp of vertical barrier and unit width of the horizontal barriers
                  profitTakingStopLoss,  #list of two non-negative float values:to set the width of the upper and lower barrier.
                  molecule): #A list with the subset of event indices
    eventsFiltered = events.loc[molecule]
    # dataframe of output with vertical barrier and two horizontal barriers
    output = eventsFiltered[['timestamp']].copy(deep = True)
    # here, profitTakingStopLoss multiplies target to set the width of the upper and lower barrier.
    if profitTakingStopLoss[0] > 0:
        profittaking = profitTakingStopLoss[0]*eventsFiltered['target'] # factors multiply targt to set the width of the upper barrier.
    else:
        profittaking = pd.Series(index = events.index) # NaNs
    if profitTakingStopLoss[1] > 0:
        stoploss = -profitTakingStopLoss[1]*eventsFiltered['target']  # factors multiply targt to set the width of the lower barrier.
    else:
        stoploss = pd.Series(index = events.index) # NaNs
    for location, timestamp in eventsFiltered['timestamp'].fillna(close.index[-1]).iteritems():
        dataframe = close[location:timestamp]  # path prices
        dataframe = (dataframe/close[location] - 1)*eventsFiltered.at[location, 'side']  # path returns
        output.loc[location, 'stoploss'] = dataframe[dataframe < stoploss[location]].index.min()  # earliest stop loss.
        output.loc[location, 'profittaking'] = dataframe[dataframe > profittaking[location]].index.min()  # earliest profit taking.
    return output

"""----------------------------------------------------------------------
    function: finds the time of the first barrier touch
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: 48
----------------------------------------------------------------------"""
def events(close,  # dataframe of dates and close price
           timeEvents,  # a the pandas timeindex containing the timestamps that will seed every triple barrier
           ptsl, # a non-negative float that sets the width of the two barriers
           target,  # a pandas series of targets, expressed in terms of absolute returns
           returnMin, # the minimum target return required for running a triple barrier search
           numThreads, # the number of threads
           timestamp=False): # a pandas series with the timestamps of the vertical barriers. We pass a False when we want to disable vertical barrier
    target = target.loc[timeEvents] # get target dataframe
    target = target[target > returnMin]  # returnMin
    if timestamp is False: 
        timestamp = pd.Series(pd.NaT, index = timeEvents) #  get timestamp (max holding period)
    # form events object, apply stop loss on timestamp
    sidePosition = pd.Series(1., index = target.index) # create side array
    events = pd.concat({'timestamp': timestamp, 'target': target, 'side': sidePosition}, axis=1).dropna(subset = ['target']) # create events dataframe
    dataframe = mpPandasObj(func = tripleBarrier, pdObj = ('molecule', events.index), 
                      numThreads = numThreads, close=close, events=events, profitTakingStopLoss=[ptsl, ptsl]) # use multithreading
    events['timestamp'] = dataframe.dropna(how = 'all').min(axis = 1)  # pd.min ignores nan
    events = events.drop('side', axis = 1)
    return events


"""----------------------------------------------------------------------
    function: shows one way to define a vertical barrier
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: 49
----------------------------------------------------------------------"""
def verticalBarrier(close, # dataframe of prices and dates
                    timeEvents, # vecotr of timestamps
                    numberDays): # a number of days for vertical barrier)
  timestampArray = close.index.searchsorted(timeEvents + pd.Timedelta(days = numberDays)) # searchsort a lag of numberDays dates column
  timestampArray = timestampArray[timestampArray<close.shape[0]]
  timestampArray = pd.Series(close.index[timestampArray], index = timeEvents[:timestampArray.shape[0]]) # NaNs at end
  return timestampArray

"""----------------------------------------------------------------------
    function: label the observations
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: 49
----------------------------------------------------------------------"""
def label(events, # dataframe with timestamp of vertical barrier and unit width of the horizontal barriers
          close): # dataframe of dates and close price
    eventsFiltered = events.dropna(subset = ['timestamp']) # filter events without NaN
    allDates = eventsFiltered.index.union(eventsFiltered['timestamp'].values).drop_duplicates() # get all dates
    closeFiltered = close.reindex(allDates, method = 'bfill') # prices aligned with events
    # create out object by calculating ret and bin
    out = pd.DataFrame(index = eventsFiltered.index) # create output object
    out['ret'] = closeFiltered.loc[eventsFiltered['timestamp'].values].values/closeFiltered.loc[eventsFiltered.index] - 1 # the return realized at the time of the first touched barrier
    out['bin'] = np.sign(out['ret']) # the label, {−1, 0, 1}, as a function of the sign of the outcome
    return out

"""----------------------------------------------------------------------
    function: expand events tO incorporate meta-labeling
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: 50
----------------------------------------------------------------------"""
def eventsMeta(close, # dataframe of prices and dates
               timeEvents, # vecotr of timestamps
               ptsl, # list of two non-negative float values that multiply targt
               target, # dataframe of targets, expressed in terms of absolute returns
               returnMin, # The minimum target return required for running a triple barrier
               numThreads, # number of threads
               timestamp = False, # vector contains the timestamps of the vertical barriers
               side = None): # when side is not nothing, the function understands that meta-labeling is in play
  target = target.loc[timeEvents] # get target dataframe
  target = target[target > returnMin] # returnMin
  if timestamp is False:
    timestamp = pd.Series(pd.NaT, index = timeEvents) # get timestamp (max holding period) based on vertical barrier
  if side is None:
    sidePosition, profitLoss = pd.Series(1.,index = target.index), [ptsl[0], ptsl[0]] # create side array
  else:
    sidePosition, profitLoss = side.loc[target.index], ptsl[:2]
  events = pd.concat({'timestamp': timestamp,'target': target,'side': sidePosition}, axis=1).dropna(subset = ['target']) #create events dataframe
  df0 = mpPandasObj(func=tripleBarrier, pdObj=('molecule',events.index), 
                    numThreads=numThreads, close=close, events=events, profitTakingStopLoss=profitLoss) # using multithreads
  events['timestamp'] = df0.dropna(how = 'all').min(axis = 1) # pd.min ignores nan
  if side is None: # when side is not None, the function understand that meta-labeling is in play
    events = events.drop('side',axis = 1)
  return events


"""----------------------------------------------------------------------
    function: expand label tO incorporate meta-labeling
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: 51
----------------------------------------------------------------------"""
def labelMeta(events, # dataframe with timestamp of vertical barrier and unit width of the horizontal barriers
              close): # dataframe of dates and close price
    eventsFiltered = events.dropna(subset = ['timestamp']) # filter events without NaN
    allDates = eventsFiltered.index.union(eventsFiltered['timestamp'].values).drop_duplicates() # get all dates
    closeFiltered = close.reindex(allDates, method = 'bfill') # prices aligned with events
    out = pd.DataFrame(index = eventsFiltered.index) # create output object
    out['ret'] = closeFiltered.loc[eventsFiltered['timestamp'].values].values/closeFiltered.loc[eventsFiltered.index] - 1 # calculate returns
    if 'side' in eventsFiltered: 
      out['ret']*=eventsFiltered['side']  # meta-labeling
    out['bin'] = np.sign(out['ret']) # get sign of returns
    if 'side' in eventsFiltered: 
      out.loc[out['ret'] <= 0, 'bin'] = 0  # meta-labeling
    return out


"""----------------------------------------------------------------------
    function: presents a procedure that recursively drops observations associated with extremely rare labels
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: 54
----------------------------------------------------------------------"""
def dropLabel(events, # dataframe, with columns: Dates, ret, and bin
              percentMin = .05): # minimum percentage
  # apply weights, drop labels with insufficient examples
  while True:
    dataframe = events['bin'].value_counts(normalize = True) # calculate percentage frequency
    if dataframe.min() > percentMin or dataframe.shape[0] < 3: # check for eliminating
      break
    print('dropped label', dataframe.argmin(), dataframe.min()) # print results
    events = events[events['bin'] != dataframe.argmin()] # update events
  return events


"Multi-threading----------------------------------------"

# SNIPPET 20.5 THE linParts FUNCTION
def linParts(numAtoms,numThreads):
  # partition of atoms with a single loop
  parts=np.linspace(0,numAtoms,min(numThreads,numAtoms)+1)
  parts=np.ceil(parts).astype(int)
  return parts

# SNIPPET 20.6 THE nestedParts FUNCTION
def nestedParts(numAtoms,numThreads,upperTriang=False):
  # partition of atoms with an inner loop
  parts,numThreads_=[0],min(numThreads,numAtoms)
  for num in range(numThreads_):
    part=1 + 4*(parts[-1]**2+parts[-1]+numAtoms*(numAtoms+1.)/numThreads_)
    part=(-1+part**.5)/2.
    parts.append(part)
  parts=np.round(parts).astype(int)
  if upperTriang: # the first rows are the heaviest
    parts=np.cumsum(np.diff(parts)[::-1])
    parts=np.append(np.array([0]),parts)
  return parts

# SNIPPET 20.7 THE mpPandasObj, USED AT VARIOUS POINTS IN THE BOOK
def mpPandasObj(func,pdObj,numThreads=24,mpBatches=1,linMols=True,**kargs):
  """
  Parallelize jobs, return a DataFrame or Series
  + func: function to be parallelized. Returns a DataFrame
  + pdObj[0]: Name of argument used to pass the molecule
  + pdObj[1]: List of atoms that will be grouped into molecules
  + kargs: any other argument needed by func
  Example: df1=mpPandasObj(func,(’molecule’,df0.index),24,**kargs)
  """
  argList = list(kargs.values()) #?
  if linMols:
    parts=linParts(len(argList[1]),numThreads*mpBatches)
  else:
    parts=nestedParts(len(argList[1]),numThreads*mpBatches)
  jobs=[] 
  for i in range(1,len(parts)):
    job={pdObj[0]:pdObj[1][parts[i-1]:parts[i]],'func':func}
    job.update(kargs)
    jobs.append(job)
  if numThreads==1:
    out=processJobs_(jobs)
  else:
    out=processJobs(jobs,numThreads=numThreads)
  if isinstance(out[0],pd.DataFrame):
    df0=pd.DataFrame()
  elif isinstance(out[0],pd.Series):
    df0=pd.Series()
  else:
    return out
  for i in out:
    df0=df0.append(i)
  df0=df0.sort_index()
  return df0

# SNIPPET 20.8 SINGLE-THREAD EXECUTION, FOR DEBUGGING
def processJobs_(jobs):
  # Run jobs sequentially, for debugging
  out=[]
  for job in jobs:
    out_=expandCall(job)
    out.append(out_)
  return out

# SNIPPET 20.9 EXAMPLE OF ASYNCHRONOUS CALL TO PYTHON’S MULTIPROCESSING LIBRARY
def reportProgress(jobNum,numJobs,time0,task):
  # Report progress as asynch jobs are completed
  msg=[float(jobNum)/numJobs,(time.time()-time0)/60.]
  msg.append(msg[1]*(1/msg[0]-1))
  timeStamp=str(datetime.datetime.fromtimestamp(time.time()))
  msg= timeStamp+' '+str(round(msg[0]*100,2))+'% '+task+' done after '+ str(round(msg[1],2))+' minutes. Remaining '+str(round(msg[2],2))+' minutes.'
  if jobNum<numJobs:
    sys.stderr.write(msg+'\r')
  else:
    sys.stderr.write(msg+'\n')
  return

def processJobs(jobs,task=None,numThreads=24):
  # Run in parallel.
  # jobs must contain a ’func’ callback, for expandCall
  if task is None:task=jobs[0]['func'].__name__
  pool=mp.Pool(processes=numThreads)
  outputs,out,time0=pool.imap_unordered(expandCall,jobs),[],time.time()
  # Process asynchronous output, report progress
  for i,out_ in enumerate(outputs,1):
    out.append(out_)
    reportProgress(i,len(jobs),time0,task)
  pool.close()
  pool.join() # this is needed to prevent memory leaks
  return out

# SNIPPET 20.10 PASSING THE JOB (MOLECULE) TO THE CALLBACK FUNCTION
def expandCall(kargs):
  # Expand the arguments of a callback function, kargs[’func’]
  func=kargs['func']
  del kargs['func']
  out=func(**kargs)
  return out



