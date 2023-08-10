import pandas as pd
import numpy as np 
import datetime
import time
import sys
from scipy import stats
from statsmodels.stats import stattools
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

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
    function: The sequence of weights used to compute each value of the fractionally differentiated series.
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: 79
----------------------------------------------------------------------"""
def weighting(degree, # degree
              size): #size
    weight = [1.] # array of weights
    for k in range(1, size):
        thisWeight = -weight[-1]/k*(degree - k + 1)  # calculate each weight
        weight.append(thisWeight) # append weight into array
    weight = np.array(weight[ : : -1]).reshape(-1, 1) # reshape weight array
    return weight

"""----------------------------------------------------------------------
    function: plot weights
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: 79
----------------------------------------------------------------------"""
def plotWeights(degreeRange, # range for degree
                numberDegrees, # number of degrees
                size): # number of weights
    weight = pd.DataFrame() # dataframe of weights
    for degree in np.linspace(degreeRange[0], degreeRange[1], numberDegrees):
        degree = np.round(degree,2) # round degree with digits = 2
        thisweight = weighting(degree, size = size) # calculate weights for each degree
        thisweight = pd.DataFrame(thisweight, index = range(thisweight.shape[0])[ : : -1], columns = [degree])
        # dataframe of weights for each degree
        weight = weight.join(thisweight, how = 'outer') # append into weight
    fig = weight.plot() # plot weights
    fig.show() # show weights
    return


"""----------------------------------------------------------------------
    function: standard fractionally differentiated
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: 82
----------------------------------------------------------------------"""
def fracDiff(series, # dataframe of dates and prices
             degree, # degree of binomial series
             threshold = .01): # threshold for weight-loss
    weights = weighting(degree, series.shape[0]) # calculate weights
    weights_ = np.cumsum(abs(weights)) # cumulate weights
    weights_ /= weights_[-1] # determine the relative weight-loss
    skip = weights_[weights_ > threshold].shape[0]  # number of results where the weight-loss is beyond the acceptable value
    dataframe = {} # dataframe of output
    for name in series.columns:
        seriesFiltered = series[[name]].fillna(method = 'ffill').dropna() # filter for missing data
        dataframeFiltered = pd.Series(dtype = "float64") # create a pd series
        for iloc in range(skip, seriesFiltered.shape[0]):
            date = seriesFiltered.index[iloc] # find date of obs 
            price = series.loc[date,name] # price for that date
            if isinstance(price, (pd.Series, pd.DataFrame)):
                price = price.resample('1m').mean()
            if not np.isfinite(price).any(): # check for being finite
                 continue # exclude NAs
            try: # the (iloc)^th obs will use all the weights from the start to the (iloc)^th
                dataframeFiltered.loc[date] = np.dot(weights[-(iloc + 1):, :].T, seriesFiltered.loc[:date])[0, 0] # calculate values
            except:
                continue
        dataframe[name] = dataframeFiltered.copy(deep = True) # update dataframe
    dataframe = pd.concat(dataframe, axis = 1) # concat dataframes into dataframe
    return dataframe

"""----------------------------------------------------------------------
    function: weights for fixed-width window method
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: 83
----------------------------------------------------------------------"""
def weightingFFD(degree, 
                 threshold):
    weights = [1.] # array of weights
    k = 1 # initial value of k
    while abs(weights[-1]) >= threshold:  
        thisweight = -weights[-1]/k*(degree - k + 1) # calculate each weight
        weights.append(thisweight) # append into array
        k += 1 # update k
    weights = np.array(weights[ : : -1]).reshape(-1, 1)[1 : ] # reshape into a vector
    return weights

"""----------------------------------------------------------------------
    function: Fixed-width window fractionally differentiated method
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: 83
----------------------------------------------------------------------"""
def fracDiffFixed(series, # dataframe of dates and prices
                  degree, # degree of binomial series
                  threshold = 1e-5): # threshold for weight-loss
    weights = weightingFFD(degree, threshold) # compute weights for the longest series
    width = len(weights) - 1 # length of weights
    dataframe = {} # empty dict
    for name in series.columns:
        seriesFiltered = series[[name]].fillna(method = 'ffill').dropna()  # filter for missing data
        dataframeFiltered = pd.Series(dtype = "float64") # empty pd.series
        for iloc in range(width, seriesFiltered.shape[0]):
            day1 = seriesFiltered.index[iloc - width] # first day
            day2 = seriesFiltered.index[iloc] # last day
            if not np.isfinite(series.loc[day2, name]):
                continue # exclude NAs
            dataframeFiltered[day2] = np.dot(weights.T, seriesFiltered.loc[day1 : day2])[0, 0] # calculate value
    
        dataframe[name] = dataframeFiltered.copy(deep = True) # copy dataframeFiltered into dataframe
    dataframe = pd.concat(dataframe, axis = 1) # concat all into dataframe
    return dataframe

"""----------------------------------------------------------------------
    function: Find the minimum degree value that passes the ADF test
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: 85
----------------------------------------------------------------------"""
def minFFD(input):
    out = pd.DataFrame(columns = ['adfStat', 'pVal', 'lags', 'nObs', '95% conf', 'corr']) # dataframe of output
    for d in np.linspace(0, 1, 11):
        dataframe = np.log(input[['close']]).resample('1D').last() # dataframe of price and dates
        differentiated = fracDiffFixed(dataframe, d, threshold = .01) # call fixed-width frac diff method
        corr = np.corrcoef(dataframe.loc[differentiated.index, 'close'], differentiated['close'])[0, 1] # correlation 
        differentiated = adfuller(differentiated['close'], maxlag = 1, regression = 'c', autolag = None) # ADF test
        out.loc[d] = list(differentiated[ : 4]) + [differentiated[4]['5%']] + [corr] # push new observation with critical value
    return out


