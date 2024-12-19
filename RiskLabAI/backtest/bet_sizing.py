import importlib
import numpy as np
import pandas as pd 
from numba import jit
from RiskLabAI.hpc import *
from scipy.stats import norm

def probability_bet_size(
    probabilities: np.ndarray,
    sides: np.ndarray
) -> np.ndarray:
    """
    Calculate the bet size based on probabilities and side.

    :param probabilities: array of probabilities
    :param sides: array indicating the side of the bet (e.g., long/short or buy/sell)
    :return: array of bet sizes

    .. math::

       \text{bet size} = \text{side} \times (2 \times \text{CDF}(\text{probabilities}) - 1)
    """
    return sides * (2 * norm.cdf(probabilities) - 1)

@jit(nopython=True)
def average_bet_sizes(
    price_dates: np.ndarray,
    start_dates: np.ndarray,
    end_dates: np.ndarray,
    bet_sizes: np.ndarray
) -> np.ndarray:
    """
    Compute average bet sizes for each date.

    :param price_dates: array of price dates
    :param start_dates: array of start dates for bets
    :param end_dates: array of end dates for bets
    :param bet_sizes: array of bet sizes for each date range
    :return: array of average bet sizes for each price date
    """
    num_dates = len(price_dates)
    avg_bet_sizes = np.zeros(num_dates)
    
    for i in range(num_dates):
        total = 0
        count = 0
        for j in range(len(start_dates)):
            if start_dates[j] <= price_dates[i] <= end_dates[j]:
                total += bet_sizes[j]
                count += 1
        if count > 0:
            avg_bet_sizes[i] = total / count
            
    return avg_bet_sizes

def strategy_bet_sizing(
    price_timestamps: pd.Series,
    times: pd.Series,
    sides: pd.Series,
    probabilities: pd.Series
) -> pd.Series:
    """
    Calculate the average bet size for a trading strategy given price timestamps.

    :param price_timestamps: series of price timestamps
    :param times: series with start times as indices and end times as values
    :param sides: series indicating the side of the position (e.g., long/short)
    :param probabilities: series of probabilities associated with each position
    :return: series of average bet sizes for each price timestamp
    """
    
    bet_sizes = probability_bet_size(probabilities.to_numpy(), sides.to_numpy())
    
    avg_bet_sizes = average_bet_sizes(price_timestamps.to_numpy(), times.index.to_numpy(), times.values, bet_sizes)

    return pd.Series(avg_bet_sizes, index=price_timestamps)

"""----------------------------------------------------------------------
    function: Calculation of Average Active Signals 
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: SNIPPET 10.2
----------------------------------------------------------------------"""
def avgActiveSignals(signals,# DataFrame of signal
                     nThreads): #number of threads 
    # compute the average signal among those active
    #1) time points where signals change (either one starts or one ends)
    timePoints=set(signals['t1'].dropna().values) #drop nan value from dataframe
    timePoints=timePoints.union(signals.index.values) #union incex 
    timePoints=list(timePoints) #list them!
    timePoints.sort() #sort them!
    out=mpPandasObj(mpAvgActiveSignals,('molecule',timePoints),nThreads,signals=signals) #generate final signal
    return out
#———————————————————————————————————————

"""
function: Calculation of Average Active Signals 
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: SNIPPET 10.2
At time loc, average signal among those still active.
Signal is active if:
a) issued before or at loc AND
b) loc before signal's endtime, or endtime is still unknown (NaT).
"""
def mpAvgActiveSignals(signals, #DataFrame of signal 
                       molecule): #index of dataFrame that function act on it 
    out=pd.Series()
    for loc in molecule:
        signal_=(signals.index.values<=loc)&((loc<signals['t1'])|pd.isnull(signals['t1'])) #keep signal that contain loc
        act=signals[signal_].index #store index 
        if len(act)>0:
            out[loc]=signals.loc[act,'signal'].mean() #get mean!
        else:
            out[loc]=0 # no signals active at this time
    return out

"""----------------------------------------------------------------------
    function: Discretize Signals 
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: SNIPPET 10.3
----------------------------------------------------------------------"""
def discreteSignal(signal, #DataFrame of signal  
                   stepSize): #stepsize for dicretize signal 
    # discretize signal
    discreteSignal=(signal/stepSize).round()*stepSize # discretize
    discreteSignal[discreteSignal>1]=1 # cap
    discreteSignal[discreteSignal<-1]=-1 # floor
    return discreteSignal

"""----------------------------------------------------------------------
    function: generate Signal 
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: SNIPPET 10.1
----------------------------------------------------------------------"""
def Signal(events, #DataFrame of events that genrate in labelling
           stepSize, #Step size for discretize signal 
           probability, #probability that come from meta labelling 
           prediction, # prediction of primary model 
           nClasses, #number of class of prediction  
           nThreads,): # number of threads  # o
    # get signals from predictions
    if probability.shape[0]==0:return pd.Series()
    #1) generate signals from multinomial classification (one-vs-rest, OvR)
    discreteSignal=(probability-1./nClasses)/(probability*(1.-probability))**.5 # t-value of OvR
    newSignal=prediction*(2*norm.cdf(discreteSignal)-1) # signal=side*size
    if 'side' in events:newSignal*=events.loc[newSignal.index,'side'] # meta-labeling
    #2) compute average signal among those concurrently open
    newSignal=newSignal.to_frame('signal').join(events[['t1']],how='left')
    newSignal=avgActiveSignals(newSignal,nThreads) #average signal that contain one events 
    newSignal=discreteSignal(signal=newSignal,stepSize=stepSize) #discretize  signal 
    return newSignal

"""----------------------------------------------------------------------
    function: DYNAMIC POSITION SIZE AND LIMIT PRICE 
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: p.145 SNIPPET 10.4 
----------------------------------------------------------------------"""
def betSize(w,x):
    return x*(w + x**2)**-.5
#———————————————————————————————————————
def TPos(w, # coefficient that regulates the width of the sigmoid function
         f, # predicted price
         acctualPrice , # acctual price 
         maximumPositionSize): #  maximum absolute position size
    return int(betSize(w, f-acctualPrice)*maximumPositionSize)
#———————————————————————————————————————
def inversePrice(f, # predicted price
                 w, #coefficient that regulates the width of the sigmoid function
                 m): # betsize 
    return f - m*(w/(1-m^2))^(.5)
#———————————————————————————————————————
def limitPrice(targetPositionSize, #  target position size
               cPosition, # current position
               f, #  predicted price
               w, # coefficient that regulates the width of the sigmoid function
               maximumPositionSize): # maximum absolute position size
    if targetPositionSize >=  cPosition:
        sgn = 1
    else:
        sgn = -1
    lP = 0
    for i in range(abs(cPosition+sgn),abs(targetPositionSize)) :
        lP += inversePrice(f,w,i/float(maximumPositionSize))
    lP /= targetPositionSize-cPosition
    return lP    
#———————————————————————————————————————
def getW(x, # divergence between the current market price and the forecast
         m): # bet size 
    return x**2*(1/m**(-2)-1)