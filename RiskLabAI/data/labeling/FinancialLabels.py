from scipy import stats
import numpy as np
import pandas as pd

"""
function: calculates the t-value of a linear trend
refernce: De Prado, M (2020) Machine Learning for Asset Managers
methodology: page 68, snippet 5.1
"""
def t_value_linear_regression(price:pd.Series): # time series of prices
    x = np.arange(price.shape[0]) # create regression data
    ols = stats.linregress(x, price.values) # fit linear regression
    t_value = ols.slope / ols.stderr # calculate t-value
    
    return t_value

"""
function: implements the trend scanning method
refernce: De Prado, M (2020) Machine Learning for Asset Managers
methodology: page 68, snippet 5.2
"""
def bins_from_trend(molecule, # index of observations we wish to label
                    close, # time series of prices
                    span): # the range arguments of span lenghts that the algorithm will evaluate, in search for the maximum absolute t-value
    
    outputs = pd.DataFrame(index=molecule, columns=['End Time', 't-Value', 'Trend']) # initialize outputs
    spans = range(*span) # get spans
    
    for index in molecule:
        t_values = pd.Series(dtype='float64') # initialize t-value series
        location = close.index.get_loc(index) # find observation location
        
        if location + max(spans) > close.shape[0]: # check if the window goes out of range
            continue
        
        for span in spans:
            tail = close.index[location + span - 1] # get window tail index
            window_prices = close.loc[index:tail] # get window prices
            t_values.loc[tail] = t_value_linear_regression(window_prices) # get trend t-value 
            
        tail = t_values.replace([-np.inf, np.inf, np.nan],0).abs().idxmax() # modify for validity and find the t-value's window tail index
        outputs.loc[index,['End Time', 't-Value', 'Trend']] = t_values.index[-1], t_values[tail], np.sign(t_values[tail]) # prevent leakage and get best t-value
        
    outputs['End Time'] = pd.to_datetime(outputs['End Time']) # convert to datetime
    outputs['Trend'] = pd.to_numeric(outputs['Trend'], downcast='signed') # convert to numeric
    
    return outputs.dropna(subset=['Trend']) # drop nan values from trends