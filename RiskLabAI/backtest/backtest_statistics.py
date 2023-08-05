import numpy as np
import pandas as pd

"""
    function: returns timing of bets when positions flatten or flip
    reference: De Prado, M (2018) Advances in financial machine learning
    methodology: page 197, snippet 14.1
"""
def bet_timing(target_positions: pd.Series):  # series of target positions
    zero_positions = target_positions[target_positions == 0].index # get zero positions indices

    lagged_non_zero_positions = target_positions.shift(1) # lag the positions
    lagged_non_zero_positions = lagged_non_zero_positions[lagged_non_zero_positions != 0].index # get lagged non zero positions indices
    
    bets = zero_positions.intersection(lagged_non_zero_positions) # get flattening indices
    zero_positions = target_positions.iloc[1:]*target_positions.iloc[:-1].values # find flips
    bets = bets.union(zero_positions[zero_positions < 0].index).sort_values() # get flips' indices
    
    if target_positions.index[-1] not in bets:
        bets = bets.append(target_positions.index[-1:]) # add the last bet

    return bets

"""
    function: derives avgerage holding period (in days) using avg entry time pairing algo
    refernce: De Prado, M (2018) Advances in financial machine learning
    methodology: page 197, snippet 14.2
"""
def holding_period(target_positions: pd.Series):  # series of target positions
    hold_period, time_entry = pd.DataFrame(columns = ['dT', 'w']), 0.0 # initialize holding periods and entry time
    position_difference, time_difference = target_positions.diff(), (target_positions.index - target_positions.index[0]) / np.timedelta64(1,'D') # find position difference and elapssed time
    
    for i in range(1, target_positions.shape[0]):
        if position_difference.iloc[i]*target_positions.iloc[i-1] >= 0: # find if position is increased or unchanged
            if target_positions.iloc[i] != 0: # find if target position is non zero
                time_entry = (time_entry*target_positions.iloc[i-1] + time_difference[i]*position_difference.iloc[i]) / target_positions.iloc[i] # update entry time
        else: # find if position is decreased
            if target_positions.iloc[i]*target_positions.iloc[i-1] < 0: # find if there is a flip
                hold_period.loc[target_positions.index[i],['dT','w']] = (time_difference[i] - time_entry,abs(target_positions.iloc[i-1])) # add the new holding period
                time_entry = time_difference[i] # reset entry time
            else:
                hold_period.loc[target_positions.index[i],['dT','w']] = (time_difference[i] - time_entry,abs(position_difference.iloc[i])) # add the new holding period
    
    if hold_period['w'].sum() > 0: # find if there are holding periods
        mean = (hold_period['dT']*hold_period['w']).sum() / hold_period['w'].sum() # calculate the average holding period
    else:
        mean = np.nan

    return hold_period, mean

"""
    function: derives the algorithm for deriving hhi concentration
    refernce: De Prado, M (2018) Advances in financial machine learning
    methodology: page 201, snippet 14.3
"""
def hhi_concentration(returns: pd.Series):  # series of returns
    returns_hhi_positive = hhi(returns[returns >= 0]) # get concentration of positive returns per bet
    ruturns_hhi_negative = hhi(returns[returns < 0]) # get concentration of negative returns per bet
    time_concentrated_hhi = hhi(returns.groupby(pd.Grouper(freq='M')).count()) # get concentr. bets/month

    return returns_hhi_positive, ruturns_hhi_negative, time_concentrated_hhi

def hhi(bet_returns: pd.Series):  # bet returns
    if bet_returns.shape[0] <= 2: # find returns length is less than 3
        return np.nan

    weight = bet_returns / bet_returns.sum() # Calculate weights
    hhi_ = (weight**2).sum() # sum of squared weights
    hhi_ = (hhi_ - bet_returns.shape[0]**-1)/(1. - bet_returns.shape[0]**-1) # calculate hhi with squared weights

    return hhi_

"""
    function: computes series of drawdowns and the time under water associated with them
    refernce: De Prado, M (2018) Advances in financial machine learning
    methodology: page 201, snippet 14.4
"""
def compute_drawdowns_time_under_water(series: pd.Series, # series of returns or dollar performance
                                       dollars=False): # returns or dollar performance

    series_df = series.to_frame('PnL').reset_index(names='Datetime') # convert to DataFrame
    series_df['HWM'] = series.expanding().max().values # find max of expanding window

    def process_groups(group): # proces drawdowns

        if len(group) <= 1: # check if there is a drawdown 
            return None
        
        result = pd.Series()
        result.loc['Start'] = group['Datetime'].iloc[0] # find drawdown beginning
        result.loc['Stop'] = group['Datetime'].iloc[-1] # find drawdown ending
        result.loc['HWM'] = group['HWM'].iloc[0] # find drawdown high watermark
        result.loc['Min'] = group['PnL'].min() # find the maximum drawdown 
        result.loc['Min. Time'] = group['Datetime'][group['PnL'] == group['PnL'].min()].iloc[0] # find the maximum drawdown time

        return result
    
    groups = series_df.groupby('HWM') # group by high water mark
    drawdown_analysis = pd.DataFrame() # initiate dataframe    

    for _, group in groups:
        drawdown_analysis = drawdown_analysis.append(process_groups(group), ignore_index=True) # process and aggregate drawdowns

    if dollars:
        drawdown = drawdown_analysis['HWM'] - drawdown_analysis['Min'] # calculate drawdowns
    else:
        drawdown = 1 - drawdown_analysis['Min'] / drawdown_analysis['HWM'] # calculate drawdowns

    drawdown.index = drawdown_analysis['Start'] # set index
    drawdown.index.name = 'Datetime' # set index name

    time_under_water = ((drawdown_analysis['Stop'] - drawdown_analysis['Start']) / np.timedelta64(1, 'Y')).values # convert time under water to years
    time_under_water = pd.Series(time_under_water, index=drawdown_analysis['Start']) # create Series

    return drawdown, time_under_water, drawdown_analysis