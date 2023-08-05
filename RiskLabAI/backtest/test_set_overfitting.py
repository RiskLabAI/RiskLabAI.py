import numpy as np
import pandas as pd
from scipy.stats import norm
import scipy.stats as ss

"""
    function: validates the False Strategy Theorem experimentally
    refernce: De Prado, M (2020) Machine Learning for Asset Managers
    methodology: page 110, snippet 8.1
"""
def expected_max_sharpe_ratio(nTrials, # number of trials
                              mean_sharpe_ratio, # mean Sharpe Ratio
                              std_sharpe_ratio): # standard deviation of Sharpe Ratios

    # Expected max SR, controlling for SBuMT
    emc = 0.577215664901532860606512090082402431042159336 # euler gamma constant

    sharpe_ratio = (1 - emc)*norm.ppf(1 - 1.0/nTrials) + emc*norm.ppf(1 - (nTrials*np.e)**-1) # get expected value of sharpe ratio by using false strategy theorem
    sharpe_ratio = mean_sharpe_ratio + std_sharpe_ratio*sharpe_ratio # get max Sharpe Ratio, controlling for SBuMT

    return sharpe_ratio

def generated_max_sharpe_ratio(nSims, # number of simulations
                               nTrials, # number of trials
                               std_sharpe_ratio, # mean Sharpe Ratio
                               mean_sharpe_ratio): # standard deviation of Sharpe Ratios

    # Monte Carlo of max{SR} on nTrials, from nSims simulations
    rng = np.random.RandomState()
    output = pd.DataFrame()

    for nTrials_ in nTrials:
        #1) Simulated Sharpe ratios
        sharpe_ratio = pd.DataFrame(rng.randn(nSims, nTrials_)) # generate random numbers for Sharpe Ratios
        sharpe_ratio = sharpe_ratio.sub(sharpe_ratio.mean(axis=1), axis=0) # standardize Sharpe Ratios 
        sharpe_ratio = sharpe_ratio.div(sharpe_ratio.std(axis=1), axis=0) # standardize Sharpe Ratios 
        sharpe_ratio = mean_sharpe_ratio + sharpe_ratio*std_sharpe_ratio # set the mean and standard deviation

        #2) Store output
        output_ = sharpe_ratio.max(axis=1).to_frame('max{SR}') # generate output
        output_['nTrials'] = nTrials_ # generate output
        output = output.append(output_, ignore_index=True) # append output

    return output
"""
    function: calculates mean and standard deviation of the predicted errors
    refernce: De Prado, M (2020) Machine Learning for Asset Managers
    methodology: page 112, snippet 8.2
"""
def mean_std_error(nSims0, # number of max{SR} used to estimate E[max{SR}]
                   nSims1, # number of errors on which std is computed
                   nTrials, # array of numbers of SR used to derive max{SR}
                   std_sharpe_ratio=1, # standard deviation Sharpe Ratio
                   mean_sharpe_ratio=0): # mean of Sharpe Ratios

    sharpe_ratio0 = pd.Series({i:expected_max_sharpe_ratio(i,mean_sharpe_ratio,std_sharpe_ratio) for i in nTrials}) # compute expected max Sharpe Ratios
    sharpe_ratio0 = sharpe_ratio0.to_frame('E[max{SR}]') # compute expected max Sharpe Ratios
    sharpe_ratio0.index.name = 'nTrials' # compute expected max Sharpe Ratios
    error = pd.DataFrame() # initialize errors

    for i in range(int(nSims1)):
        sharpe_ratio1 = generated_max_sharpe_ratio(nSims=nSims0, nTrials=nTrials, mean_sharpe_ratio=mean_sharpe_ratio, std_sharpe_ratio=std_sharpe_ratio) # generate max Sharpe Ratios 
        sharpe_ratio1 = sharpe_ratio1.groupby('nTrials').mean() # calculate mean max Sharpe Ratios 
        error_ = sharpe_ratio0.join(sharpe_ratio1).reset_index() # create DataFrame of generated Max Sharpe Ratios with errors
        error_['error'] = error_['max{SR}'] / error_['E[max{SR}]'] - 1.0 # add expected max Sharpe Ratios and calculate errors
        error=error.append(error_) # append errors

    output = {'meanErr' : error.groupby('nTrials')['error'].mean()} # calculate mean errors
    output['stdErr'] = error.groupby('nTrials')['error'].std() # calculate standard deviation of errors
    output = pd.DataFrame.from_dict(output, orient='columns') # create output

    return output

"""
    function: calculates type I error probability of stratgies
    refernce: De Prado, M (2020) Machine Learning for Asset Managers
    methodology: page 119, snippet 8.3
"""
def estimated_sharpe_ratio_z_statistics(sharpe_ratio, # estimated Sharpe Ratio
                                        t, # number of observations
                                        sharpe_ratio_=0, # true Sharpe Ratio
                                        skew=0, # skewness of returns
                                        kurt=3): # kurtosis of returns
    
    z = (sharpe_ratio - sharpe_ratio_)*(t - 1)**0.5 # calculate first part of z statistic
    z /= (1 - skew*sharpe_ratio + (kurt - 1) / 4.0*sharpe_ratio**2)**0.5 # calculate z statistic

    return z
def strategy_type1_error_probability(z, # z statistic for the estimated Sharpe Ratios
                                     k=1): # number of tests
    # false positive rate
    α = ss.norm.cdf(-z) # find false positive rate
    α_k = 1 - (1 - α)**k # correct for multi-testing 

    return α_k

"""
    function: calculates type II error probability of stratgies
    refernce: De Prado, M (2020) Machine Learning for Asset Managers
    methodology: page 121, snippet 8.4
"""
def theta_for_type2_error(sharpe_ratio, # estimated Sharpe Ratio
                          t, # number of observations
                          sharpe_ratio_=0, # true Sharpe Ratio
                          skew=0, # skewness of returns
                          kurt=3): # kurtosis of returns
 
    θ = sharpe_ratio_*(t - 1)**.5
    θ /= (1 - skew*sharpe_ratio + (kurt - 1) / 4.0*sharpe_ratio**2)**0.5
    
    return θ
def strategy_type2_error_probability(α_k, # type I error
                                     k, # number of tests
                                     θ): # calculated theta parameter
    # false negative rate
    z = ss.norm.ppf((1 - α_k)**(1.0 / k)) # perform Sidak’s correction
    β = ss.norm.cdf(z - θ) # calculate false negative rate

    return β


