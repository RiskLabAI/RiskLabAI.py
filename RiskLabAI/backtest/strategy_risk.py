import numpy as np 
from sympy import *
import scipy.stats as ss

"""
	function: targets a Sharpe ratio as a function of the number of bets
	reference: De Prado, M. (2018) Advances in financial machine learning.
	methodology: page 213, snippet 15.1
"""
def sharpe_ratio_trials(p, # probability of success
                        n_run): # number of runs

    output = [] # initial results list

    for i in range(n_run):
        random = np.random.binomial(n=1, p=p) # generate random number with binomial distribution with probability p
        x = (1 if random == 1 else -1) # find if the generated number is 1 or -1
        output.append(x) # append result

    return np.mean(output), np.std(output), np.mean(output) / np.std(output)

"""
	function: uses the SymPy library for symbolic operations 
	reference: De Prado, M. (2018) Advances in financial machine learning.
	methodology: page 214, snippet 15.2
"""
def target_sharpe_ratio_symbolic():

    p,u,d = symbols("p u d") # Create symbols

    m2 = p*u**2 + (1 - p)*d**2 # do symbolic operations
    m1 = p*u + (1 - p)*d # do symbolic operations
    v = m2 - m1**2 # do symbolic operations

    return factor(v) 

"""
	function: computes implied precision 
	reference: De Prado, M. (2018) Advances in financial machine learning.
	methodology: page 214, snippet 15.3
"""
def implied_precision(stop_loss, # stop loss threshold
			   		  profit_taking, # profit taking threshold
			   		  frequency, # number of bets per year
			   	 	  target_sharpe_ratio): # target annual Sharpe ratio

	a = (frequency + target_sharpe_ratio**2)*(profit_taking - stop_loss)**2 # calculate the "a" parameter for the quadratic euation
	b = (2*frequency*stop_loss - target_sharpe_ratio**2*(profit_taking - stop_loss))*(profit_taking - stop_loss) # calculate the "b" parameter for the quadratic euation
	c = frequency*stop_loss**2 # calculate the "c" parameter for the quadratic euation
	precision = (-b + (b**2 - 4*a*c)**0.5)/(2*a) # solve the quadratic euation

	return precision

"""
	function: computes the number of bets/year needed to achieve a Sharpe ratio with a certain precision rate
	reference: De Prado, M. (2018) Advances in financial machine learning.
	methodology: page 215, snippet 15.4
"""
def bin_frequency(stop_loss, # stop loss threshold
				  profit_taking, # profit taking threshold
				  precision, # precision rate p
				  target_sharpe_ratio): # target annual Sharpe ratio

	frequency = (target_sharpe_ratio*(profit_taking - stop_loss))**2*precision*(1 - precision) / ((profit_taking - stop_loss)*precision + stop_loss)**2  # calculate possible extraneous
	
	if not np.isclose(binSR(stop_loss, profit_taking, frequency, precision), target_sharpe_ratio): # check if it's near the target Sharpe Ratio
		return None
    
	return frequency

def binSR(sl, pt, frequency, p):
    return ((pt - sl)*p + sl) / ((pt - sl)*(p*(1 - p))**0.5)*frequency**0.5  # Define Sharpe Ratio function        

"""
	function: calculates the strategy risk in practice
	reference: De Prado, M. (2018) Advances in financial machine learning.
	methodology: page 215, snippet 15.4
"""
def mixGaussians(μ1, # mean of the first gaussian distribution to generate bet outcomes
				 μ2, # mean of the second gaussian distribution to generate bet outcomes
			     σ1, # standard deviation of the first gaussian distribution to generate bet outcomes
			     σ2, # standard deviation of the second gaussian distribution to generate bet outcomes
				 probability, # probability of success
				 nObs): # number of observations

    return1 = np.random.normal(μ1, σ1, size=int(nObs*probability)) # draw random bet outcomes from a gaussian distribution
    return2 = np.random.normal(μ2, σ2, size=int(nObs) - return1.shape[0]) # draw random bet outcomes from a gaussian distribution
	
    returns = np.append(return1, return2, axis=0) # append returns
    np.random.shuffle(returns) # shuffle returns

    return returns


def failure_probability(returns, # returns list
					    frequency, # number of bets per year
					    target_sharpe_ratio): # target annual Sharpe ratio

    # derive the probability that the strategy may fail
    rPositive, rNegative = returns[returns > 0].mean(), returns[returns <= 0].mean() # divide returns
    p = returns[returns > 0].shape[0] / float(returns.shape[0]) # find success rate
    thresholdP = implied_precision(rNegative, rPositive, frequency, target_sharpe_ratio) # calculate success rate threshold
    risk = ss.norm.cdf(thresholdP, p, p*(1 - p)) # approximate to bootstrap

    return risk

def calculate_strategy_risk(μ1, # mean of the first gaussian distribution to generate bet outcomes
						    μ2, # mean of the second gaussian distribution to generate bet outcomes
						    σ1, # standard deviation of the first gaussian distribution to generate bet outcomes
						    σ2, # standard deviation of the second gaussian distribution to generate bet outcomes
						    probability, # probability of success
						    nObs, # number of observations
						    frequency, # number of bets per year
						    target_sharpe_ratio): # target annual Sharpe ratio
							   #1) Parameters

    returns = mixGaussians(μ1, μ2, σ1, σ2, probability, nObs) # 2) Generate sample from mixture

    probabilityFail = failure_probability(returns, frequency, target_sharpe_ratio) # 3) Compute failure probability
    print("Prob strategy will fail ", probabilityFail) # print result

    return probabilityFail

    