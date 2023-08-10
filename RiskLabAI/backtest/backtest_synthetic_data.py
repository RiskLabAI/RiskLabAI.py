import numpy as np
from random import gauss
from itertools import product

"""----------------------------------------------------------------------
function: backTesting with synthetic data
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: p.175 snippet 13.1
----------------------------------------------------------------------"""
def syntheticBackTesting(forecast,# forecast price 
                         halfLife, # halflife time need to reach half 
                         sigma,nIteration =1e5, # std of noise
                         maximumHoldingPeriod=100,# number of iteration
                         profitTakingRange=np.linspace(.5,10,20),# maximumHoldingPeriod
                         stopLossRange=np.linspace(.5,10,20),
                         seed=0):

    phi,backTest=2**(-1./halfLife),[] # compute ρ coeficient from halflife
    for combination_ in product(profitTakingRange,stopLossRange):
        stopPrices=[]
        for iter_ in range(int(nIteration)):
            price,holdingPeriod=seed,0 # initial price 
            while True:
                price=(1-phi)*forecast+phi*price+sigma*gauss(0,1) # update price according to  O_U process 
                gain=price-seed # compute gain
                holdingPeriod+=1
                if gain>combination_[0] or gain<-combination_[1] or holdingPeriod>maximumHoldingPeriod: # check stop condition 
                    stopPrices.append(gain)
                break
        mean,std=np.mean(stopPrices),np.std(stopPrices) # compute mean and std of samples 
        backTest.append((combination_[0],combination_[1],mean,std,mean/std)) # add mean and std and sharp ratio to backTest data 
    return backTest