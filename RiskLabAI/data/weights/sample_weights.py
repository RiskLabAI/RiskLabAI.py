
import imp
import numpy as np 
import pandas as pd 

"""
    function: expand label tO incorporate meta-labeling
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: SNIPPET 4.1
"""
def concurrencyEvents(closeIndex, # DataFrame that has events
                      timestamp, # DateFrame that has return and label of each period
                      molecule, # index that function must apply on it
    ): #1) find events that span the period [molecule[0],molecule[-1]]
    timestamp=timestamp.fillna(closeIndex[-1]) # unclosed events still must impact other weights
    timestamp=timestamp[timestamp>=molecule[0]] # events that end at or after molecule[0]
    timestamp=timestamp.loc[:timestamp[molecule].max()] # events that start at or before t1[molecule].max()
    #2) count events spanning a bar
    iloc=closeIndex.searchsorted(np.array([timestamp.index[0],timestamp.max()]))
    count=pd.Series(0,index=closeIndex[iloc[0]:iloc[1]+1])
    for tIn,tOut in timestamp.iteritems():
        count.loc[tIn:tOut]+=1. #add for new events 

    return count.loc[molecule[0]:timestamp[molecule].max()]

"""
    function: SampleWeight with triple barrier
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: SNIPPET 4.2
"""
def mpSampleWeight(
        timestamp, # DataFrame of events start and end for labelling 
        concurrencyEvents, # Data frame of concurrent events for each events 
        molecule # index that function must apply on it
    ): # Derive average uniqueness over the event's lifespan
    weight=pd.Series(index=molecule) # create pandas object fot weights 
    for tIn,tOut in timestamp.loc[weight.index].iteritems():
        weight.loc[tIn]=(1./concurrencyEvents.loc[tIn:tOut]).mean() #compute sample weight according to book equation 
    return weight

"""
    function: Creating Index matrix 
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: SNIPPET 4.3 
"""
def index_matrix(
        barIndex, #index of all data 
        timestamp #times of events contain starting and ending time 
    ): # Get indicator matrix
    indM=pd.DataFrame(0,index=barIndex,columns=range(timestamp.shape[0]))
    for row in timestamp.itertuples():
        t0 =  int(row.date) 
        t1 = int(row.timestamp)
        indM.loc[t0:t1,row.Index]=1.
    return indM

"""
    function: compute average uniqueness
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: SNIPPET 4.4
"""   
def averageUniqueness(indexMatrix):
    # Average uniqueness from indicator matrix
    c=indexMatrix.sum(axis=1) # concurrency
    u=indexMatrix.div(c,axis=0) # uniqueness
    averageUniqueness_=u[u>0].mean() # average uniqueness
    return averageUniqueness_

"""
    function:  SequentialBootstrap implementation 
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: SNIPPET 4.5
"""
def sequential_bootstrap(
        indexMatrix, #matrix that Indicator for events 
        sampleLength # number of sample
    ): # Generate a sample via sequential bootstrap
    if sampleLength is None:sampleLength=indexMatrix.shape[1]
    phi=[]
    while len(phi)<sampleLength:
        averageUniqueness_=pd.Series(dtype=np.float64)
        for i in indexMatrix:
            indexMatrix_=indexMatrix[phi+[i]] # reduce indM
            averageUniqueness_.loc[i]=averageUniqueness(indexMatrix_).iloc[-1]
        prob=averageUniqueness_/averageUniqueness_.sum() # draw prob
        phi+=[np.random.choice(indexMatrix.columns,p=prob)]
    return phi

"""
    function:  sample weight with returns 
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: SNIPPET 4.10
"""
def mpSampleWeightAbsoluteReturn(
        timestamp, # dataFrame for events 
        concurrencyEvents, # dataframe that contain number of concurrent events for each events
        returns, # data frame that contains returns
        molecule # molecule
    ): # Derive sample weight by return attribution
    
    return_=np.log(returns).diff() # log-returns, so that they are additive
    weight=pd.Series(index=molecule)
    for tIn,tOut in timestamp.loc[weight.index].iteritems():
        weight.loc[tIn]=(return_.loc[tIn:tOut]/concurrencyEvents.loc[tIn:tOut]).sum() # compute sample weight 
    return weight.abs()

"""
    function:  compute TimeDecay
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: SNIPPET 4.11
"""
def timeDecay(
        weight, #weight that compute for each event 
        clfLastW = 1.0 # weight of oldest observation
    ): # apply piecewise-linear decay to observed uniqueness (weight)

    # newest observation gets weight=1, oldest observation gets weight=clfLastW
    clfW=weight.sort_index().cumsum() # compute cumulative sum of weights 
    if clfLastW>=0:
        slope=(1.-clfLastW)/clfW.iloc[-1] # compute slope of line 
    else:
        slope=1./((clfLastW+1)*clfW.iloc[-1]) # compute slope of line 
    const=1.-slope*clfW.iloc[-1] # compute b in y =ax + b 
    clfW=const+slope*clfW # compute points on line 
    clfW[clfW<0]=0 # if clfw is less than zero set that entry to zero 
    return clfW

