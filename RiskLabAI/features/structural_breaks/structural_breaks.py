
import numpy as np 
import pandas as pd 

"""----------------------------------------------------------------------
    function: apply lags to dataframe 
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: snippet 17.3
----------------------------------------------------------------------"""
def lagDF(marketData,# data of price or log price
          lags): # arrays of lag or integer that show number of lags 
    laggedData=pd.DataFrame() # create lagged Dataframe 
    if isinstance(lags,int): #check lags is array of integer or is integer  and if it is integer change it to array 
        lags=range(lags+1)
    else:
        lags=[int(lag) for lag in lags]
    for lag in lags:
        tempData=pd.DataFrame(marketData.shift(lag).copy(deep=True)) # shift market data 
        tempData.columns=[str(i)+'_'+str(lag) for i in tempData.columns] # change column names 
        laggedData=laggedData.join(tempData,how='outer') # add tempData to laggedData
    return laggedData

"""----------------------------------------------------------------------
    function: preparing the datasets
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: snippet 17.2
----------------------------------------------------------------------"""
def prepareData(series, # data of price or log price
                constant, # string thant must be "nc" or "ct" or "ctt"
                lags): # arrays of lag or integer that show number of lags 
    series_=series.diff().dropna() #compute data difference  
    x=lagDF(series_,lags).dropna()  # compute lagged data 
    x.iloc[:,0]=series.values[-x.shape[0]-1:-1,0] # lagged level
    y=series_.iloc[-x.shape[0]:].values # dependent variable 
    if constant!='nc': # check type of model 
        x=np.append(x,np.ones((x.shape[0],1)),axis=1) #add a column with 1 entry 
    if constant[:2]=='ct':
        trend=np.arange(x.shape[0]).reshape(-1,1)  #add first moment of trend 
        x=np.append(x,trend,axis=1)
    if constant=='ctt':
        x=np.append(x,trend**2,axis=1)  # add second moment of trend 
    return y,x

"""----------------------------------------------------------------------
    function: fitting the adf specification
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: snippet 17.4
----------------------------------------------------------------------"""
def computeBeta(y, # dependent variable
                x):# matrix of independent variable 
    xInvers = np.linalg.inv(np.dot(x.T,x))
    betaMean = xInvers * np.dot(x.T,y)  # Compute β with OLS estimator 
    epsilon=y-np.dot(x,betaMean) # compute error 
    betaVariance=np.dot(epsilon.T,epsilon)/(x.shape[0]-x.shape[1])*xInvers # compute variance of β
    return betaMean ,betaVariance

"""----------------------------------------------------------------------
    function: sadf’s inner loop
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: snippet 17.1
----------------------------------------------------------------------"""
def ADF(logPrice, #  pandas dataframe of logprice 
        minSampleLength, #minimum sample length
        constant,  # string thant must be "nc" or "ct" or "ctt"
        lags): # arrays of lag or integer that show number of lags 
    y,x=prepareData(logPrice,constant=constant,lags=lags)  # preparing data 
    startPoints,bsadf,allADF=range(0,y.shape[0]+lags-minSampleLength+1),-1 *np.inf,[] #initial variable bsadf is best supremum ADF 
    for start in startPoints:
        y_,x_=y[start:],x[start:] # select data 
        betaMean_,betaStd_=computeBeta(y_,x_) # compute beta mean and beta std with OLS
        betaMean_,betaStd_=betaMean_[0,0],betaStd_[0,0]**.5 # select coefficient of first independent variable 
        allADF.append(betaMean_/betaStd_) # append it to allADF
        if allADF[-1]>bsadf:
            bsadf=allADF[-1] # update  bsadf 
    out={'Time':logPrice.index[-1],'gsadf':bsadf}
    return out



