import matplotlib.pyplot as mpl
import scipy.cluster.hierarchy as sch,random,numpy as np,pandas as pd
import yfinance as yf

"""----------------------------------------------------------------------
function: inverse variance weights
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: Snipet 16.4, Page 240
----------------------------------------------------------------------""" 
def IVP(cov, # covariance matrix
        **kargs):
  # Compute the inverse-variance portfolio
  ivp = 1./np.diag(cov) # inverse of diag of cov matrix
  ivp /= ivp.sum() # divide by sum(ivp)
  return ivp

"""----------------------------------------------------------------------
function: Compute variance of cluster
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: Snipet 16.4, Page 240
----------------------------------------------------------------------""" 
def varianceCluster(cov, # covariance matrix
                    clusteredItems): # clustered items
  # Compute variance per cluster
  covSlice = cov.loc[clusteredItems, clusteredItems] # matrix slice
  weights = IVP(covSlice).reshape(-1, 1) # make a vector of porfolio weights
  clusterVariance = np.dot(np.dot(weights.T, covSlice), weights)[0, 0] #compute variance
  return clusterVariance

"""----------------------------------------------------------------------
    function: The output is a sorted list of original items to reshape corr matrix
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: Snipet 16.2, Page 229
----------------------------------------------------------------------"""
def quasiDiagonal(link): # linkage matrix
  # Sort clustered items by distance
  link = link.astype(int) # int each element
  sortedItems = pd.Series([link[-1, 0], link[-1, 1]]) # initialize sorted array
  numItems = link[-1, 3] # number of original items
  
  while sortedItems.max() >= numItems:
    sortedItems.index = range(0, sortedItems.shape[0]*2, 2) # make space
    dataframe = sortedItems[sortedItems >= numItems] # find clusters 
    i = dataframe.index; j = dataframe.values - numItems
    sortedItems[i] = link[j, 0] # item 1
    dataframe = pd.Series(link[j, 1], index = i + 1)
    sortedItems = sortedItems.append(dataframe) # item 2
    sortedItems = sortedItems.sort_index() # re-sort
    sortedItems.index = range(sortedItems.shape[0]) # re-index
    
  return sortedItems.tolist()

"""----------------------------------------------------------------------
    function: The output is a dataframe including weights of assets
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: Snipet 16.2, Page 230
----------------------------------------------------------------------"""
def recursiveBisection(cov, # covariance matrix
                       sortedItems): # sorted items from quasiDiagonal
# Compute HRP alloc
  weights = pd.Series(1, index = sortedItems)
  clusteredItems = [sortedItems] # initialize all items in one cluster
  while len(clusteredItems)>0:
    clusteredItems = [i[j:k] for i in clusteredItems for j, k in ((0, len(i)//2), (len(i)//2, len(i))) if len(i)>1] # bi-section
    #print(clusteredItems) 
    for i in range(0, len(clusteredItems), 2): # parse in pairs
      clusteredItems0 = clusteredItems[i] # cluster 1
      clusteredItems1 = clusteredItems[i + 1] # cluster 2
      clusterVariance0 = varianceCluster(cov, clusteredItems0) # variance of cluster 1
      clusterVariance1 = varianceCluster(cov, clusteredItems1) # variance of cluster 2
      ALPHA =1 - clusterVariance0/(clusterVariance0 + clusterVariance1) # set alpha
      weights[clusteredItems0] *= ALPHA # weight 1
      weights[clusteredItems1] *= 1 - ALPHA # weight 2
  return weights

"""----------------------------------------------------------------------
    function: Distance from corr matrix
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: Snipet 16.4, Page 241
----------------------------------------------------------------------"""
def distanceCorr(corr): #corr matrix
  # A distance matrix based on correlation, where 0<=d[i,j]<=1
  distance = ((1 - corr)/2.)**.5 # distance matrix
  return distance

"""----------------------------------------------------------------------
    function: Plot correlation matrix
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: Snipet 16.4, Page 241
----------------------------------------------------------------------"""
def plotCorrMatrix(path,
                   corr,
                   labels = None):
  # Heatmap of the correlation matrix
  if labels is None:labels=[]
  mpl.pcolor(corr)
  mpl.colorbar()
  mpl.yticks(np.arange(.5, corr.shape[0] + .5), labels)
  mpl.xticks(np.arange(.5, corr.shape[0] + .5), labels)
  mpl.savefig(path)
  mpl.clf(); mpl.close() # reset pylab
  return

"""----------------------------------------------------------------------
    function: Generates random data
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: Snipet 16.4, Page 241
----------------------------------------------------------------------"""
def randomData(numberObservations, #number of observation
               size0, # size for uncorrelated data
               size1, # size for correlated data
               sigma1): # sigma for uncorrelated data
  # generating some uncorrelated data
  np.random.seed(seed = 12345);random.seed(12345)
  data1 = np.random.normal(0, 1, size = (numberObservations, size0)) # each row is a variable
  columns = [random.randint(0, (size0 - 1)) for i in range(size1)] # select random column
  data2 = data1[:, columns] + np.random.normal(0, sigma1, size = (numberObservations, len(columns))) # correlated data
  data = np.append(data1, data2, axis = 1) # merge data
  data = pd.DataFrame(data, columns = range(1, data.shape[1] + 1)) # dataframe of data
  return data, columns

"""----------------------------------------------------------------------
function: random data for MC simulation
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: Snipet 16.5, Page 242
----------------------------------------------------------------------"""
def generalRandomData(numberObservations, # number of observation
                      lengthSample, # starting point for selecting random observation
                      size0, # size of uncorrelated data
                      size1, # size of correlated data
                      mu0, # mu for uncorrelated data
                      sigma0, # sigma for uncorrelated data
                      sigma1): # sigma for correlated data
  # generate random uncorrelated data
  data1 = np.random.normal(mu0, sigma0, size = (numberObservations, size0))
  # create correlation between the variables
  columns = [random.randint(0, size0 - 1) for i in range(size1)] # randomly select columns
  data2 = data1[:, columns] + np.random.normal(0, sigma0*sigma1, size = (numberObservations, len(columns))) # correlated data
  data = np.append(data1, data2, axis = 1) # merge data
  point = np.random.randint(lengthSample, numberObservations - 1, size = 2) # select random observations
  data[np.ix_(point, [columns[0], size0])] = np.array([[-.5, -.5], [2, 2]]) # add common random shock
  point = np.random.randint(lengthSample, numberObservations - 1, size = 2) # select random observations
  data[point, columns[-1]] = np.array([-.5, 2]) # add specific random shock
  return data, columns

"""----------------------------------------------------------------------
function: HRP method
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: Snipet 16.5, Page 243
----------------------------------------------------------------------"""  
def HRP(cov, # covariance matrix
        corr): # correlation matrix
  # Construct a hierarchical portfolio
  corr, cov = pd.DataFrame(corr), pd.DataFrame(cov) # dataframe of cov and corr
  distance = distanceCorr(corr) # distance 
  link = sch.linkage(distance, 'single') # linkage matrix
  sortedItems = quasiDiagonal(link) # sort items
  sortedItems = corr.index[sortedItems].tolist() # recover labels
  hrp = recursiveBisection(cov, sortedItems) # weights
  return hrp.sort_index()

"""----------------------------------------------------------------------
function: MC simulation for out of sample comparison
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: Snipet 16.5, Page 243
----------------------------------------------------------------------""" 
def hrpMC(numberIters = 5e3, # number of iterations
          numberObservations = 520, # number of observation
          size0 = 5, # size of uncorrelated data
          size1 = 5, # size of correlated data
          mu0 = 0, # mu for uncorrelated data
          sigma0 = 1e-2, # sigma for uncorrelated data
          sigma1 = .25, # sigma for correlated data
          lengthSample = 260, # length for in sample
          testSize = 22): # observation for test set
  # Monte Carlo experiment on HRP
  methods = [IVP,HRP] #  methods
  results, numIter = {i.__name__:pd.Series() for i in methods}, 0 # initialize results and number of iteration
  pointers = range(lengthSample, numberObservations, testSize) # pointers for inSample and outSample
  while numIter<numberIters:
    #print (numIter)
    data, columns = generalRandomData(numberObservations, lengthSample, size0, size1, mu0, sigma0, sigma1) # prepare data for one experiment
    returns = {i.__name__:pd.Series() for i in methods} # initialize returns
    # Compute portfolios in-sample
    for pointer in pointers:
      inSample = data[pointer - lengthSample:pointer] # in sample
      cov_, corr_ = np.cov(inSample, rowvar = 0), np.corrcoef(inSample, rowvar = 0) # cov and corr
      # Compute performance out-of-sample
      outSample = data[pointer:pointer + testSize] # out of sample
      for func in methods:
        weight = func(cov = cov_, corr = corr_) # call methods
        ret = pd.Series(outSample @ (weight.transpose())) # return
        returns[func.__name__] = returns[func.__name__].append(ret) # update returns
    # Evaluate and store results
    for func in methods:
      ret = returns[func.__name__].reset_index(drop = True) # return column of each method
      cumprodReturn = (1 + ret).cumprod() # cumprod of returns
      results[func.__name__].loc[numIter] = cumprodReturn.iloc[-1] - 1 # update results
    numIter += 1 # next iteration
  # Report results
  results = pd.DataFrame.from_dict(results, orient = 'columns') # dataframe of results
  results.save_batch_run('results.csv') # csv file
  stdResults, varResults = results.std(), results.var() # std and var for each method
  print(pd.concat([stdResults, varResults, varResults/varResults['HRP'] - 1], axis = 1))
