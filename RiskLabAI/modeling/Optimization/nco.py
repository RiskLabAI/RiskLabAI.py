import matplotlib.pyplot as mpl
import random,numpy as np,pandas as pd
import matplotlib.pyplot as mpl,seaborn as sns
import numpy as np
from scipy.linalg import block_diag
import yfinance as yf
from sklearn.metrics import silhouette_samples
from sklearn.cluster import KMeans

"""----------------------------------------------------------------------
    function: Derive the correlation matrix from a covariance matrix
    reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    methodology: Snipet 2.3, Page 27
----------------------------------------------------------------------"""
def covToCorr(cov): # covariance matrix
    std = np.sqrt(np.diag(cov)) # standard deviations
    corr = cov/np.outer(std, std) # create correlation matrix
    corr[corr < -1], corr[corr > 1] = -1, 1 # numerical error
    return corr


"""----------------------------------------------------------------------
    function: Monte Carlo simulation
    reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    methodology: Snipet 2.10, Page 34
----------------------------------------------------------------------"""
def optPort(cov, # covariance matrix
            mu = None): # mean vector
    inv = np.linalg.inv(cov) # inverse of cov 
    ones = np.ones(shape = (inv.shape[0], 1)) # ones matrix for mean vector
    if mu is None:mu = ones
    w = np.dot(inv, mu) # compute weights
    w /= np.dot(ones.T, w) # normalize weights
    return w
    
"""----------------------------------------------------------------------
    function: NCO algorithm
    reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    methodology: Snipet 7.6, Page 100
----------------------------------------------------------------------"""
def optPortNCO(covariance, # covariance matrix
               mu = None, # mean vector
               numberClusters = None): # maximum number of clusters
  covariance = pd.DataFrame(covariance)
  if mu is not None:mu = pd.Series(mu[:, 0])
  correlation = covToCorr(covariance) # corr dataframe
  if numberClusters == None:
    numberClusters = int(correlation.shape[0]/2) # set maximum number of clusters
  correlation, clusters, _ = clusterKMeansBase(correlation, numberClusters, iterations = 10) # clustering
  weightsIntraCluster = pd.DataFrame(0, index = covariance.index, columns = clusters.keys()) # dataframe of intraclustering weights
  for i in clusters:
    covarianceIntraCluster = covariance.loc[clusters[i], clusters[i]].values # slice cov matrix
    if mu is None:muIntraCluster = None
    else:muIntraCluster = mu.loc[clusters[i]].values.reshape(-1, 1)
    weightsIntraCluster.loc[clusters[i], i] = optPort(covarianceIntraCluster, muIntraCluster).flatten() # calculate weights of intraclustering
  covarianceInterCluster = weightsIntraCluster.T.dot(np.dot(covariance, weightsIntraCluster)) # reduce covariance matrix
  muInterCluster = (None if mu is None else weightsIntraCluster.T.dot(mu)) # calculate mean for each cluster
  weightsInterCluster = pd.Series(optPort(covarianceInterCluster, muInterCluster).flatten(), index = covarianceInterCluster.index) # calculate weights of interclustering
  weightsNCO = weightsIntraCluster.mul(weightsInterCluster, axis = 1).sum(axis = 1).values.reshape(-1, 1) # calculate weights
  return weightsNCO

"""----------------------------------------------------------------------
    function: Clustering
    reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    methodology: Snipet 4.1, Page 56
----------------------------------------------------------------------"""
def clusterKMeansBase(correlation, # corr pd.dataframe
                      numberClusters = 10, # maximum number of clusters 
                      iterations = 10): # iterations
  distance, silh=((1 - correlation.fillna(0))/2.)**.5,pd.Series() # observations matrix
  for init in range(iterations):
    for i in range(2,numberClusters + 1):
      kmeans_ = KMeans(n_clusters = i, n_jobs = 1, iterations = 1) # clustering distance with maximum cluster i
      kmeans_ = kmeans_.fit(distance) # fit kmeans 
      silh_ = silhouette_samples(distance, kmeans_.labels_) # calculate silh scores
      statistic = (silh_.mean()/silh_.std(), silh.mean()/silh.std()) # calculate t-statistic
      if np.isnan(statistic[1]) or statistic[0]>statistic[1]:
        silh, kmeans = silh_, kmeans_ # choose better clustering
  indexNew = np.argsort(kmeans.labels_) # sort arguments
  correlationNew = correlation.iloc[indexNew] # reorder rows
  correlationNew = correlationNew.iloc[:, indexNew] # reorder columns
  clusters = {i:correlation.columns[np.where(kmeans.labels_==i)[0]].tolist() for i in np.unique(kmeans.labels_)} # cluster members
  silh = pd.Series(silh, index = distance.index) # silh scores
  return correlationNew, clusters, silh




