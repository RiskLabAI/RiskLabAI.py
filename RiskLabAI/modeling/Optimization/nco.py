import numpy as np
import pandas as pd
from scipy.linalg import block_diag
import yfinance as yf
from sklearn.metrics import silhouette_samples
from sklearn.cluster import KMeans

def cov_to_corr(cov):
    """
    Derive the correlation matrix from a covariance matrix.

    Parameters:
        cov (numpy.ndarray): Covariance matrix.

    Returns:
        numpy.ndarray: Correlation matrix.
    """
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr[corr < -1] = -1
    corr[corr > 1] = 1
    return corr

def opt_port(cov, mu=None):
    """
    Optimal portfolio weights.

    Parameters:
        cov (numpy.ndarray): Covariance matrix.
        mu (numpy.ndarray, optional): Mean vector. Defaults to None.

    Returns:
        numpy.ndarray: Portfolio weights.
    """
    inv = np.linalg.inv(cov)
    ones = np.ones(shape=(inv.shape[0], 1))
    if mu is None:
        mu = ones
    w = np.dot(inv, mu)
    w /= np.dot(ones.T, w)
    return w

def opt_port_nco(covariance, mu=None, number_clusters=None):
    """
    NCO algorithm for optimal portfolio weights.

    Parameters:
        covariance (numpy.ndarray): Covariance matrix.
        mu (numpy.ndarray, optional): Mean vector. Defaults to None.
        number_clusters (int, optional): Maximum number of clusters. Defaults to None.

    Returns:
        numpy.ndarray: Optimal portfolio weights using NCO algorithm.
    """
    covariance = pd.DataFrame(covariance)
    if mu is not None:
        mu = pd.Series(mu[:, 0])
    correlation = cov_to_corr(covariance)
    if number_clusters is None:
        number_clusters = int(correlation.shape[0] / 2)
    correlation, clusters, _ = cluster_k_means_base(correlation, number_clusters, iterations=10)
    weights_intra_cluster = pd.DataFrame(0, index=covariance.index, columns=clusters.keys())
    for i in clusters:
        covariance_intra_cluster = covariance.loc[clusters[i], clusters[i]].values
        if mu is None:
            mu_intra_cluster = None
        else:
            mu_intra_cluster = mu.loc[clusters[i]].values.reshape(-1, 1)
        weights_intra_cluster.loc[clusters[i], i] = opt_port(covariance_intra_cluster, mu_intra_cluster).flatten()
    covariance_inter_cluster = weights_intra_cluster.T.dot(np.dot(covariance, weights_intra_cluster))
    mu_inter_cluster = (None if mu is None else weights_intra_cluster.T.dot(mu))
    weights_inter_cluster = pd.Series(opt_port(covariance_inter_cluster, mu_inter_cluster).flatten(),
                                      index=covariance_inter_cluster.index)
    weights_nco = weights_intra_cluster.mul(weights_inter_cluster, axis=1).sum(axis=1).values.reshape(-1, 1)
    return weights_nco

def cluster_k_means_base(correlation, number_clusters=10, iterations=10):
    """
    Clustering using K-means algorithm.

    Parameters:
        correlation (pd.DataFrame): Correlation matrix.
        number_clusters (int, optional): Maximum number of clusters. Defaults to 10.
        iterations (int, optional): Number of iterations. Defaults to 10.

    Returns:
        pd.DataFrame, dict, pd.Series: Updated correlation matrix, cluster members, silhouette scores.
    """
    distance = ((1 - correlation.fillna(0)) / 2.) ** 0.5
    silh = pd.Series()
    
    for init in range(iterations):
        for i in range(2, number_clusters + 1):
            kmeans_ = KMeans(n_clusters=i, n_jobs=1, iterations=1)
            kmeans_ = kmeans_.fit(distance)
            silh_ = silhouette_samples(distance, kmeans_.labels_)
            statistic = (silh_.mean() / silh_.std(), silh.mean() / silh.std())
            
            if np.isnan(statistic[1]) or statistic[0] > statistic[1]:
                silh, kmeans = silh_, kmeans_

    index_new = np.argsort(kmeans.labels_)
    correlation_new = correlation.iloc[index_new]
    correlation_new = correlation_new.iloc[:, index_new]
    
    clusters = {i: correlation.columns[np.where(kmeans.labels_ == i)[0]].tolist()
                for i in np.unique(kmeans.labels_)}
    
    silh = pd.Series(silh, index=distance.index)
    return correlation_new, clusters, silh
