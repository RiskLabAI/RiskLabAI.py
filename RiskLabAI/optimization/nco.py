import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

def covariance_to_correlation_matrix(covariance: np.ndarray) -> np.ndarray:
    """
    Derive the correlation matrix from a covariance matrix.
    
    :param covariance: Covariance matrix.
    :type covariance: numpy.ndarray
    :return: Correlation matrix.
    :rtype: numpy.ndarray
    """
    standard_deviation = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(standard_deviation, standard_deviation)
    correlation[correlation < -1] = -1
    correlation[correlation > 1] = 1
    return correlation


def get_optimal_portfolio_weights(covariance: np.ndarray, mu: np.ndarray = None) -> np.ndarray:
    """
    Compute the optimal portfolio weights.
    
    :param covariance: Covariance matrix.
    :type covariance: numpy.ndarray
    :param mu: Mean vector, defaults to None.
    :type mu: numpy.ndarray, optional
    :return: Portfolio weights.
    :rtype: numpy.ndarray
    """
    inverse_covariance = np.linalg.inv(covariance)
    ones = np.ones(shape=(inverse_covariance.shape[0], 1))
    if mu is None:
        mu = ones
    weights = np.dot(inverse_covariance, mu)
    weights /= np.dot(ones.T, weights)
    return weights


def get_optimal_portfolio_weights_nco(
        covariance: np.ndarray,
        mu: np.ndarray = None,
        number_clusters: int = None) -> np.ndarray:
    """
    Compute the optimal portfolio weights using the NCO algorithm.
    
    :param covariance: Covariance matrix.
    :type covariance: numpy.ndarray
    :param mu: Mean vector, defaults to None.
    :type mu: numpy.ndarray, optional
    :param number_clusters: Maximum number of clusters, defaults to None.
    :type number_clusters: int, optional
    :return: Optimal portfolio weights using NCO algorithm.
    :rtype: numpy.ndarray
    """
    covariance = pd.DataFrame(covariance)
    correlation = covariance_to_correlation_matrix(covariance)
    if mu is not None:
        mu = pd.Series(mu[:, 0])
    if number_clusters is None:
        number_clusters = int(correlation.shape[0] / 2)
    correlation, clusters, _ = cluster_k_means_base(correlation, number_clusters)
    weights_intra_cluster = pd.DataFrame(0, index=covariance.index, columns=clusters.keys())
    for i in clusters:
        covariance_intra_cluster = covariance.loc[clusters[i], clusters[i]].values
        if mu is None:
            mu_intra_cluster = None
        else:
            mu_intra_cluster = mu.loc[clusters[i]].values.reshape(-1, 1)
        weights_intra_cluster.loc[clusters[i], i] = get_optimal_portfolio_weights(
            covariance_intra_cluster, mu_intra_cluster).flatten()
    covariance_inter_cluster = weights_intra_cluster.T.dot(np.dot(covariance, weights_intra_cluster))
    mu_inter_cluster = None if mu is None else weights_intra_cluster.T.dot(mu)
    weights_inter_cluster = pd.Series(
        get_optimal_portfolio_weights(covariance_inter_cluster, mu_inter_cluster).flatten(),
        index=covariance_inter_cluster.index)
    weights_nco = weights_intra_cluster.mul(weights_inter_cluster, axis=1).sum(axis=1).values.reshape(-1, 1)
    return weights_nco


def cluster_k_means_base(
        correlation: pd.DataFrame,
        number_clusters: int = 10,
        iterations: int = 10) -> tuple:
    """
    Perform clustering using the K-means algorithm.
    
    :param correlation: Correlation matrix.
    :type correlation: pd.DataFrame
    :param number_clusters: Maximum number of clusters, defaults to 10.
    :type number_clusters: int, optional
    :param iterations: Number of iterations, defaults to 10.
    :type iterations: int, optional
    :return: Updated correlation matrix, cluster members, silhouette scores.
    :rtype: tuple
    """
    distance = ((1 - correlation.fillna(0)) / 2.0) ** 0.5
    silhouette_scores = pd.Series()

    for init in range(iterations):
        for i in range(2, number_clusters + 1):
            kmeans_ = KMeans(n_clusters=i, n_init=1)
            kmeans_ = kmeans_.fit(distance)
            silhouette_scores_ = silhouette_samples(distance, kmeans_.labels_)
            statistic = (silhouette_scores_.mean() / silhouette_scores_.std(),
                         silhouette_scores.mean() / silhouette_scores.std())
            if np.isnan(statistic[1]) or statistic[0] > statistic[1]:
                silhouette_scores, kmeans = silhouette_scores_, kmeans_

    index_new = np.argsort(kmeans.labels_)
    correlation_new = correlation.iloc[index_new]
    correlation_new = correlation_new.iloc[:, index_new]
    clusters = {i: correlation.columns[np.where(kmeans.labels_ == i)[0]].tolist() for i in np.unique(kmeans.labels_)}
    silhouette_scores = pd.Series(silhouette_scores, index=distance.index)
 
    return correlation_new, clusters, silhouette_scores
