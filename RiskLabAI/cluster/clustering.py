import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from scipy.linalg import block_diag
from sklearn.utils import check_random_state
from typing import Tuple

def covariance_to_correlation(
        covariance: np.ndarray
) -> np.ndarray:
    """
    Derive the correlation matrix from a covariance matrix.

    .. math::
        \\text{correlation}_{ij} = \\frac{\\text{covariance}_{ij}}{\\sqrt{\\text{covariance}_{ii} \\text{covariance}_{jj}}}

    Reference:
        De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
        Snippet 2.3, Page 27

    :param covariance: Covariance matrix.

    :return: Correlation matrix.
    """
    std = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(std, std)
    correlation[correlation < -1], correlation[correlation > 1] = -1, 1  # Handle numerical error

    return correlation

def cluster_k_means_base(
        correlation: pd.DataFrame,
        max_clusters: int = 10,
        iterations: int = 10
) -> Tuple[pd.DataFrame, dict, pd.Series]:
    """
    Clustering using K-Means.

    Reference:
        De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
        Snippet 4.1, Page 56

    :param correlation: Correlation matrix.
    :param max_clusters: Maximum number of clusters.
    :param iterations: Number of iterations for clustering.

    :return: Tuple containing the sorted correlation matrix, clusters, and silhouette scores.
    """
    distance = ((1 - correlation.fillna(0)) / 2.0)**0.5
    silhouette = pd.Series()

    for init in range(iterations):
        for i in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=i, n_init=1)
            kmeans = kmeans.fit(distance)
            silhouette_ = silhouette_samples(distance, kmeans.labels_)
            statistic = (silhouette_.mean() / silhouette_.std(), silhouette.mean() / silhouette.std())
            

            if np.isnan(statistic[1]) or statistic[0] > statistic[1]:
                silhouette, best_kmeans = silhouette_, kmeans
    index_sorted = np.argsort(best_kmeans.labels_)
    correlation_sorted = correlation.iloc[index_sorted, index_sorted]
    clusters = {i: correlation.columns[np.where(best_kmeans.labels_ == i)[0]].tolist() for i in np.unique(best_kmeans.labels_)}
    silhouette = pd.Series(silhouette, index=distance.index)

    return correlation_sorted, clusters, silhouette


def make_new_outputs(
        correlation: pd.DataFrame,
        clusters_1: dict,
        clusters_2: dict
) -> Tuple[pd.DataFrame, dict, pd.Series]:
    """
    Merge two clusters and produce new correlation matrix and silhouette scores.
    Clusters are disjoint.
    
    Reference:
        De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
        Snippet 4.2, Page 58

    :param correlation: Correlation matrix.
    :param clusters_1: First cluster.
    :param clusters_2: Second cluster.

    :return: Tuple containing the new correlation matrix, new clusters, and new silhouette scores.
    """
    clusters_new = {}
    for i in clusters_1.keys():
        clusters_new[len(clusters_new.keys())] = list(clusters_1[i])
    for i in clusters_2.keys():
        clusters_new[len(clusters_new.keys())] = list(clusters_2[i])

    index_new = [j for i in clusters_new for j in clusters_new[i]]
    correlation_new = correlation.loc[index_new, index_new]
    distance = ((1 - correlation.fillna(0)) / 2.0)**0.5
    labels_kmeans = np.zeros(len(distance.columns))

    for i in clusters_new.keys():
        index = [distance.index.get_loc(k) for k in clusters_new[i]]
        labels_kmeans[index] = i

    silhouette_new = pd.Series(silhouette_samples(distance, labels_kmeans), index=distance.index)
    return correlation_new, clusters_new, silhouette_new

def cluster_k_means_top(
        correlation: pd.DataFrame,
        max_clusters: int = None,
        iterations: int = 10
) -> Tuple[pd.DataFrame, dict, pd.Series]:
    """
    Clustering using ONC method.
    
    Reference:
        De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
        Snippet 4.2, Page 58

    :param correlation: Correlation matrix.
    :param max_clusters: Maximum Number of clusters.
    :param iterations: Number of iterations.

    :return: Tuple containing the sorted correlation matrix, clusters, and silhouette scores.
    """
    if max_clusters is None:
        max_clusters = correlation.shape[1] - 1

    correlation_sorted, clusters, silhouette = cluster_k_means_base(correlation, max_clusters=min(max_clusters, correlation.shape[1] - 1), iterations=iterations)
    cluster_t_stats = {i: np.mean(silhouette[clusters[i]]) / np.std(silhouette[clusters[i]]) for i in clusters.keys()}
    t_stat_mean = sum(cluster_t_stats.values()) / len(cluster_t_stats)
    redo_clusters = [i for i in cluster_t_stats.keys() if cluster_t_stats[i] < t_stat_mean]

    if len(redo_clusters) <= 1:
        return correlation_sorted, clusters, silhouette
    else:
        keys_redo = [j for i in redo_clusters for j in clusters[i]]
        correlation_temp = correlation.loc[keys_redo, keys_redo]
        t_stat_mean = np.mean([cluster_t_stats[i] for i in redo_clusters])

        remained_n_clusters = max_clusters - len(clusters) + len(redo_clusters)
        correlation_sorted2, clusters2, silh2 = cluster_k_means_top(correlation_temp, max_clusters=min(remained_n_clusters, correlation_temp.shape[1] - 1), iterations=iterations)
        correlation_new, clusters_new, silh_new = make_new_outputs(correlation, {i: clusters[i] for i in clusters.keys() if i not in redo_clusters}, clusters2)

        new_t_stat_mean = np.mean([np.mean(silh_new[clusters_new[i]]) / np.std(silh_new[clusters_new[i]]) for i in clusters_new.keys()])

        if new_t_stat_mean <= t_stat_mean:
            return correlation_sorted, clusters, silhouette
        else:
            return correlation_new, clusters_new, silh_new
        
        
def random_covariance_sub(
        n_observations: int,
        n_columns: int,
        sigma: float,
        random_state: int = None
) -> np.ndarray:
    """
    Generates covariance matrix for n_columns same normal random variables with a nomral noise scaled by sigma.
    Variables have n_observations observations.
    
    Reference:
        De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
        Snippet 4.3, Page 61

    :param n_observations: Number of observations.
    :param n_columns: Number of columns.
    :param sigma: Sigma for normal distribution.
    :param random_state: Random state for reproducibility.

    :return: Sub covariance matrix.
    """
    domain = check_random_state(random_state)

    if n_columns == 1:
        return np.ones((1, 1))

    data = domain.normal(size=(n_observations, 1))
    data = np.repeat(data, n_columns, axis=1)
    data += domain.normal(scale=sigma, size=data.shape)
    covariance = np.cov(data, rowvar=False)

    return covariance


def random_block_covariance(
        n_columns: int,
        n_blocks: int,
        block_size_min: int = 1,
        sigma: float = 1.0,
        random_state: int = None
) -> np.ndarray:
    """
    Compute random block covariance matrix.
    
    Reference:
        De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
        Snippet 4.3, Page 61

    :param n_columns: Number of columns.
    :param n_blocks: Number of blocks.
    :param block_size_min: Minimum size of block.
    :param sigma: Sigma for normal distribution.
    :param random_state: Random state for reproducibility.

    :return: Random block covariance matrix.
    """
    domain = check_random_state(random_state)

    parts = domain.choice(range(1, n_columns - (block_size_min - 1) * n_blocks), n_blocks - 1, replace=False)
    parts.sort()
    parts = np.append(parts, n_columns - (block_size_min - 1) * n_blocks)
    parts = np.append(parts[0], np.diff(parts)) - 1 + block_size_min

    cov = None
    for column in parts:
        this_covariance = random_covariance_sub(int(max(column * (column + 1) / 2., 100)), column, sigma, random_state=domain)
        if cov is None:
            cov = this_covariance.copy()
        else:
            cov = block_diag(cov, this_covariance)

    return cov


def random_block_correlation(
        n_columns: int,
        n_blocks: int,
        random_state: int = None,
        block_size_min: int = 1
) -> pd.DataFrame:
    """
    Compute random block correlation matrix.
    
    Reference:
        De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
        Snippet 4.3, Page 61

    :param n_columns: Number of columns.
    :param n_blocks: Number of blocks.
    :param random_state: Random state for reproducibility.
    :param block_size_min: Minimum size of block.

    :return: Random block correlation matrix.
    """

    domain = check_random_state(random_state)

    covariance1 = random_block_covariance(n_columns, n_blocks, block_size_min=block_size_min, sigma=0.5, random_state=domain)
    covariance2 = random_block_covariance(n_columns, 1, block_size_min=block_size_min, sigma=1.0, random_state=domain)

    covariance1 += covariance2
    correlation = covariance_to_correlation(covariance1)
    correlation = pd.DataFrame(correlation)

    return correlation