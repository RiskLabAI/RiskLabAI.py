import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from scipy.linalg import block_diag
from sklearn.utils import check_random_state


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
) -> (pd.DataFrame, dict, pd.Series):
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
    silh = pd.Series()

    for init in range(iterations):
        for i in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=i, n_init=1)
            kmeans = kmeans.fit(distance)
            silh_ = silhouette_samples(distance, kmeans.labels_)
            statistic = (silh_.mean() / silh_.std(), silh.mean() / silh.std())

            if np.isnan(statistic[1]) or statistic[0] > statistic[1]:
                silh, best_kmeans = silh_, kmeans

    index_sorted = np.argsort(best_kmeans.labels_)
    correlation_sorted = correlation.iloc[index_sorted]
    correlation_sorted = correlation_sorted.iloc[:, index_sorted]
    clusters = {i: correlation.columns[np.where(best_kmeans.labels_ == i)[0]].tolist() for i in np.unique(best_kmeans.labels_)}
    silh = pd.Series(silh, index=distance.index)

    return correlation_sorted, clusters, silh

import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_samples


def make_new_outputs(
        correlation: pd.DataFrame,
        clusters_1: dict,
        clusters_2: dict
) -> (pd.DataFrame, dict, pd.Series):
    """
    Merge two clusters and produce new correlation matrix and silhouette scores.

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

    silh_new = pd.Series(silhouette_samples(distance, labels_kmeans), index=distance.index)
    return correlation_new, clusters_new, silh_new


def cluster_k_means_top(
        correlation: pd.DataFrame,
        num_clusters: int = None,
        iterations: int = 10
) -> (pd.DataFrame, dict, pd.Series):
    """
    Clustering using ONC method.

    :param correlation: Correlation matrix.
    :param num_clusters: Number of clusters.
    :param iterations: Number of iterations.

    :return: Tuple containing the sorted correlation matrix, clusters, and silhouette scores.
    """
    if num_clusters is None:
        num_clusters = correlation.shape[1] - 1

    correlation_sorted, clusters, silh = cluster_k_means_base(correlation, num_clusters=min(num_clusters, correlation.shape[1] - 1), iterations=iterations)
    cluster_t_stats = {i: np.mean(silh[clusters[i]]) / np.std(silh[clusters[i]]) for i in clusters.keys()}
    t_stat_mean = sum(cluster_t_stats.values()) / len(cluster_t_stats)
    redo_clusters = [i for i in cluster_t_stats.keys() if cluster_t_stats[i] < t_stat_mean]

    if len(redo_clusters) <= 1:
        return correlation_sorted, clusters, silh
    else:
        keys_redo = [j for i in redo_clusters for j in clusters[i]]
        correlation_temp = correlation.loc[keys_redo, keys_redo]
        t_stat_mean = np.mean([cluster_t_stats[i] for i in redo_clusters])

        correlation_sorted2, clusters2, silh2 = cluster_k_means_top(correlation_temp, num_clusters=min(num_clusters, correlation_temp.shape[1] - 1), iterations=iterations)
        correlation_new, clusters_new, silh_new = make_new_outputs(correlation, {i: clusters[i] for i in clusters.keys() if i not in redo_clusters}, clusters2)

        new_t_stat_mean = np.mean([np.mean(silh_new[clusters_new[i]]) / np.std(silh_new[clusters_new[i]]) for i in clusters_new.keys()])

        if new_t_stat_mean <= t_stat_mean:
            return correlation_sorted, clusters, silh
        else:
            return correlation_new, clusters_new, silh_new

import numpy as np
import pandas as pd
from scipy.linalg import block_diag


def random_covariance_sub(
        number_observations: int,
        number_columns: int,
        sigma: float,
        random_state: int = None
) -> np.ndarray:
    """
    Compute sub covariance matrix.

    :param number_observations: Number of observations.
    :param number_columns: Number of columns.
    :param sigma: Sigma for normal distribution.
    :param random_state: Random state for reproducibility.

    :return: Sub covariance matrix.
    """
    domain = check_random_state(random_state)

    if number_columns == 1:
        return np.ones((1, 1))

    data = domain.normal(size=(number_observations, 1))
    data = np.repeat(data, number_columns, axis=1)
    data += domain.normal(scale=sigma, size=data.shape)
    covariance = np.cov(data, rowvar=False)

    return covariance


def random_block_covariance(
        number_columns: int,
        number_blocks: int,
        block_size_min: int = 1,
        sigma: float = 1.0,
        random_state: int = None
) -> np.ndarray:
    """
    Compute random block covariance matrix.

    :param number_columns: Number of columns.
    :param number_blocks: Number of blocks.
    :param block_size_min: Minimum size of block.
    :param sigma: Sigma for normal distribution.
    :param random_state: Random state for reproducibility.

    :return: Random block covariance matrix.
    """
    domain = check_random_state(random_state)

    parts = domain.choice(range(1, number_columns - (block_size_min - 1) * number_blocks), number_blocks - 1, replace=False)
    parts.sort()
    parts = np.append(parts, number_columns - (block_size_min - 1) * number_blocks)
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
        number_columns: int,
        number_blocks: int,
        random_state: int = None,
        block_size_min: int = 1
) -> pd.DataFrame:
    """
    Compute random block correlation matrix.

    :param number_columns: Number of columns.
    :param number_blocks: Number of blocks.
    :param random_state: Random state for reproducibility.
    :param block_size_min: Minimum size of block.

    :return: Random block correlation matrix.
    """
    def cov_to_corr(covariance: np.ndarray) -> np.ndarray:
        """Convert covariance matrix to correlation matrix."""
        d = np.sqrt(np.diag(covariance))
        correlation = covariance / np.outer(d, d)
        return correlation

    domain = check_random_state(random_state)

    covariance1 = random_block_covariance(number_columns, number_blocks, block_size_min=block_size_min, sigma=0.5, random_state=domain)
    covariance2 = random_block_covariance(number_columns, 1, block_size_min=block_size_min, sigma=1.0, random_state=domain)

    covariance1 += covariance2
    correlation = cov_to_corr(covariance1)
    correlation = pd.DataFrame(correlation)

    return correlation
