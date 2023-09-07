import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from scipy.linalg import block_diag
from sklearn.utils import check_random_state


def covariance_to_correlation(covariance: np.ndarray) -> np.ndarray:
    """
    Derive the correlation matrix from a covariance matrix.

    :param covariance: numpy ndarray
        The covariance matrix to convert to a correlation matrix.
    :return: numpy ndarray
        The correlation matrix derived from the covariance matrix.

    The conversion is done based on the following mathematical formula:
    correlation = covariance / (std_i * std_j)
    where std_i and std_j are the standard deviations of the i-th and j-th elements.
    """
    std = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(std, std)
    correlation[correlation < -1], correlation[correlation > 1] = -1, 1
    return correlation


def cluster_kmeans_base(correlation: pd.DataFrame, number_clusters: int = 10, iterations: int = 10) -> tuple:
    """
    Apply the K-means clustering algorithm.

    :param correlation: pandas DataFrame
        The correlation matrix.
    :param number_clusters: int, optional
        The maximum number of clusters. Default is 10.
    :param iterations: int, optional
        The number of iterations to run the clustering. Default is 10.
    :return: tuple
        A tuple containing the sorted correlation matrix, cluster membership, and silhouette scores.

    This function is based on Snippet 4.1 from De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    """
    distance, silh = ((1 - correlation.fillna(0)) / 2.0) ** 0.5, pd.Series()
    for init in range(iterations):
        for i in range(2, number_clusters + 1):
            kmeans = KMeans(n_clusters=i, n_init=1)
            kmeans = kmeans.fit(distance)
            silh_ = silhouette_samples(distance, kmeans.labels_)
            statistic = (silh_.mean() / silh_.std(), silh.mean() / silh.std())
            if np.isnan(statistic[1]) or statistic[0] > statistic[1]:
                silh, kmeans = silh_, kmeans
    index_sorted = np.argsort(kmeans.labels_)
    correlation_sorted = correlation.iloc[index_sorted]
    correlation_sorted = correlation_sorted.iloc[:, index_sorted]
    clusters = {i: correlation.columns[np.where(kmeans.labels_ == i)[0]].tolist() for i in np.unique(kmeans.labels_)}
    silh = pd.Series(silh, index=distance.index)
    return correlation_sorted, clusters, silh


def make_new_outputs(correlation: pd.DataFrame, clusters: dict, clusters2: dict) -> tuple:
    """
    Merge two sets of clusters and derive new outputs.

    :param correlation: pandas DataFrame
        The correlation matrix.
    :param clusters: dict
        The first set of clusters.
    :param clusters2: dict
        The second set of clusters.
    :return: tuple
        A tuple containing the new correlation matrix, new cluster membership, and new silhouette scores.
    """
    clusters_new = {}
    for i in clusters.keys():
        clusters_new[len(clusters_new.keys())] = list(clusters[i])
    for i in clusters2.keys():
        clusters_new[len(clusters_new.keys())] = list(clusters2[i])
    index_new = [j for i in clusters_new for j in clusters_new[i]]
    correlation_new = correlation.loc[index_new, index_new]
    distance = ((1 - correlation.fillna(0)) / 2.0) ** 0.5
    labels_kmeans = np.zeros(len(distance.columns))
    for i in clusters_new.keys():
        index = [distance.index.get_loc(k) for k in clusters_new[i]]
        labels_kmeans[index] = i
    silh_new = pd.Series(silhouette_samples(distance, labels_kmeans), index=distance.index)
    return correlation_new, clusters_new, silh_new


def cluster_kmeans_top(correlation: pd.DataFrame, number_clusters: int = None, iterations: int = 10) -> tuple:
    """
    Apply the K-means clustering algorithm with hierarchical re-clustering.

    :param correlation: pandas DataFrame
        The correlation matrix.
    :param number_clusters: int, optional
        The maximum number of clusters. Default is None.
    :param iterations: int, optional
        The number of iterations to run the clustering. Default is 10.
    :return: tuple
        A tuple containing the sorted correlation matrix, cluster membership, and silhouette scores.

    This function is based on Snippet 4.2 from De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    """
    if number_clusters is None:
        number_clusters = correlation.shape[1] - 1
    correlation_sorted, clusters, silh = cluster_kmeans_base(
        correlation, min(number_clusters, correlation.shape[1] - 1), iterations
    )
    cluster_tstats = {i: np.mean(silh[clusters[i]]) / np.std(silh[clusters[i]]) for i in clusters.keys()}
    tstat_mean = sum(cluster_tstats.values()) / len(cluster_tstats)
    redo_clusters = [i for i in cluster_tstats.keys() if cluster_tstats[i] < tstat_mean]
    if len(redo_clusters) <= 1:
        return correlation_sorted, clusters, silh
    else:
        keys_redo = [j for i in redo_clusters for j in clusters[i]]
        correlation_temp = correlation.loc[keys_redo, keys_redo]
        tstat_mean = np.mean([cluster_tstats[i] for i in redo_clusters])
        correlation_sorted2, clusters2, silh2 = cluster_kmeans_top(
            correlation_temp, min(number_clusters, correlation_temp.shape[1] - 1), iterations
        )
        correlation_new, clusters_new, silh_new = make_new_outputs(
            correlation, {i: clusters[i] for i in clusters.keys() if i not in redo_clusters}, clusters2
        )
        new_tstat_mean = np.mean(
            [np.mean(silh_new[clusters_new[i]]) / np.std(silh_new[clusters_new[i]]) for i in clusters_new.keys()]
        )
        if new_tstat_mean <= tstat_mean:
            return correlation_sorted, clusters, silh
        else:
            return correlation_new, clusters_new, silh_new


import numpy as np
from scipy.linalg import block_diag
from sklearn.utils import check_random_state


def random_covariance_sub(number_observations: int, number_columns: int, sigma: float, random_state=None) -> np.ndarray:
    """
    Compute a sub covariance matrix.

    Generates a covariance matrix based on random data.

    :param number_observations: Number of observations.
    :param number_columns: Number of columns.
    :param sigma: Sigma for normal distribution.
    :param random_state: Random state for reproducibility.
    :return: Sub covariance matrix.

    .. note:: Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
       Methodology: Snipet 4.3, Page 61.
    """
    domain = check_random_state(random_state)
    if number_columns == 1:
        return np.ones((1, 1))
    data = domain.normal(size=(number_observations, 1))  # generate data
    data = np.repeat(data, number_columns, axis=1)  # repeat data
    data += domain.normal(scale=sigma, size=data.shape)  # add data
    covariance = np.cov(data, rowvar=False)  # covariance of data
    return covariance


def random_block_covariance(
    number_columns: int, number_blocks: int, block_size_min: int = 1, sigma: float = 1.0, random_state=None
) -> np.ndarray:
    """
    Compute a random block covariance matrix.

    Generates a block random covariance matrix by combining multiple sub covariance matrices.

    :param number_columns: Number of columns.
    :param number_blocks: Number of blocks.
    :param block_size_min: Minimum size of block.
    :param sigma: Sigma for normal distribution.
    :param random_state: Random state for reproducibility.
    :return: Block random covariance matrix.

    .. note:: Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
       Methodology: Snipet 4.3, Page 61.
    """
    domain = check_random_state(random_state)
    parts = domain.choice(
        range(1, number_columns - (block_size_min - 1) * number_blocks), number_blocks - 1, replace=False
    )
    parts.sort()
    parts = np.append(parts, number_columns - (block_size_min - 1) * number_blocks)
    parts = np.append(parts[0], np.diff(parts)) - 1 + block_size_min
    cov = None
    for column in parts:
        this_covariance = random_covariance_sub(
            int(max(column * (column + 1) / 2.0, 100)), column, sigma, random_state=domain
        )  # sub covariance
        if cov is None:
            cov = this_covariance.copy()  # copy covariance matrix
        else:
            cov = block_diag(cov, this_covariance)  # block diagram covariance matrix
    return cov


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.utils import check_random_state


def random_block_correlation(
    number_columns: int, number_blocks: int, random_state=None, block_size_min: int = 1
) -> pd.DataFrame:
    """
    Compute a random block correlation matrix.

    Generates a block random correlation matrix by adding two block random covariance matrices
    and converting them to a correlation matrix.

    :param number_columns: Number of columns.
    :param number_blocks: Number of blocks.
    :param random_state: Random state for reproducibility.
    :param block_size_min: Minimum size of block.
    :return: Block random correlation matrix.

    .. note:: Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
       Methodology: Snipet 4.3, Page 61.
    """
    domain = check_random_state(random_state)
    covariance1 = random_block_covariance(
        number_columns, number_blocks, block_size_min=block_size_min, sigma=0.5, random_state=domain
    )
    covariance2 = random_block_covariance(
        number_columns, 1, block_size_min=block_size_min, sigma=1.0, random_state=domain
    )
    covariance1 += covariance2
    correlation = cov_to_corr(covariance1)
    correlation = pd.DataFrame(correlation)
    return correlation


def cov_to_corr(covariance: np.ndarray) -> np.ndarray:
    """
    Convert a covariance matrix to a correlation matrix.

    :param covariance: Covariance matrix.
    :return: Correlation matrix.

    .. math::
        correlation_{ij} = \\frac{covariance_{ij}}{\\sqrt{covariance_{ii} \cdot covariance_{jj}}}
    """
    std = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(std, std)
    correlation[correlation < -1] = -1
    correlation[correlation > 1] = 1
    return correlation


def cluster_kmeans_base(
    correlation: pd.DataFrame, number_clusters: int = 10, iterations: int = 10
) -> (pd.DataFrame, dict, pd.Series):
    """
    Perform KMeans clustering on a correlation matrix.

    :param correlation: Correlation matrix.
    :param number_clusters: Number of clusters, default is 10.
    :param iterations: Number of iterations, default is 10.
    :return: Sorted correlation matrix, clusters, silhouette scores.

    .. note::
        The silhouette score is a measure of how similar an object is to its own cluster compared to other clusters.
    """
    distance = ((1 - correlation.fillna(0)) / 2.0) ** 0.5
    silh = pd.Series()

    for init in range(iterations):
        for i in range(2, number_clusters + 1):
            kmeans_ = KMeans(n_clusters=i, n_init=1)
            kmeans_ = kmeans_.fit(distance)
            silh_ = silhouette_samples(distance, kmeans_.labels_)
            statistic = (silh_.mean() / silh_.std(), silh.mean() / silh.std())

            if np.isnan(statistic[1]) or statistic[0] > statistic[1]:
                silh, kmeans = silh_, kmeans_

    index_sorted = np.argsort(kmeans.labels_)
    correlation_sorted = correlation.iloc[index_sorted]
    correlation_sorted = correlation_sorted.iloc[:, index_sorted]

    clusters = {i: correlation.columns[np.where(kmeans.labels_ == i)[0]].tolist() for i in np.unique(kmeans.labels_)}

    silh = pd.Series(silh, index=distance.index)
    return correlation_sorted, clusters, silh


# Example usage:
if __name__ == "__main__":
    correlation_matrix = random_block_correlation(number_columns=20, number_blocks=5)
    sorted_corr, cluster_dict, silh_scores = cluster_kmeans_base(correlation_matrix)
    print(sorted_corr)
    print(cluster_dict)
    print(silh_scores)
