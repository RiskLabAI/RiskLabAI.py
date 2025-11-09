"""
Implements clustering algorithms, including the Optimized Nested Clustering (ONC)
method from Marcos Lopez de Prado.

Reference:
    De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
"""

import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.utils import check_random_state
from typing import Tuple, Dict, List, Optional

def covariance_to_correlation(covariance: np.ndarray) -> np.ndarray:
    r"""
    Derive the correlation matrix from a covariance matrix.

    .. math::
        \text{corr}_{ij} = \frac{\text{cov}_{ij}}
                                {\sqrt{\text{cov}_{ii} \text{cov}_{jj}}}

    Reference:
        Snippet 2.3, Page 27.

    Parameters
    ----------
    covariance : np.ndarray
        A square covariance matrix.

    Returns
    -------
    np.ndarray
        The corresponding correlation matrix.
    """
    std = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(std, std)
    
    # Handle numerical errors
    correlation[correlation < -1] = -1.0
    correlation[correlation > 1] = 1.0
    
    return correlation

def cluster_k_means_base(
    correlation: pd.DataFrame,
    max_clusters: int = 10,
    iterations: int = 10,
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[int, List[str]], pd.Series]:
    """
    Perform the base K-Means clustering step.

    This function iterates through different numbers of clusters (from 2 to
    `max_clusters`) and multiple initializations (`iterations`), selecting the
    clustering result that yields the highest average silhouette score
    t-statistic (mean / std).

    Reference:
        Snippet 4.1, Page 56.

    Parameters
    ----------
    correlation : pd.DataFrame
        The correlation matrix to be clustered.
    max_clusters : int, default=10
        The maximum number of clusters to try.
    iterations : int, default=10
        Number of K-Means initializations for each cluster count.
    random_state : int, optional
        Random state for K-Means reproducibility.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[int, List[str]], pd.Series]
        - correlation_sorted: The correlation matrix, reordered by cluster.
        - clusters: A dictionary mapping cluster ID to a list of item names.
        - silhouette_scores: A Series of silhouette scores for each item.
    """
    # Calculate distance matrix
    distance = ((1 - correlation.fillna(0)) / 2.0) ** 0.5
    
    best_kmeans = None
    best_silhouette_scores = None
    best_score = -np.inf

    rng = check_random_state(random_state)
    
    for _ in range(iterations):
        for n_clusters in range(2, max_clusters + 1):
            # Use a different random_state for each K-Means fit
            iter_seed = rng.randint(0, np.iinfo(np.int32).max)
            
            kmeans = KMeans(
                n_clusters=n_clusters,
                n_init=1,  # We handle iterations externally
                random_state=iter_seed,
            )
            kmeans.fit(distance)
            
            silhouette_scores_ = silhouette_samples(distance, kmeans.labels_)
            
            # Use silhouette score t-statistic (mean/std) as the quality metric
            stat_mean = silhouette_scores_.mean()
            stat_std = silhouette_scores_.std()
            
            if stat_std == 0:
                # Avoid division by zero if all silhouette scores are identical
                score = np.sign(stat_mean) * np.inf
            else:
                score = stat_mean / stat_std

            if score > best_score:
                best_score = score
                best_kmeans = kmeans
                best_silhouette_scores = silhouette_scores_

    if best_kmeans is None:
        raise ValueError("Clustering failed to find a valid solution.")

    # Sort items by cluster label
    index_sorted = np.argsort(best_kmeans.labels_)
    correlation_sorted = correlation.iloc[index_sorted, index_sorted]

    # Create cluster dictionary
    clusters = {
        i: correlation.columns[np.where(best_kmeans.labels_ == i)[0]].tolist()
        for i in np.unique(best_kmeans.labels_)
    }
    
    silhouette_series = pd.Series(best_silhouette_scores, index=distance.index)

    return correlation_sorted, clusters, silhouette_series


def make_new_outputs(
    correlation: pd.DataFrame,
    clusters_1: Dict[int, List[str]],
    clusters_2: Dict[int, List[str]],
) -> Tuple[pd.DataFrame, Dict[int, List[str]], pd.Series]:
    """
    Merge two disjoint sets of clusters and re-calculate metrics.

    Reference:
        Snippet 4.2, Page 58.

    Parameters
    ----------
    correlation : pd.DataFrame
        The *original* correlation matrix.
    clusters_1 : Dict[int, List[str]]
        The first set of clusters (e.g., the "good" clusters).
    clusters_2 : Dict[int, List[str]]
        The second set of clusters (e.g., the re-clustered "bad" ones).

    Returns
    -------
    Tuple[pd.DataFrame, Dict[int, List[str]], pd.Series]
        - correlation_new: The correlation matrix, reordered by new clusters.
        - clusters_new: The merged dictionary of all clusters.
        - silhouette_new: New silhouette scores based on the merged clustering.
    """
    clusters_new = {}
    for i in clusters_1:
        clusters_new[len(clusters_new)] = list(clusters_1[i])
    for i in clusters_2:
        clusters_new[len(clusters_new)] = list(clusters_2[i])

    # Get the new full, sorted index
    index_new = [
        item for cluster_items in clusters_new.values() for item in cluster_items
    ]
    correlation_new = correlation.loc[index_new, index_new]

    # Calculate new silhouette scores based on the *original* distance
    distance = ((1 - correlation.fillna(0)) / 2.0) ** 0.5
    labels_kmeans = np.zeros(len(distance.columns))

    # Create the label array for silhouette_samples
    for i, items in clusters_new.items():
        item_indices = [distance.index.get_loc(k) for k in items]
        labels_kmeans[item_indices] = i

    silhouette_new = pd.Series(
        silhouette_samples(distance, labels_kmeans), index=distance.index
    )
    return correlation_new, clusters_new, silhouette_new

def cluster_k_means_top(
    correlation: pd.DataFrame,
    max_clusters: Optional[int] = None,
    iterations: int = 10,
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[int, List[str]], pd.Series]:
    """
    Perform Optimized Nested Clustering (ONC).

    This method recursively re-clusters unstable clusters (those with a
    silhouette t-statistic below the average).

    Reference:
        Snippet 4.2, Page 58.

    Parameters
    ----------
    correlation : pd.DataFrame
        The correlation matrix to be clustered.
    max_clusters : int, optional
        Maximum number of clusters. Defaults to `N-1`.
    iterations : int, default=10
        Number of K-Means initializations.
    random_state : int, optional
        Random state for reproducibility.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[int, List[str]], pd.Series]
        - correlation_sorted: The final sorted correlation matrix.
        - clusters: The final dictionary of clusters.
        - silhouette_scores: The final silhouette scores for each item.
    """
    n_cols = correlation.shape[1]
    if max_clusters is None:
        max_clusters = n_cols - 1
    
    max_clusters = min(max_clusters, n_cols - 1)
    if max_clusters < 2:
        return (
            correlation,
            {0: correlation.columns.tolist()},
            pd.Series(dtype='float64'),
        )

    # 1. Run base clustering
    corr_sorted, clusters, silhouette = cluster_k_means_base(
        correlation,
        max_clusters=max_clusters,
        iterations=iterations,
        random_state=random_state,
    )

    # 2. Calculate t-statistic for each cluster
    cluster_t_stats = {
        i: silhouette[clusters[i]].mean() / silhouette[clusters[i]].std()
        for i in clusters
        if silhouette[clusters[i]].std() > 0
    }
    
    if not cluster_t_stats:
        return corr_sorted, clusters, silhouette # No valid clusters found

    t_stat_mean = np.mean(list(cluster_t_stats.values()))

    # 3. Identify clusters to re-cluster
    redo_clusters = [
        i for i, t_stat in cluster_t_stats.items() if t_stat < t_stat_mean
    ]

    if len(redo_clusters) <= 1:
        # Base case: All clusters are stable, or only one is unstable
        return corr_sorted, clusters, silhouette
    else:
        # 4. Recurse on unstable clusters
        keys_redo = [
            item for i in redo_clusters for item in clusters[i]
        ]
        corr_temp = correlation.loc[keys_redo, keys_redo]
        
        # Keep track of mean t-stat for comparison
        t_stat_mean_redo = np.mean([cluster_t_stats[i] for i in redo_clusters])
        
        # Calculate remaining clusters for recursive call
        n_clusters_good = len(clusters) - len(redo_clusters)
        remained_n_clusters = max_clusters - n_clusters_good
        
        # Recursive call
        corr_sorted_2, clusters_2, silh_2 = cluster_k_means_top(
            corr_temp,
            max_clusters=min(remained_n_clusters, corr_temp.shape[1] - 1),
            iterations=iterations,
            random_state=random_state,
        )

        # 5. Merge results
        clusters_1 = {
            i: clusters[i] for i in clusters if i not in redo_clusters
        }
        corr_new, clusters_new, silh_new = make_new_outputs(
            correlation, clusters_1, clusters_2
        )

        # 6. Decide whether to keep the re-clustered result
        new_t_stats = [
            silh_new[clusters_new[i]].mean() / silh_new[clusters_new[i]].std()
            for i in clusters_new
            if silh_new[clusters_new[i]].std() > 0
        ]
        
        if not new_t_stats:
             return corr_sorted, clusters, silhouette # Re-clustering failed
             
        new_t_stat_mean = np.mean(new_t_stats)

        if new_t_stat_mean <= t_stat_mean_redo:
            # Re-clustering did not improve, return original
            return corr_sorted, clusters, silhouette
        else:
            # Re-clustering improved, return new results
            return corr_new, clusters_new, silh_new


def random_covariance_sub(
    n_observations: int,
    n_columns: int,
    sigma: float,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a random covariance matrix for a single block.

    Reference:
        Snippet 4.3, Page 61.

    Parameters
    ----------
    n_observations : int
        Number of observations to simulate.
    n_columns : int
        Number of columns (assets) in this block.
    sigma : float
        Sigma for the idiosyncratic noise.
    random_state : int, optional
        Random state for reproducibility.

    Returns
    -------
    np.ndarray
        The (n_columns x n_columns) covariance matrix.
    """
    rng = check_random_state(random_state)

    if n_columns == 1:
        return np.ones((1, 1))

    # Common factor
    data = rng.normal(size=(n_observations, 1))
    data = np.repeat(data, n_columns, axis=1)
    
    # Idiosyncratic noise
    data += rng.normal(scale=sigma, size=data.shape)
    
    covariance = np.cov(data, rowvar=False)
    return covariance


def random_block_covariance(
    n_columns: int,
    n_blocks: int,
    block_size_min: int = 1,
    sigma: float = 1.0,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a random block-diagonal covariance matrix.

    Reference:
        Snippet 4.3, Page 61.

    Parameters
    ----------
    n_columns : int
        Total number of columns (assets).
    n_blocks : int
        Number of blocks to create.
    block_size_min : int, default=1
        The minimum size for each block.
    sigma : float, default=1.0
        Sigma for idiosyncratic noise.
    random_state : int, optional
        Random state for reproducibility.

    Returns
    -------
    np.ndarray
        The (n_columns x n_columns) block-diagonal covariance matrix.
    """
    rng = check_random_state(random_state)

    # Generate block sizes
    parts = rng.choice(
        range(1, n_columns - (block_size_min - 1) * n_blocks),
        n_blocks - 1,
        replace=False,
    )
    parts.sort()
    parts = np.append(
        parts, n_columns - (block_size_min - 1) * n_blocks
    )
    parts = np.append(parts[0], np.diff(parts)) - 1 + block_size_min

    cov_list = []
    for col_size in parts:
        # Number of observations must be > number of columns
        n_obs = int(max(col_size * (col_size + 1) / 2.0, 100))
        
        this_covariance = random_covariance_sub(
            n_obs, col_size, sigma, random_state=rng
        )
        cov_list.append(this_covariance)

    return block_diag(*cov_list)


def random_block_correlation(
    n_columns: int,
    n_blocks: int,
    random_state: Optional[int] = None,
    block_size_min: int = 1,
) -> pd.DataFrame:
    """
    Generate a random block-diagonal correlation matrix with noise.

    This adds a "market" component (a single block covariance)
    to the block-diagonal structure.

    Reference:
        Snippet 4.3, Page 61.

    Parameters
    ----------
    n_columns : int
        Total number of columns (assets).
    n_blocks : int
        Number of blocks to create.
    random_state : int, optional
        Random state for reproducibility.
    block_size_min : int, default=1
        The minimum size for each block.

    Returns
    -------
    pd.DataFrame
        The (n_columns x n_columns) correlation matrix.
    """
    rng = check_random_state(random_state)

    # Block-diagonal structure
    covariance1 = random_block_covariance(
        n_columns,
        n_blocks,
        block_size_min=block_size_min,
        sigma=0.5,
        random_state=rng,
    )
    
    # Market component (noise)
    covariance2 = random_block_covariance(
        n_columns, 1, block_size_min=n_columns, sigma=1.0, random_state=rng
    )

    covariance = covariance1 + covariance2
    correlation = covariance_to_correlation(covariance)
    correlation = pd.DataFrame(correlation)

    return correlation