"""
Implements the Hierarchical Risk Parity (HRP) portfolio
optimization algorithm.

Reference:
    De Prado, M. (2018) Advances in financial machine learning.
"""

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as scd
from typing import List

def inverse_variance_weights(covariance_matrix: pd.DataFrame) -> np.ndarray:
    """
    Compute the inverse-variance portfolio weights.

    Parameters
    ----------
    covariance_matrix : pd.DataFrame
        Covariance matrix of asset returns.

    Returns
    -------
    np.ndarray
        Array of portfolio weights.
    """
    inv_var_weights = 1.0 / np.diag(covariance_matrix.values)
    inv_var_weights /= inv_var_weights.sum()
    return inv_var_weights


def cluster_variance(
    covariance_matrix: pd.DataFrame, clustered_items: List[str]
) -> float:
    """
    Compute the variance of a cluster using inverse-variance weighting.

    Parameters
    ----------
    covariance_matrix : pd.DataFrame
        Full covariance matrix of asset returns.
    clustered_items : List[str]
        List of asset names (index/columns) in the cluster.

    Returns
    -------
    float
        Variance of the cluster.
    """
    cov_slice = covariance_matrix.loc[clustered_items, clustered_items]
    weights = inverse_variance_weights(cov_slice).reshape(-1, 1)
    
    # V_cluster = w' * C * w
    cluster_var = np.dot(np.dot(weights.T, cov_slice), weights)[0, 0]
    return cluster_var


def quasi_diagonal(linkage_matrix: np.ndarray) -> List[int]:
    """
    Return a sorted list of original item indices for a quasi-diagonal matrix.

    Parameters
    ----------
    linkage_matrix : np.ndarray
        The linkage matrix from `scipy.cluster.hierarchy.linkage`.

    Returns
    -------
    List[int]
        Sorted list of original item indices.
    """
    link = linkage_matrix.astype(int)
    num_items = link[-1, 3]  # Total number of original items

    # Get the top-level clusters
    items_to_process = [link[-1, 0], link[-1, 1]]
    sorted_items = []

    while len(items_to_process) > 0:
        item = items_to_process.pop(0)  # Process items recursively (depth-first)

        if item >= num_items:
            # This is a cluster, add its children to the processing list
            cluster_id = item - num_items
            # Add children in their linkage order
            items_to_process.insert(0, link[cluster_id, 1])  # Right child
            items_to_process.insert(0, link[cluster_id, 0])  # Left child
        else:
            # This is an original item
            sorted_items.append(item)

    return sorted_items


def recursive_bisection(
    covariance_matrix: pd.DataFrame, sorted_items: List[str]
) -> pd.Series:
    """
    Compute the Hierarchical Risk Parity (HRP) weights
    using recursive bisection.

    Parameters
    ----------
    covariance_matrix : pd.DataFrame
        Covariance matrix of asset returns.
    sorted_items : List[str]
        Sorted list of asset names from `quasi_diagonal`.

    Returns
    -------
    pd.Series
        DataFrame of asset weights.
    """
    weights = pd.Series(1.0, index=sorted_items)
    clustered_items = [sorted_items]

    while len(clustered_items) > 0:
        # Bisection
        clustered_items = [
            i[j:k]
            for i in clustered_items
            for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
            if len(i) > 1
        ]

        # Process pairs of clusters
        for i in range(0, len(clustered_items), 2):
            cluster_0 = clustered_items[i]
            cluster_1 = clustered_items[i + 1]
            
            # 1. Calculate variance for each cluster
            variance_0 = cluster_variance(covariance_matrix, cluster_0)
            variance_1 = cluster_variance(covariance_matrix, cluster_1)
            
            # 2. Calculate allocation factor (alpha)
            if variance_0 + variance_1 == 0:
                alpha = 0.5 # Default to equal weight if both variances are zero
            else:
                alpha = 1 - variance_0 / (variance_0 + variance_1)
            
            # 3. Apply weights
            weights[cluster_0] *= alpha
            weights[cluster_1] *= (1 - alpha)

    return weights


def distance_corr(corr_matrix: np.ndarray) -> np.ndarray:
    """
    Compute the distance matrix based on correlation.
    d = sqrt(0.5 * (1 - p))

    Parameters
    ----------
    corr_matrix : np.ndarray
        Correlation matrix.

    Returns
    -------
    np.ndarray
        Distance matrix.
    """
    distance_matrix = ((1 - corr_matrix) / 2.0) ** 0.5
    return distance_matrix


def hrp(cov: pd.DataFrame, corr: pd.DataFrame) -> pd.Series:
    """
    Main HRP algorithm.

    Constructs a hierarchical portfolio by:
    1. Calculating correlation-based distance.
    2. Hierarchical clustering (single linkage).
    3. Quasi-diagonalization of the linkage matrix.
    4. Recursive bisection to determine weights.

    Parameters
    ----------
    cov : pd.DataFrame
        Covariance matrix.
    corr : pd.DataFrame
        Correlation matrix.

    Returns
    -------
    pd.Series
        The HRP portfolio weights, sorted by original index.
    """
    corr_df, cov_df = corr, cov

    # 1. Calculate distance
    distance = distance_corr(corr_df.values)

    dist_condensed = scd.squareform(distance, force='tovector')

    # 2. Cluster
    link = sch.linkage(dist_condensed, "single")
    
    # 3. Quasi-diagonalize
    sorted_items_idx = quasi_diagonal(link)
    sorted_items_names = corr_df.index[sorted_items_idx].tolist()
    
    # 4. Recursive bisection
    hrp_portfolio = recursive_bisection(cov_df, sorted_items_names)
    
    return hrp_portfolio.sort_index()