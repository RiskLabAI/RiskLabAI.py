"""
Implements the Nested Clustered Optimization (NCO) algorithm.

Reference:
    De Prado, M. (2018) Advances in financial machine learning.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List

# Import canonical implementations instead of duplicating
try:
    from RiskLabAI.cluster.clustering import (
        cluster_k_means_base, covariance_to_correlation
    )
except ImportError:
    # Fallback for testing if cluster module not found
    print("Warning: RiskLabAI.cluster.clustering not found. Using dummy functions.")
    def covariance_to_correlation(cov: np.ndarray) -> np.ndarray: return cov
    def cluster_k_means_base(*args, **kwargs) -> Tuple:
        return pd.DataFrame(), {}, pd.Series()

def get_optimal_portfolio_weights(
    covariance: np.ndarray, mu: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute the optimal portfolio weights (Markowitz).

    If `mu` is not provided, computes the Global Minimum Variance (GMV) portfolio.
    If `mu` is provided, computes the Mean-Variance Optimization (MVO) portfolio.

    Parameters
    ----------
    covariance : np.ndarray
        Covariance matrix.
    mu : np.ndarray, optional
        Vector of expected returns.

    Returns
    -------
    np.ndarray
        Optimal portfolio weights.
    """
    inverse_covariance = np.linalg.inv(covariance)
    ones = np.ones(shape=(inverse_covariance.shape[0], 1))

    if mu is None:
        mu = ones  # For GMV portfolio
    
    weights = np.dot(inverse_covariance, mu)
    weights /= np.dot(ones.T, weights)  # Normalize weights to sum to 1
    return weights


def get_optimal_portfolio_weights_nco(
    covariance: np.ndarray,
    mu: Optional[np.ndarray] = None,
    number_clusters: Optional[int] = None,
) -> np.ndarray:
    """
    Compute optimal portfolio weights using the NCO algorithm.

    NCO works in three steps:
    1. Cluster assets based on their correlation matrix.
    2. Compute optimal (e.g., GMV) weights *within* each cluster.
    3. Compute optimal weights *between* the clusters (treating each
       cluster as a single asset).
    4. Combine the inter-cluster and intra-cluster weights.

    Parameters
    ----------
    covariance : np.ndarray
        Covariance matrix.
    mu : np.ndarray, optional
        Vector of expected returns. If None, uses GMV.
    number_clusters : int, optional
        Maximum number of clusters. If None, defaults to N/2.

    Returns
    -------
    np.ndarray
        The NCO-optimized portfolio weights.
    """
    covariance = pd.DataFrame(covariance)
    correlation = covariance_to_correlation(covariance.to_numpy())
    correlation = pd.DataFrame(correlation, 
                             index=covariance.index, 
                             columns=covariance.columns)
    
    if mu is not None:
        mu = pd.Series(mu.flatten(), index=covariance.index)
        
    if number_clusters is None:
        number_clusters = int(correlation.shape[0] / 2)

    # 1. Cluster assets
    _, clusters, _ = cluster_k_means_base(
        correlation, max_clusters=number_clusters, iterations=10
    )

    # 2. Compute intra-cluster weights
    weights_intra_cluster = pd.DataFrame(
        0.0, index=covariance.index, columns=clusters.keys()
    )
    for i, cluster_assets in clusters.items():
        cov_intra = covariance.loc[cluster_assets, cluster_assets].values
        
        mu_intra = None
        if mu is not None:
            mu_intra = mu.loc[cluster_assets].values.reshape(-1, 1)
            
        weights_intra_cluster.loc[cluster_assets, i] = (
            get_optimal_portfolio_weights(cov_intra, mu_intra).flatten()
        )

    # 3. Compute inter-cluster weights
    # Reduce covariance matrix using intra-cluster weights
    covariance_inter_cluster = weights_intra_cluster.T.dot(
        covariance.dot(weights_intra_cluster)
    )

    mu_inter_cluster = None
    if mu is not None:
        mu_inter_cluster = weights_intra_cluster.T.dot(mu.values)
        mu_inter_cluster = mu_inter_cluster.values.reshape(-1, 1)

    weights_inter_cluster = pd.Series(
        get_optimal_portfolio_weights(
            covariance_inter_cluster.values, mu_inter_cluster
        ).flatten(),
        index=covariance_inter_cluster.index,
    )

    # 4. Combine weights
    weights_nco = weights_intra_cluster.mul(weights_inter_cluster, axis=1).sum(
        axis=1
    )
    return weights_nco.values.reshape(-1, 1)