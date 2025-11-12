"""
Implements the Hierarchical Risk Parity (HRP) portfolio
optimization algorithm.

Reference:
    De Prado, M. (2018) Advances in financial machine learning.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as scd  
import random
from typing import List, Tuple, Optional

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
        item = items_to_process.pop(0) # Process items recursively (depth-first)

        if item >= num_items:
            # This is a cluster, add its children to the processing list
            cluster_id = item - num_items
            # Add children in their linkage order
            items_to_process.insert(0, link[cluster_id, 1]) # Right child
            items_to_process.insert(0, link[cluster_id, 0]) # Left child
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


def plot_corr_matrix(
    path: str, corr_matrix: np.ndarray, labels: Optional[List[str]] = None
) -> None:
    """
    Plot a heatmap of the correlation matrix and save to file.

    Parameters
    ----------
    path : str
        Path to save the plot.
    corr_matrix : np.ndarray
        Correlation matrix.
    labels : List[str], optional
        List of labels for the assets.
    """
    if labels is None:
        labels = []

    plt.figure()
    plt.pcolor(corr_matrix)
    plt.colorbar()
    plt.yticks(np.arange(0.5, corr_matrix.shape[0] + 0.5), labels)
    plt.xticks(np.arange(0.5, corr_matrix.shape[0] + 0.5), labels)
    plt.savefig(path)
    plt.clf()
    plt.close()


def random_data(
    num_observations: int, size_uncorr: int, size_corr: int, sigma_corr: float
) -> Tuple[pd.DataFrame, List[int]]:
    """
    Generate random sample data with correlated and uncorrelated parts.

    Parameters
    ----------
    num_observations : int
        Number of observations (rows).
    size_uncorr : int
        Number of uncorrelated features.
    size_corr : int
        Number of correlated features.
    sigma_corr : float
        Standard deviation of noise for correlated features.

    Returns
    -------
    Tuple[pd.DataFrame, List[int]]
        - The generated data.
        - List of column indices for the base of correlated data.
    """
    np.random.seed(seed=12345)
    random.seed(12345)

    # Uncorrelated data
    data1 = np.random.normal(0, 1, size=(num_observations, size_uncorr))
    
    # Correlated data
    columns_correlated = [
        random.randint(0, size_uncorr - 1) for _ in range(size_corr)
    ]
    data2 = data1[:, columns_correlated] + np.random.normal(
        0, sigma_corr, size=(num_observations, len(columns_correlated))
    )
    
    data = np.append(data1, data2, axis=1)
    data = pd.DataFrame(data, columns=range(1, data.shape[1] + 1))
    return data, columns_correlated


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
    cov : np.ndarray
        Covariance matrix.
    corr : np.ndarray
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


def hrp_mc(
    number_iterations: int = 5000,
    number_observations: int = 520,
    size_uncorrelated: int = 5,
    size_correlated: int = 5,
    mu_uncorrelated: float = 0,
    sigma_uncorrelated: float = 0.01,
    sigma_correlated: float = 0.25,
    length_sample: int = 260,
    test_size: int = 22,
) -> None:
    """
    Run a Monte Carlo simulation comparing HRP performance.
    
    (Note: This function is written as a script, not a library function.
    It will print results and save to 'results.csv'.)
    """
    methods = [hrp]  # Add other methods like NCO, IVP, etc. here
    
    results = {f.__name__: pd.Series(dtype=float) for f in methods}
    iteration_counter = 0
    
    pointers = range(length_sample, number_observations, test_size)

    while iteration_counter < number_iterations:
        # 1. Generate new data
        data, _ = random_data(
            num_observations=number_observations,
            size_uncorr=size_uncorrelated,
            size_corr=size_correlated,
            sigma_corr=sigma_correlated,
            # (Note: random_data2 was in original, using random_data)
        )
        
        returns = {f.__name__: pd.Series(dtype=float) for f in methods}

        # 2. Walk-forward
        for pointer in pointers:
            in_sample = data.iloc[pointer - length_sample : pointer].values
            cov_ = np.cov(in_sample, rowvar=False)
            corr_ = np.corrcoef(in_sample, rowvar=False)
            
            out_sample = data.iloc[pointer : pointer + test_size].values
            
            for func in methods:
                weight = func(cov=cov_, corr=corr_)  # Get weights
                ret = out_sample @ weight.values
                returns[func.__name__] = returns[func.__name__]._append(
                    pd.Series(ret)
                )
        
        # 3. Store iteration performance
        for func in methods:
            ret = returns[func.__name__].reset_index(drop=True)
            cumprod_return = (1 + ret).cumprod()
            results[func.__name__].loc[iteration_counter] = (
                cumprod_return.iloc[-1] - 1
            )
        
        iteration_counter += 1

    # 4. Save and print results
    results_df = pd.DataFrame.from_dict(results, orient="columns")
    results_df.to_csv("results.csv")
    
    std_results = results_df.std()
    var_results = results_df.var()
    relative_var = var_results / var_results["hrp"] - 1
    
    summary = pd.concat(
        [std_results, var_results, relative_var], 
        axis=1, 
        keys=["StdDev", "Variance", "Var_vs_HRP"]
    )
    print(summary)