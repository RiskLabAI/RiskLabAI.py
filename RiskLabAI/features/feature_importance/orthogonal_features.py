"""
Implements a method for feature orthogonalization using PCA.

Reference:
    De Prado, M. (2018) Advances in financial machine learning.
"""

import pandas as pd
import numpy as np
from typing import Tuple

def _compute_eigenvectors(
    dot_product: np.ndarray, explained_variance_threshold: float
) -> pd.DataFrame:
    """
    Compute eigenvalues and eigenvectors and filter by explained variance.

    Parameters
    ----------
    dot_product : np.ndarray
        The dot product matrix (e.g., X.T @ X).
    explained_variance_threshold : float
        The cumulative variance threshold to filter eigenvectors.

    Returns
    -------
    pd.DataFrame
        DataFrame with sorted eigenvalues, eigenvectors, and cumulative variance.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(dot_product)

    # Sort in descending order
    indices_sorted = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[indices_sorted]
    eigenvectors = eigenvectors[:, indices_sorted]

    eigen_dataframe = pd.DataFrame(
        {
            "EigenValue": eigenvalues,
            "EigenVector": [ev for ev in eigenvectors.T],
        },
        index=[f"PC_{i+1}" for i in range(len(eigenvalues))],
    )

    # Calculate cumulative variance
    cumulative_variance = eigenvalues.cumsum() / eigenvalues.sum()
    eigen_dataframe["CumulativeVariance"] = cumulative_variance

    # Find the index where cumulative variance crosses the threshold
    index = cumulative_variance.searchsorted(explained_variance_threshold)
    
    # Keep components up to and including the one that crosses the threshold
    eigen_dataframe = eigen_dataframe.iloc[: index + 1, :]

    return eigen_dataframe


def orthogonal_features(
    features: pd.DataFrame, variance_threshold: float = 0.95
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute orthogonal features using PCA.

    Parameters
    ----------
    features : pd.DataFrame
        The feature DataFrame.
    variance_threshold : float, default=0.95
        Cumulative explained variance threshold to keep.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - orthogonal_features_df: The transformed (orthogonal) features.
        - eigen_dataframe: DataFrame with eigenvalues and vectors.
    """
    # 1. Normalize features (z-score)
    normalized_features = (features - features.mean(axis=0)) / features.std(axis=0)
    normalized_features = normalized_features.dropna(axis=1) # Drop constant cols
    
    # 2. Compute dot product (proportional to covariance)
    dot_product = normalized_features.T @ normalized_features
    
    # 3. Get principal components
    eigen_dataframe = _compute_eigenvectors(dot_product, variance_threshold)

    # 4. Get transformation matrix (stacking eigenvectors)
    transformation_matrix = np.vstack(eigen_dataframe["EigenVector"].values).T

    # 5. Transform features
    orthogonal_features_arr = normalized_features.values @ transformation_matrix
    
    orthogonal_features_df = pd.DataFrame(
        orthogonal_features_arr,
        index=features.index,
        columns=eigen_dataframe.index,
    )

    return orthogonal_features_df, eigen_dataframe