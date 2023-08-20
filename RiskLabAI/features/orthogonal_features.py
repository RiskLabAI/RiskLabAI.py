import pandas as pd
import numpy as np


def compute_eigenvectors(
        dot_product: np.ndarray,
        explained_variance_threshold: float
) -> pd.DataFrame:
    """
    Compute eigenvalues and eigenvectors for orthogonal features.

    :param dot_product: Input dot product matrix.
    :type dot_product: np.ndarray
    :param explained_variance_threshold: Threshold for cumulative explained variance.
    :type explained_variance_threshold: float
    :return: DataFrame containing eigenvalues, eigenvectors, and cumulative explained variance.
    :rtype: pd.DataFrame
    """
    eigenvalues, eigenvectors = np.linalg.eigh(dot_product)

    eigen_dataframe = pd.DataFrame({
        "Index": [f"PC {i}" for i in range(1, len(eigenvalues) + 1)],
        "EigenValue": eigenvalues,
        "EigenVector": [ev for ev in eigenvectors]
    })

    eigen_dataframe = eigen_dataframe.sort_values("EigenValue", ascending=False)

    cumulative_variance = np.cumsum(eigen_dataframe["EigenValue"]) / np.sum(eigen_dataframe["EigenValue"])
    eigen_dataframe["CumulativeVariance"] = cumulative_variance

    index = cumulative_variance.searchsorted(explained_variance_threshold)

    eigen_dataframe = eigen_dataframe.iloc[:index + 1, :]

    return eigen_dataframe


def orthogonal_features(
        features: np.ndarray,
        variance_threshold: float = 0.95
) -> tuple:
    """
    Compute orthogonal features using eigenvalues and eigenvectors.

    :param features: Features matrix.
    :type features: np.ndarray
    :param variance_threshold: Threshold for cumulative explained variance, default is 0.95.
    :type variance_threshold: float
    :return: Tuple containing orthogonal features and eigenvalues information.
    :rtype: tuple
    """
    normalized_features = (features - features.mean(axis=0)) / features.std(axis=0)
    dot_product = normalized_features.T @ normalized_features
    eigen_dataframe = compute_eigenvectors(dot_product, variance_threshold)

    transformation_matrix = np.vstack(eigen_dataframe["EigenVector"].values).T

    orthogonal_features = normalized_features @ transformation_matrix

    return orthogonal_features, eigen_dataframe
