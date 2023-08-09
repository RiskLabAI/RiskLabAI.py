import pandas as pd
import numpy as np

def eigen_vectors(
    dot_product: np.ndarray, 
    explained_variance_threshold: float
) -> pd.DataFrame:
    """
    Compute eigenvalues and eigenvectors for Orthogonal Features.

    :param dot_product: Input dot product matrix
    :type dot_product: np.ndarray
    :param explained_variance_threshold: Threshold for cumulative explained variance
    :type explained_variance_threshold: float
    :return: DataFrame containing eigenvalues, eigenvectors, and cumulative explained variance
    :rtype: pd.DataFrame
    """
    e_values, e_vectors = np.linalg.eigh(dot_product)

    eigen_dataframe = pd.DataFrame({
        "Index": [f"PC {i}" for i in range(1, len(e_values) + 1)],
        "EigenValue": e_values,
        "EigenVector": [ev for ev in e_vectors]
    })

    eigen_dataframe = eigen_dataframe.sort_values("EigenValue", ascending=False)

    cumulative_variance = np.cumsum(eigen_dataframe["EigenValue"]) / np.sum(eigen_dataframe["EigenValue"])

    eigen_dataframe["CumulativeVariance"] = cumulative_variance

    index = cumulative_variance.searchsorted(explained_variance_threshold)

    eigen_dataframe = eigen_dataframe.iloc[:index + 1, :]

    return eigen_dataframe


def orthogonal_features(
    X: np.ndarray, 
    variance_threshold: float = 0.95
) -> tuple:
    """
    Compute Orthogonal Features using eigenvalues and eigenvectors.

    :param X: Features matrix
    :type X: np.ndarray
    :param variance_threshold: Threshold for cumulative explained variance, default is 0.95
    :type variance_threshold: float
    :return: Tuple containing Orthogonal Features and eigenvalues information
    :rtype: tuple
    """
    Z = (X - X.mean(axis=0)) / X.std(axis=0)
    dot_product = Z.T @ Z
    eigen_dataframe = eigen_vectors(dot_product, variance_threshold)

    W = np.vstack(eigen_dataframe["EigenVector"].values).T

    P = Z @ W

    return P, eigen_dataframe
