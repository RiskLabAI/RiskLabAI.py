import numpy as np
from typing import Optional

def pca_weights(
    cov: np.ndarray,
    risk_distribution: Optional[np.ndarray] = None,
    risk_target: float = 1.0
) -> np.ndarray:
    """
    Calculates hedging weights using covariance, risk distribution, and risk target.

    :param cov: Covariance matrix
    :type cov: np.ndarray
    :param risk_distribution: Risk distribution, defaults to None
    :type risk_distribution: Optional[np.ndarray], optional
    :param risk_target: Risk target, defaults to 1.0
    :type risk_target: float, optional
    :return: Weights calculated based on PCA
    :rtype: np.ndarray

    Reference:
        De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
        Methodology 36.
    """
    eigen_values, eigen_vectors = np.linalg.eigh(cov)  # must be Hermitian
    indices = eigen_values.argsort()[::-1]  # arguments for sorting eigen_values descending
    eigen_values, eigen_vectors = eigen_values[indices], eigen_vectors[:, indices]

    # If risk_distribution is not provided, it will assume all risk must be allocated to the principal component with
    # the smallest eigenvalue, and the weights will be the last eigenvector re-scaled to match σ.
    if risk_distribution is None:
        risk_distribution = np.zeros(cov.shape[0])
        risk_distribution[-1] = 1.0

    loads = risk_target * (risk_distribution / eigen_values) ** 0.5  # allocation in the new (orthogonal) basis
    weights = np.dot(eigen_vectors, np.reshape(loads, (-1, 1)))  # calculate weights

    return weights
