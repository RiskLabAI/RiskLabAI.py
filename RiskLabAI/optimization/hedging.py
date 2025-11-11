"""
Implements PCA-based portfolio hedging techniques.
"""

import numpy as np
from typing import Optional

def pca_weights(
    cov: np.ndarray,
    risk_distribution: Optional[np.ndarray] = None,
    risk_target: float = 1.0,
) -> np.ndarray:
    r"""
    Calculate hedging weights using PCA.

    This function uses PCA to determine portfolio weights that
    match a target risk profile.

    Reference:
       De Prado, M. (2018) Advances in financial machine learning.
       Methodology 36.

    .. math::
       w = V \cdot \sqrt{\frac{\rho T}{\lambda}}

    Where:
    - \(w\) are the weights.
    - \(V\) is the matrix of eigenvectors.
    - \(\rho\) is the risk distribution.
    - \(T\) is the risk target.
    - \(\lambda\) are the eigenvalues.

    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix.
    risk_distribution : np.ndarray, optional
        Target risk distribution across principal components.
        If None, all risk is allocated to the component with the
        smallest eigenvalue (minimum variance portfolio).
    risk_target : float, default=1.0
        The total risk target (T).

    Returns
    -------
    np.ndarray
        The calculated portfolio weights.
    """
    # Calculate eigenvalues and eigenvectors
    eigen_values, eigen_vectors = np.linalg.eigh(cov)
    
    # Sort in descending order
    indices = eigen_values.argsort()[::-1]
    eigen_values, eigen_vectors = eigen_values[indices], eigen_vectors[:, indices]

    # If no risk distribution, target minimum variance
    if risk_distribution is None:
        risk_distribution = np.zeros(cov.shape[0])
        risk_distribution[-1] = 1.0  # All risk on last component

    # Compute loads (allocation in the orthogonal basis)
    loads = risk_target * (risk_distribution / eigen_values) ** 0.5
    
    # Calculate weights in the original basis
    weights = np.dot(eigen_vectors, np.reshape(loads, (-1, 1)))

    return weights.flatten()