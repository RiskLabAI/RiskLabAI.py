import numpy as np
from typing import Optional

def pca_weights(
    cov: np.ndarray,
    risk_distribution: Optional[np.ndarray] = None,
    risk_target: float = 1.0
) -> np.ndarray:
    """
    Calculates hedging weights using covariance, risk distribution, and risk target.

    The function uses Principal Component Analysis (PCA) to determine the weights.
    If the risk distribution is not provided, all risk is allocated to the principal 
    component with the smallest eigenvalue.

    :param cov: Covariance matrix.
    :type cov: np.ndarray
    :param risk_distribution: Risk distribution, defaults to None.
    :type risk_distribution: Optional[np.ndarray], optional
    :param risk_target: Risk target, defaults to 1.0.
    :type risk_target: float, optional
    :return: Weights calculated based on PCA.
    :rtype: np.ndarray

    .. note:: 
       Reference:
       De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
       Methodology 36.

    .. math::
       w = EV . \sqrt{\frac{\rho T}{\lambda}}

       where:
       - :math:`w` are the weights.
       - :math:`EV` is the matrix of eigenvectors.
       - :math:`\rho` is the risk distribution.
       - :math:`T` is the risk target.
       - :math:`\lambda` is the eigenvalues.
    """

    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    eigen_values, eigen_vectors = np.linalg.eigh(cov)  # must be Hermitian
    # Sort eigenvalues in descending order
    indices = eigen_values.argsort()[::-1]  
    eigen_values, eigen_vectors = eigen_values[indices], eigen_vectors[:, indices]

    # If risk_distribution is not provided, allocate all risk to the principal component 
    # with the smallest eigenvalue.
    if risk_distribution is None:
        risk_distribution = np.zeros(cov.shape[0])
        risk_distribution[-1] = 1.0

    # Compute loads (allocation in the new orthogonal basis)
    loads = risk_target * (risk_distribution / eigen_values) ** 0.5  
    # Calculate weights
    weights = np.dot(eigen_vectors, np.reshape(loads, (-1, 1)))  

    return weights