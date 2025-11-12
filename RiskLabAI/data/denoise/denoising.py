"""
Implements covariance matrix denoising using Random Matrix Theory (RMT).

This module provides functions to:
1. Fit the Marcenko-Pastur distribution to the eigenvalues of a
   correlation matrix.
2. Identify and remove eigenvalues associated with noise.
3. Reconstruct a "denoised" correlation and covariance matrix.

Reference:
    De Prado, M. (2018) Advances in financial machine learning.
    John Wiley & Sons, Chapter 2.
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.neighbors import KernelDensity
from typing import Tuple, Union, Optional

# --- Marcenko-Pastur (MP) PDF ---

def marcenko_pastur_pdf(
    variance: float, q: float, num_points: int = 1000
) -> pd.Series:
    r"""
    Compute the Marcenko-Pastur (MP) probability density function.

    .. math::
        f(\lambda) = \frac{q}{2\pi\sigma^2\lambda} \sqrt{(\lambda_{max} - \lambda)
                     (\lambda - \lambda_{min})}

    :param variance: Variance of the observations (\(\sigma^2\))
    :type variance: float
    :param q: Ratio T/N, where T is observations and N is features.
    :type q: float
    :param num_points: Number of points in the PDF, default 1000
    :type num_points: int
    :return: The Marcenko-Pastur PDF, indexed by eigenvalues (\(\lambda\)).
    :rtype: pd.Series
    """
    lambda_min = variance * (1 - (1.0 / q) ** 0.5) ** 2
    lambda_max = variance * (1 + (1.0 / q) ** 0.5) ** 2
    
    eigenvalues = np.linspace(lambda_min, lambda_max, num_points).flatten()
    
    pdf = (q / (2 * np.pi * variance * eigenvalues)) * (
        (lambda_max - eigenvalues) * (eigenvalues - lambda_min)
    ) ** 0.5
    
    return pd.Series(pdf, index=eigenvalues)


def fit_kde(
    observations: Union[np.ndarray, pd.Series],
    bandwidth: float = 0.25,
    kernel: str = 'gaussian',
    x: Optional[Union[np.ndarray, pd.Series]] = None
) -> pd.Series:
    """
    Fit a kernel density estimator (KDE) to a series of observations.

    :param observations: Series of observations (e.g., eigenvalues)
    :type observations: Union[np.ndarray, pd.Series]
    :param bandwidth: Bandwidth of the kernel, default 0.25
    :type bandwidth: float
    :param kernel: Type of kernel to use (e.g., 'gaussian'), default 'gaussian'
    :type kernel: str
    :param x: Array of values on which the fit KDE will be evaluated.
              If None, defaults to unique observation values.
    :type x: Optional[Union[np.ndarray, pd.Series]]
    :return: Kernel density estimate as a pandas Series
    :rtype: pd.Series
    """
    observations = np.asarray(observations).reshape(-1, 1)
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(observations)

    if x is None:
        x = np.unique(observations).reshape(-1, 1)
    else:
        x = np.asarray(x).reshape(-1, 1)

    log_prob = kde.score_samples(x)
    pdf = pd.Series(np.exp(log_prob), index=x.flatten())
    return pdf


def _mp_pdf_fit_error(
    variance: float,
    eigenvalues: np.ndarray,
    q: float,
    bandwidth: float,
    num_points: int = 1000
) -> float:
    """
    Internal: Computes the SSE between theoretical and empirical MP PDFs.
    """
    pdf0 = marcenko_pastur_pdf(variance, q, num_points)
    pdf1 = fit_kde(eigenvalues, bandwidth, x=pdf0.index.values)
    sse = np.sum((pdf1 - pdf0)**2)
    return sse


def find_max_eval(
    eigenvalues: np.ndarray,
    q: float,
    bandwidth: float
) -> Tuple[float, float]:
    """
    Find the maximum random eigenvalue (\(\lambda_{max}\))
    by fitting the Marcenko-Pastur distribution.

    :param eigenvalues: 1D array of observed eigenvalues
    :type eigenvalues: np.ndarray
    :param q: Ratio T/N
    :type q: float
    :param bandwidth: Bandwidth of the kernel
    :type bandwidth: float
    :return: (lambda_max, fitted_variance)
    :rtype: Tuple[float, float]
    """
    objective_func = lambda *args: _mp_pdf_fit_error(args[0], eigenvalues, q, bandwidth)
    
    optimizer_result = minimize(
        objective_func,
        x0=np.array([0.5]), # Initial variance guess
        bounds=((1e-5, 1 - 1e-5),),
    )

    if optimizer_result.success:
        variance = optimizer_result.x[0]
    else:
        variance = 1.0 # Fallback

    # Calculate lambda_max based on the fitted variance
    lambda_max = variance * (1 + (1.0 / q) ** 0.5) ** 2
    return lambda_max, variance

# --- Denoising Methods ---

def denoised_corr(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    num_factors: int
) -> np.ndarray:
    """
    Denoise correlation matrix using the Constant Residual Eigenvalue method.

    :param eigenvalues: 1D array of eigenvalues, sorted descending
    :type eigenvalues: np.ndarray
    :param eigenvectors: 2D array of corresponding eigenvectors
    :type eigenvectors: np.ndarray
    :param num_factors: The number of factors (signal eigenvalues) to keep.
    :type num_factors: int
    :return: Denoised correlation matrix
    :rtype: np.ndarray
    """
    eval_copy = eigenvalues.copy()
    
    # Average the noise eigenvalues
    if num_factors < len(eval_copy):
        avg_noise = eval_copy[num_factors:].sum() / float(len(eval_copy) - num_factors)
        eval_copy[num_factors:] = avg_noise
        
    # Reconstruct
    eval_diag = np.diag(eval_copy)
    corr1 = np.dot(eigenvectors, eval_diag).dot(eigenvectors.T)
    corr1 = cov_to_corr(corr1) # Rescale to be a valid correlation matrix
    return corr1


def denoised_corr2(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    num_factors: int,
    alpha: float = 0
) -> np.ndarray:
    """
    Denoise correlation matrix using Targeted Shrinkage.

    :param eigenvalues: 1D array of eigenvalues, sorted descending
    :type eigenvalues: np.ndarray
    :param eigenvectors: 2D array of corresponding eigenvectors
    :type eigenvectors: np.ndarray
    :param num_factors: Number of factors for the correlation matrix
    :type num_factors: int
    :param alpha: Shrinkage parameter, default 0
    :type alpha: float
    :return: Denoised correlation matrix
    :rtype: np.ndarray
    """
    eval_L, evec_L = eigenvalues[:num_factors], eigenvectors[:, :num_factors]
    eval_R, evec_R = eigenvalues[num_factors:], eigenvectors[:, num_factors:]

    corr0 = np.dot(evec_L, np.diag(eval_L)).dot(evec_L.T)
    corr1 = np.dot(evec_R, np.diag(eval_R)).dot(evec_R.T)
    
    # Target shrinkage formula
    corr2 = corr0 + alpha * corr1 + (1 - alpha) * np.diag(np.diag(corr1))
    return corr2

# --- Main Public Function ---

def denoise_cov(
    cov0: np.ndarray,
    q: float,
    bandwidth: float = 0.01,
    denoise_method: str = 'const_resid'
) -> np.ndarray:
    """
    De-noises a covariance matrix using RMT.

    :param cov0: The original (noisy) covariance matrix.
    :type cov0: np.ndarray
    :param q: The T/N ratio.
    :type q: float
    :param bandwidth: The KDE bandwidth, default 0.01
    :type bandwidth: float
    :param denoise_method: 'const_resid' or 'targeted_shrink', 
                           default 'const_resid'
    :type denoise_method: str
    :return: The de-noised covariance matrix.
    :rtype: np.ndarray
    """
    corr0 = cov_to_corr(cov0)
    
    eigenvalues, eigenvectors = pca(corr0) # Use helper to get sorted eigenpairs

    # Find the noise cutoff
    emax0, var0 = find_max_eval(eigenvalues, q, bandwidth)
    
    # Identify number of signal factors
    n_facts0 = eigenvalues.searchsorted(emax0, side='right')
    
    # Denoise the correlation matrix
    if denoise_method == 'const_resid':
        corr1 = denoised_corr(eigenvalues, eigenvectors, n_facts0)
    elif denoise_method == 'targeted_shrink':
        # Note: targeted_shrink (denoised_corr2) is often used with alpha=0
        corr1 = denoised_corr2(eigenvalues, eigenvectors, n_facts0, alpha=0)
    else:
        raise ValueError(f"Unknown denoise_method: {denoise_method}")

    # Convert back to covariance
    cov1 = corr_to_cov(corr1, np.diag(cov0) ** 0.5)
    return cov1

# --- Utility Functions ---

def pca(
    matrix: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the principal component analysis of a Hermitian matrix.
    Ensures eigenvalues are sorted descending.

    :param matrix: Hermitian matrix (e.g., correlation matrix)
    :type matrix: np.ndarray
    :return: (eigenvalues_vector, eigenvectors_matrix)
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    indices = eigenvalues.argsort()[::-1] # Sort descending
    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]
    return eigenvalues, eigenvectors

def cov_to_corr(cov: np.ndarray) -> np.ndarray:
    """
    Convert a covariance matrix to a correlation matrix.

    :param cov: Covariance matrix
    :type cov: np.ndarray
    :return: Correlation matrix
    :rtype: np.ndarray
    """
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr = np.clip(corr, -1, 1)  # Fix numerical errors
    return corr

def corr_to_cov(
    corr: np.ndarray,
    std: np.ndarray
) -> np.ndarray:
    """
    Converts a correlation matrix to a covariance matrix.

    :param corr: Correlation matrix
    :type corr: np.ndarray
    :param std: 1D array of standard deviations
    :type std: np.ndarray
    :return: Covariance matrix
    :rtype: np.ndarray
    """
    return corr * np.outer(std, std)