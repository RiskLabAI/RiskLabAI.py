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
from typing import Tuple, Union, Optional, Dict, Any

# --- FIX 5: Removed unused imports for LedoitWolf and block_diag ---


def marcenko_pastur_pdf(
    variance: float, q: float, num_points: int = 1000
) -> pd.Series:
    r"""
    Compute the Marcenko-Pastur (MP) probability density function.

    This function defines the theoretical distribution of eigenvalues
    for a random covariance matrix.

    .. math::
        f(\lambda) = \frac{q}{2\pi\sigma^2\lambda} \sqrt{(\lambda_{max} - \lambda)
                     (\lambda - \lambda_{min})}

    Parameters
    ----------
    variance : float
        Variance of the observations (\(\sigma^2\)).
    q : float
        Ratio T/N, where T is observations and N is features.
    num_points : int, default=1000
        Number of points in the PDF.

    Returns
    -------
    pd.Series
        The Marcenko-Pastur PDF, indexed by eigenvalues (\(\lambda\)).
    """
    lambda_min = variance * (1 - (1.0 / q) ** 0.5) ** 2
    lambda_max = variance * (1 + (1.0 / q) ** 0.5) ** 2
    
    # --- FIX 1: Add epsilon to prevent division by zero if lambda_min=0 (when q=1) ---
    e_min = max(lambda_min, 1e-10) 
    eigenvalues = np.linspace(e_min, lambda_max, num_points)
    
    pdf = (q / (2 * np.pi * variance * eigenvalues)) * (
        (lambda_max - eigenvalues) * (eigenvalues - lambda_min)
    ) ** 0.5
    
    # Set PDF to 0 where eigenvalues are outside the valid range (e.g., due to numerical precision)
    pdf[np.isnan(pdf)] = 0
    
    return pd.Series(pdf.flatten(), index=eigenvalues.flatten())


def fit_kde(
    observations: np.ndarray,
    bandwidth: float = 0.01,
    kernel: str = "gaussian",
) -> KernelDensity:
    """
    Fit a Kernel Density Estimator (KDE) to observations.

    Parameters
    ----------
    observations : np.ndarray
        The observed data (e.g., eigenvalues).
    bandwidth : float, default=0.01
        The bandwidth for the kernel.
    kernel : str, default="gaussian"
        The kernel to use.

    Returns
    -------
    KernelDensity
        A fitted `sklearn.neighbors.KernelDensity` object.
    """
    observations = observations.reshape(-1, 1)
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(observations)
    return kde


def _mp_pdf_fit_error(
    variance: float, q: float, eigenvalues: np.ndarray, bandwidth: float
) -> float: # <-- FIX 2: Added bandwidth
    """
    Error function for fitting the MP PDF to observed eigenvalues.
    
    Calculates the sum of squared errors between the theoretical
    MP PDF and the empirical PDF (from KDE).

    Parameters
    ----------
    variance : float
        The \(\sigma^2\) parameter to test.
    q : float
        The T/N ratio.
    eigenvalues : np.ndarray
        The observed eigenvalues.
    bandwidth : float
        The KDE bandwidth.

    Returns
    -------
    float
        The sum of squared errors.
    """
    # Ensure eigenvalues is 1D for PDF generation
    if eigenvalues.ndim == 2:
        eigenvalues = np.diag(eigenvalues)
        
    theoretical_pdf = marcenko_pastur_pdf(variance, q, num_points=eigenvalues.shape[0])
    
    # Fit empirical PDF
    # --- FIX 2: Pass bandwidth to fit_kde ---
    kde = fit_kde(eigenvalues, bandwidth=bandwidth) 
    empirical_pdf = np.exp(kde.score_samples(theoretical_pdf.index.values.reshape(-1, 1)))
    
    # Calculate SSE
    sse = np.sum((empirical_pdf - theoretical_pdf.values) ** 2)
    return sse


def find_max_eval(
    eigenvalues: np.ndarray, q: float, bandwidth: float
) -> Tuple[float, float]:
    """
    Find the maximum theoretical eigenvalue (\(\lambda_{max}\))
    by fitting the Marcenko-Pastur distribution.

    Parameters
    ----------
    eigenvalues : np.ndarray
        The diagonal matrix (or 1D vector) of observed eigenvalues.
    q : float
        The T/N ratio.
    bandwidth : float
        The KDE bandwidth.

    Returns
    -------
    Tuple[float, float]
        - lambda_max: The maximum theoretical eigenvalue.
        - variance: The fitted variance (\(\sigma^2\)).
    """
    # --- FIX 3: Ensure we have a 1D array for fitting ---
    if eigenvalues.ndim == 2:
        eigenvalues_1d = np.diag(eigenvalues)
    else:
        eigenvalues_1d = eigenvalues
    
    # Minimize the SSE to find the best-fit variance
    # --- FIX 2: Pass bandwidth to the objective function ---
    objective_func = lambda *args: _mp_pdf_fit_error(args[0], q, eigenvalues_1d, bandwidth)
    
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


def denoised_corr(
    eigenvalues: np.ndarray, eigenvectors: np.ndarray, num_facts: int
) -> np.ndarray:
    """
    Reconstruct the correlation matrix using only the eigenvalues
    associated with signal (i.e., > lambda_max).
    
    Note: Assumes eigenvalues are sorted in descending order.

    Parameters
    ----------
    eigenvalues : np.ndarray
        The diagonal matrix of *all* eigenvalues (sorted descending).
    eigenvectors : np.ndarray
        The matrix of eigenvectors (corresponding to descending eigenvalues).
    num_facts : int
        The number of factors (signal eigenvalues) to keep.

    Returns
    -------
    np.ndarray
        The denoised correlation matrix.
    """
    # 1. Get the eigenvalues and eigenvectors for signal
    # --- FIX 3: This logic is now correct as eigenvalues are descending ---
    eigenvalues_1d = np.diag(eigenvalues) 
    eigenvalues_signal = np.diag(eigenvalues_1d[:num_facts]) 

    eigenvectors_signal = eigenvectors[:, :num_facts]
    
    # 2. Reconstruct the signal-only correlation matrix
    corr1 = eigenvectors_signal @ eigenvalues_signal @ eigenvectors_signal.T
    
    # 3. Get the eigenvalues for noise and average them
    if num_facts < eigenvalues.shape[0]:
        # --- FIX 3: Correctly averages the smaller (noise) eigenvalues ---
        avg_noise_eigenvalue = eigenvalues_1d[num_facts:].mean()
        eigenvectors_noise = eigenvectors[:, num_facts:]
        
        # 4. Reconstruct the noise-only correlation matrix
        corr2 = eigenvectors_noise @ (
            np.diag([avg_noise_eigenvalue] * (eigenvalues.shape[0] - num_facts))
        ) @ eigenvectors_noise.T
        
        # 5. Add them back together
        corr1 = corr1 + corr2
        
    # 6. Rescale to be a valid correlation matrix
    diag_inv_sqrt = 1. / np.sqrt(np.diag(corr1))
    corr1 = np.diag(diag_inv_sqrt) @ corr1 @ np.diag(diag_inv_sqrt)
    np.fill_diagonal(corr1, 1.0) # Clean up numerical errors
    return corr1

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
    """Convert covariance matrix to correlation matrix."""
    std = np.sqrt(np.diag(cov))
    # Handle division by zero if any std is 0
    std[std == 0] = 1.0 
    corr = cov / np.outer(std, std)
    corr[corr < -1] = -1.0 # Handle numerical errors
    corr[corr > 1] = 1.0
    np.fill_diagonal(corr, 1.0) # Ensure diagonal is 1
    return corr


def corr_to_cov(corr: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Convert correlation matrix to covariance matrix."""
    return corr * np.outer(std, std)

def denoise_cov(
    cov0: np.ndarray, q: float, bandwidth: float = 0.01
) -> np.ndarray:
    """
    De-noises a covariance matrix.

    Parameters
    ----------
    cov0 : np.ndarray
        The original (noisy) covariance matrix.
    q : float
        The T/N ratio.
    bandwidth : float, default=0.01
        The KDE bandwidth.

    Returns
    -------
    np.ndarray
        The de-noised covariance matrix.
    """

    corr0 = cov_to_corr(cov0)
    
    # --- FIX 3: Use pca helper to get DESCENDING eigenvalues/vectors ---
    eigenvalues, eigenvectors = pca(corr0)
    eigenvalues_diag = np.diag(eigenvalues) # 2D diag matrix (desc)

    # Find the noise cutoff
    # --- FIX 2: Pass bandwidth down to find_max_eval ---
    emax0, var0 = find_max_eval(eigenvalues_diag, q, bandwidth)
    
    # --- FIX 3: Correctly find num factors as count of evals > emax0 ---
    n_facts0 = np.sum(eigenvalues > emax0)
    
    # Denoise the correlation matrix
    corr1 = denoised_corr(eigenvalues_diag, eigenvectors, n_facts0)
    
    # Convert back to covariance
    cov1 = corr_to_cov(corr1, np.diag(cov0) ** 0.5)
    return cov1


def optimal_portfolio(
    cov: np.ndarray, mu: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute the optimal (e.g., minimum variance) portfolio weights.
    
    (Note: This is duplicated in `optimization/nco.py`)

    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix.
    mu : np.ndarray, optional
        Vector of expected returns. If None, computes GMV portfolio.

    Returns
    -------
    np.ndarray
        The optimal portfolio weights.
    """
    inv_cov = np.linalg.inv(cov)
    ones = np.ones(shape=(inv_cov.shape[0], 1))
    
    if mu is None:
        mu = ones
    
    w = inv_cov @ mu
    w /= (ones.T @ w)
    return w.flatten()


def optimal_portfolio_denoised(
    cov: np.ndarray,
    q: float,
    mu: Optional[np.ndarray] = None,
    bandwidth: float = 0.01,
) -> np.ndarray:
    """
    Compute the optimal portfolio weights from a denoised covariance matrix.

    Parameters
    ----------
    cov : np.ndarray
        The *original* (noisy) covariance matrix.
    q : float
        The T/N ratio.
    mu : np.ndarray, optional
        Vector of expected returns.
    bandwidth : float, default=0.01
        The KDE bandwidth.

    Returns
    -------
    np.ndarray
        The optimal, denoised portfolio weights.
    """
    # --- FIX 4: Pass bandwidth to denoise_cov ---
    cov_denoised = denoise_cov(cov, q, bandwidth)
    
    # Compute optimal portfolio on the denoised matrix
    return optimal_portfolio(cov_denoised, mu)