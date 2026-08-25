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

from numbers import Integral, Real
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.neighbors import KernelDensity

# --- FIX 5: Removed unused imports for LedoitWolf and block_diag ---


def marcenko_pastur_pdf(variance: float, q: float, num_points: int = 1000) -> pd.Series:
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
) -> float:  # <-- FIX 2: Added bandwidth
    r"""
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
    empirical_pdf = np.exp(
        kde.score_samples(theoretical_pdf.index.values.reshape(-1, 1))
    )

    # Calculate SSE
    sse = np.sum((empirical_pdf - theoretical_pdf.values) ** 2)
    return sse


def find_max_eval(
    eigenvalues: np.ndarray, q: float, bandwidth: float
) -> tuple[float, float]:
    r"""
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
    def objective_func(*args):
        return _mp_pdf_fit_error(args[0], q, eigenvalues_1d, bandwidth)

    optimizer_result = minimize(
        objective_func,
        x0=np.array([0.5]),  # Initial variance guess
        bounds=((1e-5, 1 - 1e-5),),
    )

    if optimizer_result.success:
        variance = optimizer_result.x[0]
    else:
        variance = 1.0  # Fallback

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
        corr2 = (
            eigenvectors_noise
            @ (np.diag([avg_noise_eigenvalue] * (eigenvalues.shape[0] - num_facts)))
            @ eigenvectors_noise.T
        )

        # 5. Add them back together
        corr1 = corr1 + corr2

    # 6. Rescale to be a valid correlation matrix
    diag_inv_sqrt = 1.0 / np.sqrt(np.diag(corr1))
    corr1 = np.diag(diag_inv_sqrt) @ corr1 @ np.diag(diag_inv_sqrt)
    np.fill_diagonal(corr1, 1.0)  # Clean up numerical errors
    return corr1


def denoised_corr2(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    num_factors: int,
    alpha: float = 0.0,
) -> np.ndarray:
    r"""Denoise a correlation matrix by targeted eigenvector shrinkage.

    The leading ``num_factors`` eigenpairs are retained as signal. The
    remaining eigenpairs are treated as noise, and their off-diagonal
    contribution is multiplied by ``alpha`` while their diagonal contribution
    is preserved. Thus ``alpha=0`` applies full targeted shrinkage and
    ``alpha=1`` reconstructs the unshrunk matrix.

    This is the targeted-shrinkage construction in López de Prado (2020),
    *Machine Learning for Asset Managers*, Section 2.5.2, Code Snippet 2.6.

    Parameters
    ----------
    eigenvalues : np.ndarray
        Square diagonal matrix of eigenvalues, ordered consistently with
        ``eigenvectors`` and conventionally from largest to smallest.
    eigenvectors : np.ndarray
        Square matrix whose columns are the corresponding eigenvectors.
    num_factors : int
        Number of leading eigenpairs classified as signal. Values from zero
        through the matrix dimension are allowed.
    alpha : float, default=0.0
        Noise shrinkage intensity in the closed interval ``[0, 1]``.

    Returns
    -------
    np.ndarray
        The targeted-shrinkage matrix.

    Raises
    ------
    TypeError
        If the arrays are not real numeric arrays, ``num_factors`` is not an
        integer, or ``alpha`` is not real.
    ValueError
        If the arrays are empty or incompatible, the eigenvalue matrix is not
        diagonal, an input is non-finite, or a parameter lies outside its
        permitted range.
    """
    eigenvalues_array = np.asarray(eigenvalues)
    eigenvectors_array = np.asarray(eigenvectors)

    if (
        eigenvalues_array.ndim != 2
        or eigenvalues_array.shape[0] != eigenvalues_array.shape[1]
    ):
        raise ValueError("eigenvalues must be a non-empty square diagonal matrix")
    if eigenvalues_array.shape[0] == 0:
        raise ValueError("eigenvalues must be a non-empty square diagonal matrix")
    if eigenvectors_array.shape != eigenvalues_array.shape:
        raise ValueError("eigenvectors must be square and match eigenvalues")

    for name, array in (
        ("eigenvalues", eigenvalues_array),
        ("eigenvectors", eigenvectors_array),
    ):
        if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
            raise TypeError(f"{name} must be a real numeric array")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")

    diagonal = np.diag(np.diag(eigenvalues_array))
    if not np.allclose(eigenvalues_array, diagonal, rtol=0.0, atol=1e-12):
        raise ValueError("eigenvalues must be a diagonal matrix")

    if isinstance(num_factors, (bool, np.bool_)) or not isinstance(
        num_factors, Integral
    ):
        raise TypeError("num_factors must be an integer")
    num_factors = int(num_factors)
    dimension = eigenvalues_array.shape[0]
    if not 0 <= num_factors <= dimension:
        raise ValueError("num_factors must be between zero and the matrix dimension")

    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real):
        raise TypeError("alpha must be a real number")
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be finite and between zero and one")

    signal_values = eigenvalues_array[:num_factors, :num_factors]
    signal_vectors = eigenvectors_array[:, :num_factors]
    noise_values = eigenvalues_array[num_factors:, num_factors:]
    noise_vectors = eigenvectors_array[:, num_factors:]

    signal = signal_vectors @ signal_values @ signal_vectors.T
    noise = noise_vectors @ noise_values @ noise_vectors.T
    return signal + alpha * noise + (1.0 - alpha) * np.diag(np.diag(noise))


# --- Utility Functions ---


def pca(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the principal component analysis of a Hermitian matrix.
    Ensures eigenvalues are sorted descending.

    :param matrix: Hermitian matrix (e.g., correlation matrix)
    :type matrix: np.ndarray
    :return: (eigenvalues_vector, eigenvectors_matrix)
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    indices = eigenvalues.argsort()[::-1]  # Sort descending
    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]
    return eigenvalues, eigenvectors


def cov_to_corr(cov: np.ndarray) -> np.ndarray:
    """Convert covariance matrix to correlation matrix."""
    std = np.sqrt(np.diag(cov))
    # Handle division by zero if any std is 0
    std[std == 0] = 1.0
    corr = cov / np.outer(std, std)
    corr[corr < -1] = -1.0  # Handle numerical errors
    corr[corr > 1] = 1.0
    np.fill_diagonal(corr, 1.0)  # Ensure diagonal is 1
    return corr


def corr_to_cov(corr: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Convert correlation matrix to covariance matrix."""
    return corr * np.outer(std, std)


def denoise_cov(cov0: np.ndarray, q: float, bandwidth: float = 0.01) -> np.ndarray:
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
    eigenvalues_diag = np.diag(eigenvalues)  # 2D diag matrix (desc)

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


def optimal_portfolio(cov: np.ndarray, mu: Optional[np.ndarray] = None) -> np.ndarray:
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
    w /= ones.T @ w
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
