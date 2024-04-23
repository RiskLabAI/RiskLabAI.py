import pandas as pd
import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from sklearn.neighbors._kde import KernelDensity
from typing import Tuple, Union, Optional


def marcenko_pastur_pdf(
        variance: float,
        q: float,
        num_points: int
    ) -> pd.Series:
    """
    Computes the Marcenko-Pastur probability density function (pdf).

    :param variance: Variance of the observations
    :type variance: float
    :param q: Ratio T/N
    :type q: float
    :param num_points: Number of points in the pdf
    :type num_points: int
    :return: The Marcenko-Pastur pdf as a pandas Series
    :rtype: pd.Series

    The Marcenko-Pastur pdf is given by the formula:
    .. math::
       \frac{q}{{2 \pi \sigma \lambda}} \sqrt{(\lambda_{max} - \lambda)(\lambda - \lambda_{min})}

    where:
    - :math:`\lambda` is the eigenvalue
    - :math:`\sigma` is the variance of the observations
    - :math:`q` is the ratio T/N
    - :math:`\lambda_{max}` and :math:`\lambda_{min}` are the maximum and minimum eigenvalues respectively

    """
    lambda_min = variance * (1 - 1./q**0.5)**2
    lambda_max = variance * (1 + 1./q**0.5)**2
    eigenvalues = np.linspace(lambda_min, lambda_max, num_points).flatten()
    pdf = q / (2 * np.pi * variance * eigenvalues) * ((lambda_max - eigenvalues) * (eigenvalues - lambda_min))**0.5
    return pd.Series(pdf, index=eigenvalues)


def pca(
        matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the principal component analysis of a Hermitian matrix.

    :param matrix: Hermitian matrix
    :type matrix: np.ndarray
    :return: Eigenvalues and eigenvectors
    :rtype: Tuple[np.ndarray, np.ndarray]

    The principal component analysis is computed using the eigen decomposition of the Hermitian matrix.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    indices = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[indices], eigenvectors[:, indices]
    return np.diagflat(eigenvalues), eigenvectors

def fit_kde(
        observations: Union[np.ndarray, pd.Series],
        bandwidth: float = 0.25,
        kernel: str = 'gaussian',
        x: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> pd.Series:
    """
    Fit a kernel density estimator to a series of observations.

    :param observations: Series of observations
    :type observations: Union[np.ndarray, pd.Series]
    :param bandwidth: Bandwidth of the kernel
    :type bandwidth: float
    :param kernel: Type of kernel to use (e.g., 'gaussian')
    :type kernel: str
    :param x: Array of values on which the fit KDE will be evaluated
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


def random_cov(
        num_columns: int,
        num_factors: int
    ) -> np.ndarray:
    """
    Generate a random covariance matrix.

    :param num_columns: Number of columns in the covariance matrix
    :type num_columns: int
    :param num_factors: Number of factors for random covariance matrix
    :type num_factors: int
    :return: Random covariance matrix
    :rtype: np.ndarray
    """
    w = np.random.normal(size=(num_columns, num_factors))
    cov = np.dot(w, w.T)
    cov += np.diag(np.random.uniform(size=num_columns))
    return cov


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
    corr = np.clip(corr, -1, 1)  # fix numerical errors
    return corr

def error_pdfs(
        variance: float,
        eigenvalues: np.ndarray,
        q: float,
        bandwidth: float,
        num_points: int = 1000
    ) -> float:
    """
    Computes the sum of squared errors between the theoretical and empirical PDFs.

    :param variance: Variance of the observations
    :type variance: float
    :param eigenvalues: Eigenvalues of the correlation matrix
    :type eigenvalues: np.ndarray
    :param q: Ratio T/N
    :type q: float
    :param bandwidth: Bandwidth of the kernel
    :type bandwidth: float
    :param num_points: Number of points in the PDF
    :type num_points: int
    :return: Sum of squared errors between the theoretical and empirical PDFs
    :rtype: float
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
    Find the maximum random eigenvalue by fitting the Marcenko-Pastur distribution.

    :param eigenvalues: Eigenvalues of the correlation matrix
    :type eigenvalues: np.ndarray
    :param q: Ratio T/N
    :type q: float
    :param bandwidth: Bandwidth of the kernel
    :type bandwidth: float
    :return: Maximum random eigenvalue and its variance
    :rtype: Tuple[float, float]
    """
    result = minimize(lambda *x: error_pdfs(*x), 0.5, args=(eigenvalues, q, bandwidth), bounds=((1E-5, 1-1E-5),))
    variance = result['x'][0] if result['success'] else 1
    emax = variance * (1 + (1./q)**0.5)**2
    return emax, variance


def denoised_corr(
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
        num_factors: int
    ) -> np.ndarray:
    """
    Remove noise from the correlation matrix by fixing random eigenvalues.

    :param eigenvalues: Eigenvalues of the correlation matrix
    :type eigenvalues: np.ndarray
    :param eigenvectors: Eigenvectors of the correlation matrix
    :type eigenvectors: np.ndarray
    :param num_factors: Number of factors for the correlation matrix
    :type num_factors: int
    :return: Denoised correlation matrix
    :rtype: np.ndarray
    """
    eval_copy = np.diag(eigenvalues).copy()
    eval_copy[num_factors:] = eval_copy[num_factors:].sum() / float(eval_copy.shape[0] - num_factors)
    eval_copy = np.diag(eval_copy)
    corr1 = np.dot(eigenvectors, eval_copy).dot(eigenvectors.T)
    corr1 = cov_to_corr(corr1)
    return corr1


def denoised_corr2(
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
        num_factors: int,
        alpha: float = 0
    ) -> np.ndarray:
    """
    Remove noise from the correlation matrix through targeted shrinkage.

    :param eigenvalues: Eigenvalues of the correlation matrix
    :type eigenvalues: np.ndarray
    :param eigenvectors: Eigenvectors of the correlation matrix
    :type eigenvectors: np.ndarray
    :param num_factors: Number of factors for the correlation matrix
    :type num_factors: int
    :param alpha: Shrinkage parameter
    :type alpha: float
    :return: Denoised correlation matrix
    :rtype: np.ndarray
    """
    eval_L, evec_L = eigenvalues[:num_factors, :num_factors], eigenvectors[:, :num_factors]
    eval_R, evec_R = eigenvalues[num_factors:, num_factors:], eigenvectors[:, num_factors:]
    corr0 = np.dot(evec_L, eval_L).dot(evec_L.T)
    corr1 = np.dot(evec_R, eval_R).dot(evec_R.T)
    corr2 = corr0 + alpha*corr1 + (1 - alpha)*np.diag(np.diag(corr1))
    return corr2

def form_block_matrix(
        n_blocks: int,
        block_size: int,
        block_correlation: float
    ) -> np.ndarray:
    """
    Forms a block diagonal correlation matrix.

    :param n_blocks: Number of blocks
    :type n_blocks: int
    :param block_size: Size of each block
    :type block_size: int
    :param block_correlation: Correlation within each block
    :type block_correlation: float
    :return: Block diagonal correlation matrix
    :rtype: np.ndarray
    """
    block = np.ones((block_size, block_size)) * block_correlation
    block[range(block_size), range(block_size)] = 1
    corr = block_diag(*([block] * n_blocks))
    return corr

def form_true_matrix(
        n_blocks: int,
        block_size: int,
        block_correlation: float
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forms a shuffled block diagonal correlation matrix and the corresponding covariance matrix.

    :param n_blocks: Number of blocks
    :type n_blocks: int
    :param block_size: Size of each block
    :type block_size: int
    :param block_correlation: Correlation within each block
    :type block_correlation: float
    :return: Mean and covariance matrix
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    corr0 = form_block_matrix(n_blocks, block_size, block_correlation)
    corr0 = pd.DataFrame(corr0)
    cols = corr0.columns.tolist()
    np.random.shuffle(cols)
    corr0 = corr0[cols].loc[cols].copy(deep=True)
    std0 = np.random.uniform(.05, .2, corr0.shape[0])
    cov0 = corr_to_cov(corr0, std0)
    mu0 = np.random.normal(std0, std0, cov0.shape[0]).reshape(-1, 1)
    return mu0, cov0

def simulates_cov_mu(
        mu0: np.ndarray,
        cov0: np.ndarray,
        n_obs: int,
        shrink: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulates multivariate normal observations and computes the sample mean and covariance.

    :param mu0: True mean
    :type mu0: np.ndarray
    :param cov0: True covariance matrix
    :type cov0: np.ndarray
    :param n_obs: Number of observations
    :type n_obs: int
    :param shrink: Whether to use Ledoit-Wolf shrinkage
    :type shrink: bool
    :return: Sample mean and covariance matrix
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    x = np.random.multivariate_normal(mu0.flatten(), cov0, size=n_obs)
    mu1 = x.mean(axis=0).reshape(-1, 1)
    cov1 = LedoitWolf().fit(x).covariance_ if shrink else np.cov(x, rowvar=0)
    return mu1, cov1

def corr_to_cov(
        corr: np.ndarray,
        std: np.ndarray
    ) -> np.ndarray:
    """
    Converts a correlation matrix to a covariance matrix.

    :param corr: Correlation matrix
    :type corr: np.ndarray
    :param std: Standard deviations
    :type std: np.ndarray
    :return: Covariance matrix
    :rtype: np.ndarray
    """
    return corr * np.outer(std, std)

def denoise_cov(
        cov0: np.ndarray,
        q: float,
        bandwidth: float
    ) -> np.ndarray:
    """
    De-noises the covariance matrix.

    :param cov0: Covariance matrix
    :type cov0: np.ndarray
    :param q: Ratio of number of observations to number of variables
    :type q: float
    :param bandwidth: Bandwidth parameter
    :type bandwidth: float
    :return: De-noised covariance matrix
    :rtype: np.ndarray
    """
    corr0 = cov_to_corr(cov0)
    eval0, evec0 = np.linalg.eigh(corr0)
    eval0 = np.diag(eval0)
    emax0, var0 = find_max_eval(eval0, q, bandwidth)
    n_facts0 = eval0.shape[0] - np.diag(eval0)[::-1].searchsorted(emax0)
    corr1 = denoised_corr(eval0, evec0, n_facts0)
    cov1 = corr_to_cov(corr1, np.diag(cov0)**0.5)
    return cov1

def optimal_portfolio(
        cov: np.ndarray,
        mu: np.ndarray = None
    ) -> np.ndarray:
    """
    Computes the optimal portfolio weights.

    :param cov: Covariance matrix
    :type cov: np.ndarray
    :param mu: Expected returns
    :type mu: np.ndarray
    :return: Optimal portfolio weights
    :rtype: np.ndarray
    """
    inv = np.linalg.inv(cov)
    ones = np.ones(shape=(inv.shape[0], 1))
    if mu is None:
        mu = ones
    w = np.dot(inv, mu)
    w /= np.dot(ones.T, w)
    return w
