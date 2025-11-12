"""
Functions for generating synthetic, structured data for financial simulations.
Includes functions for creating random covariance matrices and
simulating multivariate normal observations.
"""

import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from sklearn.covariance import LedoitWolf
from typing import Tuple

# Import the utility from the denoising module
from RiskLabAI.data.denoise.denoising import corr_to_cov

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
    Forms a shuffled block diagonal correlation matrix and the
    corresponding covariance matrix.

    :param n_blocks: Number of blocks
    :type n_blocks: int
    :param block_size: Size of each block
    :type block_size: int
    :param block_correlation: Correlation within each block
    :type block_correlation: float
    :return: (mu0, cov0) Mean vector and Covariance matrix
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    corr0 = form_block_matrix(n_blocks, block_size, block_correlation)
    corr0 = pd.DataFrame(corr0)
    cols = corr0.columns.tolist()
    np.random.shuffle(cols)
    corr0 = corr0[cols].loc[cols].copy(deep=True).values
    
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
    Simulates multivariate normal observations and computes the
    sample mean and covariance.

    :param mu0: True mean
    :type mu0: np.ndarray
    :param cov0: True covariance matrix
    :type cov0: np.ndarray
    :param n_obs: Number of observations
    :type n_obs: int
    :param shrink: Whether to use Ledoit-Wolf shrinkage, default False
    :type shrink: bool
    :return: (mu1, cov1) Sample mean and covariance matrix
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    x = np.random.multivariate_normal(mu0.flatten(), cov0, size=n_obs)
    mu1 = x.mean(axis=0).reshape(-1, 1)
    cov1 = LedoitWolf().fit(x).covariance_ if shrink else np.cov(x, rowvar=0)
    return mu1, cov1