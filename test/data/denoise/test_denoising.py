"""
Tests for data/denoise/denoising.py
"""

import pytest
import numpy as np
import pandas as pd
from RiskLabAI.data.denoise import (
    marcenko_pastur_pdf,
    cov_to_corr,
    corr_to_cov,
    denoise_cov
)

@pytest.fixture
def noisy_cov_matrix():
    """
    Generate a noisy covariance matrix.
    T=100, N=50 => q = 100/50 = 2
    """
    T, N = 100, 50
    rng = np.random.default_rng(42)
    
    # 1. Create a true, simple correlation structure (2 factors)
    factors = rng.normal(size=(T, 2))
    loadings = rng.normal(size=(N, 2))
    
    # 2. Generate returns with noise
    true_returns = factors @ loadings.T
    noise = rng.normal(scale=0.5, size=(T, N))
    
    returns = true_returns + noise
    
    # 3. Calculate the noisy covariance matrix
    cov = np.cov(returns, rowvar=False)
    
    return cov, T/N

def test_marcenko_pastur_pdf():
    """Test the MP PDF calculation."""
    q = 10
    variance = 1.0
    
    pdf = marcenko_pastur_pdf(variance, q, num_points=100)
    
    # lambda_min = 1 * (1 - 1/sqrt(10))^2 = 0.467
    # lambda_max = 1 * (1 + 1/sqrt(10))^2 = 1.732
    
    assert isinstance(pdf, pd.Series)
    assert np.isclose(pdf.index.min(), 0.4675, atol=1e-3)
    assert np.isclose(pdf.index.max(), 1.7324, atol=1e-3)
    assert not pdf.isna().any()

def test_cov_corr_conversion():
    """Test that cov_to_corr and corr_to_cov are inverses."""
    cov = np.array([[4.0, 1.0], [1.0, 1.0]])
    std = np.array([2.0, 1.0])
    
    # Test cov -> corr
    corr = cov_to_corr(cov)
    expected_corr = np.array([[1.0, 0.5], [0.5, 1.0]])
    assert np.allclose(corr, expected_corr)
    
    # Test corr -> cov
    cov_new = corr_to_cov(corr, std)
    assert np.allclose(cov, cov_new)

def test_denoise_cov(noisy_cov_matrix):
    """Test the end-to-end denoising function."""
    cov, q = noisy_cov_matrix
    
    assert cov.shape == (50, 50)
    
    # Denoise the matrix
    cov_denoised = denoise_cov(cov, q, bandwidth=0.01)
    
    assert cov_denoised.shape == cov.shape
    
    # Get eigenvalues
    evals_orig, _ = np.linalg.eigh(cov)
    evals_denoised, _ = np.linalg.eigh(cov_denoised)
    
    # The denoising process "clips" the smallest (noise) eigenvalues
    # and boosts the signal eigenvalues.
    # The smallest denoised eigenvalue should be larger
    # than the smallest original one.
    assert evals_denoised.min() > evals_orig.min()
    
    # The largest denoised eigenvalue should be smaller
    # than the largest original one (as noise variance is removed)
    assert evals_denoised.max() < evals_orig.max()

def test_optimal_portfolio(noisy_cov_matrix):
    """Test the optimal portfolio helper."""
    cov, q = noisy_cov_matrix
    
    # GMV portfolio
    weights = optimal_portfolio(cov, mu=None)
    
    assert weights.shape == (50,)
    assert np.isclose(weights.sum(), 1.0)