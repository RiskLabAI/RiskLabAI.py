"""
Tests for optimization/nco.py
"""

import pytest
import numpy as np
import pandas as pd
from .nco import get_optimal_portfolio_weights, get_optimal_portfolio_weights_nco

@pytest.fixture
def mock_cov_matrix():
    """
    Generate a 4x4 covariance matrix with two blocks.
    Assets [0, 1] are correlated.
    Assets [2, 3] are correlated.
    Blocks are uncorrelated.
    """
    cov = np.array([
        [1.0, 0.8, 0.0, 0.0],
        [0.8, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.5],
        [0.0, 0.0, 0.5, 1.0]
    ])
    return cov

def test_get_optimal_portfolio_weights_gmv(mock_cov_matrix):
    """Test the GMV portfolio calculation."""
    weights = get_optimal_portfolio_weights(mock_cov_matrix, mu=None)
    
    assert weights.shape == (4, 1)
    assert np.isclose(weights.sum(), 1.0)
    
    # Check that weights are spread
    assert all(w > 0 for w in weights)

def test_get_optimal_portfolio_weights_mvo(mock_cov_matrix):
    """Test the MVO portfolio calculation."""
    mu = np.array([0.1, 0.2, 0.05, 0.1]).reshape(-1, 1)
    weights = get_optimal_portfolio_weights(mock_cov_matrix, mu=mu)
    
    assert weights.shape == (4, 1)
    assert np.isclose(weights.sum(), 1.0)
    
    # MVO should overweight asset 1 (highest return)
    assert weights[1] > weights[0]
    assert weights[1] > weights[2]

@pytest.mark.filterwarnings("ignore:Warning: RiskLabAI.cluster.clustering")
def test_get_optimal_portfolio_weights_nco(mock_cov_matrix):
    """Test the NCO algorithm."""
    # This will use the dummy clusterer, but it should still run.
    # To test properly, the RiskLabAI.cluster module must be available.
    
    weights = get_optimal_portfolio_weights_nco(
        mock_cov_matrix, number_clusters=2
    )
    
    assert weights.shape == (4, 1)
    assert np.isclose(weights.sum(), 1.0)