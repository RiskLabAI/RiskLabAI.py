"""
Tests for hedging.py
"""

import numpy as np
import pytest
from RiskLabAI.data.structures.hedging import pca_weights

@pytest.fixture
def sample_cov_matrix():
    """A simple 2x2 covariance matrix."""
    # corr = 0.5, std1=1, std2=2
    # cov(0,0) = 1*1 = 1
    # cov(1,1) = 2*2 = 4
    # cov(0,1) = 0.5 * 1 * 2 = 1
    return np.array([[1.0, 1.0], [1.0, 4.0]])

def test_pca_weights_min_variance(sample_cov_matrix):
    """Test PCA weights for minimum variance (default)."""
    cov = sample_cov_matrix
    weights = pca_weights(cov, risk_distribution=None, risk_target=1.0)
    
    # Eigenvectors of [[1, 1], [1, 4]]
    # (solve (1-L)(4-L) - 1 = 0 => L^2 - 5L + 3 = 0)
    # L = (5 +/- sqrt(25-12))/2 = (5 +/- sqrt(13))/2
    # L1 = 4.302, L2 = 0.697
    # Default targets min-variance, so all risk on L2.
    # [v1, v2]
    # v1 (L=4.302): [1, 3.302] -> norm -> [0.293, 0.956]
    # v2 (L=0.697): [1, -0.302] -> norm -> [0.956, -0.293]
    
    # Risk distribution = [0, 1] (all on L2)
    # Loads = 1.0 * [0, 1] / [4.302, 0.697]**0.5 = [0, 1.196]
    # Weights = V * Loads'
    # w0 = v1[0]*0 + v2[0]*1.196 = 0.956 * 1.196 = 1.143
    # w1 = v1[1]*0 + v2[1]*1.196 = -0.293 * 1.196 = -0.350
    # (Note: This is just one solution, sign can be flipped)
    
    # The important part: the resulting portfolio variance
    # w = [1.143, -0.350]
    # Var = w' * C * w = [1.143, -0.350] * [[1,1],[1,4]] * [1.143, -0.350]'
    #     = [1.143, -0.350] * [1.143 - 0.350, 1.143 - 1.400]'
    #     = [1.143, -0.350] * [0.793, -0.257]'
    #     = 1.143*0.793 + (-0.350)*(-0.257) = 0.906 + 0.09 = 0.996
    # This variance (0.996) should be 1.0 (the risk target)
    
    port_var = weights.T @ cov @ weights
    assert np.isclose(port_var, 1.0)
    
    # Check that it's not the max variance
    risk_dist_max = np.array([1.0, 0.0])
    weights_max = pca_weights(cov, risk_dist_max, risk_target=1.0)
    port_var_max = weights_max.T @ cov @ weights_max
    assert np.isclose(port_var_max, 1.0)
    assert not np.allclose(weights, weights_max)