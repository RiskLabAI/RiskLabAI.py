"""
Tests for optimization/hrp.py
"""

import pytest
import numpy as np
import pandas as pd
from .hrp import (
    inverse_variance_weights,
    cluster_variance,
    quasi_diagonal,
    hrp
)
import scipy.cluster.hierarchy as sch

@pytest.fixture
def mock_cov_matrix():
    """
    Generate a 4x4 covariance matrix with two blocks.
    Assets ['A', 'B'] are correlated.
    Assets ['C', 'D'] are correlated.
    Blocks are uncorrelated.
    """
    cov = np.array([
        [1.0, 0.8, 0.0, 0.0],
        [0.8, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.5],
        [0.0, 0.0, 0.5, 1.0]
    ])
    return pd.DataFrame(cov, columns=['A', 'B', 'C', 'D'], index=['A', 'B', 'C', 'D'])

def test_inverse_variance_weights(mock_cov_matrix):
    """Test inverse variance weights."""
    weights = inverse_variance_weights(mock_cov_matrix)
    # Diag = [1, 1, 1, 1]
    # Inv Diag = [1, 1, 1, 1]
    # Sum = 4
    # Weights = [0.25, 0.25, 0.25, 0.25]
    assert np.allclose(weights, [0.25, 0.25, 0.25, 0.25])

def test_cluster_variance(mock_cov_matrix):
    """Test cluster variance calculation."""
    # Cluster 0 = ['A', 'B']
    # Cov = [[1, 0.8], [0.8, 1]]
    # IVP weights = [0.5, 0.5]
    # Var = [0.5, 0.5] @ [[1, 0.8], [0.8, 1]] @ [0.5, 0.5]'
    #     = [0.5, 0.5] @ [0.9, 0.9]' = 0.45 + 0.45 = 0.9
    var_0 = cluster_variance(mock_cov_matrix, ['A', 'B'])
    assert np.isclose(var_0, 0.9)

    # Cluster 1 = ['C', 'D']
    # Cov = [[1, 0.5], [0.5, 1]]
    # IVP weights = [0.5, 0.5]
    # Var = [0.5, 0.5] @ [[1, 0.5], [0.5, 1]] @ [0.5, 0.5]'
    #     = [0.5, 0.5] @ [0.75, 0.75]' = 0.375 + 0.375 = 0.75
    var_1 = cluster_variance(mock_cov_matrix, ['C', 'D'])
    assert np.isclose(var_1, 0.75)

def test_hrp(mock_cov_matrix):
    """Test the full HRP algorithm."""
    cov = mock_cov_matrix.values
    corr = mock_cov_matrix.values # Since stds are 1
    
    weights = hrp(cov, corr)
    
    assert isinstance(weights, pd.Series)
    assert weights.shape == (4,)
    assert np.isclose(weights.sum(), 1.0)
    
    # Test allocation
    # Var(A,B) = 0.9, Var(C,D) = 0.75
    # Alpha = 1 - 0.9 / (0.9 + 0.75) = 1 - 0.545 = 0.4545
    # Weight(A,B) = 0.4545, Weight(C,D) = 0.5454
    # Bisection of (A,B): Var(A)=1, Var(B)=1. Alpha=0.5
    #   W(A) = 0.4545 * 0.5 = 0.227
    #   W(B) = 0.4545 * 0.5 = 0.227
    # Bisection of (C,D): Var(C)=1, Var(D)=1. Alpha=0.5
    #   W(C) = 0.5454 * 0.5 = 0.272
    #   W(D) = 0.5454 * 0.5 = 0.272
    
    # Note: The 'sorted_items' in the test fixture might be ['A', 'B', 'C', 'D']
    # The code's `hrp` function will sort them by index name at the end.
    expected_weights = pd.Series(
        [0.22727, 0.22727, 0.27272, 0.27272],
        index=['A', 'B', 'C', 'D']
    )
    pd.testing.assert_series_equal(
        weights, expected_weights, atol=1e-5, check_names=False
    )