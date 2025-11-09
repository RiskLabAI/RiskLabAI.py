"""
Tests for orthogonal_features.py
"""

import pytest
import pandas as pd
import numpy as np
from .orthogonal_features import orthogonal_features

@pytest.fixture
def mock_features():
    """Mock correlated features."""
    N = 100
    X = pd.DataFrame(np.random.normal(0, 1, size=(N, 3)),
                     columns=['A', 'B', 'C'])
    # Make B and C highly correlated with A
    X['B'] = X['A'] + np.random.normal(0, 0.01, N)
    X['C'] = X['A'] + np.random.normal(0, 0.01, N)
    # Add an independent feature
    X['D'] = np.random.normal(0, 1, size=(N,))
    return X

def test_orthogonal_features(mock_features):
    """Test orthogonal feature generation."""
    X = mock_features
    
    ortho_X, eigen_df = orthogonal_features(X, variance_threshold=0.95)
    
    # Check shapes
    # With 3 highly correlated features + 1 independent, we expect 2 main PCs
    assert ortho_X.shape[1] == 2
    assert eigen_df.shape[0] == 2
    
    # Check column names
    assert 'PC_1' in ortho_X.columns
    assert 'PC_2' in ortho_X.columns
    
    # Check cumulative variance
    assert eigen_df.iloc[0]['CumulativeVariance'] < 0.95
    assert eigen_df.iloc[1]['CumulativeVariance'] >= 0.95
    
    # Check for orthogonality
    corr_matrix = ortho_X.corr()
    # Off-diagonal should be near 0
    assert np.isclose(corr_matrix.loc['PC_1', 'PC_2'], 0.0, atol=1e-10)