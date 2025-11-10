"""
Tests for generate_synthetic_data.py
"""

import pytest
import pandas as pd
from RiskLabAI.features.feature_importance.generate_synthetic_data import get_test_dataset

def test_get_test_dataset():
    """Test the synthetic data generation."""
    n_features = 50
    n_informative = 10
    n_redundant = 20
    n_samples = 100
    
    X, y = get_test_dataset(
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_samples=n_samples,
        random_state=42,
        sigma_std=0.1
    )

    # Check shapes
    assert X.shape == (n_samples, n_features)
    assert y.shape == (n_samples,)
    
    # Check column names
    assert len([col for col in X.columns if col.startswith('I_')]) == n_informative
    assert len([col for col in X.columns if col.startswith('R_')]) == n_redundant
    assert len([col for col in X.columns if col.startswith('N_')]) == n_features - n_informative - n_redundant
    
    # Check for determinism
    X2, y2 = get_test_dataset(
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_samples=n_samples,
        random_state=42,
        sigma_std=0.1
    )
    pd.testing.assert_frame_equal(X, X2)
    pd.testing.assert_series_equal(y, y2)