"""
Tests for data/differentiation/differentiation.py
"""

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import adfuller
from RiskLabAI.data.differentiation import (
    calculate_weights_std,
    calculate_weights_ffd,
    fractional_difference_std,
    fractional_difference_fixed,
    fractional_difference_fixed_single,
    fractionally_differentiated_log_price,
)

@pytest.fixture
def sample_series():
    """A simple linear series."""
    return pd.Series(np.arange(1, 21, dtype=float), name="close")

@pytest.fixture
def random_walk_series():
    """A non-stationary random walk."""
    rng = np.random.default_rng(42)
    log_price = np.log(100 + rng.normal(0, 1, 1000).cumsum())
    return pd.Series(log_price, name="close")

def test_calculate_weights_std():
    """Test standard weights calculation."""
    # d=0, w=[1]
    w0 = calculate_weights_std(degree=0, size=5)
    assert np.allclose(w0, [[0], [0], [0], [0], [1]])

    # d=1, w=[-1, 1]
    w1 = calculate_weights_std(degree=1, size=5)
    # [w4, w3, w2, w1, w0] = [0, 0, 0, -1, 1]
    assert np.allclose(w1, [[0], [0], [0], [-1], [1]])

def test_calculate_weights_ffd():
    """Test fixed-width weights calculation."""
    # d=0, w=[1]
    w0 = calculate_weights_ffd(degree=0, threshold=1e-5)
    assert np.allclose(w0, [[1.0]])

    # d=1, w=[-1, 1]
    w1 = calculate_weights_ffd(degree=1, threshold=1e-5)
    assert np.allclose(w1, [[-1.0], [1.0]])

def test_fractional_difference_std(sample_series):
    """Test standard differentiation."""
    df = sample_series.to_frame()
    
    # d=1 should be equivalent to .diff(1)
    diff_std = fractional_difference_std(df, degree=1.0, threshold=0.01)
    
    # Standard .diff()
    diff_pd = df.diff(1).dropna()
    
    # Compare common indices
    common_idx = diff_std.index.intersection(diff_pd.index)
    assert np.allclose(diff_std.loc[common_idx, 'close'], 
                       diff_pd.loc[common_idx, 'close'])

def test_fractional_difference_fixed(sample_series):
    """Test fixed-width differentiation."""
    df = sample_series.to_frame()
    
    # d=1 should be equivalent to .diff(1)
    diff_ffd = fractional_difference_fixed(df, degree=1.0, threshold=1e-5)
    
    # Standard .diff()
    diff_pd = df.diff(1).dropna()
    
    # Compare common indices
    common_idx = diff_ffd.index.intersection(diff_pd.index)
    assert np.allclose(diff_ffd.loc[common_idx, 'close'], 
                       diff_pd.loc[common_idx, 'close'])

def test_fractional_difference_fixed_single(sample_series):
    """Test fixed-width differentiation on a single Series."""
    # d=1
    diff_ffd = fractional_difference_fixed_single(
        sample_series, degree=1.0, threshold=1e-5
    )
    diff_pd = sample_series.diff(1).dropna()
    
    common_idx = diff_ffd.index.intersection(diff_pd.index)
    assert np.allclose(diff_ffd.loc[common_idx], 
                       diff_pd.loc[common_idx])
    
    # d=0.5
    diff_d05 = fractional_difference_fixed_single(
    sample_series, degree=0.5, threshold=0.01
    )
    assert not diff_d05.empty
    assert diff_d05.iloc[0] > 1.0 # Should be > diff(1)

def test_fractionally_differentiated_log_price(random_walk_series):
    """Test the minimum 'd' finding function."""
    # Original series should not be stationary
    adf_orig = adfuller(random_walk_series.dropna(), maxlag=1, regression='c')
    assert adf_orig[1] > 0.05 # p-value > 0.05 (not stationary)

    # Differentiated series should be stationary
    differentiated_series = fractionally_differentiated_log_price(
        random_walk_series, p_value_threshold=0.05
    )
    adf_diff = adfuller(differentiated_series.dropna(), maxlag=1, regression='c')
    assert adf_diff[1] < 0.05 # p-value < 0.05 (stationary)