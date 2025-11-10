"""
Tests for utils/momentum_mean_reverting_strategy_sides.py
"""

import pytest
import pandas as pd
import numpy as np
from RiskLabAI.utils.momentum_mean_reverting_strategy_sides import determine_strategy_side

@pytest.fixture
def price_series():
    """A simple price series."""
    return pd.Series([100, 101, 102, 103, 104, 105, 104, 103, 102, 101])

def test_momentum_strategy(price_series):
    """Test momentum (mean_reversion=False)."""
    # fast=2, slow=5
    # fast_ma = [100, 100.5, 101.5, 102.5, 103.5, 104.5, 104.5, 103.5, 102.5, 101.5]
    # slow_ma = [100, 100.5, 101, 102, 103, 104, 103.6, 103, 102.4, 102]
    # fast > slow? [F, F, T, T, T, T, T, T, T, F] (ignoring min_periods=1)
    # Using min_periods=1...
    # fast = [100, 100.5, 101.5, 102.5, 103.5, 104.5, 104.5, 103.5, 102.5, 101.5]
    # slow = [100, 100.5, 101, 101.5, 102, 103, 103.6, 103.8, 103.6, 103]
    # fast >= slow: [T, T, T, T, T, T, T, F, F, F]
    # signal: [1, 1, 1, 1, 1, 1, 1, -1, -1, -1]
    
    sides = determine_strategy_side(
        price_series, fast_window=2, slow_window=5, mean_reversion=False
    )
    expected = pd.Series([1, 1, 1, 1, 1, 1, 1, -1, -1, -1])
    pd.testing.assert_series_equal(sides, expected)

def test_mean_reversion_strategy(price_series):
    """Test mean reversion (mean_reversion=True)."""
    # Should be the inverse of the momentum test
    sides = determine_strategy_side(
        price_series, fast_window=2, slow_window=5, mean_reversion=True
    )
    expected = pd.Series([-1, -1, -1, -1, -1, -1, -1, 1, 1, 1])
    pd.testing.assert_series_equal(sides, expected)

def test_exponential_ma(price_series):
    """Test that exponential=True runs."""
    sides = determine_strategy_side(
        price_series, fast_window=2, slow_window=5, exponential=True
    )
    # Different values from SMA, but should still have 1s and -1s
    assert sides.shape == (10,)
    assert sides.isin([1, -1]).all()

def test_window_error(price_series):
    """Test that fast_window >= slow_window raises an error."""
    with pytest.raises(ValueError, match="fast_window must be smaller"):
        determine_strategy_side(price_series, fast_window=5, slow_window=2)
    
    with pytest.raises(ValueError, match="fast_window must be smaller"):
        determine_strategy_side(price_series, fast_window=5, slow_window=5)