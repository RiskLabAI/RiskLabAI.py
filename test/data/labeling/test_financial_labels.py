"""
Tests for data/labeling/financial_labels.py
"""

import numpy as np
import pandas as pd
import pytest
from RiskLabAI.data.labeling import (
    calculate_t_value_linear_regression,
    find_trend_using_trend_scanning,
)

def test_calculate_t_value_linear_regression():
    """Test the t-value calculation."""
    # Perfect positive trend
    prices_pos = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    t_val_pos = calculate_t_value_linear_regression(prices_pos)
    assert t_val_pos > 0
    assert np.isinf(t_val_pos) # OLS stderr is 0

    # Perfect negative trend
    prices_neg = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0])
    t_val_neg = calculate_t_value_linear_regression(prices_neg)
    assert t_val_neg < 0
    assert np.isinf(t_val_neg)

    # Noisy trend
    prices_noisy = pd.Series([1.0, 2.1, 2.9, 4.0, 5.1])
    t_val_noisy = calculate_t_value_linear_regression(prices_noisy)
    assert t_val_noisy > 0
    assert np.isfinite(t_val_noisy)
    
    # No trend
    prices_flat = pd.Series([2.0, 2.0, 2.0, 2.0, 2.0])
    t_val_flat = calculate_t_value_linear_regression(prices_flat)
    assert np.isnan(t_val_flat) # StdErr is 0, slope is 0

def test_find_trend_using_trend_scanning():
    """Test the trend scanning function."""
    dates = pd.date_range("2020-01-01", periods=20)
    prices = pd.Series(
        [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, # Strong uptrend
            9, 8, 7, 6, 5, 4, 3, 2, 1, 0  # Strong downtrend
        ],
        index=dates,
        dtype=float
    )
    
    # Scan from 2020-01-01. Span [5, 10]
    # Window [0, 4] (len 5): 1..5 -> t = inf
    # Window [0, 9] (len 10): 1..10 -> t = inf
    # It will pick the last one, t_value at '2020-01-10'
    
    molecule = pd.to_datetime(["2020-01-01", "2020-01-10"])
    span = (5, 11) # span(5, 10)
    
    trends = find_trend_using_trend_scanning(molecule, prices, span)
    
    # Check event 1
    assert trends.loc["2020-01-01", "Trend"] == 1.0
    assert trends.loc["2020-01-01", "End Time"] == pd.to_datetime("2020-01-11")
    assert np.isinf(trends.loc["2020-01-01", "t-Value"])
    
    # Check event 2
    # Scan from 2020-01-10 (price 10)
    # Window [9, 13] (len 5): 10, 9, 8, 7, 6 -> t = -inf
    # Window [9, 18] (len 10): 10..2 -> t = -inf
    assert trends.loc["2020-01-10", "Trend"] == -1.0
    assert trends.loc["2020-01-10", "End Time"] == pd.to_datetime("2020-01-20")
    assert np.isinf(trends.loc["2020-01-10", "t-Value"])