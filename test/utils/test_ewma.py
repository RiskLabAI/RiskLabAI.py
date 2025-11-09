"""
Tests for utils/ewma.py
"""

import pytest
import numpy as np
import pandas as pd
from .ewma import ewma

def test_ewma_vs_pandas():
    """
    Test that the Numba ewma function matches pandas
    ewm(adjust=True).mean().
    """
    series_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0])
    window = 3
    
    # Our Numba implementation
    ewma_ours = ewma(series_np, window=window)
    
    # Pandas implementation
    ewma_pandas = (
        pd.Series(series_np)
        .ewm(span=window, adjust=True)
        .mean()
        .values
    )
    
    assert np.allclose(ewma_ours, ewma_pandas)

def test_ewma_single_value():
    """Test ewma with a single value."""
    series_np = np.array([5.0])
    ewma_ours = ewma(series_np, window=3)
    assert np.allclose(ewma_ours, [5.0])

def test_ewma_empty_value():
    """Test ewma with an empty array."""
    series_np = np.array([])
    ewma_ours = ewma(series_np, window=3)
    assert ewma_ours.shape == (0,)