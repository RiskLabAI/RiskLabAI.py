"""
Tests for data/synthetic_data/drift_burst_hypothesis.py
"""

import numpy as np
import pytest
from RiskLabAI.data.synthetic_data.drift_burst_hypothesis import drift_volatility_burst

def test_drift_volatility_burst_shape():
    """Test the output shape."""
    drifts, vols = drift_volatility_burst(
        bubble_length=100,
        a_before=1, a_after=1,
        b_before=1, b_after=1,
        alpha=0.5, beta=0.5
    )
    assert drifts.shape == (100,)
    assert vols.shape == (100,)

def test_drift_volatility_burst_midpoint_handling():
    """Test that the midpoint explosion is handled."""
    # bubble_length=101 creates a perfect midpoint at index 50
    drifts, vols = drift_volatility_burst(
        bubble_length=101,
        a_before=1, a_after=1,
        b_before=1, b_after=1,
        alpha=0.5, beta=0.5,
        explosion_filter_width=0.01 # Small width
    )
    
    midpoint_index = 50
    
    # Check that steps[50] is 0.5
    steps = np.linspace(0, 1, 101)
    assert np.isclose(steps[midpoint_index], 0.5)
    
    # Drift at midpoint should be 0
    assert np.isclose(drifts[midpoint_index], 0.0)
    
    # Volatility at midpoint should be copied from [49]
    assert np.isclose(vols[midpoint_index], vols[midpoint_index - 1])
    
    # Check that values just before midpoint use the filter width
    # step[49] = 0.49
    # denominator = abs(0.49 - 0.5) = 0.01
    # expected vol = 1 / sqrt(0.01) = 10
    assert np.isclose(vols[midpoint_index - 1], 1.0 / np.sqrt(0.01))

def test_drift_volatility_burst_asymmetry():
    """Test that before/after parameters are used correctly."""
    drifts, vols = drift_volatility_burst(
        bubble_length=101,
        a_before=1, a_after=2,
        b_before=3, b_after=4,
        alpha=1.0, beta=1.0,
        explosion_filter_width=0.1
    )
    
    midpoint_index = 50
    
    # Check 'a' and 'b' values before midpoint
    # denominator at [49] = 0.1 (due to filter)
    # drift = a_before / denom = 1 / 0.1 = 10
    # vol = b_before / denom = 3 / 0.1 = 30
    assert np.isclose(drifts[midpoint_index - 1], 1.0 / 0.1)
    assert np.isclose(vols[midpoint_index - 1], 3.0 / 0.1)

    # Check 'a' and 'b' values after midpoint
    # denominator at [51] = 0.1 (due to filter)
    # drift = a_after / denom = 2 / 0.1 = 20
    # vol = b_after / denom = 4 / 0.1 = 40
    assert np.isclose(drifts[midpoint_index + 1], 2.0 / 0.1)
    assert np.isclose(vols[midpoint_index + 1], 4.0 / 0.1)