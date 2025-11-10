"""
Tests for backtest_synthetic_data.py
"""

import numpy as np
import pytest
from RiskLabAI.backtest.backtest_synthetic_data import synthetic_back_testing

def test_synthetic_back_testing_structure():
    """
    Test the output structure with minimal iterations.
    """
    pt_range = np.linspace(1, 2, 2)
    sl_range = np.linspace(1, 2, 2)
    
    results = synthetic_back_testing(
        forecast=10.0,
        half_life=10.0,
        sigma=0.1,
        n_iteration=5,
        maximum_holding_period=10,
        profit_taking_range=pt_range,
        stop_loss_range=sl_range,
        seed=10
    )
    
    # Expected number of results = len(pt_range) * len(sl_range) = 4
    assert len(results) == 4
    
    # Check the structure of the first result
    first_result = results[0]
    assert isinstance(first_result, tuple)
    assert len(first_result) == 5
    
    # Check the first tuple corresponds to the first (pt, sl) combo
    assert first_result[0] == 1.0  # profit_taking
    assert first_result[1] == 1.0  # stop_loss
    
    # Check types
    assert isinstance(first_result[2], float) # mean
    assert isinstance(first_result[3], float) # std
    assert isinstance(first_result[4], float) # sharpe

def test_synthetic_back_testing_logic():
    """
    Test the logic with deterministic parameters (no noise).
    """
    pt_range = np.array([0.5])
    sl_range = np.array([0.5])
    
    # forecast = 10, seed = 5, half_life = 1 -> rho = 0.5
    # P_1 = (1-0.5)*10 + 0.5*5 + 0*gauss = 5 + 2.5 = 7.5
    # gain = 7.5 - 5 = 2.5
    # This gain (2.5) is > profit_taking (0.5), so it stops.
    results = synthetic_back_testing(
        forecast=10.0,
        half_life=1.0,  # rho = 0.5
        sigma=0.0,      # No noise
        n_iteration=5,
        maximum_holding_period=10,
        profit_taking_range=pt_range,
        stop_loss_range=sl_range,
        seed=5
    )
    
    assert len(results) == 1
    mean_ret = results[0][2]
    std_ret = results[0][3]
    
    # All 5 iterations should yield the same gain of 2.5
    assert np.isclose(mean_ret, 2.5)
    assert np.isclose(std_ret, 0.0)
    assert np.isclose(results[0][4], 0.0) # Sharpe is 0 if std is 0