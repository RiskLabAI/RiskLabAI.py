"""
Tests for backtest_statistics.py
"""

import numpy as np
import pandas as pd
import pytest
from RiskLabAI.backtest.backtest_statistics import (
    bet_timing,
    calculate_holding_period,
    calculate_hhi,
    calculate_hhi_concentration,
    compute_drawdowns_time_under_water,
)

@pytest.fixture
def sample_positions():
    """Fixture for a sample position series."""
    dates = pd.to_datetime(
        [
            "2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04",
            "2020-01-05", "2020-01-06", "2020-01-07", "2020-01-08",
        ]
    )
    # [0, 1, 1, 0, -1, -1, 1, 0]
    pos = [0, 1, 1, 0, -1, -1, 1, 0]
    return pd.Series(pos, index=dates)

def test_bet_timing(sample_positions):
    """Test the bet_timing function."""
    # Bets should be at:
    # 2020-01-04 (1 -> 0)
    # 2020-01-05 (0 -> -1, no, this is not a flip *at* 0)
    # 2020-01-07 (-1 -> 1, sign flip)
    # 2020-01-08 (1 -> 0)
    
    # Rerunning logic:
    # zero_positions = ['01-01', '01-04', '01-08']
    # lagged_non_zero = ['01-02', '01-03', '01-05', '01-06', '01-07']
    # bets (intersection) = [] ... wait, '01-04' is not in lagged_non_zero
    #   lagged_non_zero = target_positions.shift(1) -> [nan, 0, 1, 1, 0, -1, -1, 1]
    #   lagged_non_zero (filtered) -> ['01-03', '01-04', '01-06', '01-07', '01-08']
    #   zero_positions = ['01-01', '01-04', '01-08']
    #   bets (intersection) = ['01-04', '01-08']
    #
    # sign_flips = [0, 1, 0, 0, 1, -1, 0]
    # sign_flips < 0 -> ['01-07']
    # bets (union) = ['01-04', '01-07', '01-08']
    # last day ('01-08') is already in bets.
    
    expected_dates = pd.to_datetime(["2020-01-04", "2020-01-07", "2020-01-08"])
    
    bet_times = bet_timing(sample_positions)
    pd.testing.assert_index_equal(bet_times, expected_dates)

def test_calculate_holding_period():
    """Test calculate_holding_period."""
    dates = pd.to_datetime(
        ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]
    )
    # [0, 1, 1, 0]
    pos = pd.Series([0, 1, 1, 0], index=dates)
    
    # t=1: pos=1, prev=0, diff=1. diff*prev=0. time_entry = (0*0 + 1*1)/1 = 1
    # t=2: pos=1, prev=1, diff=0. diff*prev=0. time_entry = (1*1 + 2*0)/1 = 1
    # t=3: pos=0, prev=1, diff=-1. diff*prev=-1. (close)
    #   hold_period = (dT=3-1=2, w=abs(-1)=1)
    
    df, mean_hold = calculate_holding_period(pos)
    
    assert np.isclose(mean_hold, 2.0)
    assert df.shape == (1, 2)
    assert np.isclose(df['dT'].iloc[0], 2.0)
    assert np.isclose(df['w'].iloc[0], 1.0)

def test_calculate_hhi():
    """Test the HHI calculation."""
    # Perfectly diversified
    returns_div = pd.Series([1, 1, 1, 1])
    # w = [0.25, 0.25, 0.25, 0.25], hhi = 4 * (0.25**2) = 0.25
    # n = 4, hhi_norm = (0.25 - 1/4) / (1 - 1/4) = 0
    assert np.isclose(calculate_hhi(returns_div), 0.0)

    # Perfectly concentrated
    returns_conc = pd.Series([10, 0, 0, 0])
    # w = [1, 0, 0, 0], hhi = 1
    # n = 4, hhi_norm = (1 - 1/4) / (1 - 1/4) = 1
    assert np.isclose(calculate_hhi(returns_conc), 1.0)

def test_compute_drawdowns_time_under_water():
    """Test drawdown and time under water calculation."""
    dates = pd.date_range("2020-01-01", periods=10)
    # HWM: [10, 10, 10, 10, 11, 11, 12, 12, 12, 12]
    pnl = pd.Series([10, 9, 8, 10, 11, 10, 12, 11, 10, 12], index=dates)

    # Group 1 (HWM=10):
    #   Start='01-01', Stop='01-04', HWM=10, Min=8
    # Group 2 (HWM=11):
    #   Start='01-05', Stop='01-06', HWM=11, Min=10
    # Group 3 (HWM=12):
    #   Start='01-07', Stop='01-10', HWM=12, Min=10
    
    dd, tuw = compute_drawdowns_time_under_water(pnl, dollars=True)
    
    assert len(dd) == 3
    assert len(tuw) == 3
    
    # Check drawdowns (dollars)
    assert np.isclose(dd.loc['2020-01-01'], 2.0) # 10 - 8
    assert np.isclose(dd.loc['2020-01-05'], 1.0) # 11 - 10
    assert np.isclose(dd.loc['2020-01-07'], 2.0) # 12 - 10
    
    # Check time under water (in years)
    days_in_year = 365.25
    assert np.isclose(tuw.loc['2020-01-01'], 3.0 / days_in_year) # 01-04 - 01-01
    assert np.isclose(tuw.loc['2020-01-05'], 1.0 / days_in_year) # 01-06 - 01-05
    assert np.isclose(tuw.loc['2020-01-07'], 3.0 / days_in_year) # 01-10 - 01-07