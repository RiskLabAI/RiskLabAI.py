"""
Tests for data/structures/run_bars.py
"""
import numpy as np
import pandas as pd
import pytest
from RiskLabAI.data.structures.run_bars import (
    FixedRunBars, ExpectedRunBars
)

@pytest.fixture
def sample_tick_data_for_run():
    """
    Fixture for sample tick data.
    Prices: 100, 101, 102, 103, 102, 101, 100
    Tick Rules: 0, 1, 1, 1, -1, -1, -1
    Imbalance (tick): 0, 1, 1, 1, -1, -1, -1
    Cumul. Buy: 0, 1, 2, 3
    Cumul. Sell: 0, 0, 0, 0, 1, 2, 3
    """
    return [
        (pd.to_datetime('2020-01-01 10:00:00'), 100, 10),
        (pd.to_datetime('2020-01-01 10:00:01'), 101, 10),
        (pd.to_datetime('2020-01-01 10:00:02'), 102, 10),
        (pd.to_datetime('2020-01-01 10:00:03'), 103, 10), # Buy run bar
        (pd.to_datetime('2020-01-01 10:00:04'), 102, 10),
        (pd.to_datetime('2020-01-01 10:00:05'), 101, 10),
        (pd.to_datetime('2020-01-01 10:00:06'), 100, 10), # Sell run bar
    ]

def test_fixed_run_bars(sample_tick_data_for_run):
    """Test FixedRunBars."""
    bars = FixedRunBars(
        bar_type='tick_run',
        initial_estimate_of_expected_n_ticks_in_bar=3, # E[T] = 3
        window_size_for_expected_imbalance_estimation=10, # E[theta] window
        analyse_thresholds=False
    )
    
    # Warm-up (Tick 0, 1):
    # E[T]=3, E[P_buy]=0.5 (initial guess), E[theta_buy]=0.5, E[theta_sell]=nan
    # Ticks 0, 1, 2:
    # E[P_buy]=2/3, E[theta_buy]=ewma([1,1])=1, E[theta_sell]=nan
    
    # Tick 3 (price 103): b=1, buy_theta=3, sell_theta=0
    # E[T]=3, E[P_buy]=3/4=0.75, E[theta_buy]=1, E[theta_sell]=nan
    # buy_thresh = 0.75 * 1 = 0.75
    # sell_thresh = 0.25 * nan = nan
    # threshold = E[T] * E[P_buy] * E[theta_buy] = 3 * 0.75 = 2.25
    # max_theta (3) >= 2.25. Bar 1 constructed.
    # Reset: buy_theta=0, sell_theta=0
    
    # Tick 6 (price 100): b=-1, buy_theta=0, sell_theta=3
    # E[T]=3 (fixed)
    # E[P_buy] = ewma([0.75]) = 0.75
    # E[theta_buy]=ewma([1,1,1])=1, E[theta_sell]=ewma([1,1,1])=1
    # buy_thresh = 0.75 * 1 = 0.75
    # sell_thresh = (1-0.75) * 1 = 0.25
    # threshold = E[T] * max(0.75, 0.25) = 3 * 0.75 = 2.25
    # max_theta (3) >= 2.25. Bar 2 constructed.
    
    bar_list = bars.construct_bars_from_data(sample_tick_data_for_run)
    
    assert len(bar_list) == 2
    assert bar_list[0][9] == 4 # Ticks in bar 1 (0, 1, 2, 3)
    assert bar_list[1][9] == 3 # Ticks in bar 2 (4, 5, 6)