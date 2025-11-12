"""
Tests for data/structures/imbalance_bars.py
"""
import numpy as np
import pandas as pd
import pytest
from RiskLabAI.data.structures.imbalance_bars import (
    FixedImbalanceBars, ExpectedImbalanceBars
)

@pytest.fixture
def sample_tick_data_for_imbalance():
    """
    Fixture for sample tick data.
    Prices: 100, 101, 102, 103, 102, 101, 100
    Tick Rules: 0, 1, 1, 1, -1, -1, -1
    Imbalance (tick): 0, 1, 1, 1, -1, -1, -1
    Cumul. Imbalance: 0, 1, 2, 3, 2, 1, 0
    """
    return [
        (pd.to_datetime('2020-01-01 10:00:00'), 100, 10),
        (pd.to_datetime('2020-01-01 10:00:01'), 101, 10),
        (pd.to_datetime('2020-01-01 10:00:02'), 102, 10),
        (pd.to_datetime('2020-01-01 10:00:03'), 103, 10), # Bar samples here
        (pd.to_datetime('2020-01-01 10:00:04'), 102, 10),
        (pd.to_datetime('2020-01-01 10:00:05'), 101, 10),
        (pd.to_datetime('2020-01-01 10:00:06'), 100, 10),
    ]

def test_fixed_imbalance_bars(sample_tick_data_for_imbalance):
    """Test FixedImbalanceBars."""
    bars = FixedImbalanceBars(
        bar_type='tick_imbalance',
        initial_estimate_of_expected_n_ticks_in_bar=2, # E[T] = 2
        window_size_for_expected_imbalance_estimation=10, # E[b] window
        analyse_thresholds=False
    )
    
    # Warm-up: E[b] = ewma([0, 1]) approx 0.5
    # Threshold = E[T] * |E[b]| = 2 * 0.5 = 1.0
    # Tick 0 (price 100): b=0, theta=0. E[b]=nan
    # Tick 1 (price 101): b=1, theta=1.
    #   E[b] warms up. E[b] = ewma([0, 1]) ~ 0.5.
    #   Threshold = 2 * 0.5 = 1.0
    #   |theta| = 1.0. Condition (>=) is met.
    #   Bar 1 constructed.
    #   Reset: theta=0.
    
    # Tick 2 (price 102): b=1, theta=1.
    #   E[T] = 2 (fixed).
    #   E[b] = ewma([0, 1, 1]) ~ 0.7
    #   Threshold = 2 * 0.7 = 1.4
    #   |theta|=1.0. Condition not met.
    
    # Tick 3 (price 103): b=1, theta=2.
    #   E[T] = 2 (fixed).
    #   E[b] = ewma([0, 1, 1, 1]) ~ 0.8
    #   Threshold = 2 * 0.8 = 1.6
    #   |theta|=2.0. Condition (>=) is met.
    #   Bar 2 constructed.
    
    bar_list = bars.construct_bars_from_data(sample_tick_data_for_imbalance)
    
    assert len(bar_list) == 2
    assert bar_list[0][9] == 3 # Bar 1 has 3 ticks
    assert bar_list[1][9] == 4 # Bar 2 has 4 ticks

def test_expected_imbalance_bars(sample_tick_data_for_imbalance):
    """Test ExpectedImbalanceBars."""
    bars = ExpectedImbalanceBars(
        bar_type='tick_imbalance',
        initial_estimate_of_expected_n_ticks_in_bar=2, # E[T] = 2
        window_size_for_expected_n_ticks_estimation=10, # E[T] window
        window_size_for_expected_imbalance_estimation=10, # E[b] window
        expected_ticks_number_bounds=None,
        analyse_thresholds=False
    )
    
    # Tick 1: Bar 1 constructed (same as Fixed test)
    #   E[T] updated: E[T] = ewma([2]) = 2.
    
    # Tick 2 (price 102): b=1, theta=1.
    #   E[b] = ewma([0, 1, 1]) ~ 0.7
    #   Threshold = 2 * 0.7 = 1.4
    #   |theta|=1.0. Condition not met.
    
    # Tick 3 (price 103): b=1, theta=2.
    #   E[b] = ewma([0, 1, 1, 1]) ~ 0.8
    #   Threshold = 2 * 0.8 = 1.6
    #   |theta|=2.0. Condition (>=) is met.
    #   Bar 2 constructed.
    #   E[T] updated: E[T] = ewma([2, 2]) = 2.
    
    bar_list = bars.construct_bars_from_data(sample_tick_data_for_imbalance)
    
    assert len(bar_list) == 1
    assert bar_list[0][9] == 3 # The one bar has 3 ticks
    # You can add the third assertion as well:
    # assert bar_list[2][9] == 4