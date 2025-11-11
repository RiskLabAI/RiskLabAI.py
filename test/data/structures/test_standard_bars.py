"""
Tests for data/structures/standard_bars.py
"""
import numpy as np
import pandas as pd
import pytest
from RiskLabAI.data.structures.standard_bars import StandardBars
from RiskLabAI.utils.constants import *

@pytest.fixture
def sample_tick_data():
    """Fixture for sample tick data."""
    # (date_time, price, volume)
    return [
        (pd.to_datetime('2020-01-01 10:00:00'), 100, 10), # T=1, V=10, D=1000
        (pd.to_datetime('2020-01-01 10:00:01'), 101, 5),  # T=2, V=15, D=1505
        (pd.to_datetime('2020-01-01 10:00:02'), 100, 20), # T=3, V=35, D=3505 (Bar 1)
        (pd.to_datetime('2020-01-01 10:00:03'), 101, 10), # T=1, V=10, D=1010
        (pd.to_datetime('2020-01-01 10:00:04'), 102, 10), # T=2, V=20, D=2030
        (pd.to_datetime('2020-01-01 10:00:05'), 103, 10), # T=3, V=30, D=3060
        (pd.to_datetime('2020-01-01 10:00:06'), 102, 5),  # T=4, V=35, D=3570 (Bar 2)
    ]

def test_tick_bars(sample_tick_data):
    """Test standard tick bars."""
    bars = StandardBars(bar_type=CUMULATIVE_TICKS, threshold=3)
    bar_list = bars.construct_bars_from_data(sample_tick_data)
    
    assert len(bar_list) == 2
    
    # Check Bar 1
    # [dt, idx, open, high, low, close, vol, buy_vol, sell_vol, ticks, dollar, thresh]
    assert bar_list[0][0] == pd.to_datetime('2020-01-01 10:00:02') # end time
    assert bar_list[0][2] == 100 # open
    assert bar_list[0][3] == 101 # high
    assert bar_list[0][4] == 100 # low
    assert bar_list[0][5] == 100 # close
    assert bar_list[0][9] == 3   # ticks
    
    # Check Bar 2
    assert bar_list[1][0] == pd.to_datetime('2020-01-01 10:00:05')
    assert bar_list[1][2] == 101 # open
    assert bar_list[1][3] == 103 # high
    assert bar_list[1][4] == 101 # low
    assert bar_list[1][5] == 103 # close
    assert bar_list[1][9] == 3   # ticks



def test_volume_bars(sample_tick_data):
    """Test standard volume bars."""
    bars = StandardBars(bar_type=CUMULATIVE_VOLUME, threshold=35)
    bar_list = bars.construct_bars_from_data(sample_tick_data)
    
    assert len(bar_list) == 2
    assert bar_list[0][6] == 35 # cumulative_volume
    assert bar_list[1][6] == 35 # cumulative_volume

def test_dollar_bars(sample_tick_data):
    """Test standard dollar bars."""
    bars = StandardBars(bar_type=CUMULATIVE_DOLLAR, threshold=3500)
    bar_list = bars.construct_bars_from_data(sample_tick_data)
    
    assert len(bar_list) == 2
    assert bar_list[0][10] == 3505 # cumulative_dollar
    assert bar_list[1][10] == 3570 # cumulative_dollar