"""
Tests for data/structures/time_bars.py
"""
import numpy as np
import pandas as pd
import pytest
from RiskLabAI.data.structures.time_bars import TimeBars

@pytest.fixture
def sample_tick_data_for_time():
    """Fixture for sample tick data with fine timestamps."""
    return [
        (pd.to_datetime('2020-01-01 10:00:00.100'), 100, 10),
        (pd.to_datetime('2020-01-01 10:00:00.500'), 101, 5),
        (pd.to_datetime('2020-01-01 10:00:01.200'), 100, 20), # Bar 1 ends, Bar 2 starts
        (pd.to_datetime('2020-01-01 10:00:01.800'), 101, 10),
        (pd.to_datetime('2020-01-01 10:00:02.100'), 102, 10), # Bar 2 ends, Bar 3 starts
        (pd.to_datetime('2020-01-01 10:00:02.500'), 103, 10),
    ]

def test_time_bars(sample_tick_data_for_time):
    """Test time bars with 1-second resolution."""
    # 1-second bars
    bars = TimeBars(resolution_type='S', resolution_units=1)
    bar_list = bars.construct_bars_from_data(sample_tick_data_for_time)
    
    assert len(bar_list) == 2
    
    # Bar 1 (from 10:00:00.000 to 10:00:01.000)
    # Ticks: 0, 1
    # End time: 10:00:01.000
    # Open: 100, High: 101, Low: 100, Close: 101
    assert bar_list[0][0] == pd.to_datetime('2020-01-01 10:00:01') # end time
    assert bar_list[0][2] == 100 # open
    assert bar_list[0][3] == 101 # high
    assert bar_list[0][4] == 100 # low
    assert bar_list[0][5] == 101 # close (from tick 1)
    assert bar_list[0][9] == 2   # ticks
    
    # Bar 2 (from 10:00:01.000 to 10:00:02.000)
    # Ticks: 2, 3
    # End time: 10:00:02.000
    # Open: 100, High: 101, Low: 100, Close: 101
    assert bar_list[1][0] == pd.to_datetime('2020-01-01 10:00:02')
    assert bar_list[1][2] == 100 # open
    assert bar_list[1][3] == 101 # high
    assert bar_list[1][4] == 100 # low
    assert bar_list[1][5] == 101 # close (from tick 3)
    assert bar_list[1][9] == 2   # ticks