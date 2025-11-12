"""
Tests for data/labeling/labeling.py
"""

import numpy as np
import pandas as pd
import pytest
from RiskLabAI.data.labeling import (
    symmetric_cusum_filter,
    cusum_filter_events_dynamic_threshold,
    daily_volatility_with_log_returns,
    vertical_barrier,
    meta_events,
    meta_labeling,
)

@pytest.fixture
def price_series():
    """Fixture for a predictable price series."""
    dates = pd.to_datetime(
        pd.date_range("2020-01-01", periods=20, freq="D")
    )
    prices = [
        10, 11, 12, 13, 14, 15,  # Event 1
        14, 13, 12, 11, 10,  # Event 2
        11, 12, 11, 12, 11,  # Noise
        12, 13, 14, 15,      # Event 3
    ]
    return pd.Series(prices, index=dates, dtype=float)

def test_symmetric_cusum_filter(price_series):
    """Test the fixed-threshold CUSUM filter."""
    # Threshold of 3
    # 10->15 (diff=5) > 3. Event at 2020-01-06
    # 15->10 (diff=-5) < -3. Event at 2020-01-11
    # 11->12->11... no event
    # 11->15 (diff=4) > 3. Event at 2020-01-20
    events = symmetric_cusum_filter(price_series, threshold=3.0)
    expected_dates = pd.to_datetime(["2020-01-05", "2020-01-10", "2020-01-19"])

    pd.testing.assert_index_equal(events, expected_dates)

def test_cusum_filter_dynamic_threshold(price_series):
    """Test the dynamic-threshold CUSUM filter."""
    # Threshold = 2.0 everywhere
    thresholds = pd.Series(2.0, index=price_series.index)
    
    # 10->12 (diff=2), 12->13 (diff=1), s_pos=3 > 2. Event at 2020-01-03
    # 13->14 (diff=1), 14->15 (diff=1), s_pos=2
    # 15->14 (diff=-1), 14->13 (diff=-1), s_neg=-2
    # 13->12 (diff=-1), s_neg=-3 < -2. Event at 2020-01-09
    
    events = cusum_filter_events_dynamic_threshold(price_series, thresholds)
    # Note: CUSUM logic is slightly different, it triggers *after*
    # 10->11 (1), 11->12 (2), 12->13 (3) > 2. Event at 2020-01-04
    # 13->14 (1), 14->15 (2)
    # 15->14 (-1), 14->13 (-2), 13->12 (-3) < -2. Event at 2020-01-09
    
    expected_dates = pd.to_datetime(
        ["2020-01-04", "2020-01-09", "2020-01-18"]
    )
    pd.testing.assert_index_equal(events, expected_dates)

def test_daily_volatility(price_series):
    """Test daily volatility calculation."""
    vol = daily_volatility_with_log_returns(price_series, span=5)
    assert isinstance(vol, pd.Series)
    assert not vol.empty
    assert vol.name == "std"

def test_vertical_barrier(price_series):
    """Test the vertical barrier function."""
    events = pd.to_datetime(["2020-01-02", "2020-01-10"])
    # 2020-01-02 + 5 days = 2020-01-07
    # 2020-01-10 + 5 days = 2020-01-15
    barriers = vertical_barrier(price_series, events, number_days=5)
    
    expected_index = pd.to_datetime(["2020-01-02", "2020-01-10"])
    expected_values = pd.to_datetime(["2020-01-07", "2020-01-15"])
    
    pd.testing.assert_index_equal(barriers.index, expected_index)
    assert np.all(barriers.values == expected_values)

def test_meta_events_and_labeling(price_series):
    """Test the triple-barrier and meta-labeling functions."""
    time_events = pd.to_datetime(["2020-01-02", "2020-01-08"])
    volatility = pd.Series(0.01, index=price_series.index)
    ptsl = [2.0, 2.0] # 2 * 0.01 = 0.02
    return_min = 0.0
    num_threads = 1
    
    # Event 1: Start 2020-01-02 (price 11)
    #   Price path: 12, 13, 14, 15
    #   Log returns: log(12/11)=0.087, log(13/11)=0.167, ...
    #   0.167 > 0.02 (pt). Hit at 2020-01-04.
    
    # Event 2: Start 2020-01-08 (price 13)
    #   Price path: 12, 11, 10
    #   Log returns: log(12/13)=-0.08, log(11/13)=-0.167
    #   -0.167 < -0.02 (sl). Hit at 2020-01-10.
    
    events = meta_events(
        price_series, time_events, ptsl, volatility, 
        return_min, num_threads
    )
    
    expected_end_times = pd.to_datetime(["2020-01-03", "2020-01-09"])
    assert np.all(events["End Time"] == expected_end_times)
    
    # Test meta-labeling (long only)
    labels = meta_labeling(events, price_series)
    
    # Event 1: 13 / 11 - 1 = 0.18 > 0. Label = 1
    # Event 2: 11 / 13 - 1 = -0.15 < 0. Label = -1
    assert np.isclose(labels.loc["2020-01-02", "Label"], 1.0)
    assert np.isclose(labels.loc["2020-01-08", "Label"], -1.0)
    
    # Test meta-labeling (with side)
    events["Side"] = pd.Series([1, 1], index=time_events)
    labels_meta = meta_labeling(events, price_series)
    
    # Event 1: Return > 0. Label = 1
    # Event 2: Return < 0. Label = 0
    assert np.isclose(labels_meta.loc["2020-01-02", "Label"], 1.0)
    assert np.isclose(labels_meta.loc["2020-01-08", "Label"], 0.0)