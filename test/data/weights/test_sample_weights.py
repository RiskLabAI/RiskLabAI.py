"""
Tests for data/weights/sample_weights.py
"""

import numpy as np
import pandas as pd
import pytest
from RiskLabAI.data.weights.sample_weights import (
    expand_label_for_meta_labeling,
    calculate_average_uniqueness,
    sample_weight_absolute_return_meta_labeling,
    calculate_time_decay,
)

@pytest.fixture
def sample_events():
    """Fixture for sample events and price index."""
    close_index = pd.to_datetime(pd.date_range("2020-01-01", periods=10))
    
    # Event 1: [0, 4]
    # Event 2: [2, 6]
    # Event 3: [8, 9]
    timestamp = pd.Series(
        pd.to_datetime(["2020-01-05", "2020-01-07", "2020-01-10"]),
        index=pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-09"])
    )
    molecule = timestamp.index
    return close_index, timestamp, molecule

def test_expand_label_for_meta_labeling(sample_events):
    """Test the concurrency calculation."""
    close_index, timestamp, molecule = sample_events
    
    concurrency = expand_label_for_meta_labeling(
        close_index, timestamp, molecule
    )
    
    # Concurrency:
    # 01-01: 1
    # 01-02: 1
    # 01-03: 2
    # 01-04: 2
    # 01-05: 2
    # 01-06: 1
    # 01-07: 1
    # 01-08: 0
    # 01-09: 1
    # 01-10: 1
    expected_values = [1, 1, 2, 2, 2, 1, 1, 0, 1, 1]
    expected_index = pd.to_datetime(pd.date_range("2020-01-01", periods=10))
    
    pd.testing.assert_series_equal(
        concurrency, 
        pd.Series(expected_values, index=expected_index),
        check_dtype=False
    )

def test_calculate_average_uniqueness():
    """Test average uniqueness calculation."""
    # T=4, N=3
    # Event 0: [0, 2]
    # Event 1: [1, 3]
    # Event 2: [0, 1]
    idx_matrix = pd.DataFrame(
        [
            [1, 0, 1], # t=0, c=2
            [1, 1, 1], # t=1, c=3
            [1, 1, 0], # t=2, c=2
            [0, 1, 0]  # t=3, c=1
        ]
    )
    # Uniqueness = 
    #   [1/2, 0, 1/2]
    #   [1/3, 1/3, 1/3]
    #   [1/2, 1/2, 0]
    #   [0,   1,   0]
    
    # Avg Uniqueness (by column):
    # E0: (1/2 + 1/3 + 1/2) / 3 = (0.5 + 0.333 + 0.5) / 3 = 1.333 / 3 = 0.444
    # E1: (1/3 + 1/2 + 1) / 3 = (0.333 + 0.5 + 1) / 3 = 1.833 / 3 = 0.611
    # E2: (1/2 + 1/3) / 2 = (0.5 + 0.333) / 2 = 0.833 / 2 = 0.416
    
    avg_u = calculate_average_uniqueness(idx_matrix)
    
    assert np.isclose(avg_u[0], (0.5 + 1/3 + 0.5) / 3)
    assert np.isclose(avg_u[1], (1/3 + 0.5 + 1) / 3)
    assert np.isclose(avg_u[2], (0.5 + 1/3) / 2)

def test_sample_weight_absolute_return(sample_events):
    """Test sample weighting by absolute return."""
    close_index, timestamp, molecule = sample_events
    prices = pd.Series(
        [10, 11, 12, 13, 12, 11, 10, 11, 12, 13], index=close_index
    )
    
    weights = sample_weight_absolute_return_meta_labeling(
        timestamp, prices, molecule
    )
    
    assert weights.shape == (3,)
    assert np.isclose(weights.sum(), 3.0) # Normalized to N
    assert weights.loc['2020-01-01'] > 0
    assert weights.loc['2020-01-03'] > 0
    assert weights.loc['2020-01-09'] > 0

def test_calculate_time_decay():
    """Test time decay weighting."""
    weights = pd.Series(1.0, index=pd.date_range("2020-01-01", periods=10))
    
    # Test 1: No decay
    decayed_1 = calculate_time_decay(weights, clf_last_weight=1.0)
    assert np.allclose(decayed_1, 1.0)
    
    # Test 2: Linear decay to 0
    decayed_0 = calculate_time_decay(weights, clf_last_weight=0.0)
    # cumsum = [1, 2, ..., 10]
    # slope = (1-0) / 10 = 0.1
    # const = 1 - 0.1 * 10 = 0
    # new_weights = 0 + 0.1 * [1, 2, ..., 10] = [0.1, 0.2, ..., 1.0]
    expected_0 = np.arange(1, 11) * 0.1
    assert np.allclose(decayed_0, expected_0)
    
    # Test 3: Linear decay to 0.5
    decayed_05 = calculate_time_decay(weights, clf_last_weight=0.5)
    # slope = (1-0.5) / 10 = 0.05
    # const = 1 - 0.05 * 10 = 0.5
    # new_weights = 0.5 + 0.05 * [1, ..., 10] = [0.55, 0.6, ..., 1.0]
    expected_05 = 0.5 + 0.05 * np.arange(1, 11)
    assert np.allclose(decayed_05, expected_05)