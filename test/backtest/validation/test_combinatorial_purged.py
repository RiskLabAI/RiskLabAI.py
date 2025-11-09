"""
Tests for CombinatorialPurged cross-validator.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from math import comb

# Re-use the purged k-fold fixture
@pytest.fixture
def sample_data_with_times():
    """Fixture for sample data with a 'times' series."""
    n_samples = 120 # Use 120 for easier division
    X = pd.DataFrame(
        {'feature1': np.arange(n_samples)}, 
        index=pd.date_range('2020-01-01', periods=n_samples, freq='B')
    )
    y = pd.Series(
        np.random.randint(0, 2, n_samples), 
        index=X.index
    )
    times = pd.Series(
        X.index + pd.DateOffset(days=7), 
        index=X.index
    )
    return X, y, times

def test_combinatorial_purged_init(sample_data_with_times):
    """Test CPCV initialization."""
    _, _, times = sample_data_with_times
    n_splits = 6
    n_test_groups = 2
    cv = CombinatorialPurged(
        n_splits=n_splits, 
        n_test_groups=n_test_groups, 
        times=times
    )
    
    assert cv.n_splits == n_splits
    assert cv.n_test_groups == n_test_groups
    assert cv.get_n_splits() == comb(n_splits, n_test_groups) # 15

def test_path_locations():
    """Test the static _path_locations method."""
    n_splits = 4
    n_test_groups = 2
    # C(4, 2) = 6 combinations
    combinations_list = list(combinations(range(n_splits), n_test_groups))
    assert len(combinations_list) == 6
    
    locations = CombinatorialPurged._path_locations(n_splits, combinations_list)
    
    # n_splits=4, n_test_groups=2.
    # Total test sets = 4 * C(3, 1) = 12
    # Number of paths = C(4, 2) * 2 / 4 * 2 = 6? No...
    # Number of paths = n_splits - n_test_groups + 1 = 4 - 2 + 1 = 3
    assert len(locations) == 3 
    
    # Path 1 should have 4 segments
    assert len(locations[1]) == 4
    # Path 3 (last) should have 4 segments
    assert len(locations[3]) == 4
    
    # Check coordinates for Path 1
    # (group_idx, split_idx)
    # Path 1, Group 0: (0, 3) -> combo (1, 2)
    # Path 1, Group 1: (1, 0) -> combo (0, 2)
    # Path 1, Group 2: (2, 0) -> combo (0, 1)
    # ... this depends on the `combinations` order.
    
    # Let's check total segments
    total_segments = sum(len(loc) for loc in locations.values())
    assert total_segments == n_splits * comb(n_splits - 1, n_test_groups - 1)
    assert total_segments == 4 * comb(3, 1) # 12
    
def test_combinatorial_split(sample_data_with_times):
    """Test the split method."""
    X, y, times = sample_data_with_times
    n_splits = 6
    n_test_groups = 2
    cv = CombinatorialPurged(
        n_splits=n_splits, 
        n_test_groups=n_test_groups, 
        times=times,
        embargo=0.01
    )
    
    n_combinations = comb(n_splits, n_test_groups) # 15
    splits = list(cv.split(X, y))
    assert len(splits) == n_combinations
    
    # Test one split
    train_idx, test_idx = splits[0] # Combo (0, 1)
    
    # Test indices should be groups 0 and 1
    # n_samples = 120, n_splits = 6 -> 20 samples/group
    assert len(test_idx) == 40
    np.testing.assert_array_equal(test_idx, np.arange(40))
    
    # Train indices must be purged
    assert 39 not in train_idx
    assert np.all(train_idx >= 40)
    
def test_combinatorial_backtest_paths(sample_data_with_times):
    """Test the backtest_paths method."""
    X, y, times = sample_data_with_times
    n_splits = 6
    n_test_groups = 2
    cv = CombinatorialPurged(
        n_splits=n_splits, 
        n_test_groups=n_test_groups, 
        times=times
    )
    
    paths = cv.backtest_paths(X)
    
    # Num paths = n_splits - n_test_groups + 1 = 6 - 2 + 1 = 5
    assert len(paths) == 5
    assert 'Path 1' in paths
    assert 'Path 5' in paths
    
    # Each path should have n_splits = 6 segments
    assert len(paths['Path 1']) == n_splits
    
    # Check a segment
    # Path 1, Segment 0
    segment = paths['Path 1'][0]
    train_idx = segment['Train']
    test_idx = segment['Test']
    
    # Test set is just group 0
    np.testing.assert_array_equal(test_idx, np.arange(20))
    
    # Train set should be purged against its *combination*
    # (which combination this is depends on path logic)
    # But it must be a subset of 0..119
    assert train_idx.max() < 120
    
def test_combinatorial_backtest_predictions(sample_data_with_times):
    """Test the backtest_predictions method."""
    X, y, times = sample_data_with_times
    n_splits = 6
    n_test_groups = 2
    cv = CombinatorialPurged(
        n_splits=n_splits, 
        n_test_groups=n_test_groups, 
        times=times,
        embargo=0.01
    )
    model = LogisticRegression()
    
    preds_dict = cv.backtest_predictions(model, X, y, n_jobs=1)
    
    # 5 paths
    assert len(preds_dict) == 5
    assert 'Path 1' in preds_dict
    
    # Each path's predictions should cover the whole dataset
    preds1 = preds_dict['Path 1']
    assert len(preds1) == X.shape[0]
    
    preds5 = preds_dict['Path 5']
    assert len(preds5) == X.shape[0]
    
    # Predictions should be binary
    assert np.isin(preds1, [0, 1]).all()