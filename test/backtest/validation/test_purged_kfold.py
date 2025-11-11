"""
Tests for PurgedKFold cross-validator.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from RiskLabAI.backtest.validation.purged_kfold import PurgedKFold

@pytest.fixture
def sample_data_with_times():
    """Fixture for sample data with a 'times' series."""
    n_samples = 100
    X = pd.DataFrame(
        {'feature1': np.arange(n_samples)}, 
        index=pd.date_range('2020-01-01', periods=n_samples, freq='B')
    )
    y = pd.Series(
        np.random.randint(0, 2, n_samples), 
        index=X.index
    )
    
    # 'times' Series: info starts at index time, ends 5 business days later
    # 5 business days = 7 calendar days
    times = pd.Series(
        X.index + pd.DateOffset(days=7), 
        index=X.index
    )
    return X, y, times

def test_purged_kfold_init(sample_data_with_times):
    """Test PurgedKFold initialization."""
    _, _, times = sample_data_with_times
    cv = PurgedKFold(n_splits=5, times=times, embargo=0.01)
    
    assert cv.n_splits == 5
    assert cv.embargo == 0.01
    assert cv.get_n_splits() == 5
    assert cv.is_multiple_datasets is False

def test_filtered_training_indices_with_embargo():
    """Test the static purging method directly."""
    # All data: 100 days, info span is 5 days
    all_times = pd.Series(
        pd.date_range('2020-01-06', periods=100, freq='D'),
        index=pd.date_range('2020-01-01', periods=100, freq='D')
    )
    
    # Test set: days 20-30 (iloc 20 to 29)
    test_times = all_times.iloc[20:30]
    
    # Test 1: No embargo
    train_times = PurgedKFold.filtered_training_indices_with_embargo(
        all_times, test_times, embargo_fraction=0
    )
    
    # Test set info range: ['2020-01-21', '2020-02-04']
    # Purge range: ['2020-01-16' (iloc 15) to '2020-02-04' (iloc 34)]
    # Purged: 15..34 (inclusive). 34 - 15 + 1 = 20 samples.
    # 100 - 20 = 80 samples.
    assert all_times.index[14] in train_times.index
    assert all_times.index[15] not in train_times.index
    assert all_times.index[34] not in train_times.index
    assert all_times.index[35] in train_times.index
    assert len(train_times) == 80

    # Test 2: With embargo (1% of 100 = 1 day/sample embargo)
    train_times_emb = PurgedKFold.filtered_training_indices_with_embargo(
        all_times, test_times, embargo_fraction=0.01
    )
    # Embargoed end timestamp: all_times.index[35] = '2020-02-05'
    # Purge range: ['2020-01-16' (iloc 15) to '2020-02-05' (iloc 35)]
    # Purged: 15..35 (inclusive). 35 - 15 + 1 = 21 samples.
    # 100 - 21 = 79 samples.
    assert all_times.index[35] not in train_times_emb.index
    assert all_times.index[36] in train_times_emb.index
    assert len(train_times_emb) == 79

def test_purged_kfold_split(sample_data_with_times):
    """Test the split method."""
    X, y, times = sample_data_with_times
    n_splits = 5
    cv = PurgedKFold(n_splits=n_splits, times=times, embargo=0.01)

    splits = list(cv.split(X, y))
    assert len(splits) == n_splits
    
    # --- Test first fold ---
    train_idx_0, test_idx_0 = splits[0]
    np.testing.assert_array_equal(test_idx_0, np.arange(0, 20))
    
    # Test range: start '2020-01-01', end '2020-02-04' (from iloc 19)
    # Embargo (1%*100=1 sample): end_iloc = 24. embargoed_iloc = 25.
    # Embargoed end timestamp: times.index[25] = '2020-02-05'
    # Purge range: ['2020-01-01', '2020-02-05']
    # cond1 purges [0..25]
    # cond2 purges [0..20] (iloc 20 end is '2020-02-05')
    # First sample kept is iloc[26] (start '2020-02-06')
    assert 25 not in train_idx_0
    assert 26 in train_idx_0
    assert np.all(train_idx_0 >= 26)

    # --- Test last fold ---
    train_idx_4, test_idx_4 = splits[4]
    np.testing.assert_array_equal(test_idx_4, np.arange(80, 100))
    
    # Test range: start '2020-04-27' (iloc 80), end '2020-05-26' (from iloc 99)
    # Embargo (1 sample) -> end_val '2020-05-26' is OOB, so embargoed_end is '2020-05-26'
    # Purge range: ['2020-04-27', '2020-05-26']
    # cond1 purges [80..99]
    # cond2 purges [71..99] (iloc 71 end is '2020-04-27')
    # First sample kept is iloc[70]
    assert 70 in train_idx_4
    assert 71 in train_idx_4  # 71 should be in the set
    assert 74 in train_idx_4  # This is the last valid index
    assert 75 not in train_idx_4 # This is the first purged index
    assert len(train_idx_4) == 75 # The correct length is 75



def test_get_train_indices_refactor(sample_data_with_times):
    """Test the _get_train_indices refactor."""
    X, y, times = sample_data_with_times
    cv = PurgedKFold(n_splits=5, times=times, embargo=0.01)
    
    test_indices = np.arange(80, 100)
    train_indices = cv._get_train_indices(test_indices, times, True)
    
    # Logic is identical to test_purged_kfold_split[4]
    assert isinstance(train_indices, np.ndarray)
    assert 70 in train_indices
    assert 71 in train_indices # 71 should be in the set
    assert 74 in train_indices # This is the last valid index
    assert 75 not in train_indices # This is the first purged index
    assert len(train_indices) == 75 # The correct length is 75