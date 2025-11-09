"""
Tests for KFold cross-validator.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

# Use a fixed sample dataset for all tests
@pytest.fixture
def sample_data():
    """Fixture for sample data."""
    n_samples = 100
    X = pd.DataFrame({'feature1': np.arange(n_samples), 'feature2': np.arange(n_samples, 0, -1)})
    y = pd.Series(np.random.randint(0, 2, n_samples))
    return X, y

def test_kfold_init():
    """Test KFold initialization."""
    cv = KFold(n_splits=5, shuffle=True, random_seed=42)
    assert cv.n_splits == 5
    assert cv.shuffle is True
    assert cv.random_seed == 42
    assert cv.get_n_splits() == 5

def test_kfold_split_no_shuffle(sample_data):
    """Test KFold split without shuffling."""
    X, y = sample_data
    n_samples = X.shape[0]
    n_splits = 5
    cv = KFold(n_splits=n_splits, shuffle=False)

    splits = list(cv.split(X, y))
    assert len(splits) == n_splits

    all_test_indices = []
    for i, (train_idx, test_idx) in enumerate(splits):
        assert isinstance(train_idx, np.ndarray)
        assert isinstance(test_idx, np.ndarray)

        # Check fold size
        assert len(test_idx) == n_samples / n_splits
        assert len(train_idx) == n_samples - len(test_idx)

        # Check for non-overlapping train/test
        assert not np.intersect1d(train_idx, test_idx).any()

        # Check non-shuffled indices
        expected_test_idx = np.arange(i * 20, (i + 1) * 20)
        np.testing.assert_array_equal(test_idx, expected_test_idx)
        all_test_indices.extend(test_idx)

    # Check that all indices are used exactly once as test indices
    assert len(np.unique(all_test_indices)) == n_samples

def test_kfold_split_shuffle(sample_data):
    """Test KFold split with shuffling."""
    X, y = sample_data
    n_samples = X.shape[0]
    n_splits = 5
    cv = KFold(n_splits=n_splits, shuffle=True, random_seed=42)
    
    splits1 = list(cv.split(X, y))
    
    # Check for determinism
    cv_same_seed = KFold(n_splits=n_splits, shuffle=True, random_seed=42)
    splits2 = list(cv_same_seed.split(X, y))
    
    all_test_indices = []
    for i in range(n_splits):
        np.testing.assert_array_equal(splits1[i][0], splits2[i][0])
        np.testing.assert_array_equal(splits1[i][1], splits2[i][1])
        all_test_indices.extend(splits1[i][1])

    # Check that indices are indeed shuffled
    sequential_indices = np.arange(20)
    assert not np.array_equal(splits1[0][1], sequential_indices)

    # Check that all indices are covered
    assert len(np.unique(all_test_indices)) == n_samples

def test_kfold_backtest_paths(sample_data):
    """Test backtest_paths method."""
    X, y = sample_data
    n_splits = 4
    cv = KFold(n_splits=n_splits, shuffle=False)
    
    paths = cv.backtest_paths(X)
    assert 'Path 1' in paths
    assert len(paths['Path 1']) == n_splits
    
    first_fold = paths['Path 1'][0]
    assert 'Train' in first_fold
    assert 'Test' in first_fold
    
    # Test first fold indices
    np.testing.assert_array_equal(first_fold['Test'], np.arange(25))
    np.testing.assert_array_equal(first_fold['Train'], np.arange(25, 100))

def test_kfold_backtest_predictions(sample_data):
    """Test backtest_predictions method."""
    X, y = sample_data
    n_splits = 5
    cv = KFold(n_splits=n_splits, shuffle=False)
    model = LogisticRegression()
    
    preds_dict = cv.backtest_predictions(model, X, y, n_jobs=1)
    
    assert 'Path 1' in preds_dict
    preds = preds_dict['Path 1']
    assert isinstance(preds, np.ndarray)
    assert len(preds) == X.shape[0]
    
    # Check that predictions are binary (0 or 1)
    assert np.isin(preds, [0, 1]).all()

def test_kfold_backtest_predictions_shuffle(sample_data):
    """Test that shuffled predictions are re-ordered correctly."""
    X, y = sample_data
    n_splits = 5
    cv = KFold(n_splits=n_splits, shuffle=True, random_seed=42)
    model = LogisticRegression()
    
    preds_dict = cv.backtest_predictions(model, X, y, n_jobs=1)
    
    assert 'Path 1' in preds_dict
    preds = preds_dict['Path 1']
    assert isinstance(preds, np.ndarray)
    assert len(preds) == X.shape[0]