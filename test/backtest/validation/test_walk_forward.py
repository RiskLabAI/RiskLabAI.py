"""
Tests for WalkForward cross-validator.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from RiskLabAI.backtest.validation.walk_forward import WalkForward

# Re-use the sample data fixture from test_kfold
@pytest.fixture
def sample_data():
    """Fixture for sample data."""
    n_samples = 100
    X = pd.DataFrame({'feature1': np.arange(n_samples), 'feature2': np.arange(n_samples, 0, -1)})
    y = pd.Series(np.random.randint(0, 2, n_samples))
    return X, y

def test_walk_forward_init():
    """Test WalkForward initialization."""
    cv = WalkForward(n_splits=5, max_train_size=50, gap=2)
    assert cv.n_splits == 5
    assert cv.max_train_size == 50
    assert cv.gap == 2
    assert cv.shuffle is False  # WalkForward should never shuffle

def test_walk_forward_split_no_gap(sample_data):
    """Test WalkForward split without gap."""
    X, _ = sample_data
    n_splits = 5
    cv = WalkForward(n_splits=n_splits, gap=0)
    
    splits = list(cv.split(X))
    assert len(splits) == n_splits
    
    # Fold 1
    train_idx_0, test_idx_0 = splits[0]
    np.testing.assert_array_equal(train_idx_0, np.array([], dtype=int))
    np.testing.assert_array_equal(test_idx_0, np.arange(0, 20))
    
    # Fold 2
    train_idx_1, test_idx_1 = splits[1]
    np.testing.assert_array_equal(train_idx_1, np.arange(0, 20))
    np.testing.assert_array_equal(test_idx_1, np.arange(20, 40))

    # Fold 5 (last)
    train_idx_4, test_idx_4 = splits[4]
    np.testing.assert_array_equal(train_idx_4, np.arange(0, 80))
    np.testing.assert_array_equal(test_idx_4, np.arange(80, 100))

def test_walk_forward_split_with_gap(sample_data):
    """Test WalkForward split with a gap."""
    X, _ = sample_data
    n_splits = 5
    gap = 2
    cv = WalkForward(n_splits=n_splits, gap=gap)
    
    splits = list(cv.split(X))
    assert len(splits) == n_splits
    
    # Fold 1 (test starts at 0, train_end = 0 - 2 = -2)
    train_idx_0, test_idx_0 = splits[0]
    np.testing.assert_array_equal(train_idx_0, np.array([], dtype=int))
    np.testing.assert_array_equal(test_idx_0, np.arange(0, 20))
    
    # Fold 2 (test starts at 20, train_end = 20 - 2 = 18)
    train_idx_1, test_idx_1 = splits[1]
    np.testing.assert_array_equal(train_idx_1, np.arange(0, 18))
    np.testing.assert_array_equal(test_idx_1, np.arange(20, 40))

    # Fold 5 (last) (test starts at 80, train_end = 80 - 2 = 78)
    train_idx_4, test_idx_4 = splits[4]
    np.testing.assert_array_equal(train_idx_4, np.arange(0, 78))
    np.testing.assert_array_equal(test_idx_4, np.arange(80, 100))

def test_walk_forward_split_with_max_train(sample_data):
    """Test WalkForward split with max_train_size."""
    X, _ = sample_data
    n_splits = 5
    max_train_size = 30
    cv = WalkForward(n_splits=n_splits, max_train_size=max_train_size, gap=0)
    
    splits = list(cv.split(X))
    
    # Fold 1 (train_end=0, train_size=0)
    train_idx_0, _ = splits[0]
    np.testing.assert_array_equal(train_idx_0, np.array([], dtype=int))
    
    # Fold 2 (train_end=20, train_size=20)
    train_idx_1, _ = splits[1]
    np.testing.assert_array_equal(train_idx_1, np.arange(0, 20))
    assert len(train_idx_1) == 20
    
    # Fold 3 (train_end=40, train_size=40, capped at 30)
    train_idx_2, _ = splits[2]
    # train_start = 40 - 30 = 10. train_end = 40.
    np.testing.assert_array_equal(train_idx_2, np.arange(10, 40))
    assert len(train_idx_2) == max_train_size

    # Fold 5 (last) (train_end=80, train_size=80, capped at 30)
    train_idx_4, _ = splits[4]
    # train_start = 80 - 30 = 50. train_end = 80.
    np.testing.assert_array_equal(train_idx_4, np.arange(50, 80))
    assert len(train_idx_4) == max_train_size

def test_walk_forward_predictions_with_nan(sample_data):
    """Test that predictions are np.nan for the first fold with no train data."""
    X, y = sample_data
    n_splits = 5
    cv = WalkForward(n_splits=n_splits, gap=0)
    model = LogisticRegression()

    preds_dict = cv.backtest_predictions(model, X, y, n_jobs=1)
    preds = preds_dict['Path 1']
    
    assert len(preds) == X.shape[0]
    
    # First 20 predictions should be NaN
    assert np.isnan(preds[:20]).all()
    # Remaining predictions should be valid
    assert not np.isnan(preds[20:]).any()