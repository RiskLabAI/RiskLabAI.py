"""
Tests for BaggedCombinatorialPurged cross-validator.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, LinearRegression

# Re-use the purged k-fold fixture
@pytest.fixture
def sample_data_with_times():
    """Fixture for sample data with a 'times' series."""
    n_samples = 120
    idx = pd.date_range('2020-01-01', periods=n_samples, freq='B')
    X = pd.DataFrame(
        {'feature1': np.arange(n_samples)}, 
        index=idx
    )
    y_class = pd.Series(
        np.random.randint(0, 2, n_samples), 
        index=idx
    )
    y_reg = pd.Series(
        np.random.randn(n_samples), 
        index=idx
    )
    times = pd.Series(
        idx + pd.DateOffset(days=7), 
        index=idx
    )
    return X, y_class, y_reg, times

def test_bagged_init(sample_data_with_times):
    """Test B-CPCV initialization."""
    _, _, _, times = sample_data_with_times
    cv = BaggedCombinatorialPurged(
        n_splits=6,
        n_test_groups=2,
        times=times,
        classifier=True,
        n_estimators=5
    )
    assert cv.classifier is True
    assert cv.n_estimators == 5

def test_bagged_predictions_classifier(sample_data_with_times):
    """Test B-CPCV predictions with a classifier."""
    X, y_class, _, times = sample_data_with_times
    cv = BaggedCombinatorialPurged(
        n_splits=6,
        n_test_groups=2,
        times=times,
        classifier=True,
        n_estimators=5,
        random_state=42
    )
    model = LogisticRegression()
    
    preds_dict = cv.backtest_predictions(model, X, y_class, n_jobs=1)
    
    assert len(preds_dict) == 5 # 5 paths
    preds1 = preds_dict['Path 1']
    assert len(preds1) == X.shape[0]
    assert np.isin(preds1, [0, 1]).all()

def test_bagged_predictions_regressor(sample_data_with_times):
    """Test B-CPCV predictions with a regressor."""
    X, _, y_reg, times = sample_data_with_times
    cv = BaggedCombinatorialPurged(
        n_splits=6,
        n_test_groups=2,
        times=times,
        classifier=False,
        n_estimators=5,
        random_state=42
    )
    model = LinearRegression()
    
    preds_dict = cv.backtest_predictions(model, X, y_reg, n_jobs=1)
    
    assert len(preds_dict) == 5 # 5 paths
    preds1 = preds_dict['Path 1']
    assert len(preds1) == X.shape[0]
    assert preds1.dtype == float
    
    # Test that predict_proba fails
    with pytest.raises(ValueError):
        cv.backtest_predictions(
            model, X, y_reg, predict_probability=True, n_jobs=1
        )