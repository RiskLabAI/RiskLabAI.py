"""
Tests for AdaptiveCombinatorialPurged cross-validator.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from itertools import combinations  


from RiskLabAI.backtest.validation.adaptive_combinatorial_purged import AdaptiveCombinatorialPurged

from RiskLabAI.backtest.validation.combinatorial_purged import CombinatorialPurged 
from itertools import combinations 


# Re-use the purged k-fold fixture and add an external feature
@pytest.fixture
def sample_data_with_times_and_feature():
    """Fixture for sample data with 'times' and 'external_feature'."""
    n_samples = 120
    idx = pd.date_range('2020-01-01', periods=n_samples, freq='B')
    X = pd.DataFrame(
        {'feature1': np.arange(n_samples)}, 
        index=idx
    )
    y = pd.Series(
        np.random.randint(0, 2, n_samples), 
        index=idx
    )
    times = pd.Series(
        idx + pd.DateOffset(days=7), 
        index=idx
    )
    # External feature: a sine wave to create predictable quantiles
    external_feature = pd.Series(
        np.sin(np.linspace(0, 10, n_samples)), 
        index=idx
    )
    return X, y, times, external_feature

def test_adaptive_init(sample_data_with_times_and_feature):
    """Test A-CPCV initialization."""
    _, _, times, feature = sample_data_with_times_and_feature
    cv = AdaptiveCombinatorialPurged(
        n_splits=6,
        n_test_groups=2,
        times=times,
        external_feature=feature,
        lower_quantile=0.3,
        upper_quantile=0.7
    )
    assert cv.n_splits == 6
    assert cv.lower_quantile == 0.3
    assert cv.external_feature is not None

def test_adaptive_split_segments(sample_data_with_times_and_feature):
    """Test the adaptive segment splitting."""
    X, _, _, feature = sample_data_with_times_and_feature
    
    cv_adaptive = AdaptiveCombinatorialPurged(
        n_splits=6, n_test_groups=2, times=pd.Series(), # Dummy times
        external_feature=feature
    )
    cv_normal = CombinatorialPurged(
         n_splits=6, n_test_groups=2, times=pd.Series() # Dummy times
    )
    
    adaptive_segments = cv_adaptive._get_split_segments(X, feature)
    normal_segments = cv_normal._get_split_segments(X)
    
    # Check that segments are different
    assert len(adaptive_segments) == 6
    assert len(normal_segments) == 6
    
    # Normal segments should all be size 20
    assert all(len(seg) == 20 for seg in normal_segments)
    
    # Adaptive segments should NOT all be size 20
    adaptive_lengths = [len(seg) for seg in adaptive_segments]
    assert not all(length == 20 for length in adaptive_lengths)
    
    # Check they still cover all indices
    assert sum(adaptive_lengths) == 120
    
def test_adaptive_predictions(sample_data_with_times_and_feature):
    """Test the backtest_predictions method."""
    X, y, times, feature = sample_data_with_times_and_feature
    cv = AdaptiveCombinatorialPurged(
        n_splits=6,
        n_test_groups=2,
        times=times,
        external_feature=feature,
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
    
    # Check total length of predictions
    total_preds = sum(len(p) for p in preds_dict.values())
    assert total_preds == 5 * 120