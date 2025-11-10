"""
Tests for optimization/hyper_parameter_tuning.py
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from RiskLabAI.optimization.hyper_parameter_tuning import MyPipeline, clf_hyper_fit

@pytest.fixture
def mock_data():
    """Mock data for tuning."""
    X = pd.DataFrame(np.random.randn(100, 3), columns=['A', 'B', 'C'])
    y = pd.Series(np.random.randint(0, 2, 100))
    times = pd.Series(
        pd.date_range('2020-01-01', periods=100),
        index=pd.date_range('2020-01-01', periods=100)
    )
    return X, y, times

def test_my_pipeline_fit_sample_weight(mock_data):
    """Test that MyPipeline correctly passes sample_weight."""
    X, y, _ = mock_data
    sample_weight = np.random.rand(100)
    
    # Mock classifier to capture fit_params
    class MockLR(LogisticRegression):
        def fit(self, X, y, **kwargs):
            self.fit_kwargs = kwargs
            super().fit(X, y)

    pipe = MyPipeline([
        ('scaler', StandardScaler()),
        ('clf', MockLR())
    ])
    
    pipe.fit(X, y, sample_weight=sample_weight)
    
    # Check if 'clf__sample_weight' was passed
    assert 'clf__sample_weight' in pipe.named_steps['clf'].fit_kwargs
    assert np.array_equal(
        pipe.named_steps['clf'].fit_kwargs['clf__sample_weight'],
        sample_weight
    )

def test_clf_hyper_fit_gridsearch(mock_data):
    """Test the grid search functionality."""
    X, y, times = mock_data
    
    pipe_clf = MyPipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression())
    ])
    
    param_grid = {'clf__C': [0.1, 1.0]}
    
    # Use 'kfold' for simplicity, as purgedkfold requires 'times'
    validator_params = {'n_splits': 3}
    
    best_model = clf_hyper_fit(
        X, y, times, pipe_clf, param_grid,
        validator_type='kfold', # Use standard KFold for this test
        validator_params=validator_params,
        bagging=[0, 0, 0] # No bagging
    )
    
    assert isinstance(best_model, Pipeline)
    assert 'clf' in best_model.named_steps
    assert best_model.named_steps['clf'].C in [0.1, 1.0]

def test_clf_hyper_fit_bagging(mock_data):
    """Test the bagging functionality."""
    X, y, times = mock_data
    
    pipe_clf = MyPipeline([('clf', LogisticRegression())])
    param_grid = {'clf__C': [1.0]}
    validator_params = {'n_splits': 3}
    
    # Bagging: 5 estimators, 50% samples, 100% features
    bagging_params = [5, 0.5, 1.0] 
    
    bagged_model = clf_hyper_fit(
        X, y, times, pipe_clf, param_grid,
        validator_type='kfold',
        validator_params=validator_params,
        bagging=bagging_params
    )
    
    assert isinstance(bagged_model, Pipeline)
    assert 'bag' in bagged_model.named_steps
    assert isinstance(bagged_model.named_steps['bag'], BaggingClassifier)
    assert bagged_model.named_steps['bag'].n_estimators == 5