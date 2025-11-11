"""
Tests for the features/feature_importance module.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from RiskLabAI.features.feature_importance.feature_importance_controller import FeatureImportanceController

@pytest.fixture
def mock_data():
    """Generate mock data for testing."""
    N = 100
    P = 10
    X = pd.DataFrame(np.random.normal(0, 1, size=(N, P)),
                     columns=[f'feat_{i}' for i in range(P)])
    
    # Make features 0, 1, 2 correlated
    X['feat_1'] = X['feat_0'] + np.random.normal(0, 0.1, N)
    X['feat_2'] = X['feat_0'] + np.random.normal(0, 0.1, N)
    
    # Target depends on feat_0 (and its cluster) and feat_5
    y = pd.Series(np.where(X['feat_0'] + X['feat_5'] > 0, 1, 0))
    
    clusters = {
        'cluster_0': ['feat_0', 'feat_1', 'feat_2'],
        'cluster_1': ['feat_3', 'feat_4'],
        'cluster_2': ['feat_5', 'feat_6', 'feat_7'],
        'cluster_3': ['feat_8', 'feat_9'],
    }
    
    classifier = RandomForestClassifier(n_estimators=10, random_state=42)
    
    return X, y, classifier, clusters

def test_controller_mdi(mock_data):
    """Test MDI via the controller."""
    X, y, classifier, _ = mock_data
    
    controller = FeatureImportanceController("MDI", classifier=classifier)
    importance = controller.calculate_importance(X, y)
    
    assert isinstance(importance, pd.DataFrame)
    assert importance.shape == (10, 2)
    assert 'Mean' in importance.columns
    assert np.isclose(importance['Mean'].sum(), 1.0)
    assert importance['Mean'].idxmax() in ['feat_0', 'feat_1', 'feat_2', 'feat_5']

def test_controller_clustered_mdi(mock_data):
    """Test Clustered MDI via the controller."""
    X, y, classifier, clusters = mock_data
    
    controller = FeatureImportanceController(
        "ClusteredMDI", classifier=classifier, clusters=clusters
    )
    importance = controller.calculate_importance(X, y)
    
    assert isinstance(importance, pd.DataFrame)
    assert importance.shape == (4, 2) # 4 clusters
    assert 'C_cluster_0' in importance.index
    assert np.isclose(importance['Mean'].sum(), 1.0)
    # cluster_0 should have the highest importance
    assert importance['Mean'].idxmax() in ['C_cluster_0', 'C_cluster_2']

def test_controller_mda(mock_data):
    """Test MDA via the controller."""
    X, y, classifier, _ = mock_data
    
    controller = FeatureImportanceController(
        "MDA", classifier=classifier, n_splits=3
    )
    importance = controller.calculate_importance(X, y)

    assert isinstance(importance, pd.DataFrame)
    assert importance.shape == (10, 2)

    top_2 = set(importance['Mean'].nlargest(2).index)
    assert 'feat_5' in top_2
    assert top_2.intersection({'feat_0', 'feat_1', 'feat_2'})

def test_controller_clustered_mda(mock_data):
    """Test Clustered MDA via the controller."""
    X, y, classifier, clusters = mock_data
    
    controller = FeatureImportanceController(
        "ClusteredMDA", classifier=classifier, clusters=clusters, n_splits=3
    )
    importance = controller.calculate_importance(X, y)
    
    assert isinstance(importance, pd.DataFrame)
    assert importance.shape == (4, 2)
    # cluster_0 and cluster_2 should be most important
    top_2 = importance['Mean'].nlargest(2).index
    assert 'C_cluster_0' in top_2
    assert 'C_cluster_2' in top_2

def test_controller_sfi(mock_data):
    """Test SFI via the controller."""
    X, y, classifier, _ = mock_data
    
    controller = FeatureImportanceController(
        "SFI", classifier=classifier, n_splits=3, scoring="accuracy"
    )
    importance = controller.calculate_importance(X, y)
    
    assert isinstance(importance, pd.DataFrame)
    assert importance.shape == (10, 2)

    top_5 = set(importance['Mean'].nlargest(5).index)
    assert 'feat_0' in top_5
    assert 'feat_1' in top_5
    assert 'feat_2' in top_5
    assert 'feat_5' in top_5