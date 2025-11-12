"""
RiskLabAI Feature Importance Module

Provides a framework for calculating feature importance using
various methods (MDI, MDA, SFI) and clustering.
"""

# Import from other RiskLabAI modules
from RiskLabAI.cluster.clustering import (
    cluster_k_means_top,
    random_block_correlation,
)
from RiskLabAI.data.synthetic_data.simulation import (
    form_true_matrix,
    simulates_cov_mu,
)

# Core Strategy Pattern
from .feature_importance_strategy import FeatureImportanceStrategy
from .feature_importance_factory import FeatureImportanceFactory
from .feature_importance_controller import FeatureImportanceController

# Strategy Implementations
from .feature_importance_mdi import FeatureImportanceMDI
from .clustered_feature_importance_mdi import ClusteredFeatureImportanceMDI
from .feature_importance_mda import FeatureImportanceMDA
from .clustered_feature_importance_mda import ClusteredFeatureImportanceMDA
from .feature_importance_sfi import FeatureImportanceSFI

# Utility Functions for this module
from .generate_synthetic_data import get_test_dataset
from .orthogonal_features import orthogonal_features, _compute_eigenvectors
from .weighted_tau import calculate_weighted_tau

# Placeholder for future modules (from original file)
# from . import swefi

__all__ = [
    # Core Pattern
    "FeatureImportanceStrategy",
    "FeatureImportanceFactory",
    "FeatureImportanceController",
    
    # Strategy Implementations
    "FeatureImportanceMDI",
    "ClusteredFeatureImportanceMDI",
    "FeatureImportanceMDA",
    "ClusteredFeatureImportanceMDA",
    "FeatureImportanceSFI",
    
    # Imported utilities from *other* modules
    "cluster_k_means_top",
    "random_block_correlation",
    "form_true_matrix",
    "simulates_cov_mu",
    
    # Utilities *from this* module
    "get_test_dataset",
    "orthogonal_features",
    "calculate_weighted_tau",
]