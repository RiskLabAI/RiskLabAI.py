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

from .clustered_feature_importance_mda import ClusteredFeatureImportanceMDA
from .clustered_feature_importance_mdi import ClusteredFeatureImportanceMDI
from .debiased_importance import (
    conditional_predictive_impact,
    mdi_plus_importance,
)
from .feature_importance_controller import FeatureImportanceController
from .feature_importance_factory import FeatureImportanceFactory
from .feature_importance_mda import FeatureImportanceMDA

# Strategy Implementations
from .feature_importance_mdi import FeatureImportanceMDI
from .feature_importance_sfi import FeatureImportanceSFI

# Core Strategy Pattern
from .feature_importance_strategy import FeatureImportanceStrategy

# Utility Functions for this module
from .generate_synthetic_data import get_test_dataset
from .orthogonal_features import _compute_eigenvectors, orthogonal_features
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
    # Debiased / conditional importance (Appraisal 10)
    "mdi_plus_importance",
    "conditional_predictive_impact",
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
