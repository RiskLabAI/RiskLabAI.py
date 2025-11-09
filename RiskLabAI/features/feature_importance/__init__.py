"""
RiskLabAI Feature Importance Module

Provides a framework for calculating feature importance using
various methods (MDI, MDA, SFI) and clustering.
"""

from .feature_importance_strategy import FeatureImportanceStrategy
from .feature_importance_factory import FeatureImportanceFactory
from .feature_importance_controller import FeatureImportanceController

from .feature_importance_mdi import FeatureImportanceMDI
from .clustered_feature_importance_mdi import ClusteredFeatureImportanceMDI
from .feature_importance_mda import FeatureImportanceMDA
from .clustered_feature_importance_mda import ClusteredFeatureImportanceMDA
from .feature_importance_sfi import FeatureImportanceSFI

# Other utilities from the original __init__
# (Assuming they exist, as they were not provided)
# from .clustering import *
# from .generate_synthetic_data import *
# from .orthogonal_features import *
# from .weighted_tau import *
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
]