"""
RiskLabAI Cross-Validation Module

Implements advanced financial cross-validation techniques from
"Advances in Financial Machine Learning" by de Prado, including:
- Purged K-Fold (for removing group leakage)
- Walk-Forward
- Combinatorial Purged K-Fold (CPSC-V)

This module is built on an AbstractCrossValidator interface and
a CrossValidatorFactory for easy instantiation.
"""

from .cross_validator_interface import CrossValidator
from .kfold import KFold
from .purged_kfold import PurgedKFold
from .walk_forward import WalkForward
from .combinatorial_purged import CombinatorialPurged
from .bagged_combinatorial_purged import BaggedCombinatorialPurged
from .adaptive_combinatorial_purged import AdaptiveCombinatorialPurged
from .cross_validator_factory import CrossValidatorFactory
from .cross_validator_controller import CrossValidatorController

__all__ = [
    # Interface
    "CrossValidator",
    
    # Validators
    "KFold",
    "PurgedKFold",
    "WalkForward",  # <-- Fix
    "CombinatorialPurged",  # <-- Fix
    "BaggedCombinatorialPurged",  # <-- Fix
    "AdaptiveCombinatorialPurged",  # <-- Fix
    
    # Utilities
    "CrossValidatorFactory",
    "CrossValidatorController",
]