"""
RiskLabAI Backtest Validation Module

This module provides a suite of advanced cross-validation tools
for financial machine learning, specializing in methods that
prevent information leakage, such as Purged K-Fold and
Combinatorial Purged Cross-Validation (CPCV).
"""

from .cross_validator_interface import CrossValidator
from .kfold import KFold
from .walk_forward import WalkForward
from .purged_kfold import PurgedKFold
from .combinatorial_purged import CombinatorialPurged
from .adaptive_combinatorial_purged import AdaptiveCombinatorialPurged
from .bagged_combinatorial_purged import BaggedCombinatorialPurged
from .cross_validator_factory import CrossValidatorFactory
from .cross_validator_controller import CrossValidatorController

__all__ = [
    "CrossValidator",
    "KFold",
    "WalkForward",
    "PurgedKFold",
    "CombinatorialPurged",
    "AdaptiveCombinatorialPurged",
    "BaggedCombinatorialPurged",
    "CrossValidatorFactory",
    "CrossValidatorController"
]