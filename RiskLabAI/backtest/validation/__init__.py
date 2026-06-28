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

from .adaptive_combinatorial_purged import AdaptiveCombinatorialPurged
from .bagged_combinatorial_purged import BaggedCombinatorialPurged
from .combinatorial_purged import CombinatorialPurged
from .cross_validator_controller import CrossValidatorController
from .cross_validator_factory import CrossValidatorFactory
from .cross_validator_interface import CrossValidator
from .kfold import KFold
from .leakage_aware_hpo import deflated_sharpe_gate, leakage_aware_hpo
from .path_adaptive_cpcv import (
    adaptive_probability_of_backtest_overfitting,
    estimate_volatility_regimes,
)
from .path_bagged_cpcv import (
    bagged_probability_of_backtest_overfitting,
    moving_block_bootstrap_indices,
)
from .purged_kfold import PurgedKFold
from .walk_forward import WalkForward

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
    # Path-level bagged CPCV PBO (Appraisal 09)
    "bagged_probability_of_backtest_overfitting",
    "moving_block_bootstrap_indices",
    # Path-level adaptive (regime-weighted) CPCV PBO (Appraisal 09b)
    "adaptive_probability_of_backtest_overfitting",
    "estimate_volatility_regimes",
    # Leakage-aware HPO methodology (Appraisal 20)
    "leakage_aware_hpo",
    "deflated_sharpe_gate",
    # Utilities
    "CrossValidatorFactory",
    "CrossValidatorController",
]
