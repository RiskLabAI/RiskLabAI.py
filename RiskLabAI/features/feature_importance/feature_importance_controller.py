"""
Deprecated controller for managing feature-importance strategies.

Use the core component registry (:data:`RiskLabAI.core.FEATURE_IMPORTANCE`)
directly instead. This controller is retained for backward compatibility and
delegates to that registry; it is removed in 2.1.0.
"""

import warnings
from typing import Any

import pandas as pd

from .feature_importance_strategy import FeatureImportanceStrategy


class FeatureImportanceController:
    """
    Deprecated. Use ``RiskLabAI.core.FEATURE_IMPORTANCE.create(...)`` instead.

    Removed in 2.1.0.
    """

    def __init__(self, strategy_type: str, **kwargs: Any):
        warnings.warn(
            "FeatureImportanceController is deprecated and will be removed in "
            "2.1.0; use RiskLabAI.core.FEATURE_IMPORTANCE.create(strategy_type, "
            "...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from RiskLabAI.core import FEATURE_IMPORTANCE

        self.strategy_instance: FeatureImportanceStrategy = FEATURE_IMPORTANCE.create(
            strategy_type, filter_unknown_kwargs=True, **kwargs
        )

    def calculate_importance(
        self, x: pd.DataFrame, y: pd.Series, **kwargs: Any
    ) -> pd.DataFrame:
        """Calculate feature importance using the configured strategy."""
        return self.strategy_instance.compute(x, y, **kwargs)
