"""
Deprecated factory for creating feature-importance strategy objects.

The core component registry (:data:`RiskLabAI.core.FEATURE_IMPORTANCE`) is now
the single source of truth. This factory is retained for backward compatibility
and delegates to that registry; it is removed in 2.1.0.
"""

import warnings
from typing import Any

from .feature_importance_strategy import FeatureImportanceStrategy


class FeatureImportanceFactory:
    """
    Deprecated. Use ``RiskLabAI.core.FEATURE_IMPORTANCE.create(...)`` instead.

    Removed in 2.1.0.
    """

    @staticmethod
    def create_feature_importance(
        strategy_type: str, **kwargs: Any
    ) -> FeatureImportanceStrategy:
        """
        Deprecated. Create a feature-importance strategy via the core registry.

        Raises
        ------
        ValueError
            If ``strategy_type`` is not a known strategy.
        """
        warnings.warn(
            "FeatureImportanceFactory is deprecated and will be removed in "
            "2.1.0; use RiskLabAI.core.FEATURE_IMPORTANCE.create(strategy_type, "
            "...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from RiskLabAI.core import FEATURE_IMPORTANCE

        if strategy_type.lower() not in FEATURE_IMPORTANCE:
            valid = list(FEATURE_IMPORTANCE.available())
            raise ValueError(
                f"Invalid strategy_type: {strategy_type}. Valid types are: {valid}"
            )
        return FEATURE_IMPORTANCE.create(
            strategy_type, filter_unknown_kwargs=True, **kwargs
        )
