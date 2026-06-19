"""
Deprecated controller for creating and holding a cross-validator.

Use the core component registry (:data:`RiskLabAI.core.CROSS_VALIDATORS`)
directly instead. This controller is retained for backward compatibility and
delegates to that registry; it is removed in 2.1.0.
"""

import warnings
from typing import Any

from .cross_validator_interface import CrossValidator


class CrossValidatorController:
    """
    Deprecated. Use ``RiskLabAI.core.CROSS_VALIDATORS.create(...)`` instead.

    Removed in 2.1.0.
    """

    def __init__(self, validator_type: str, **kwargs: Any):
        warnings.warn(
            "CrossValidatorController is deprecated and will be removed in "
            "2.1.0; use RiskLabAI.core.CROSS_VALIDATORS.create(validator_type, "
            "...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from RiskLabAI.core import CROSS_VALIDATORS

        self.cross_validator: CrossValidator = CROSS_VALIDATORS.create(
            validator_type, filter_unknown_kwargs=True, **kwargs
        )

    def get_validator(self) -> CrossValidator:
        """Return the created cross-validator instance."""
        return self.cross_validator
