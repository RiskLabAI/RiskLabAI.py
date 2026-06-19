"""
Deprecated factory for creating cross-validator instances.

The core component registry (:data:`RiskLabAI.core.CROSS_VALIDATORS`) is now the
single source of truth for creating cross-validators. This factory is retained
for backward compatibility and simply delegates to that registry; it is removed
in 2.1.0.
"""

import warnings
from typing import Any

from .adaptive_combinatorial_purged import AdaptiveCombinatorialPurged
from .bagged_combinatorial_purged import BaggedCombinatorialPurged
from .combinatorial_purged import CombinatorialPurged
from .cross_validator_interface import CrossValidator
from .kfold import KFold
from .purged_kfold import PurgedKFold
from .walk_forward import WalkForward


class CrossValidatorFactory:
    """
    Deprecated. Use ``RiskLabAI.core.CROSS_VALIDATORS.create(...)`` instead.

    Retained for backward compatibility (removed in 2.1.0). ``VALIDATORS`` is
    kept so existing introspection keeps working; ``create_cross_validator``
    now delegates to the core registry.
    """

    VALIDATORS = {
        "kfold": KFold,
        "walkforward": WalkForward,
        "purgedkfold": PurgedKFold,
        "combinatorialpurged": CombinatorialPurged,
        "baggedcombinatorialpurged": BaggedCombinatorialPurged,
        "adaptivecombinatorialpurged": AdaptiveCombinatorialPurged,
    }

    @staticmethod
    def create_cross_validator(validator_type: str, **kwargs: Any) -> CrossValidator:
        """
        Deprecated. Create a cross-validator via the core registry.

        Parameters
        ----------
        validator_type : str
            One of the keys in :attr:`VALIDATORS` (case-insensitive).
        **kwargs : Any
            Forwarded to the validator's constructor; arguments it does not
            accept are dropped (matching the historical behaviour).

        Raises
        ------
        ValueError
            If ``validator_type`` is not a known validator.
        """
        warnings.warn(
            "CrossValidatorFactory is deprecated and will be removed in 2.1.0; "
            "use RiskLabAI.core.CROSS_VALIDATORS.create(validator_type, ...) "
            "instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from RiskLabAI.core import CROSS_VALIDATORS

        if validator_type.lower() not in CROSS_VALIDATORS:
            raise ValueError(
                f"Invalid validator_type: {validator_type}. "
                f"Valid types are: {list(CrossValidatorFactory.VALIDATORS.keys())}"
            )
        return CROSS_VALIDATORS.create(
            validator_type, filter_unknown_kwargs=True, **kwargs
        )
