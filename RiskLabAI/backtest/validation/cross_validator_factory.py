"""
Factory for creating cross-validator instances.
"""

import inspect
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
    Factory class for creating cross-validator objects.
    
    This class uses a static method to encapsulate the logic for
    instantiating different cross-validator strategies.
    """

    VALIDATORS = {
        'kfold': KFold,
        'walkforward': WalkForward,
        'purgedkfold': PurgedKFold,
        'combinatorialpurged': CombinatorialPurged,
        'baggedcombinatorialpurged': BaggedCombinatorialPurged,
        'adaptivecombinatorialpurged': AdaptiveCombinatorialPurged,
    }

    @staticmethod
    def create_cross_validator(
            validator_type: str,
            **kwargs: Any
    ) -> CrossValidator:
        """
        Factory method to create and return an instance of a cross-validator.

        Parameters
        ----------
        validator_type : str
            Type of cross-validator to create. Must be one of:
            'kfold', 'walkforward', 'purgedkfold', 'combinatorialpurged',
            'baggedcombinatorialpurged', 'adaptivecombinatorialpurged'.
        **kwargs : Any
            Keyword arguments to be passed to the cross-validator's
            constructor.

        Returns
        -------
        CrossValidator
            An instance of the specified cross-validator.

        Raises
        ------
        ValueError
            If an invalid `validator_type` is provided.
        """
        validator_type = validator_type.lower()
        validator_class = CrossValidatorFactory.VALIDATORS.get(validator_type)

        if validator_class:
            sig = inspect.signature(validator_class.__init__)
            valid_kwargs = {
                k: v for k, v in kwargs.items() if k in sig.parameters
            }
            return validator_class(**valid_kwargs)

        raise ValueError(
            f"Invalid validator_type: {validator_type}. "
            f"Valid types are: {list(CrossValidatorFactory.VALIDATORS.keys())}"
        )