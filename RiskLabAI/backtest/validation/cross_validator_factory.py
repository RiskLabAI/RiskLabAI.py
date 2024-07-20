from .kfold import KFold
from .walk_forward import WalkForward
from .purged_kfold import PurgedKFold
from .combinatorial_purged import CombinatorialPurged
from .bagged_combinatorial_purged import BaggedCombinatorialPurged
from .adaptive_combinatorial_purged import AdaptiveCombinatorialPurged
from .cross_validator_interface import CrossValidator

class CrossValidatorFactory:
    """
    Factory class for creating cross-validator objects.
    """

    @staticmethod
    def create_cross_validator(
            validator_type: str,
            **kwargs
    ) -> CrossValidator:
        """
        Factory method to create and return an instance of a cross-validator
        based on the provided type.

        :param validator_type: Type of cross-validator to create. Options include
            'kfold', 'walkforward', 'purgedkfold', and 'combinatorialpurged'.
        :type validator_type: str

        :param kwargs: Additional keyword arguments to be passed to the cross-validator's constructor.
        :type kwargs: Type

        :return: An instance of the specified cross-validator.
        :rtype: CrossValidator

        :raises ValueError: If an invalid validator type is provided.
        """
        if validator_type == 'kfold':
            return KFold(**kwargs)
            
        elif validator_type == 'walkforward':
            return WalkForward(**kwargs)

        elif validator_type == 'purgedkfold':
            return PurgedKFold(**kwargs)

        elif validator_type == 'combinatorialpurged':
            return CombinatorialPurged(**kwargs)
        
        elif validator_type == 'baggedcombinatorialpurged':
            return BaggedCombinatorialPurged(**kwargs)
        
        elif validator_type == 'adaptivecombinatorialpurged':
            return AdaptiveCombinatorialPurged(**kwargs)

        else:
            raise ValueError(f"Invalid validator_type: {validator_type}")
