from typing import Union
import pandas as pd
from purged_k_fold import PurgedKFold
from combinatorial_purged_k_fold import CombinatorialPurgedKFold

class CrossValidatorFactory:
    """
    Factory class for creating cross-validator objects.
    """

    @staticmethod
    def create_cross_validator(
            validator_type: str,
            n_splits: int,
            times: pd.Series,
            percent_embargo: float = 0.0
    ) -> Union[PurgedKFold, CombinatorialPurgedKFold]:
        """
        Factory method for creating a cross-validator object based on the type.

        :param str validator_type: Type of the cross-validator ('purged' or 'combinatorial_purged').
        :param int n_splits: Number of KFold splits.
        :param pd.Series times: Series representing the entire observation times.
        :param float percent_embargo: Embargo size as a percentage divided by 100.
        :return: An instance of a cross-validator.
        :rtype: Union[PurgedKFold, CombinatorialPurgedKFold]
        """
        if validator_type == 'purged':
            return PurgedKFold(n_splits=n_splits, times=times, percent_embargo=percent_embargo)
        elif validator_type == 'combinatorial_purged':
            return CombinatorialPurgedKFold(n_splits=n_splits, times=times, percent_embargo=percent_embargo)
        else:
            raise ValueError(f"Invalid validator_type: {validator_type}")

# Usage
# cross_validator = CrossValidatorFactory.create_cross_validator('purged', 5, times_series, 0.1)
# or
# cross_validator = CrossValidatorFactory.create_cross_validator('combinatorial_purged', 5, times_series, 0.1)
