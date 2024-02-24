# from feature_importance_strategy import FeatureImportanceStrategy
from RiskLabAI.features.feature_importance.feature_importance_strategy import FeatureImportanceStrategy

from typing import Optional
import pandas as pd


class FeatureImportanceFactory:
    """
    Factory class for building and fetching feature importance computation results.

    Usage:

    .. code-block:: python

        factory = FeatureImportanceFactory()
        factory.build(my_feature_importance_strategy_instance)
        results = factory.get_results()

    """
    def __init__(self) -> None:
        """Initialize the FeatureImportanceFactory class."""
        self._results: Optional[pd.DataFrame] = None

    def build(
        self, 
        feature_importance_strategy: FeatureImportanceStrategy
    ) -> 'FeatureImportanceFactory':
        """
        Build the feature importance based on the provided strategy.

        :param feature_importance_strategy: An instance of a strategy 
            inheriting from FeatureImportanceStrategy.
        
        :return: Current instance of the FeatureImportanceFactory.
        """
        self._results = feature_importance_strategy.compute()
        return self

    def get_results(self) -> pd.DataFrame:
        """
        Fetch the computed feature importance results.

        :return: Dataframe containing the feature importance results.
        """
        if self._results is None:
            raise ValueError("Feature importance not yet computed.")
        
        return self._results
