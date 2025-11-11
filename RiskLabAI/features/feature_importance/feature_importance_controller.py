"""
Controller class to manage various feature importance strategies.
"""

from typing import Any, Dict
import pandas as pd
from .feature_importance_factory import FeatureImportanceFactory
from .feature_importance_strategy import FeatureImportanceStrategy

class FeatureImportanceController:
    """
    Controller class to manage and execute feature importance strategies.

    Example
    -------
    .. code-block:: python

       from sklearn.ensemble import RandomForestClassifier
       
       my_classifier = RandomForestClassifier(n_estimators=10, seed=42)
       my_clusters = {'cluster_0': ['feat_0', 'feat_1']}

       # Initialize the controller
       controller = FeatureImportanceController(
           'ClusteredMDA',
           classifier=my_classifier, 
           clusters=my_clusters,
           n_splits=10
       )

       # Calculate feature importance
       result = controller.calculate_importance(my_x, my_y)
    """

    def __init__(self, strategy_type: str, **kwargs: Any):
        """
        Initialize the controller with a specific feature importance strategy.

        Parameters
        ----------
        strategy_type : str
            The type of strategy to use (e.g., 'MDI', 'ClusteredMDA').
        **kwargs : Any
            Configuration arguments to pass to the strategy's
            constructor (e.g., `classifier`, `clusters`, `n_splits`).
        """
        self.strategy_instance: FeatureImportanceStrategy = (
            FeatureImportanceFactory.create_feature_importance(
                strategy_type, **kwargs
            )
        )

    def calculate_importance(
        self, x: pd.DataFrame, y: pd.Series, **kwargs: Any
    ) -> pd.DataFrame:
        """
        Calculate feature importance based on the initialized strategy.

        Parameters
        ----------
        x : pd.DataFrame
            Feature data.
        y : pd.Series
            Target data.
        **kwargs : Any
            Additional arguments to pass to the strategy's `compute`
            method (e.g., `sample_weight`).
        
        Returns
        -------
        pd.DataFrame
            Feature importance results.
        """
        return self.strategy_instance.compute(x, y, **kwargs)