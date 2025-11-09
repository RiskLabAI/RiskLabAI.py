"""
Factory class for creating feature importance strategy objects.
"""

from typing import Any, Dict, List, Type
from .feature_importance_strategy import FeatureImportanceStrategy
from .feature_importance_mdi import FeatureImportanceMDI
from .clustered_feature_importance_mdi import ClusteredFeatureImportanceMDI
from .feature_importance_mda import FeatureImportanceMDA
from .clustered_feature_importance_mda import ClusteredFeatureImportanceMDA
from .feature_importance_sfi import FeatureImportanceSFI

class FeatureImportanceFactory:
    """
    Factory class to create feature importance strategy instances.
    """

    @staticmethod
    def create_feature_importance(
        strategy_type: str, **kwargs: Any
    ) -> FeatureImportanceStrategy:
        """
        Factory method to create and return an instance of a feature
        importance strategy.

        Parameters
        ----------
        strategy_type : str
            Type of strategy to create. Options include:
            'MDI', 'ClusteredMDI', 'MDA', 'ClusteredMDA', 'SFI'.
        **kwargs : Any
            Keyword arguments to be passed to the strategy's
            constructor (e.g., `classifier`, `clusters`, `n_splits`).

        Returns
        -------
        FeatureImportanceStrategy
            An instance of the specified strategy.

        Raises
        ------
        ValueError
            If an invalid `strategy_type` is provided.
        """
        
        strategies: Dict[str, Type[FeatureImportanceStrategy]] = {
            "MDI": FeatureImportanceMDI,
            "ClusteredMDI": ClusteredFeatureImportanceMDI,
            "MDA": FeatureImportanceMDA,
            "ClusteredMDA": ClusteredFeatureImportanceMDA,
            "SFI": FeatureImportanceSFI,
        }
        
        strategy_class = strategies.get(strategy_type)
        
        if strategy_class:
            # Pass only the relevant arguments to the constructor
            # This uses introspection to be robust
            import inspect
            sig = inspect.signature(strategy_class.__init__)
            valid_kwargs = {
                k: v for k, v in kwargs.items() if k in sig.parameters
            }
            return strategy_class(**valid_kwargs)
        
        raise ValueError(
            f"Invalid strategy_type: {strategy_type}. "
            f"Valid types are: {list(strategies.keys())}"
        )