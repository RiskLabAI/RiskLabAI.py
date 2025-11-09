"""
Abstract Base Class for feature importance strategies.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Any

class FeatureImportanceStrategy(ABC):
    """
    Abstract Base Class for computing feature importance.

    Derived classes must implement the `compute` method.
    """

    @abstractmethod
    def compute(
        self, x: pd.DataFrame, y: pd.Series, **kwargs: Any
    ) -> pd.DataFrame:
        """
        Abstract method to compute feature importance.

        Parameters
        ----------
        x : pd.DataFrame
            The feature data.
        y : pd.Series
            The target data.
        **kwargs : Any
            Additional keyword arguments (e.g., sample_weights).

        Returns
        -------
        pd.DataFrame
            A DataFrame containing feature importances.
        """
        pass