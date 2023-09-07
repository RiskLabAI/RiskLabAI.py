from abc import ABC, abstractmethod
import pandas as pd


class FeatureImportanceStrategy(ABC):
    """
    Abstract Base Class for computing feature importance.

    Derived classes must implement the `compute` method to 
    provide their own logic for computing feature importance.
    """

    @abstractmethod
    def compute(
            self,
            *args,
            **kwargs
    ) -> pd.DataFrame:
        """
        Abstract method to compute feature importance.

        :param args: Positional arguments.
        :param kwargs: Keyword arguments.
        :return: A pandas DataFrame containing feature importances.

        Note: Derived classes should provide a concrete implementation 
        of this method with specific parameters and docstrings relevant 
        to their implementation.
        """
        pass
