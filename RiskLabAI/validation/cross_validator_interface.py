from abc import ABC, abstractmethod
from typing import Iterator, Tuple, Union
import pandas as pd
import numpy as np

class CrossValidator(ABC):
    """
    Abstract Base Class (ABC) that defines an interface 
    for all cross-validation strategies. All other cross-validation 
    strategies should inherit from this class and implement
    the abstract methods.

    Methods
    -------
    split(
        data: pd.DataFrame,
        labels: Union[None, pd.Series] = None,
        groups: Union[None, pd.Series] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]
        Yields the indices for the training and test data.
    """
    
    @abstractmethod
    def split(
        self, 
        data: pd.DataFrame, 
        labels: Union[None, pd.Series] = None,
        groups: Union[None, pd.Series] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        data : pd.DataFrame
            The sample data that is going to be split.
        labels : Union[None, pd.Series], optional
            The labels that are going to be split, by default None
        groups : Union[None, pd.Series], optional
            Group labels for the samples used while splitting the dataset, by default None

        Returns
        -------
        Iterator[Tuple[np.ndarray, np.ndarray]]
            Yields the indices for the training and test data.

        .. note:: 
            This method must be overridden by all subclasses.
        """
        pass
