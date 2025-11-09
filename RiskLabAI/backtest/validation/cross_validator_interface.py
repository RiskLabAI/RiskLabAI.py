"""
Defines the abstract base class (ABC) for all cross-validation strategies
in the RiskLabAI library.
"""

from abc import ABC, abstractmethod
from typing import (
    Any, Dict, Generator, List, Optional, Tuple, Union
)

import numpy as np
import pandas as pd

# For type hinting sklearn-like estimators
Estimator = Any

class CrossValidator(ABC):
    """
    Abstract Base Class (ABC) for cross-validation strategies.

    This interface defines the common methods required for all cross-validators,
    ensuring they can handle both single and multiple datasets (passed as dicts)
    and provide methods for splitting, generating backtest paths, and
    running backtest predictions.
    """

    @abstractmethod
    def get_n_splits(
        self,
        data: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]] = None,
        labels: Optional[Union[pd.Series, Dict[str, pd.Series]]] = None,
        groups: Optional[np.ndarray] = None
    ) -> int:
        """
        Return the total number of splits.

        Parameters
        ----------
        data : pd.DataFrame or dict, optional
            Dataset or dictionary of datasets.
        labels : pd.Series or dict, optional
            Labels or dictionary of labels.
        groups : np.ndarray, optional
            Group labels for the samples.

        Returns
        -------
        int
            The total number of splits.
        """
        raise NotImplementedError

    @abstractmethod
    def _single_split(
        self,
        single_data: pd.DataFrame,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Split a single dataset into train-test indices.

        This function provides train-test indices to split a single dataset
        into train/test sets.

        Parameters
        ----------
        single_data : pd.DataFrame
            Input dataset.

        Yields
        -------
        Generator[Tuple[np.ndarray, np.ndarray], None, None]
            A generator where each item is a tuple of (train_indices, test_indices).
        """
        raise NotImplementedError

    @abstractmethod
    def split(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        labels: Optional[Union[pd.Series, Dict[str, pd.Series]]] = None,
        groups: Optional[np.ndarray] = None
    ) -> Union[
        Generator[Tuple[np.ndarray, np.ndarray], None, None],
        Generator[Tuple[str, Tuple[np.ndarray, np.ndarray]], None, None]
    ]:
        """
        Split data (or dictionary of data) into train-test indices.

        This function returns a generator that yields train-test indices. If a
        dictionary of data is provided, the generator yields a key followed
        by the (train_indices, test_indices) tuple.

        Parameters
        ----------
        data : pd.DataFrame or dict
            Dataset or dictionary of datasets.
        labels : pd.Series or dict, optional
            Labels or dictionary of labels.
        groups : np.ndarray, optional
            Group labels for the samples.

        Yields
        -------
        Generator
            - If data is a pd.DataFrame: (train_indices, test_indices)
            - If data is a dict: (key, (train_indices, test_indices))
        """
        raise NotImplementedError

    @abstractmethod
    def _single_backtest_paths(
        self,
        single_data: pd.DataFrame
    ) -> Dict[str, List[Dict[str, np.ndarray]]]:
        """
        Generate backtest paths for a single dataset.

        A "path" is a specific sequence of train/test folds. For simple
        K-Fold, there is only one path. For combinatorial methods, there
        can be multiple.

        Parameters
        ----------
        single_data : pd.DataFrame
            Input dataset.

        Returns
        -------
        Dict[str, List[Dict[str, np.ndarray]]]
            A dictionary where keys are path identifiers (e.g., "Path 1")
            and values are lists of splits. Each split is a dictionary
            with "Train" and "Test" keys holding their respective indices.
        """
        raise NotImplementedError

    @abstractmethod
    def backtest_paths(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    ) -> Union[
        Dict[str, List[Dict[str, np.ndarray]]],
        Dict[str, Dict[str, List[Dict[str, np.ndarray]]]]
    ]:
        """
        Generate backtest paths for data or a dictionary of data.

        Parameters
        ----------
        data : pd.DataFrame or dict
            Dataset or dictionary of datasets.

        Returns
        -------
        Union[Dict, Dict[str, Dict]]
            - If data is a pd.DataFrame: A dictionary of backtest paths.
            - If data is a dict: A nested dictionary, where the first-level
              key is the dataset key, and the value is its
              dictionary of backtest paths.
        """
        raise NotImplementedError

    @abstractmethod
    def _single_backtest_predictions(
        self,
        single_estimator: Estimator,
        single_data: pd.DataFrame,
        single_labels: pd.Series,
        single_weights: Optional[np.ndarray] = None,
        predict_probability: bool = False,
        n_jobs: int = 1
    ) -> Dict[str, np.ndarray]:
        """
        Obtain backtest predictions for a single dataset.

        Parameters
        ----------
        single_estimator : Estimator
            A scikit-learn-like estimator to be trained and used for predictions.
        single_data : pd.DataFrame
            Data for the single dataset.
        single_labels : pd.Series
            Labels for the single dataset.
        single_weights : np.ndarray, optional
            Sample weights for the observations. Defaults to equal weights.
        predict_probability : bool, default=False
            If True, call `predict_proba()` instead of `predict()`.
        n_jobs : int, default=1
            The number of jobs to run in parallel.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary where keys are path identifiers (e.g., "Path 1")
            and values are the contiguous arrays of out-of-sample predictions.
        """
        raise NotImplementedError

    @abstractmethod
    def backtest_predictions(
        self,
        estimator: Union[Estimator, Dict[str, Estimator]],
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        labels: Union[pd.Series, Dict[str, pd.Series]],
        sample_weights: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        predict_probability: bool = False,
        n_jobs: int = 1
    ) -> Union[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
        """
        Generate backtest predictions for single or multiple datasets.

        Parameters
        ----------
        estimator : Estimator or dict
            Model(s) to be trained. Can be a single estimator or a
            dictionary of estimators for multiple datasets.
        data : pd.DataFrame or dict
            Input data. Can be a single dataset or a dictionary of datasets.
        labels : pd.Series or dict
            Target labels. Can be a single series or a dictionary of series.
        sample_weights : np.ndarray or dict, optional
            Sample weights. Can be a single array or a dictionary of arrays.
            Defaults to None (equal weights).
        predict_probability : bool, default=False
            If True, call `predict_proba()` instead of `predict()`.
        n_jobs : int, default=1
            The number of jobs to run in parallel.

        Returns
        -------
        Union[Dict, Dict[str, Dict]]
            - If data is a pd.DataFrame: A dictionary of predictions.
            - If data is a dict: A nested dictionary, where the first-level
              key is the dataset key, and the value is its
              dictionary of predictions.
        """
        raise NotImplementedError