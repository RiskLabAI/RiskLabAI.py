from typing import Dict, Union, Tuple, List, Any, Optional, Generator
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class CrossValidator(ABC):
    """
    Abstract Base Class (ABC) for cross-validation strategies.
    Handles both single data inputs and dictionary inputs.

    :param data: The input data, either as a single DataFrame or a dictionary of DataFrames.
    :type data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]

    :param labels: The labels corresponding to the data, either as a single Series or a dictionary of Series.
    :type labels: Optional[Union[pd.Series, Dict[str, pd.Series]]]

    :param groups: Optional group labels for stratified splitting.
    :type groups: Optional[np.ndarray]
    """


    @abstractmethod
    def get_n_splits(
        self,
        data: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]] = None,
        labels: Optional[Union[pd.Series, Dict[str, pd.Series]]] = None,
        groups: Optional[np.ndarray] = None
    ) -> int:
        """
        Return number of splits.

        :param data: Dataset or dictionary of datasets.
        :type data: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]]

        :param labels: Labels or dictionary of labels.
        :type labels: Optional[Union[pd.Series, Dict[str, pd.Series]]]

        :param groups: Group labels for the samples.
        :type groups: Optional[np.ndarray]

        :return: Number of splits.
        :rtype: int
        """


    @abstractmethod
    def _single_split(
        self,
        single_data: pd.DataFrame,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Splits a single data set into train-test indices.

        This function provides train-test indices to split the data into train/test sets
        by respecting the time order (if applicable) and the specified number of splits.

        :param single_data: Input dataset.
        :type single_data: pd.DataFrame

        :return: Generator yielding train-test indices.
        :rtype: Generator[Tuple[np.ndarray, np.ndarray], None, None]
        """


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
        Splits data or a dictionary of data into train-test indices.

        This function returns a generator that yields train-test indices. If a dictionary
        of data is provided, the generator yields a key followed by the train-test indices.

        :param data: Dataset or dictionary of datasets.
        :type data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
        :param labels: Labels or dictionary of labels.
        :type labels: Optional[Union[pd.Series, Dict[str, pd.Series]]]
        :param groups: Group labels for the samples.
        :type groups: Optional[np.ndarray]

        :return: Generator yielding either train-test indices directly or a key
                followed by train-test indices.
        :rtype: Union[
            Generator[Tuple[np.ndarray, np.ndarray], None, None],
            Generator[Tuple[str, Tuple[np.ndarray, np.ndarray]], None, None]
        ]
        """


    @abstractmethod
    def _single_backtest_paths(
        self,
        single_data: pd.DataFrame
    ) -> Dict[str, List[Dict[str, List[np.ndarray]]]]:
        """
        Generates backtest paths for a single dataset.

        This function creates and returns backtest paths (i.e., combinations of training and test sets)
        for a single dataset by applying k-fold splitting or any other splitting strategy defined
        by the `_single_split` function.

        :param single_data: Input dataset.
        :type single_data: pd.DataFrame

        :return: Dictionary of backtest paths.
        :rtype: Dict[str, List[Dict[str, List[np.ndarray]]]]
        """


    @abstractmethod
    def backtest_paths(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    ) -> Union[
        Dict[str, List[Dict[str, np.ndarray]]],
        Dict[str, Dict[str, List[Dict[str, List[np.ndarray]]]]]
    ]:
        """
        Generates backtest paths for data.

        This function returns backtest paths for either a single dataset or a dictionary
        of datasets. Each backtest path consists of combinations of training and test sets.

        :param data: Dataset or dictionary of datasets.
        :type data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
        :param labels: Labels or dictionary of labels.
        :type labels: Union[pd.Series, Dict[str, pd.Series]]

        :return: Dictionary of backtest paths or dictionary of dictionaries for multiple datasets.
        :rtype: Union[
            Dict[str, List[Dict[str, np.ndarray]]],
            Dict[str, Dict[str, List[Dict[str, List[np.ndarray]]]]]
        ]
        """


    @abstractmethod
    def _single_backtest_predictions(
        self,
        single_estimator: Any,
        single_data: pd.DataFrame,
        single_labels: pd.Series,
        single_weights: Optional[np.ndarray] = None,
        n_jobs: int = 1
    ) -> Dict[str, np.ndarray]:
        """
        Obtain predictions for a single dataset during backtesting.

        This function leverages parallel computation to train and predict on different train-test splits
        of a single dataset using a given estimator. It utilizes the `_single_split` method to generate
        the train-test splits.

        :param single_estimator: Estimator or model to be trained and used for predictions.
        :type single_estimator: Any
        :param single_data: Data of the single dataset.
        :type single_data: pd.DataFrame
        :param single_labels: Labels corresponding to the single dataset.
        :type single_labels: pd.Series
        :param single_weights: Weights for the observations in the single dataset.
                            Defaults to equally weighted if not provided.
        :type single_weights: np.ndarray, optional
        :param n_jobs: The number of jobs to run in parallel. Default is 1.
        :type n_jobs: int, optional
        :return: Predictions structured in a dictionary for the backtest paths.
        :rtype: Dict[str, np.ndarray]
        """

    @abstractmethod
    def backtest_predictions(
        self,
        estimator: Union[Any, Dict[str, Any]],
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        labels: Union[pd.Series, Dict[str, pd.Series]],
        sample_weights: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        predict_probability: bool = False,
        n_jobs: int = 1
    ) -> Union[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
        """
        Generate backtest predictions for single or multiple datasets.

        For each dataset, this function leverages the `_single_backtest_predictions` method to obtain
        predictions for different train-test splits using the given estimator.

        :param estimator: Model or estimator to be trained and used for predictions.
                        Can be a single estimator or a dictionary of estimators for multiple datasets.
        :type estimator: Union[Any, Dict[str, Any]]
        :param data: Input data for training and testing. Can be a single dataset or
                    a dictionary of datasets for multiple datasets.
        :type data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
        :param labels: Target labels for training and testing. Can be a single series or
                    a dictionary of series for multiple datasets.
        :type labels: Union[pd.Series, Dict[str, pd.Series]]
        :param sample_weights: Weights for the observations in the dataset(s).
                            Can be a single array or a dictionary of arrays for multiple datasets.
                            Defaults to None, which means equal weights for all observations.
        :type sample_weights: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]
        :param n_jobs: The number of jobs to run in parallel. Default is 1.
        :type n_jobs: int, optional
        :return: Backtest predictions structured in a dictionary (or nested dictionaries for multiple datasets).
        :rtype: Union[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]
        """
