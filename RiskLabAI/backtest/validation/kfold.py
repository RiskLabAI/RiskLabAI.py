from copy import deepcopy
from typing import (
    Any, Generator, List, Optional, Tuple, Union, Dict
)
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning

from .cross_validator_interface import CrossValidator

class KFold(CrossValidator):
    """
    K-Fold cross-validator.

    This class implements the K-Fold cross-validation strategy, where the dataset is
    divided into `k` consecutive folds. Each fold is then used once as a validation set
    while the `k - 1` remaining folds form the training set.
    """

    def __init__(
        self,
        n_splits: int,
        shuffle: bool = False,
        random_seed: int = None
    ) -> None:
        """
        Initialize the K-Fold cross-validator.

        :param n_splits: Number of splits or folds for the cross-validation.
                         The dataset will be divided into `n_splits` consecutive parts.
        :type n_splits: int
        :param shuffle: Whether to shuffle the data before splitting it into folds.
                        If `shuffle` is set to True, the data will be shuffled before splitting.
        :type shuffle: bool, optional
        :param random_seed: Seed used for random shuffling. Set this seed for reproducibility.
                            Only used when `shuffle` is True.
        :type random_seed: int, optional
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_seed = random_seed

    def get_n_splits(
        self,
        data: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]] = None,
        labels: Optional[Union[pd.Series, Dict[str, pd.Series]]] = None,
        groups: Optional[np.ndarray] = None
    ) -> int:
        """
        Return number of splits.

        :param data: Dataset or dictionary of datasets.
        :type data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]

        :param labels: Labels or dictionary of labels.
        :type labels: Optional[Union[pd.Series, Dict[str, pd.Series]]]

        :param groups: Group labels for the samples.
        :type groups: Optional[np.ndarray]

        :return: Number of splits.
        :rtype: int
        """
        return self.n_splits

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

        indices = np.arange(single_data.shape[0])
        if self.shuffle:
            np.random.default_rng(self.random_seed).shuffle(indices)

        for test_indices in np.array_split(indices, self.n_splits):
            train_indices = np.setdiff1d(indices, test_indices)

            yield train_indices, test_indices


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

        if isinstance(data, dict):
            for key in data:
                for train_indices, test_indices in self._single_split(data[key]):
                    yield key, (train_indices, test_indices)
        else:
            for train_indices, test_indices in self._single_split(data):
                yield train_indices, test_indices

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

        path_data = []
        paths = {}

        for train_indices, test_indices in self._single_split(single_data):
            path_data.append({
                "Train": np.array(train_indices),
                "Test": test_indices,
            })

        paths['Path 1'] = path_data

        return paths

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

        :return: Dictionary of backtest paths or dictionary of dictionaries for multiple datasets.
        :rtype: Union[
            Dict[str, List[Dict[str, np.ndarray]]],
            Dict[str, Dict[str, List[Dict[str, List[np.ndarray]]]]]
        ]
        """

        if isinstance(data, dict):
            multiple_paths = {}
            for key in data:
                multiple_paths[key] = self._single_backtest_paths(data[key])

            return multiple_paths
        else:

            return self._single_backtest_paths(data)

    def _single_backtest_predictions(
        self,
        single_estimator: Any,
        single_data: pd.DataFrame,
        single_labels: pd.Series,
        single_weights: Optional[np.ndarray] = None,
        predict_probability: bool = False,
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
        :param predict_probability: If True, predict the probability of forecasts.
        :type predict_probability: bool
        :param n_jobs: The number of jobs to run in parallel. Default is 1.
        :type n_jobs: int, optional
        :return: Predictions structured in a dictionary for the backtest paths.
        :rtype: Dict[str, np.ndarray]
        """

        if single_weights is None:
            single_weights = np.ones(len(single_data))

        def train_test_single_estimator(
            single_estimator_: Any,
            train_indices: np.ndarray,
            test_indices: np.ndarray
        ) -> np.ndarray:
            X_train = single_data.iloc[train_indices]
            y_train = single_labels.iloc[train_indices]
            weights_train = single_weights[train_indices]

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=ConvergenceWarning)
                try:
                    single_estimator_.fit(X_train, y_train, sample_weight=weights_train)

                except TypeError:
                    single_estimator_.fit(X_train, y_train)

            X_test = single_data.iloc[test_indices]

            if predict_probability:
                return single_estimator_.predict_proba(X_test)

            else:
                return single_estimator_.predict(X_test)

        path_data = Parallel(n_jobs=n_jobs)(
            delayed(train_test_single_estimator)(
                deepcopy(single_estimator), train_indices, test_indices
            ) for train_indices, test_indices in self._single_split(single_data)
        )
        path_data = np.concatenate(path_data)

        if self.shuffle:
            shuffled_indices = np.arange(single_data.shape[0])
            np.random.default_rng(self.random_seed).shuffle(shuffled_indices)
            # To reorder shuffled_array back to original_array
            reorder_indices = np.argsort(shuffled_indices)
            path_data = path_data[reorder_indices]

        paths_predictions = {'Path 1': path_data}

        return paths_predictions


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
        :param predict_probability: If True, predict the probability of forecasts.
        :type predict_probability: bool
        :param n_jobs: The number of jobs to run in parallel. Default is 1.
        :type n_jobs: int, optional

        :return: Backtest predictions structured in a dictionary (or nested dictionaries for multiple datasets).
        :rtype: Union[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]
        """

        if isinstance(data, dict):
            multiple_paths_predictions = {}
            for key in data:
                multiple_paths_predictions[key] = self._single_backtest_predictions(
                    estimator[key], data[key], labels[key],
                    sample_weights[key] if sample_weights else None, predict_probability, n_jobs
                )
            return multiple_paths_predictions

        return self._single_backtest_predictions(
            estimator, data, labels, sample_weights, predict_probability, n_jobs
        )
