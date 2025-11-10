"""
Implements a standard K-Fold cross-validator.
"""

import warnings
from copy import deepcopy
from typing import (
    Any, Dict, Generator, List, Optional, Tuple, Union
)

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning

from .cross_validator_interface import CrossValidator

# For type hinting sklearn-like estimators
Estimator = Any

class KFold(CrossValidator):
    """
    K-Fold cross-validator.

    This class implements the standard K-Fold cross-validation strategy, where
    the dataset is divided into `k` consecutive folds. Each fold is then
    used once as a validation set while the `k - 1` remaining folds
    form the training set.

    This implementation allows for optional shuffling.
    """

    def __init__(
        self,
        n_splits: int,
        shuffle: bool = False,
        random_seed: int = None
    ) -> None:
        """
        Initialize the K-Fold cross-validator.

        Parameters
        ----------
        n_splits : int
            Number of splits or folds for the cross-validation.
            The dataset will be divided into `n_splits` consecutive parts.
        shuffle : bool, default=False
            Whether to shuffle the data before splitting it into folds.
        random_seed : int, optional
            Seed used for random shuffling. Only used when `shuffle` is True.
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.rng_ = np.random.default_rng(random_seed)

    def get_n_splits(
        self,
        data: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]] = None,
        labels: Optional[Union[pd.Series, Dict[str, pd.Series]]] = None,
        groups: Optional[np.ndarray] = None
    ) -> int:
        """
        Return the number of splits.

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
            The number of splits (`n_splits`).
        """
        return self.n_splits

    def _get_shuffled_indices(self, n_samples: int) -> np.ndarray:
        """Helper to get shuffled or sequential indices."""
        indices = np.arange(n_samples)
        if self.shuffle:
            self.rng_.shuffle(indices)
        return indices

    def _single_split(
        self,
        single_data: pd.DataFrame,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Split a single dataset into train-test indices.

        Parameters
        ----------
        single_data : pd.DataFrame
            Input dataset.

        Yields
        -------
        Generator[Tuple[np.ndarray, np.ndarray], None, None]
            A generator where each item is a tuple of (train_indices, test_indices).
        """
        indices = self._get_shuffled_indices(single_data.shape[0])

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
        Split data (or dictionary of data) into train-test indices.

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
    ) -> Dict[str, List[Dict[str, np.ndarray]]]:
        """
        Generate backtest paths for a single dataset.
        For K-Fold, there is only one "path".

        Parameters
        ----------
        single_data : pd.DataFrame
            Input dataset.

        Returns
        -------
        Dict[str, List[Dict[str, np.ndarray]]]
            A dictionary with one key "Path 1", containing a list of all
            (Train, Test) index dictionaries.
        """
        path_data = []
        paths = {}

        for train_indices, test_indices in self._single_split(single_data):
            path_data.append({
                "Train": train_indices,
                "Test": test_indices,
            })

        paths['Path 1'] = path_data
        return paths

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
        if isinstance(data, dict):
            multiple_paths = {}
            for key in data:
                multiple_paths[key] = self._single_backtest_paths(data[key])
            return multiple_paths
        
        return self._single_backtest_paths(data)

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
            A dictionary with one key "Path 1", containing the contiguous
            array of out-of-sample predictions, ordered by the original
            dataset index.
        """
        if single_weights is None:
            single_weights = np.ones(len(single_data))

        def train_test_single_estimator(
            estimator_: Estimator,
            train_indices: np.ndarray,
            test_indices: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
            """Train model and return (predictions, test_indices)."""
            X_train = single_data.iloc[train_indices]
            y_train = single_labels.iloc[train_indices]
            weights_train = single_weights[train_indices]

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=ConvergenceWarning)
                try:
                    estimator_.fit(X_train, y_train, sample_weight=weights_train)
                except TypeError:
                    # Fallback for estimators without sample_weight
                    estimator_.fit(X_train, y_train)

            X_test = single_data.iloc[test_indices]

            if predict_probability:
                preds = estimator_.predict_proba(X_test)
            else:
                preds = estimator_.predict(X_test)

            return preds, test_indices

        # Run all folds in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(train_test_single_estimator)(
                deepcopy(single_estimator), train_indices, test_indices
            ) for train_indices, test_indices in self._single_split(single_data)
        )

        # Unpack predictions and their corresponding original indices
        all_preds = [res[0] for res in results]
        all_indices = [res[1] for res in results]

        # Concatenate results
        path_data = np.concatenate(all_preds)
        original_indices = np.concatenate(all_indices)

        # Re-order predictions to match the original dataset order
        # This is critical if shuffle=True
        if self.shuffle:
            reorder_indices = np.argsort(original_indices)
            path_data = path_data[reorder_indices]

        paths_predictions = {'Path 1': path_data}
        return paths_predictions

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
            Model(s) to be trained.
        data : pd.DataFrame or dict
            Input data.
        labels : pd.Series or dict
            Target labels.
        sample_weights : np.ndarray or dict, optional
            Sample weights. Defaults to None.
        predict_probability : bool, default=False
            If True, call `predict_proba()` instead of `predict()`.
        n_jobs : int, default=1
            The number of jobs to run in parallel.

        Returns
        -------
        Union[Dict, Dict[str, Dict]]
            - If data is a pd.DataFrame: A dictionary of predictions.
            - If data is a dict: A nested dictionary of predictions.
        """
        if isinstance(data, dict):
            if not isinstance(estimator, dict) or \
               not isinstance(labels, dict) or \
               (sample_weights and not isinstance(sample_weights, dict)):
                raise ValueError(
                    "If data is a dict, estimator, labels, "
                    "and sample_weights (if provided) must also be dicts."
                )
            
            multiple_paths_predictions = {}
            for key in data:
                multiple_paths_predictions[key] = self._single_backtest_predictions(
                    estimator[key],
                    data[key],
                    labels[key],
                    sample_weights[key] if sample_weights else None,
                    predict_probability,
                    n_jobs
                )
            return multiple_paths_predictions

        # Handle single dataset case
        if not (isinstance(data, pd.DataFrame) and 
                isinstance(labels, pd.Series)):
            raise ValueError(
                "If data is a DataFrame, estimator must be a single estimator "
                "and labels must be a single Series."
            )

        return self._single_backtest_predictions(
            estimator,
            data,
            labels,
            sample_weights,
            predict_probability,
            n_jobs
        )