"""
Implements a Purged K-Fold cross-validator to prevent information leakage
in financial time-series data.
"""

import warnings
from copy import deepcopy
from typing import (
    Any, Dict, Generator, List, Optional, Set, Tuple, Union
)

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning

from .cross_validator_interface import CrossValidator

# For type hinting sklearn-like estimators
Estimator = Any

class PurgedKFold(CrossValidator):
    """
    Purged K-Fold cross-validator.

    Implements a K-Fold strategy where each fold is purged of observations
    that overlap in time with the test set. An "embargo" period can also be
    added after the test set to prevent information leakage from the training
    set immediately following the test set.

    This is essential for financial data where labels are defined over a
    future period (e.g., T+k returns) and simple K-Fold would leak
    information.

    Parameters
    ----------
    n_splits : int
        Number of splits or folds.
    times : pd.Series or Dict[str, pd.Series]
        A Series where the index is the time the observation starts and
        the values are the time the observation ends. This defines the
        "information range" of each observation.
    embargo : float, default=0
        The embargo fraction. A fraction of the dataset length (e.g., 0.01)
        to be removed *after* the end of the test set information range.
    """

    @staticmethod
    def filtered_training_indices_with_embargo(
        data_info_range: pd.Series,
        test_time_range: pd.Series,
        embargo_fraction: float = 0,
        continous_test_times: bool = False,
    ) -> pd.Series:
        r"""
        Purge observations in the training set with embargo.

        Finds training set indices that do not overlap with the test set's
        information range.

        .. math::
            \text{embargo\_length} = \text{len(data\_info_range)} \times \text{embargo\_fraction}

        Parameters
        ----------
        data_info_range : pd.Series
            Series detailing the information range for *all* data.
            - Index: Time when the information extraction started.
            - Value: Time when the information extraction ended.
        test_time_range : pd.Series
            Series detailing the information range for *test set* data.
            - Index: Time when the information extraction started.
            - Value: Time when the information extraction ended.
        embargo_fraction : float, default=0
            Fraction of the dataset trailing the test observations to
            exclude from training.
        continous_test_times : bool, default=False
            If True, considers the test time range as one continuous block
            from the earliest start to the latest end.

        Returns
        -------
        pd.Series
            A view of `data_info_range` containing only the filtered
            training data (i.e., observations that do not overlap with
            test + embargo).
        """
        indices_to_drop: Set[int] = set()
        embargo_length = int(len(data_info_range) * embargo_fraction)
        
        if test_time_range.empty:
            return data_info_range

        sorted_test_time_range = test_time_range.sort_index().copy()

        if not continous_test_times:
            # Create blocks of contiguous test ranges
            test_ranges = pd.DataFrame({
                'start': sorted_test_time_range.index,
                'end': sorted_test_time_range.values
            })
            gaps = test_ranges['start'] > test_ranges['end'].shift(1)
            blocks = gaps.cumsum()
            effective_test_time_range = test_ranges.groupby(blocks).agg(
                {'start': 'min', 'end': 'max'}
            )
            effective_test_time_range = pd.Series(
                effective_test_time_range['end'].values,
                index=effective_test_time_range['start']
            )
        else:
            effective_test_time_range = pd.Series(
                sorted_test_time_range.values[-1],
                index=[sorted_test_time_range.index[0]]
            )

        if embargo_length == 0:
            embargoed_ranges = effective_test_time_range
        else:
            embargoed_values = []
            for end_val in effective_test_time_range.values:
                end_iloc = data_info_range.index.searchsorted(end_val, side='left')

                if end_iloc >= len(data_info_range):
                    embargoed_values.append(end_val)
                else:
                    embargoed_iloc = min(end_iloc + embargo_length, len(data_info_range) - 1)
                    embargoed_values.append(data_info_range.index[embargoed_iloc])

            embargoed_ranges = pd.Series(
                embargoed_values, 
                index=effective_test_time_range.index
            )
            # === END OF FIX ===

        # Purge
        for test_start, test_end_embargoed in embargoed_ranges.items():
            # 1. Overlap: train starts during test/embargo
            cond1 = (data_info_range.index >= test_start) & \
                    (data_info_range.index <= test_end_embargoed)
            # 2. Overlap: train ends during test/embargo
            cond2 = (data_info_range.values >= test_start) & \
                    (data_info_range.values <= test_end_embargoed)
            # 3. Overlap: train envelops test/embargo
            cond3 = (data_info_range.index <= test_start) & \
                    (data_info_range.values >= test_end_embargoed)

            indices_to_drop.update(data_info_range[cond1 | cond2 | cond3].index)

        return data_info_range.drop(indices_to_drop)

    def __init__(
            self,
            n_splits: int,
            times: Union[pd.Series, Dict[str, pd.Series]],
            embargo: float = 0,
    ) -> None:
        """
        Initialize the PurgedKFold cross-validator.

        Parameters
        ----------
        n_splits : int
            Number of splits or folds.
        times : pd.Series or Dict[str, pd.Series]
            A Series (or dict of Series) where the index is the observation
            start time and the value is the observation end time.
        embargo : float, default=0
            The embargo fraction (e.g., 0.01 for 1%).
        """
        self.n_splits = n_splits
        self.times = times
        self.embargo = embargo
        self.is_multiple_datasets = isinstance(times, dict)

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

    def _validate_input(
        self,
        single_times: pd.Series,
        single_data: pd.DataFrame
    ) -> None:
        """Validate that data and times indices match."""
        if not single_data.index.equals(single_times.index):
            raise ValueError(
                "Data and 'times' (info range) must have the same index"
            )

    def _get_train_indices(
        self,
        test_indices: np.ndarray,
        single_times: pd.Series,
        continous_test_times: bool = False,
    ) -> np.ndarray:
        """
        Get training indices after purging and embargo.

        Parameters
        ----------
        test_indices : np.ndarray
            The integer-location indices of the test set.
        single_times : pd.Series
            The 'times' (info range) Series for *all* data.
        continous_test_times : bool, default=False
            Passed to `filtered_training_indices_with_embargo`.

        Returns
        -------
        np.ndarray
            The integer-location indices of the training set.
        """
        if len(test_indices) == 0:
            return np.arange(len(single_times))
            
        test_time_range = single_times.iloc[test_indices]
        
        train_times = self.filtered_training_indices_with_embargo(
            single_times, test_time_range, self.embargo, continous_test_times
        )
        
        # Convert index labels back to integer locations
        train_indices = single_times.index.get_indexer(train_times.index)
        return train_indices

    def _single_split(
        self,
        single_times: pd.Series,
        single_data: pd.DataFrame,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Split a single dataset into purged train-test indices.

        Parameters
        ----------
        single_times : pd.Series
            The 'times' (info range) Series for this dataset.
        single_data : pd.DataFrame
            Input dataset.

        Yields
        -------
        Generator[Tuple[np.ndarray, np.ndarray], None, None]
            A generator of (train_indices, test_indices).
        """
        self._validate_input(single_times, single_data)

        indices = np.arange(len(single_data))
        
        for test_indices in np.array_split(indices, self.n_splits):
            train_indices = self._get_train_indices(
                test_indices, single_times, True
            )
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
        Split data (or dictionary of data) into purged train-test indices.

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
        if self.is_multiple_datasets:
            if not isinstance(data, dict):
                raise ValueError("If 'times' is a dict, 'data' must also be a dict.")
            for key in self.times:
                for train_indices, test_indices in self._single_split(
                    self.times[key], data[key]
                ):
                    yield key, (train_indices, test_indices)
        else:
            if not isinstance(data, pd.DataFrame):
                raise ValueError("If 'times' is a Series, 'data' must be a DataFrame.")
            for train_indices, test_indices in self._single_split(
                self.times, data
            ):
                yield train_indices, test_indices

    def _single_backtest_paths(
        self,
        single_times: pd.Series,
        single_data: pd.DataFrame,
    ) -> Dict[str, List[Dict[str, np.ndarray]]]:
        """
        Generate backtest paths for a single dataset.
        For PurgedKFold, there is only one "path".

        Parameters
        ----------
        single_times : pd.Series
            The 'times' (info range) Series for this dataset.
        single_data : pd.DataFrame
            Input dataset.

        Returns
        -------
        Dict[str, List[Dict[str, np.ndarray]]]
            A dictionary with one key "Path 1", containing a list of all
            (Train, Test) index dictionaries.
        """
        self._validate_input(single_times, single_data)

        path_data = []
        paths = {}

        for train_indices, test_indices in self._single_split(
            single_times, single_data
        ):
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
            - If data is a dict: A nested dictionary of backtest paths.
        """
        if self.is_multiple_datasets:
            if not isinstance(data, dict):
                raise ValueError("If 'times' is a dict, 'data' must also be a dict.")
            multiple_paths = {}
            for key in self.times:
                multiple_paths[key] = self._single_backtest_paths(
                    self.times[key], data[key]
                )
            return multiple_paths
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("If 'times' is a Series, 'data' must be a DataFrame.")
        return self._single_backtest_paths(self.times, data)

    def _single_backtest_predictions(
        self,
        single_estimator: Estimator,
        single_times: pd.Series,
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
            A scikit-learn-like estimator.
        single_times : pd.Series
            The 'times' (info range) Series for this dataset.
        single_data : pd.DataFrame
            Data for the single dataset.
        single_labels : pd.Series
            Labels for the single dataset.
        single_weights : np.ndarray, optional
            Sample weights. Defaults to equal weights.
        predict_probability : bool, default=False
            If True, call `predict_proba()` instead of `predict()`.
        n_jobs : int, default=1
            The number of jobs to run in parallel.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary with one key "Path 1", containing the contiguous
            array of out-of-sample predictions.
        """
        self._validate_input(single_times, single_data)

        if single_weights is None:
            single_weights = np.ones(len(single_data))

        def train_test_single_estimator(
            estimator_: Estimator,
            train_indices: np.ndarray,
            test_indices: np.ndarray
        ) -> np.ndarray:
            """Train model and return predictions."""
            X_train = single_data.iloc[train_indices]
            y_train = single_labels.iloc[train_indices]
            weights_train = single_weights[train_indices]

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=ConvergenceWarning)
                try:
                    estimator_.fit(X_train, y_train, sample_weight=weights_train)
                except TypeError:
                    estimator_.fit(X_train, y_train)

            X_test = single_data.iloc[test_indices]

            if predict_probability:
                return estimator_.predict_proba(X_test)
            
            return estimator_.predict(X_test)

        path_data = Parallel(n_jobs=n_jobs)(
            delayed(train_test_single_estimator)(
                deepcopy(single_estimator), train_indices, test_indices
            ) for train_indices, test_indices in self._single_split(
                single_times, single_data
            )
        )

        # Since PurgedKFold is not shuffled, we can just concatenate
        paths_predictions = {'Path 1': np.concatenate(path_data)}
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
        if self.is_multiple_datasets:
            if not (isinstance(data, dict) and 
                    isinstance(estimator, dict) and 
                    isinstance(labels, dict)):
                raise ValueError(
                    "If 'times' is a dict, 'data', 'estimator', and 'labels' "
                    "must also be dicts."
                )
            
            multiple_paths_predictions = {}
            for key in self.times:
                s_weights = sample_weights[key] if sample_weights else None
                multiple_paths_predictions[key] = self._single_backtest_predictions(
                    estimator[key], self.times[key], data[key], labels[key],
                    s_weights, predict_probability, n_jobs
                )
            return multiple_paths_predictions

        # Handle single dataset case
        if not (isinstance(data, pd.DataFrame) and 
                isinstance(labels, pd.Series)):
             raise ValueError(
                "If 'times' is a Series, 'data' must be a DataFrame "
                "and 'labels' must be a Series."
            )

        return self._single_backtest_predictions(
            estimator,
            self.times,
            data,
            labels,
            sample_weights,
            predict_probability,
            n_jobs
        )