"""
Implements Combinatorial Purged Cross-Validation (CPCV) as described by
Marcos Lopez de Prado.
"""

import warnings
from collections import ChainMap, defaultdict
from copy import deepcopy
from itertools import combinations
from math import comb
from typing import (
    Any, Dict, Generator, List, Optional, Tuple, Union
)

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning

from .purged_kfold import PurgedKFold

# For type hinting sklearn-like estimators
Estimator = Any

class CombinatorialPurged(PurgedKFold):
    """
    Combinatorial Purged Cross-Validation (CPCV).

    This method generates multiple backtest paths by creating all possible
    combinations of train/test splits, given `n_splits` groups and
    `n_test_groups` for the test set size.

    It inherits from `PurgedKFold` to ensure that all generated training
    sets are properly purged and embargoed against their corresponding
    test set.

    Parameters
    ----------
    n_splits : int
        Total number of groups to partition the data into.
    n_test_groups : int
        The number of groups to use for testing in each combination.
        e.g., n_splits=8, n_test_groups=2 -> C(8, 2) = 28 splits.
    times : pd.Series or Dict[str, pd.Series]
        A Series where the index is the observation start time and
        the value is the observation end time.
    embargo : float, default=0
        The embargo fraction (e.g., 0.01 for 1%).
    """

    @staticmethod
    def _path_locations(
        n_splits: int,
        combinations_list: List[Tuple[int, ...]]
    ) -> Dict[int, List[Tuple[int, int]]]:
        """
        Generate a labeled path matrix to map splits to backtest paths.

        Parameters
        ----------
        n_splits : int
            Total number of groups.
        combinations_list : List[Tuple[int, ...]]
            The list of all C(n_splits, n_test_groups) combinations.

        Returns
        -------
        Dict[int, List[Tuple[int, int]]]
            A dictionary mapping each path ID (int) to a list of its
            coordinates (group, split_num) in the path matrix.
        """
        n_combinations = len(combinations_list)
        
        # Initialize a zero matrix
        matrix = np.zeros((n_splits, n_combinations), dtype=int)

        # Populate matrix: 1 if group `i` is in combination `j`
        for j, indices in enumerate(combinations_list):
            matrix[indices, j] = 1

        # Label path numbers
        def label_path_row(row: np.ndarray) -> np.ndarray:
            """Assigns a path number to each test split in a group."""
            path_counter = 1
            labeled_row = np.zeros_like(row)
            for i, val in enumerate(row):
                if val == 1:
                    labeled_row[i] = path_counter
                    path_counter += 1
            return labeled_row

        path_numbers = np.apply_along_axis(label_path_row, 1, matrix)

        # Extract path locations
        locations = defaultdict(list)
        for group_idx in range(n_splits):
            for split_idx in range(n_combinations):
                path_id = path_numbers[group_idx, split_idx]
                if path_id != 0:
                    locations[path_id].append((group_idx, split_idx))

        return dict(locations)

    @staticmethod
    def _combinatorial_splits(
        combinations_list: List[Tuple[int, ...]],
        split_segments: List[np.ndarray]
    ) -> Generator[np.ndarray, None, None]:
        """
        Generate combinatorial test sets.

        Parameters
        ----------
        combinations_list : List[Tuple[int, ...]]
            List of all C(n_splits, n_test_groups) combinations.
        split_segments : List[np.ndarray]
            List of index arrays, one for each of the `n_splits` groups.

        Yields
        -------
        Generator[np.ndarray, None, None]
            A generator yielding the concatenated test set indices for
            each combination.
        """
        for test_groups in combinations_list:
            test_sets = [
                split for i, split in enumerate(split_segments) 
                if i in test_groups
            ]
            yield np.concatenate(test_sets)

    def __init__(
        self,
        n_splits: int,
        n_test_groups: int,
        times: Union[pd.Series, Dict[str, pd.Series]],
        embargo: float = 0
    ) -> None:
        """
        Initialize the CombinatorialPurged class.

        Parameters
        ----------
        n_splits : int
            Number of splits/groups to partition the data into.
        n_test_groups : int
            Size of the testing set in terms of groups.
        times : pd.Series or Dict[str, pd.Series]
            The timestamp series associated with the labels.
        embargo : float, default=0
            The embargo rate for purging.
        """
        super().__init__(n_splits, times, embargo)
        if n_test_groups >= n_splits:
            raise ValueError(
                "n_test_groups must be strictly less than n_splits"
            )
        self.n_test_groups = n_test_groups

    def get_n_splits(
        self,
        data: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]] = None,
        labels: Optional[Union[pd.Series, Dict[str, pd.Series]]] = None,
        groups: Optional[np.ndarray] = None
    ) -> int:
        """
        Return the total number of combinatorial splits.

        This is C(n_splits, n_test_groups).

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
            The total number of splits, C(n_splits, n_test_groups).
        """
        return comb(self.n_splits, self.n_test_groups)

    def _get_split_segments(self, single_data: pd.DataFrame) -> List[np.ndarray]:
        """Helper to get the base K-Fold segments."""
        indices = np.arange(single_data.shape[0])
        return np.array_split(indices, self.n_splits)

    def _single_split(
        self,
        single_times: pd.Series,
        single_data: pd.DataFrame,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Split a single dataset into C(n, k) purged train-test indices.

        Parameters
        ----------
        single_times : pd.Series
            The 'times' (info range) Series for this dataset.
        single_data : pd.DataFrame
            Input dataset.

        Yields
        -------
        Generator[Tuple[np.ndarray, np.ndarray], None, None]
            A generator of (train_indices, test_indices) for all
            C(n_splits, n_test_groups) combinations.
        """
        self._validate_input(single_times, single_data)

        split_segments = self._get_split_segments(single_data)
        combinations_list = list(
            combinations(range(self.n_splits), self.n_test_groups)
        )

        all_combinatorial_splits = self._combinatorial_splits(
            combinations_list, split_segments
        )

        for test_indices in all_combinatorial_splits:
            # Purge against the *non-contiguous* test set
            train_indices = self._get_train_indices(
                test_indices, single_times, continous_test_times=False
            )
            yield train_indices, test_indices

    def _combinations_and_path_locations_and_split_segments(
        self,
        data: pd.DataFrame
    ) -> Tuple[List[Tuple[int, ...]], Dict[int, List[Tuple[int, int]]], List[np.ndarray]]:
        """
        Helper to compute all necessary components for CPCV.

        Parameters
        ----------
        data : pd.DataFrame
            Input dataset.

        Returns
        -------
        Tuple[List, Dict, List]
            - combinations_list: List of all C(n, k) combinations.
            - locations: The path location dictionary from `_path_locations`.
            - split_segments: List of index arrays for each of the `n_splits` groups.
        """
        combinations_list = list(
            combinations(range(self.n_splits), self.n_test_groups)
        )
        locations = self._path_locations(self.n_splits, combinations_list)
        split_segments = self._get_split_segments(data)

        return combinations_list, locations, split_segments

    def _single_backtest_paths(
            self,
            single_times: pd.Series,
            single_data: pd.DataFrame,
    ) -> Dict[str, List[Dict[str, np.ndarray]]]:
        """
        Generate all combinatorial backtest paths for a single dataset.

        Parameters
        ----------
        single_times : pd.Series
            The 'times' (info range) Series for this dataset.
        single_data : pd.DataFrame
            Input dataset.

        Returns
        -------
        Dict[str, List[Dict[str, np.ndarray]]]
            A dictionary where each key is a "Path {id}" and the value
            is the list of (Train, Test) splits for that path.
        """
        self._validate_input(single_times, single_data)

        paths = {}
        combinations_list, locations, split_segments = \
            self._combinations_and_path_locations_and_split_segments(single_data)
        
        all_combinatorial_splits = list(
            self._combinatorial_splits(combinations_list, split_segments)
        )

        for path_num, locs in locations.items():
            path_data = []
            for (group_idx, split_idx) in locs:
                # Get the full test set for this *combination*
                combinatorial_test_indices = all_combinatorial_splits[split_idx]
                
                # Get the train set purged against this *combination*
                train_indices = self._get_train_indices(
                    combinatorial_test_indices, 
                    single_times, 
                    continous_test_times=False
                )
                
                # The test set for this *path segment* is just one group
                test_indices_segment = split_segments[group_idx]

                path_data.append({
                    "Train": train_indices,
                    "Test": test_indices_segment,
                })
            
            # This path is complete
            paths[f"Path {path_num}"] = path_data

        return paths

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
        Obtain backtest predictions for all CPCV paths.

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
            A dictionary where keys are "Path {id}" and values are
            the contiguous arrays of out-of-sample predictions for each path.
        """
        self._validate_input(single_times, single_data)

        if single_weights is None:
            single_weights = np.ones(len(single_data))

        combinations_list, locations, split_segments = \
            self._combinations_and_path_locations_and_split_segments(single_data)

        def train_single_estimator(
            estimator_: Estimator,
            combinatorial_test_indices: np.ndarray
        ) -> Estimator:
            """Train one estimator for one C(n,k) split."""
            train_indices = self._get_train_indices(
                combinatorial_test_indices,
                single_times,
                continous_test_times=False
            )

            X_train = single_data.iloc[train_indices]
            y_train = single_labels.iloc[train_indices]
            weights_train = single_weights[train_indices]

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=ConvergenceWarning)
                try:
                    estimator_.fit(X_train, y_train, sample_weight=weights_train)
                except TypeError:
                    estimator_.fit(X_train, y_train)

            return estimator_

        # 1. Train all C(n, k) estimators in parallel
        combinatorial_trained_estimators = Parallel(n_jobs=n_jobs)(
            delayed(train_single_estimator)(
                deepcopy(single_estimator), test_indices
            )
            for test_indices in self._combinatorial_splits(
                combinations_list, split_segments
            )
        )

        def get_path_data(
            path_num: int,
            locs: List[Tuple[int, int]]
        ) -> Dict[str, np.ndarray]:
            """Assemble predictions for one path."""
            path_predictions = []

            for (group_idx, split_idx) in locs:
                # Get the test segment for this path
                test_indices_segment = split_segments[group_idx]
                X_test = single_data.iloc[test_indices_segment]
                
                # Get the estimator trained for this split
                estimator = combinatorial_trained_estimators[split_idx]

                if predict_probability:
                    preds = estimator.predict_proba(X_test)
                else:
                    preds = estimator.predict(X_test)

                path_predictions.append(preds)
            
            return {f"Path {path_num}": np.concatenate(path_predictions)}

        # 2. Assemble predictions for all paths in parallel
        path_results = Parallel(n_jobs=n_jobs)(
            delayed(get_path_data)(path_num, locs)
            for path_num, locs in locations.items()
        )
        
        # Combine list of dicts into one dict
        paths_predictions = dict(ChainMap(*reversed(path_results)))
        return paths_predictions