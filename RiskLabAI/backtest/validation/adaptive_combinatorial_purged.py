"""
Implements an Adaptive Combinatorial Purged Cross-Validation (A-CPCV),
which adjusts split boundaries based on an external feature.
"""

import warnings
from collections import ChainMap
from typing import (
    Any, Dict, Generator, List, Optional, Tuple, Union
)

from itertools import combinations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning

from .combinatorial_purged import CombinatorialPurged

# For type hinting sklearn-like estimators
Estimator = Any

class AdaptiveCombinatorialPurged(CombinatorialPurged):
    """
    Adaptive Combinatorial Purged Cross-Validation (A-CPCV).

    This method extends CPCV by making the group splits "adaptive".
    Instead of fixed-size groups, it adjusts the group boundaries
    based on the quantiles of an external feature (e.g., volatility,
    market stress indicator).

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
    n_subsplits : int, default=3
        Number of subsplits within each group to check for boundary
        adjustment. Higher numbers allow for finer adjustments.
    external_feature : pd.Series or Dict[str, pd.Series]
        The external feature used to adjust split boundaries.
        Must share the same index as `data` and `times`.
    lower_quantile : float, default=0.25
        The lower quantile threshold. If a boundary point's
        feature value is below this, the boundary is shifted.
    upper_quantile : float, default=0.75
        The upper quantile threshold. If a boundary point's
        feature value is above this, the boundary is shifted.
    subtract_border_adjustments : bool, default=True
        If True, shifts boundaries left (subtracts) for low
        quantile values and right (adds) for high. If False,
        does the opposite.
    """

    def __init__(
        self,
        n_splits: int,
        n_test_groups: int,
        times: Union[pd.Series, Dict[str, pd.Series]],
        embargo: float = 0,
        n_subsplits: int = 3,
        external_feature: Union[pd.Series, Dict[str, pd.Series]] = None,
        lower_quantile: float = 0.25,
        upper_quantile: float = 0.75,
        subtract_border_adjustments: bool = True
    ):
        """
        Initialize the AdaptiveCombinatorialPurged class.
        """
        super().__init__(n_splits, n_test_groups, times, embargo)
        
        if external_feature is None:
            raise ValueError("external_feature must be provided for A-CPCV")
            
        self.n_subsplits = n_subsplits
        self.external_feature = external_feature
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.subtract_border_adjustments = subtract_border_adjustments
        
        # Check for multiple datasets consistency
        if self.is_multiple_datasets and not isinstance(external_feature, dict):
            raise ValueError(
                "If 'times' is a dict, 'external_feature' must also be a dict."
            )
        if not self.is_multiple_datasets and not isinstance(external_feature, pd.Series):
             raise ValueError(
                "If 'times' is a Series, 'external_feature' must be a Series."
            )

    def _validate_input(
        self,
        single_times: pd.Series,
        single_data: pd.DataFrame,
        single_external_feature: pd.Series
    ) -> None:
        """
        Validate that data, times, and external feature share the same index.
        """
        # Call parent validation
        super()._validate_input(single_times, single_data)
        
        if not single_data.index.equals(single_external_feature.index):
            raise ValueError(
                "Data and external_feature must have the same index"
            )

    def _single_adaptive_split_segments(
        self,
        indices: np.ndarray,
        single_external_feature: pd.Series
    ) -> List[np.ndarray]:
        """
        Adaptively split data indices based on the external feature.

        Parameters
        ----------
        indices : np.ndarray
            Array of data indices to be split.
        single_external_feature : pd.Series
            The external feature for the entire dataset.

        Returns
        -------
        List[np.ndarray]
            List of adaptively split data indices (the `n_splits` groups).
        """
        # Get feature values for the relevant indices
        external_values = single_external_feature.iloc[indices].values

        # Calculate quantile thresholds
        lower_threshold = np.percentile(external_values, self.lower_quantile * 100)
        upper_threshold = np.percentile(external_values, self.upper_quantile * 100)

        # Partition indices into fine-grained subsplits
        n_total_subsplits = self.n_splits * self.n_subsplits
        subsplits = np.array_split(indices, n_total_subsplits)

        # Get the start index of each subsplit
        subsplit_starts_loc = np.array([split[0] for split in subsplits if len(split) > 0])
        
        # Get the feature values at the start of each subsplit
        # We need to map iloc back to the feature series's index
        subsplit_start_indices = single_external_feature.index[subsplit_starts_loc]
        subsplit_start_features = single_external_feature[subsplit_start_indices].values

        # Determine the initial (non-adaptive) borders
        # These are the indices of the *subsplits* that start a new group
        borders = np.arange(self.n_subsplits, n_total_subsplits, self.n_subsplits)

        # Vectorized comparison of external feature values at borders
        border_feature_values = subsplit_start_features[borders]

        # Determine adjustments
        border_adjustments = np.zeros_like(borders)
        border_adjustments[border_feature_values < lower_threshold] = -1
        border_adjustments[border_feature_values > upper_threshold] = 1

        if self.subtract_border_adjustments:
            adjusted_borders = borders - border_adjustments
        else:
            adjusted_borders = borders + border_adjustments

        # Ensure borders are within valid range
        adjusted_borders = np.clip(adjusted_borders, 1, len(subsplit_starts_loc) - 1)
        
        # Get the integer-location split points
        split_points = subsplit_starts_loc[adjusted_borders]

        # Create the final N-split segments
        # We split the original `indices` array at the calculated `split_points`
        # `np.searchsorted` maps the split point values back to their locations in `indices`
        split_point_locs = np.searchsorted(indices, split_points)
        split_segments = np.split(indices, split_point_locs)

        return split_segments

    def _get_split_segments(
        self, 
        single_data: pd.DataFrame,
        single_external_feature: Optional[pd.Series] = None
    ) -> List[np.ndarray]:
        """Override to use adaptive splitting."""
        if single_external_feature is None:
            raise ValueError("_get_split_segments requires external_feature")
        
        indices = np.arange(single_data.shape[0])
        return self._single_adaptive_split_segments(indices, single_external_feature)

    def _single_split(
        self,
        single_times: pd.Series,
        single_data: pd.DataFrame,
        single_external_feature: Optional[pd.Series] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Split a single dataset into C(n, k) adaptively purged indices.
        """
        if single_external_feature is None:
             raise ValueError("_single_split requires single_external_feature")
             
        self._validate_input(
            single_times, single_data, single_external_feature
        )

        split_segments = self._get_split_segments(
            single_data, single_external_feature
        )
        combinations_list = list(
            combinations(range(self.n_splits), self.n_test_groups)
        )

        all_combinatorial_splits = self._combinatorial_splits(
            combinations_list, split_segments
        )

        for test_indices in all_combinatorial_splits:
            train_indices = self._get_train_indices(
                test_indices, single_times, continous_test_times=False
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
        Split data (or dictionary of data) into adaptively purged indices.
        """
        if self.is_multiple_datasets:
            if not isinstance(data, dict):
                raise ValueError("If 'times' is a dict, 'data' must be a dict.")
            for key in self.times:
                for train_indices, test_indices in self._single_split(
                    self.times[key], data[key], self.external_feature[key]
                ):
                    yield key, (train_indices, test_indices)
        else:
            if not isinstance(data, pd.DataFrame):
                raise ValueError("If 'times' is a Series, 'data' must be a DataFrame.")
            for train_indices, test_indices in self._single_split(
                self.times, data, self.external_feature
            ):
                yield train_indices, test_indices

    def _combinations_and_path_locations_and_split_segments(
        self,
        data: pd.DataFrame,
        single_external_feature: Optional[pd.Series] = None
    ) -> Tuple[List[Tuple[int, ...]], Dict[int, List[Tuple[int, int]]], List[np.ndarray]]:
        """Helper to compute all components, now including adaptive splits."""
        if single_external_feature is None:
             raise ValueError("Method requires single_external_feature")

        combinations_list = list(
            combinations(range(self.n_splits), self.n_test_groups)
        )
        locations = self._path_locations(self.n_splits, combinations_list)
        split_segments = self._get_split_segments(data, single_external_feature)

        return combinations_list, locations, split_segments

    def _single_backtest_paths(
            self,
            single_times: pd.Series,
            single_data: pd.DataFrame,
            single_external_feature: Optional[pd.Series] = None
    ) -> Dict[str, List[Dict[str, np.ndarray]]]:
        """
        Generate all adaptive combinatorial backtest paths.
        """
        if single_external_feature is None:
             raise ValueError("Method requires single_external_feature")
             
        self._validate_input(
            single_times, single_data, single_external_feature
        )

        paths = {}
        combinations_list, locations, split_segments = \
            self._combinations_and_path_locations_and_split_segments(
                single_data, single_external_feature
            )
        
        all_combinatorial_splits = list(
            self._combinatorial_splits(combinations_list, split_segments)
        )

        for path_num, locs in locations.items():
            path_data = []
            for (group_idx, split_idx) in locs:
                combinatorial_test_indices = all_combinatorial_splits[split_idx]
                train_indices = self._get_train_indices(
                    combinatorial_test_indices, 
                    single_times, 
                    continous_test_times=False
                )
                test_indices_segment = split_segments[group_idx]

                path_data.append({
                    "Train": train_indices,
                    "Test": test_indices_segment,
                })
            
            paths[f"Path {path_num}"] = path_data
        return paths

    def backtest_paths(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    ) -> Union[
        Dict[str, List[Dict[str, np.ndarray]]],
        Dict[str, Dict[str, List[Dict[str, np.ndarray]]]]
    ]:
        """
        Generate adaptive backtest paths for data or a dictionary of data.
        """
        if self.is_multiple_datasets:
            if not isinstance(data, dict):
                raise ValueError("If 'times' is a dict, 'data' must be a dict.")
            multiple_paths = {}
            for key in self.times:
                multiple_paths[key] = self._single_backtest_paths(
                    self.times[key], data[key], self.external_feature[key]
                )
            return multiple_paths
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("If 'times' is a Series, 'data' must be a DataFrame.")
        return self._single_backtest_paths(
            self.times, data, self.external_feature
        )

    def _single_backtest_predictions(
        self,
        single_estimator: Estimator,
        single_times: pd.Series,
        single_data: pd.DataFrame,
        single_labels: pd.Series,
        single_weights: Optional[np.ndarray] = None,
        single_external_feature: Optional[pd.Series] = None, # New arg
        predict_probability: bool = False,
        n_jobs: int = 1
    ) -> Dict[str, np.ndarray]:
        """
        Obtain backtest predictions for all A-CPCV paths.
        """
        if single_external_feature is None:
             raise ValueError("Method requires single_external_feature")

        self._validate_input(
            single_times, single_data, single_external_feature
        )

        if single_weights is None:
            single_weights = np.ones(len(single_data))

        combinations_list, locations, split_segments = \
            self._combinations_and_path_locations_and_split_segments(
                single_data, single_external_feature
            )

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
                except (TypeError, ValueError):
                    estimator_.fit(X_train, y_train)

            return estimator_

        # 1. Train all C(n, k) estimators in parallel
        combinatorial_trained_estimators = Parallel(n_jobs=n_jobs)(
            delayed(train_single_estimator)(
                clone(single_estimator), test_indices
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
                test_indices_segment = split_segments[group_idx]
                X_test = single_data.iloc[test_indices_segment]
                estimator = combinatorial_trained_estimators[split_idx]

                if predict_probability:
                    preds = estimator.predict_proba(X_test)
                else:
                    preds = estimator.predict(X_test)

                path_predictions.append(preds)
            
            return {f"Path {path_num}": np.concatenate(path_predictions)}

        # 2. Assemble predictions for all paths
        path_results = Parallel(n_jobs=n_jobs)(
            delayed(get_path_data)(path_num, locs)
            for path_num, locs in locations.items()
        )
        
        paths_predictions = dict(ChainMap(*reversed(path_results)))
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
        Generate adaptive backtest predictions.
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
                    s_weights, self.external_feature[key],
                    predict_probability, n_jobs
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
            self.external_feature,
            predict_probability,
            n_jobs
        )