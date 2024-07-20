from typing import Generator, Optional, Tuple, Union, Dict, List, Any
from itertools import combinations
import pandas as pd
import numpy as np
from collections import ChainMap
from joblib import Parallel, delayed
import warnings
from sklearn.exceptions import ConvergenceWarning
from joblib_progress import joblib_progress
from sklearn.base import clone

from .combinatorial_purged import CombinatorialPurged

class AdaptiveCombinatorialPurged(CombinatorialPurged):
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

        Parameters
        ----------
        n_splits : int
            Number of splits/groups to partition the data into.
        n_test_groups : int
            Size of the testing set in terms of groups.
        times : Union[pd.Series, Dict[str, pd.Series]]
            The timestamp series associated with the labels.
        embargo : float
            The embargo rate for purging.
        n_subsplits : int
            Number of subsplits within each split segment.
        external_feature : Union[pd.Series, Dict[str, pd.Series]]
            The external feature based on which the adaptive splitting is performed.
        lower_quantile : float
            The lower quantile threshold for adjusting the split segments.
        upper_quantile : float
            The upper quantile threshold for adjusting the split segments.
        subtract_border_adjustments : bool
            Flag to determine whether to subtract border adjustments instead of adding.
        """
        super().__init__(n_splits, n_test_groups, times, embargo)
        self.n_subsplits = n_subsplits
        self.external_feature = external_feature
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.subtract_border_adjustments = subtract_border_adjustments


    def _validate_input(
        self,
        single_times: pd.Series,
        single_data: pd.DataFrame,
        single_external_feature: pd.Series
    ) -> None:
        """
        Validate that the input data, times, and external feature share the same index.

        This function checks if the provided data, times, and external feature have the same index.
        If they do not match, it raises a `ValueError`.

        :param single_times: Time series data to be validated.
        :type single_times: pd.Series
        :param single_data: Dataset with which the times should align.
        :type single_data: pd.DataFrame
        :param single_external_feature: External feature series to be validated.
        :type single_external_feature: pd.Series
        :raises ValueError: If the indices of the data, times, and external feature do not match.
        :return: None
        """
        if not single_data.index.equals(single_times.index) or not single_data.index.equals(single_external_feature.index):
            raise ValueError('Data, through date values, and external feature must have the same index')
        

    def _single_adaptive_split_segments(
        self, 
        indices: np.ndarray, 
        single_external_feature: pd.Series
    ) -> List[np.ndarray]:
        """
        Adaptively split data indices based on the external feature's values and quantile thresholds.

        Parameters
        ----------
        indices : np.ndarray
            Array of data indices to be split.
        single_external_feature : pd.Series
            The external feature based on which the adaptive splitting is performed.

        Returns
        -------
        split_segments : List[np.ndarray]
            List of adaptively split data indices.
        """
        external_values = single_external_feature.iloc[indices].values

        # Calculate the quantile thresholds
        lower_threshold = np.percentile(external_values, self.lower_quantile * 100)
        upper_threshold = np.percentile(external_values, self.upper_quantile * 100)

        # Partition each group into n_subsplits
        subsplits = np.array_split(indices, self.n_splits * self.n_subsplits)

        # Get the starting points of each subsplit
        subsplit_starts = np.array([split[0] for split in subsplits])

        # Determine the initial borders between subsplits
        borders = np.arange(self.n_subsplits, len(subsplit_starts), self.n_subsplits)

        # Vectorized comparison of external feature values at borders
        border_values = external_values[subsplit_starts[borders]]

        # Determine adjustments based on quantiles
        border_adjustments = np.zeros_like(borders)
        border_adjustments[border_values < lower_threshold] = -1
        border_adjustments[border_values > upper_threshold] = 1

        # Adjust the borders based on the subtract_border_adjustments flag
        if self.subtract_border_adjustments:
            adjusted_borders = borders - border_adjustments
        else:
            adjusted_borders = borders + border_adjustments

        # Ensure the borders are within valid range
        adjusted_borders = np.clip(adjusted_borders, 1, len(subsplit_starts) - 1)

        # Create the final split segments
        split_segments = np.split(indices, adjusted_borders)

        return split_segments
    

    def _single_split(
        self,
        single_times: pd.Series,
        single_data: pd.DataFrame,
        single_external_feature: pd.Series
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Splits data into train and test indices based on the defined combinatorial splits.

        This function is used to generate multiple train-test splits based on the combinatorial
        cross-validation method. It ensures that each train-test split is properly purged and
        embargoed to prevent data leakage.

        :param single_times: Timestamp series associated with the labels.
        :param single_data: The input data to be split.
        :param single_external_feature: External feature series used for adaptive splitting.

        :return: Generator that yields tuples of (train indices, test indices).

        .. note:: The function validates the input, and uses combinatorial cross-validation method to
                produce the train-test splits.
        """
        self._validate_input(single_times, single_data, single_external_feature)

        indices = np.arange(single_data.shape[0])
        split_segments = self._single_adaptive_split_segments(indices, single_external_feature)
        combinations_ = list(combinations(range(self.n_splits), self.n_test_groups))

        all_combinatorial_splits = list(
            CombinatorialPurged._combinatorial_splits(combinations_, split_segments)
        )

        for test_indices in all_combinatorial_splits:
            train_indices = self._get_train_indices(test_indices, single_times)
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
        Split multiple datasets into train and test sets.

        This function either splits a single dataset or multiple datasets considering
        purging and embargo.

        :param data: Dataset or dictionary of datasets.
        :type data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
        :param labels: Labels corresponding to the datasets, if available.
        :type labels: Optional[Union[pd.Series, Dict[str, pd.Series]]]
        :param groups: Group information, if available.
        :type groups: Optional[np.ndarray]
        :return: Train and test indices or key with train and test indices for multiple datasets.
        :rtype: Union[Generator[Tuple[np.ndarray, np.ndarray], None, None],
                    Generator[Tuple[str, Tuple[np.ndarray, np.ndarray]], None, None]]
        """
        if self.is_multiple_datasets:
            for key in self.times:
                for train_indices, test_indices in self._single_split(
                    self.times[key], data[key], self.external_feature[key]
                ):
                    yield key, (train_indices, test_indices)
        else:
            for train_indices, test_indices in self._single_split(
                self.times, data, self.external_feature
            ):
                yield train_indices, test_indices        


    def _combinations_and_path_locations_and_split_segments(
        self,
        data: pd.DataFrame,
        single_external_feature: pd.Series
    ) -> Tuple[List[Tuple[int, ...]], Dict[str, np.ndarray], List[np.ndarray]]:
        """
        Generate combinations, path locations, and split segments for the data.

        This function is a helper that computes necessary components for combinatorial cross-validation.

        :param data: The input dataframe to generate combinations, path locations, and split segments.
        :param single_external_feature: External feature series used for adaptive splitting.

        :return: Tuple containing combinations, path locations, and split segments.

        .. math::
        \\text{combinations} = \\binom{n}{k}
        """
        combinations_ = list(combinations(range(self.n_splits), self.n_test_groups))
        locations = CombinatorialPurged._path_locations(self.n_splits, combinations_)

        indices = np.arange(data.shape[0])
        split_segments = self._single_adaptive_split_segments(indices, single_external_feature)

        return combinations_, locations, split_segments

    def _single_backtest_paths(
        self,
        single_times: pd.Series,
        single_data: pd.DataFrame,
        single_external_feature: pd.Series
    ) -> Dict[str, List[Dict[str, List[np.ndarray]]]]:
        """
        Generate the backtest paths for given input data.

        This function creates multiple backtest paths based on combinatorial splits, where
        each path represents a sequence of train-test splits. It ensures that data leakage
        is prevented by purging and applying embargo to the train-test splits.

        :param single_times: Timestamp series associated with the data.
        :param single_data: Input data on which the backtest paths are based.
        :param single_external_feature: External feature series used for adaptive splitting.

        :return: A dictionary where each key is a backtest path name, and the value is
                a list of dictionaries with train and test index arrays.

        .. note:: This function relies on combinatorial cross-validation for backtesting to
                generate multiple paths of train-test splits.
        """
        self._validate_input(single_times, single_data, single_external_feature)

        paths = {}
        combinations_, locations, split_segments = self._combinations_and_path_locations_and_split_segments(single_data, single_external_feature)
        all_combinatorial_splits = list(
            CombinatorialPurged._combinatorial_splits(combinations_, split_segments)
        )

        for path_num, locs in locations.items():
            path_data = []

            for (G, S) in locs:
                # Use all_combinatorial_splits[S] to determine potential training indices
                # Now, we derive the actual training set after purge and embargo
                train_indices = self._get_train_indices(all_combinatorial_splits[S], single_times)

                path_data.append({
                    "Train": np.array(train_indices),
                    "Test": split_segments[G],
                })

            paths[f"Path {path_num}"] = np.array(path_data)

        return paths
    

    def backtest_paths(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    ) -> Union[
        Dict[str, List[Dict[str, np.ndarray]]],
        Dict[str, Dict[str, List[Dict[str, List[np.ndarray]]]]]
    ]:
        """
        Generate backtest paths for single or multiple datasets.

        This function checks whether multiple datasets are being used. If so, it iterates through each
        dataset, generating backtest paths using the `_single_backtest_paths` method. Otherwise, it directly
        returns the backtest paths for the single dataset.

        :param data: Input data on which the backtest paths are based.
                    Can be either a single DataFrame or a dictionary of DataFrames for multiple datasets.
        :type data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]

        :return: A dictionary where each key is a backtest path name, and the value is
                a list of dictionaries with train and test index arrays. For multiple datasets,
                a nested dictionary structure is returned.
        :rtype: Union[
            Dict[str, List[Dict[str, np.ndarray]]],
            Dict[str, Dict[str, List[Dict[str, List[np.ndarray]]]]]
        ]

        """
        if self.is_multiple_datasets:
            multiple_paths = {}
            for key in self.times:
                multiple_paths[key] = self._single_backtest_paths(
                    self.times[key], data[key], self.external_feature[key]
                )

            return multiple_paths
        else:
            return self._single_backtest_paths(self.times, data, self.external_feature)


    def _single_backtest_predictions(
        self,
        single_estimator: Any,
        single_times: pd.Series,
        single_data: pd.DataFrame,
        single_labels: pd.Series,
        single_weights: np.ndarray,
        single_external_feature: pd.Series,
        predict_probability: bool = False,
        n_jobs: int = 1
    ) -> Dict[str, np.ndarray]:
        """
        Generate predictions for a single backtest using combinatorial splits.

        This method calculates predictions across various paths created by combinatorial splits
        of the data. For each combinatorial split, a separate estimator is trained and then used
        to predict on the corresponding test set.

        :param single_estimator: The machine learning model or estimator to be trained.
        :param single_times: Timestamps corresponding to the data points.
        :param single_data: Input data on which the model is trained and predictions are made.
        :param single_labels: Labels corresponding to the input data.
        :param single_weights: Weights for each data point.
        :param single_external_feature: External feature series used for adaptive splitting.
        :param predict_probability: If True, predict the probability of forecasts.
        :type predict_probability: bool
        :param n_jobs: Number of CPU cores to use for parallelization. Default is 1.

        :return: A dictionary where keys are path names and values are arrays of predictions.

        .. note:: This function relies on internal methods (e.g., `_get_train_indices`)
                to manage data splits and training.

        .. note:: Parallelization is used to speed up the training of models for different splits.
        """
        self._validate_input(single_times, single_data, single_external_feature)

        if single_weights is None:
            single_weights = pd.Series(np.ones((len(single_data),)))

        paths_predictions = {}
        combinations_, locations, split_segments = self._combinations_and_path_locations_and_split_segments(single_data, single_external_feature)

        def train_single_estimator(
            single_estimator_,
            combinatorial_test_indices
        ):
            combinatorial_train_indices = self._get_train_indices(combinatorial_test_indices, single_times)

            X_train = single_data.iloc[combinatorial_train_indices]
            y_train = single_labels.iloc[combinatorial_train_indices]
            weights_train = single_weights.iloc[combinatorial_train_indices]

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=ConvergenceWarning)
                try:
                    single_estimator_.fit(X_train, y_train, sample_weight=weights_train)

                except (TypeError, ValueError):
                    single_estimator_.fit(X_train, y_train)

            return single_estimator_

        # with joblib_progress("Backtesting...", total=self.get_n_splits()): 
        combinatorial_trained_estimators = Parallel(n_jobs=n_jobs)(
            delayed(train_single_estimator)(clone(single_estimator), combinatorial_test_indices)
            for combinatorial_test_indices in CombinatorialPurged._combinatorial_splits(combinations_, split_segments))


        def get_path_data(
            path_num,
            locs
        ):
            path_data = []

            for (G, S) in locs:
                test_indices = split_segments[G]
                X_test = single_data.iloc[test_indices]

                if predict_probability:
                    predictions = combinatorial_trained_estimators[S].predict_proba(X_test)

                else:
                    predictions = combinatorial_trained_estimators[S].predict(X_test)

                path_data.extend(predictions)

            path_data = {f"Path {path_num}" : np.array(path_data)}

            return path_data

        paths_predictions = Parallel(n_jobs=n_jobs)(delayed(get_path_data)(path_num, locs) for path_num, locs in locations.items())
        paths_predictions = dict(ChainMap(*reversed(paths_predictions)))

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
        if self.is_multiple_datasets:
            multiple_paths_predictions = {}
            for key in self.times:
                multiple_paths_predictions[key] = self._single_backtest_predictions(
                    estimator[key], self.times[key], data[key], labels[key],
                    sample_weights[key] if sample_weights else None, self.external_feature[key],
                    predict_probability, n_jobs
                )
            return multiple_paths_predictions

        return self._single_backtest_predictions(
            estimator, self.times, data, labels, sample_weights, self.external_feature, predict_probability, n_jobs
        )
