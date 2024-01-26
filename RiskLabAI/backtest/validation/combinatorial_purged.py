from copy import deepcopy
from typing import Generator, Optional, Tuple, Union, Dict, List, Any
from itertools import combinations
import pandas as pd
import numpy as np
from collections import ChainMap, defaultdict
from joblib import Parallel, delayed
from math import comb
import warnings
from sklearn.exceptions import ConvergenceWarning

from .purged_kfold import PurgedKFold

class CombinatorialPurged(PurgedKFold):
    """
    Combinatorial Purged Cross-Validation (CPCV) implementation based on Marcos Lopez de Prado's method.

    This class provides a cross-validation scheme that aims to address the main drawback of the Walk Forward
    and traditional Cross-Validation methods by testing multiple paths. Given a number of backtest paths,
    CPCV generates the precise number of combinations of training/testing sets needed to generate those paths,
    while purging training observations that might contain leaked information.

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
    """


    @staticmethod
    def _path_locations(
        n_splits: int,
        combinations_: List[Tuple[int]]
    ) -> Dict[int, List[Tuple[int, int]]]:
        """
        Generate a labeled path matrix and return path locations for N choose K.

        This method generates a matrix where each entry corresponds to a specific combination of
        training/testing sets, and helps in mapping these combinations to specific backtest paths.

        Parameters
        ----------
        n_splits : int
            Number of splits/groups to partition the data into.
        combinations_ : list
            List of combinations for training/testing sets.

        Returns
        -------
        dict
            A dictionary mapping each backtest path to its corresponding train/test combination.
        """

        # Initialize a zero matrix
        matrix = np.zeros((n_splits, len(combinations_)), dtype=int)

        # Set appropriate entries to 1 and label path numbers in the same loop
        for col, indices in enumerate(combinations_):
            matrix[indices, col] = 1

        # Label path numbers
        def label_path(row):
            counter = iter(range(1, row.sum() + 1))
            return [next(counter) if x == 1 else 0 for x in row]

        path_numbers = np.array(list(map(label_path, matrix)))

        # Extract path locations using loops
        def map_to_location(coord):
            i, j = coord
            value = path_numbers[i, j]
            locations[value].append((i, j))

        rows, cols = path_numbers.shape
        locations = defaultdict(list)
        list(map(map_to_location, [(x, y) for x in range(rows) for y in range(cols) if path_numbers[x, y] != 0]))

        return dict(locations)


    @staticmethod
    def _combinatorial_splits(
        combinations_: List[Tuple[int]],
        split_segments: np.ndarray
    ) -> Generator[np.ndarray, None, None]:
        """
        Generate combinatorial test sets based on the number of test groups (n_test_groups).

        This method creates test sets by considering all possible combinations of group splits, allowing
        for the creation of multiple test paths, as described in the CPCV methodology.

        Parameters
        ----------
        combinations_ : list
            List of combinations for training/testing sets.
        split_segments : np.ndarray
            Array of data split segments.

        Returns
        -------
        Generator[np.ndarray]
            A generator yielding the combinatorial test sets.
        """

        for test_groups in combinations_:
            test_sets = [split for i, split in enumerate(split_segments) if i in test_groups]
            yield np.concatenate(test_sets)


    def __init__(
        self,
        n_splits: int,
        n_test_groups: int,
        times: Union[pd.Series, Dict[str, pd.Series]],
        embargo: float = 0
    ):
        """
        Initialize the CombinatorialPurged class.

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
        """

        super().__init__(n_splits, times, embargo)
        self.n_test_groups = n_test_groups

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
        return comb(self.n_splits, self.n_test_groups)

    def _single_split(
        self,
        single_times: np.ndarray,
        single_data: np.ndarray,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Splits data into train and test indices based on the defined combinatorial splits.

        This function is used to generate multiple train-test splits based on the combinatorial
        cross-validation method. It ensures that each train-test split is properly purged and
        embargoed to prevent data leakage.

        :param single_times: Timestamp series associated with the labels.
        :param single_data: The input data to be split.

        :return: Generator that yields tuples of (train indices, test indices).

        .. note:: The function validates the input, and uses combinatorial cross-validation method to
                produce the train-test splits.
        """

        self._validate_input(single_times, single_data)

        indices = np.arange(single_data.shape[0])
        split_segments = np.array_split(indices, self.n_splits)
        combinations_ = list(combinations(range(self.n_splits), self.n_test_groups))

        all_combinatorial_splits = list(
            CombinatorialPurged._combinatorial_splits(combinations_, split_segments)
        )

        for test_indices in all_combinatorial_splits:
            train_indices = self._get_train_indices(test_indices, single_times)
            yield train_indices, test_indices


    def _combinations_and_path_locations_and_split_segments(
        self,
        data: pd.DataFrame
    ) -> Tuple[List[Tuple[int, ...]], Dict[str, np.ndarray], List[np.ndarray]]:
        """
        Generate combinations, path locations, and split segments for the data.

        This function is a helper that computes necessary components for combinatorial cross-validation.

        :param data: The input dataframe to generate combinations, path locations, and split segments.

        :return: Tuple containing combinations, path locations, and split segments.

        .. math::
        \\text{combinations} = \\binom{n}{k}
        """

        combinations_ = list(combinations(range(self.n_splits), self.n_test_groups))
        locations = CombinatorialPurged._path_locations(self.n_splits, combinations_)

        indices = np.arange(data.shape[0])
        split_segments = np.array_split(indices, self.n_splits)

        return combinations_, locations, split_segments


    def _single_backtest_paths(
            self,
            single_times: pd.Series,
            single_data: pd.DataFrame,
    ) -> Dict[str, List[Dict[str, List[np.ndarray]]]]:
        """
        Generate the backtest paths for given input data.

        This function creates multiple backtest paths based on combinatorial splits, where
        each path represents a sequence of train-test splits. It ensures that data leakage
        is prevented by purging and applying embargo to the train-test splits.

        :param single_times: Timestamp series associated with the data.
        :param single_data: Input data on which the backtest paths are based.

        :return: A dictionary where each key is a backtest path name, and the value is
                a list of dictionaries with train and test index arrays.

        .. note:: This function relies on combinatorial cross-validation for backtesting to
                generate multiple paths of train-test splits.
        """

        self._validate_input(single_times, single_data)

        paths = {}
        combinations_, locations, split_segments = self._combinations_and_path_locations_and_split_segments(single_data)
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


    def _single_backtest_predictions(
        self,
        single_estimator: Any,
        single_times: pd.Series,
        single_data: pd.DataFrame,
        single_labels: pd.Series,
        single_weights: np.ndarray,
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
        :param predict_probability: If True, predict the probability of forecasts.
        :type predict_probability: bool
        :param n_jobs: Number of CPU cores to use for parallelization. Default is 1.

        :return: A dictionary where keys are path names and values are arrays of predictions.

        .. note:: This function relies on internal methods (e.g., `_get_train_indices`)
                to manage data splits and training.

        .. note:: Parallelization is used to speed up the training of models for different splits.
        """

        self._validate_input(single_times, single_data)

        if single_weights is None:
            single_weights = np.ones((len(single_data),))

        paths_predictions = {}
        combinations_, locations, split_segments = self._combinations_and_path_locations_and_split_segments(single_data)

        def train_single_estimator(
            single_estimator_,
            combinatorial_test_indices
        ):
            combinatorial_train_indices = self._get_train_indices(combinatorial_test_indices, single_times)

            X_train = single_data.iloc[combinatorial_train_indices]
            y_train = single_labels.iloc[combinatorial_train_indices]
            weights_train = single_weights[combinatorial_train_indices]

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=ConvergenceWarning)
                try:
                    single_estimator_.fit(X_train, y_train, sample_weight=weights_train)

                except TypeError:
                    single_estimator_.fit(X_train, y_train)

            return single_estimator_

        combinatorial_trained_estimators = Parallel(n_jobs=n_jobs)(
            delayed(train_single_estimator)(deepcopy(single_estimator), combinatorial_test_indices)
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
