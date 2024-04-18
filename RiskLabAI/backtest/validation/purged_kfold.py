from copy import deepcopy
from typing import Generator, Optional, Tuple, Union, Dict, List, Any, Set
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning

from .cross_validator_interface import CrossValidator

class PurgedKFold(CrossValidator):

    @staticmethod
    def filtered_training_indices_with_embargo(
        data_info_range: pd.Series,
        test_time_range: pd.Series,
        embargo_fraction: float = 0,
        continous_test_times: bool = False,
    ) -> pd.Series:
        """
        Purge observations in the training set with embargo.

        Finds the training set indices based on the information on each record
        and the test set range. It purges the training set of observations that
        overlap with the test set in the time dimension and adds an embargo period
        to further prevent potential information leakage.

        .. math::
            \text{embargo\_length} = \text{len(data\_info\_range)} \times \text{embargo\_fraction}

        :param data_info_range: Series detailing the information range for each record.
            - *data_info_range.index*: Time when the information extraction started.
            - *data_info_range.value*: Time when the information extraction ended.
        :type data_info_range: pd.Series
        :param test_time_range: Series containing times for the test dataset.
        :type test_time_range: pd.Series
        :param embargo_fraction: Fraction of the dataset trailing the test observations to exclude from training.
        :type embargo_fraction: float
        :param continuous_test_times: If set to True, considers the test time range as continuous.
        :type continuous_test_times: bool

        :return: Series of filtered training data after applying embargo.
        :rtype: pd.Series
        """

        indices_to_drop: Set[int] = set()
        embargo_length = int(len(data_info_range) * embargo_fraction)
        sorted_test_time_range = test_time_range.sort_index().copy()

        if not continous_test_times:
            sorted_test_time_range = pd.DataFrame({
                'start' : sorted_test_time_range.index,
                'end' : sorted_test_time_range.values
            })
            # Identify where the new range starts immediately after or before the previous range ends
            gaps = sorted_test_time_range['start'] > sorted_test_time_range['end'].shift(1)
            # Cumulative sum to identify contiguous blocks
            blocks = gaps.cumsum()
            # Aggregate to find the min start and max end for each block
            effective_test_time_range = sorted_test_time_range.groupby(blocks).agg({'start': 'min', 'end': 'max'})
            effective_test_time_range = pd.Series(effective_test_time_range['end'].values, index=effective_test_time_range['start'])

        else:
            effective_test_time_range = pd.Series(sorted_test_time_range.iloc[-1], index=[sorted_test_time_range.index[0]])

        if embargo_length == 0:
            embargoed_data_info_range = pd.Series(effective_test_time_range.values, index=effective_test_time_range.values)

        else:
            effective_sample = data_info_range.loc[effective_test_time_range.index.min():].copy().drop_duplicates()
            embargoed_data_info_range = pd.Series(effective_sample.values, index=effective_sample.values)
            embargoed_data_info_range = embargoed_data_info_range.shift(-embargo_length).fillna(embargoed_data_info_range.values[-1])   

        effective_ranges = pd.Series(embargoed_data_info_range.loc[effective_test_time_range].values, index=effective_test_time_range.index)

        for start_ix, end_ix_embargoed in effective_ranges.items():

            indices_to_drop.update(
                data_info_range[
                    ((start_ix <= data_info_range.index) & (data_info_range.index <= end_ix_embargoed)) |
                    ((start_ix <= data_info_range) & (data_info_range <= end_ix_embargoed)) |
                    ((data_info_range.index <= start_ix) & (end_ix_embargoed <= data_info_range))
                ].index
            )

        return data_info_range.drop(indices_to_drop)


    def __init__(
            self,
            n_splits: int,
            times: Union[pd.Series, Dict[str, pd.Series]],
            embargo: float = 0,
    ) -> None:
        """
        Purged k-fold cross-validation to prevent information leakage.

        Implements a cross-validation strategy where each fold is purged
        of observations overlapping with the training set in the time dimension.
        An embargo period is also introduced to further prevent potential
        information leakage.

        Attributes:
            n_splits (int): Number of splits/folds.
            times (Union[pd.Series, Dict[str, pd.Series]]): Series or dict containing time data.
            embargo (float): The embargo period.
            is_multiple_datasets (bool): True if `times` is a dict, else False.

        :param n_splits: Number of splits or folds.
        :type n_splits: int

        :param times: Series detailing the information range for each record.
            - *times.index*: Time when the information extraction started.
            - *times.value*: Time when the information extraction ended.
        :type times: pd.Series

        :param embargo: The embargo period to further prevent potential
                        information leakage.
        :type embargo: float
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
        return self.n_splits


    def _validate_input(
        self,
        single_times: pd.Series,
        single_data: pd.DataFrame
    ) -> None:
        """
        Validate that the input data and times share the same index.

        This function checks if the provided data and its corresponding times
        have the same index. If they do not match, it raises a `ValueError`.

        :param single_times: Time series data to be validated.
        :type single_times: pd.Series
        :param single_data: Dataset with which the times should align.
        :type single_data: pd.DataFrame
        :raises ValueError: If the indices of the data and times do not match.
        :return: None
        """
        if not single_data.index.equals(single_times.index):
            raise ValueError('Data and through date values must have the same index')


    def _get_train_indices(
        self,
        test_indices: np.ndarray,
        single_times: pd.Series,
        continous_test_times: bool = False,
    ) -> np.ndarray:
        """
        Obtain the training indices considering purging and embargo.

        This function retrieves the training set indices based on the given test indices
        while considering the purging and embargo strategy.

        :param test_indices: Indices used for the test set.
        :type test_indices: np.ndarray
        :param single_times: Time series data used for purging and embargo.
        :type single_times: pd.Series
        :return: Training indices after applying purging and embargo.
        :rtype: np.ndarray
        """
        test_time_range = single_times.iloc[test_indices]
        train_times = self.filtered_training_indices_with_embargo(
            single_times, test_time_range, self.embargo, continous_test_times
        )
        return pd.Series(train_times.index).map(single_times.index.get_loc).tolist()


    def _single_split(
        self,
        single_times: pd.Series,
        single_data: pd.DataFrame,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Split the data into train and test sets.

        This function splits the data for a single dataset considering purging and embargo.

        :param single_times: Time series data used for purging and embargo.
        :type single_times: pd.Series
        :param single_data: Dataset to split.
        :type single_data: pd.DataFrame
        :return: Train and test indices.
        :rtype: Generator[Tuple[np.ndarray, np.ndarray], None, None]
        """
        self._validate_input(single_times, single_data)

        indices = np.arange(len(single_data))

        for test_indices in np.array_split(indices, self.n_splits):
            train_indices = self._get_train_indices(test_indices, single_times, True)
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
                    self.times[key], data[key]
                ):
                    yield key, (train_indices, test_indices)
        else:
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
        Generate backtest paths based on training and testing indices.

        This function first validates the input data and times. Then, it generates
        the training and testing indices for backtesting. These paths are organized
        into a dictionary with a designated name for each backtest path.

        :param single_times: Time series data for validation.
        :type single_times: pd.Series
        :param single_data: Dataset with which the times should align.
        :type single_data: pd.DataFrame
        :return: Dictionary containing the backtest paths with training and testing indices.
        :rtype: Dict[str, List[Dict[str, np.ndarray]]]
        """
        self._validate_input(single_times, single_data)

        path_data = []
        paths = {}

        for train_indices, test_indices in self._single_split(single_times, single_data):
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
                multiple_paths[key] = self._single_backtest_paths(self.times[key], data[key])

            return multiple_paths
        else:

            return self._single_backtest_paths(self.times, data)


    def _single_backtest_predictions(
        self,
        single_estimator: Any,
        single_times: pd.Series,
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
        :param single_times: Timestamps for the single dataset.
        :type single_times: pd.Series
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
        self._validate_input(single_times, single_data)

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
            ) for train_indices, test_indices in self._single_split(single_times, single_data)
        )

        paths_predictions = {'Path 1': np.concatenate(path_data)}

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
                    sample_weights[key] if sample_weights else None, predict_probability, n_jobs
                )
            return multiple_paths_predictions

        return self._single_backtest_predictions(
            estimator, self.times, data, labels, sample_weights, predict_probability, n_jobs
        )
