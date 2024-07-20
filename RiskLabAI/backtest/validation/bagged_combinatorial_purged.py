from typing import Union, Dict, Any
import pandas as pd
import numpy as np
from collections import ChainMap
from joblib import Parallel, delayed
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from joblib_progress import joblib_progress
from sklearn.base import clone

from .combinatorial_purged import CombinatorialPurged

class BaggedCombinatorialPurged(CombinatorialPurged):
    def __init__(
        self,
        n_splits: int,
        n_test_groups: int,
        times: Union[pd.Series, Dict[str, pd.Series]],
        embargo: float = 0,
        classifier: bool = True,
        n_estimators: int = 10,
        max_samples: float = 1.0,
        max_features: float = 1.0,
        bootstrap: bool = True,
        bootstrap_features: bool = False,
        random_state: int = None 
    ):
        """
        Initialize the BaggedCombinatorialPurged class.

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
        classifier : bool
            Determines whether to use a BaggingClassifier or BaggingRegressor.
        n_estimators : int
            The number of base estimators in the ensemble.
        max_samples : float
            The number of samples to draw from X to train each base estimator.
        max_features : float
            The number of features to draw from X to train each base estimator.
        bootstrap : bool
            Whether samples are drawn with replacement.
        bootstrap_features : bool
            Whether features are drawn with replacement.
        random_state : int
            The seed used by the random number generator.
        """
        super().__init__(n_splits, n_test_groups, times, embargo)
        self.classifier = classifier
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.random_state = random_state

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
        Generate predictions for a single backtest using combinatorial splits with bagging.

        This method calculates predictions across various paths created by combinatorial splits
        of the data. For each combinatorial split, a bagged estimator is trained and then used
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
            single_weights = pd.Series(np.ones((len(single_data),)))

        paths_predictions = {}
        combinations_, locations, split_segments = self._combinations_and_path_locations_and_split_segments(single_data)

        def train_single_estimator(
            single_estimator_,
            combinatorial_test_indices
        ):
            combinatorial_train_indices = self._get_train_indices(combinatorial_test_indices, single_times)

            X_train = single_data.iloc[combinatorial_train_indices]
            y_train = single_labels.iloc[combinatorial_train_indices]
            weights_train = single_weights.iloc[combinatorial_train_indices]

            if self.classifier:
                bagging_estimator = BaggingClassifier(
                    estimator=single_estimator_,
                    n_estimators=self.n_estimators,
                    max_samples=self.max_samples,
                    max_features=self.max_features,
                    bootstrap=self.bootstrap,
                    bootstrap_features=self.bootstrap_features,
                    random_state=self.random_state
                )
            else:
                bagging_estimator = BaggingRegressor(
                    estimator=single_estimator_,
                    n_estimators=self.n_estimators,
                    max_samples=self.max_samples,
                    max_features=self.max_features,
                    bootstrap=self.bootstrap,
                    bootstrap_features=self.bootstrap_features,
                    random_state=self.random_state
                )

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=ConvergenceWarning)
                try:
                    bagging_estimator.fit(X_train, y_train, sample_weight=weights_train)

                except (TypeError, ValueError):
                    bagging_estimator.fit(X_train, y_train)

            return bagging_estimator

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
