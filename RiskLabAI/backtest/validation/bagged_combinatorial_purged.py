"""
Implements Bagged Combinatorial Purged Cross-Validation (B-CPCV).
"""

import warnings
from collections import ChainMap
from typing import (
    Any, Dict, Optional, Union
)

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.exceptions import ConvergenceWarning

from typing import List, Tuple

from .combinatorial_purged import CombinatorialPurged

# For type hinting sklearn-like estimators
Estimator = Any

class BaggedCombinatorialPurged(CombinatorialPurged):
    """
    Bagged Combinatorial Purged Cross-Validation (B-CPCV).

    This class extends CPCV by applying a bagging (Bootstrap Aggregating)
    wrapper to the estimator for each combinatorial split. This helps
    to reduce variance and improve model stability.

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
    classifier : bool, default=True
        If True, use `sklearn.ensemble.BaggingClassifier`.
        If False, use `sklearn.ensemble.BaggingRegressor`.
    n_estimators : int, default=10
        The number of base estimators in the ensemble.
    max_samples : float, default=1.0
        The number of samples to draw to train each base estimator.
    max_features : float, default=1.0
        The number of features to draw to train each base estimator.
    bootstrap : bool, default=True
        Whether samples are drawn with replacement.
    bootstrap_features : bool, default=False
        Whether features are drawn with replacement.
    random_state : int, optional
        Controls the randomness of the bootstrapping.
    """

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
        single_estimator: Estimator,
        single_times: pd.Series,
        single_data: pd.DataFrame,
        single_labels: pd.Series,
        single_weights: Optional[np.ndarray] = None,
        predict_probability: bool = False,
        n_jobs: int = 1
    ) -> Dict[str, np.ndarray]:
        """
        Obtain backtest predictions for all B-CPCV paths.

        This method overrides the parent by wrapping the `single_estimator`
        in a `BaggingClassifier` or `BaggingRegressor` before training.

        Parameters
        ----------
        single_estimator : Estimator
            The *base* estimator to be used inside the bagging ensemble.
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
            Only valid if `classifier=True`.
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
            
        if predict_probability and not self.classifier:
            raise ValueError(
                "Cannot use predict_probability=True when classifier=False"
            )

        combinations_list, locations, split_segments = \
            self._combinations_and_path_locations_and_split_segments(single_data)

        def train_single_bagging_estimator(
            base_estimator_: Estimator,
            combinatorial_test_indices: np.ndarray
        ) -> Estimator:
            """Train one Bagging estimator for one C(n,k) split."""
            train_indices = self._get_train_indices(
                combinatorial_test_indices,
                single_times,
                continous_test_times=False
            )

            X_train = single_data.iloc[train_indices]
            y_train = single_labels.iloc[train_indices]
            weights_train = single_weights[train_indices]

            # --- Bagging Wrapper ---
            if self.classifier:
                bagging_estimator = BaggingClassifier(
                    estimator=base_estimator_,
                    n_estimators=self.n_estimators,
                    max_samples=self.max_samples,
                    max_features=self.max_features,
                    bootstrap=self.bootstrap,
                    bootstrap_features=self.bootstrap_features,
                    random_state=self.random_state,
                    n_jobs=n_jobs  # Parallelize bagging itself
                )
            else:
                bagging_estimator = BaggingRegressor(
                    estimator=base_estimator_,
                    n_estimators=self.n_estimators,
                    max_samples=self.max_samples,
                    max_features=self.max_features,
                    bootstrap=self.bootstrap,
                    bootstrap_features=self.bootstrap_features,
                    random_state=self.random_state,
                    n_jobs=n_jobs # Parallelize bagging itself
                )
            # --- End Bagging Wrapper ---

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=ConvergenceWarning)
                try:
                    bagging_estimator.fit(X_train, y_train, sample_weight=weights_train)
                except (TypeError, ValueError):
                    # Fallback for models without sample_weight
                    bagging_estimator.fit(X_train, y_train)

            return bagging_estimator

        # 1. Train all C(n, k) *bagging* estimators in parallel
        # Note: We set n_jobs=1 for the *outer* parallel loop
        # and let the *inner* bagging estimator use `n_jobs`.
        # This avoids nested parallelization issues.
        
        # Determine parallelization strategy
        if n_jobs > 1 or n_jobs == -1:
            outer_n_jobs = n_jobs
            inner_n_jobs = 1 
            
            # Update bagging params to use inner_n_jobs
            if self.classifier:
                BaggingClassifier.__init__ = (
                    lambda self, **kwargs: 
                    super(BaggingClassifier, self).__init__(**kwargs, n_jobs=inner_n_jobs)
                )
            else:
                BaggingRegressor.__init__ = (
                    lambda self, **kwargs: 
                    super(BaggingRegressor, self).__init__(**kwargs, n_jobs=inner_n_jobs)
                )
        else:
            # Let bagging use all cores if outer loop is serial
            outer_n_jobs = 1
            inner_n_jobs = -1 # Use all
        
        combinatorial_trained_estimators = Parallel(n_jobs=outer_n_jobs)(
            delayed(train_single_bagging_estimator)(
                clone(single_estimator), test_indices
            )
            for test_indices in self._combinatorial_splits(
                combinations_list, split_segments
            )
        )
        
        # Restore default constructors
        BaggingClassifier.__init__ = BaggingClassifier.__init__
        BaggingRegressor.__init__ = BaggingRegressor.__init__


        # 2. Assemble predictions (this is fast, can be serial or parallel)
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

        path_results = Parallel(n_jobs=outer_n_jobs)(
            delayed(get_path_data)(path_num, locs)
            for path_num, locs in locations.items()
        )
        
        paths_predictions = dict(ChainMap(*reversed(path_results)))
        return paths_predictions