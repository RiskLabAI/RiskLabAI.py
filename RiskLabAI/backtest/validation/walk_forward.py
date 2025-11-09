"""
Implements a Walk-Forward cross-validator for time-series data.
"""

import warnings
from copy import deepcopy
from typing import (
    Any, Dict, Generator, Optional, Tuple
)

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning

from .kfold import KFold

# For type hinting sklearn-like estimators
Estimator = Any

class WalkForward(KFold):
    """
    Walk-Forward Cross-Validator for Time Series Data.

    This cross-validator splits time-series data in a "walk-forward"
    manner. In each split, the training set progressively grows (up to
    `max_train_size`) while the test set "walks" forward.

    This method respects the temporal order of data, ensuring the model is
    always trained on past data and validated on future data.

    Parameters
    ----------
    n_splits : int, default=5
        Number of splits/folds. Must be at least 2.
    max_train_size : int, optional
        Maximum number of observations allowed in the training dataset.
        If provided, the most recent `max_train_size` observations are
        used for training.
    gap : int, default=0
        Number of observations to skip between the end of the training
        data and the start of the test data.
    """

    def __init__(
        self,
        n_splits: int = 5,
        max_train_size: Optional[int] = None,
        gap: int = 0
    ) -> None:
        """
        Initialize the WalkForward cross-validator.

        Parameters
        ----------
        n_splits : int, default=5
            Number of splits/folds. Must be at least 2.
        max_train_size : int, optional
            Maximum number of observations allowed in the training dataset.
        gap : int, default=0
            Number of observations to skip between the end of the training
            data and the start of the test data.
        """
        # WalkForward cannot be shuffled
        super().__init__(n_splits=n_splits, shuffle=False)
        self.max_train_size = max_train_size
        self.gap = gap

    def _single_split(
        self,
        single_data: pd.DataFrame,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Split a single dataset into walk-forward train-test indices.

        Parameters
        ----------
        single_data : pd.DataFrame
            Input dataset.

        Yields
        -------
        Generator[Tuple[np.ndarray, np.ndarray], None, None]
            A generator where each item is a tuple of (train_indices, test_indices).
        """
        indices = np.arange(single_data.shape[0])
        
        # np.array_split handles non-divisible splits
        for test_indices in np.array_split(indices, self.n_splits):
            # The first test index
            first_test_idx_loc = test_indices[0]
            
            # Train indices end `gap` samples before the test set
            train_end_loc = first_test_idx_loc - self.gap
            
            if train_end_loc < 0:
                # No training data possible
                train_indices = np.array([], dtype=int)
            else:
                if self.max_train_size and train_end_loc > self.max_train_size:
                    train_start_loc = train_end_loc - self.max_train_size
                    train_indices = indices[train_start_loc:train_end_loc]
                else:
                    train_indices = indices[:train_end_loc]
            
            yield train_indices, test_indices

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
            array of out-of-sample predictions.
        """
        if single_weights is None:
            single_weights = np.ones(len(single_data))

        def train_test_single_estimator(
            estimator_: Estimator,
            train_indices: np.ndarray,
            test_indices: np.ndarray
        ) -> np.ndarray:
            """Train model and return predictions."""
            
            if len(train_indices) == 0:
                # No training data, return NaNs
                n_classes = len(np.unique(single_labels)) if predict_probability else 1
                shape = (len(test_indices), n_classes) if predict_probability else (len(test_indices),)
                return np.full(shape, np.nan)

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
            ) for train_indices, test_indices in self._single_split(single_data)
        )

        # Since shuffle=False, we can just concatenate
        paths_predictions = {'Path 1': np.concatenate(path_data)}
        return paths_predictions