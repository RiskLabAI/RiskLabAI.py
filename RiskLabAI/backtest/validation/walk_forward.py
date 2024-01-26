import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from typing import Dict, Tuple, Generator, Any, Optional
from copy import deepcopy
from joblib import Parallel, delayed
import warnings
from sklearn.exceptions import ConvergenceWarning
from .kfold import KFold

class WalkForward(KFold):
    """
    WalkForward Cross-Validator for Time Series Data.

    This cross-validator provides train/test indices meant to split time series data 
    in a "walk-forward" manner, which is suitable for time series forecasting tasks. 
    In each split, the training set progressively grows in size (subject to the optional
    maximum size constraint) while the test set remains roughly constant in size. 
    A gap can be optionally introduced between the training and test set to simulate 
    forecasting on unseen future data after a certain interval.

    The WalkForward cross-validator is inherently different from traditional K-Fold
    cross-validation which shuffles and splits the dataset into train/test without 
    considering the time order. In time series tasks, ensuring that the model is 
    trained on past data and validated on future data is crucial. This cross-validator 
    achieves that by progressively walking forward in time through the dataset.
    """

    def __init__(
        self,
        n_splits: int = 5,
        max_train_size: int = None,
        gap: int = 0
    ) -> None:
        """
        Initialize the TimeSeriesWalkForward cross-validator.

        Parameters:
        -----------
        n_splits : int, default=5
            Number of splits/folds. Must be at least 2.
        
        max_train_size : int, optional
            Maximum number of observations allowed in the training dataset.
            If provided, the most recent `max_train_size` observations are used 
            for training.

        gap : int, default=0
            Number of observations to skip between the end of the training data 
            and the start of the test data. Useful for simulating forecasting 
            scenarios where the test data is not immediately after the training data.
        """
        super().__init__(n_splits=n_splits)  # Initializing the parent class with n_splits
        self.max_train_size = max_train_size
        self.gap = gap


    def _single_split(
        self,
        single_data: pd.DataFrame,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Splits a single data set into train-test indices.

        This function provides train-test indices to split the data into train/test sets
        by respecting the time order (if applicable) and the specified number of splits.

        :param single_data: Input dataset.
        :type single_data: pd.DataFrame

        :return: Generator yielding train-test indices.
        :rtype: Generator[Tuple[np.ndarray, np.ndarray], None, None]
        """

        indices = np.arange(single_data.shape[0])
        gap = self.gap
        max_train_size = self.max_train_size

        for test_indices in np.array_split(indices, self.n_splits):
            # Ensure that train indices come before test indices
            last_test_index = test_indices[-1]
            train_indices = np.setdiff1d(indices, test_indices)
            train_indices = train_indices[train_indices < last_test_index]
            
            # Exclude gap from the end of train set
            if self.gap:
                train_indices = train_indices[:-gap]
            
            # Ensuring the training set does not exceed max_train_size
            if self.max_train_size and len(train_indices) > max_train_size:
                train_indices = train_indices[-max_train_size:]
            
            yield train_indices, test_indices    


    def _single_backtest_predictions(
        self,
        single_estimator: Any,
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

        if single_weights is None:
            single_weights = np.ones(len(single_data))

        def train_test_single_estimator(
            single_estimator_: Any,
            train_indices: np.ndarray,
            test_indices: np.ndarray
        ) -> np.ndarray:
            
            if len(train_indices) == 0:
                if predict_probability:
                    n_columns = len(np.unique(single_labels)) if predict_probability else 1
                    return np.full((len(test_indices), n_columns), np.nan)
        
                else:
                    return np.full(len(test_indices), np.nan)
            
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
            ) for train_indices, test_indices in self._single_split(single_data)
        )

        paths_predictions = {'Path 1': np.concatenate(path_data)}

        return paths_predictions        
