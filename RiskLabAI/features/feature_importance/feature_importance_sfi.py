"""
Computes Single Feature Importance (SFI).
"""

import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import KFold
from typing import List, Optional, Union, Any
from .feature_importance_strategy import FeatureImportanceStrategy

class FeatureImportanceSFI(FeatureImportanceStrategy):
    """
    Computes Single Feature Importance (SFI).

    This method calculates the importance of each feature by training
    and evaluating a model using *only that single feature*.
    """

    def __init__(
        self,
        classifier: object,
        n_splits: int = 10,
        scoring: str = "log_loss",
    ):
        """
        Initialize the strategy.

        Parameters
        ----------
        classifier : object
            An *untrained* scikit-learn classifier.
        n_splits : int, default=10
            Number of splits for cross-validation.
        scoring : str, default="log_loss"
            Scoring method ("log_loss" or "accuracy").
        """
        self.classifier = classifier
        self.n_splits = n_splits
        self.scoring = scoring

    def compute(self, x: pd.DataFrame, y: pd.Series, **kwargs: Any) -> pd.DataFrame:
        """
        Compute SFI feature importance.

        Parameters
        ----------
        x : pd.DataFrame
            The feature data.
        y : pd.Series
            The target data.
        **kwargs : Any
            - 'train_sample_weights': Optional sample weights for training.
            - 'score_sample_weights': Optional sample weights for scoring.

        Returns
        -------
        pd.DataFrame
            DataFrame with "FeatureName", "Mean", and "StandardDeviation"
            of the SFI scores.
        """
        train_sample_weights = kwargs.get('train_sample_weights')
        score_sample_weights = kwargs.get('score_sample_weights')
        
        if train_sample_weights is None:
            train_sample_weights = np.ones(x.shape[0])
        if score_sample_weights is None:
            score_sample_weights = np.ones(x.shape[0])

        cv_generator = KFold(n_splits=self.n_splits)
        feature_names = x.columns
        importances = []

        for feature_name in feature_names:
            scores = []
            feature_data = x[[feature_name]]

            for train_idx, test_idx in cv_generator.split(feature_data):
                x_train, y_train, w_train = (
                    feature_data.iloc[train_idx],
                    y.iloc[train_idx],
                    train_sample_weights[train_idx],
                )
                x_test, y_test, w_test = (
                    feature_data.iloc[test_idx],
                    y.iloc[test_idx],
                    score_sample_weights[test_idx],
                )

                self.classifier.fit(x_train, y_train, sample_weight=w_train)

                if self.scoring == "log_loss":
                    pred_proba = self.classifier.predict_proba(x_test)
                    score = -log_loss(
                        y_test,
                        pred_proba,
                        sample_weight=w_test,
                        labels=self.classifier.classes_,
                    )
                elif self.scoring == "accuracy":
                    pred = self.classifier.predict(x_test)
                    score = accuracy_score(
                        y_test, pred, sample_weight=w_test
                    )
                else:
                    raise ValueError(f"'{self.scoring}' method not defined.")

                scores.append(score)

            importances.append(
                {
                    "FeatureName": feature_name,
                    "Mean": np.mean(scores),
                    "StandardDeviation": np.std(scores, ddof=1)
                    * (len(scores) ** -0.5),
                }
            )

        return pd.DataFrame(importances).set_index("FeatureName")