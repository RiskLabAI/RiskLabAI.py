# from feature_importance_strategy import FeatureImportanceStrategy
from RiskLabAI.features.feature_importance.feature_importance_strategy import FeatureImportanceStrategy
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import KFold
from typing import List, Optional, Union


class FeatureImportanceSFI(FeatureImportanceStrategy):
    """
    Computes the Single Feature Importance (SFI).

    The method calculates the importance of each feature by evaluating its performance 
    individually in the classifier.

    """

    def __init__(
            self,
            classifier: object,
            x: pd.DataFrame,
            y: Union[pd.Series, List[Optional[float]]],
            n_splits: int = 10,
            score_sample_weights: Optional[List[float]] = None,
            train_sample_weights: Optional[List[float]] = None,
            scoring: str = "log_loss"
    ) -> None:
        """
        Initialize the class with parameters.

        :param classifier: The classifier object.
        :param x: The feature data.
        :param y: The target data.
        :param n_splits: The number of splits for cross-validation.
        :param score_sample_weights: Sample weights for scoring.
        :param train_sample_weights: Sample weights for training.
        :param scoring: Scoring method ("log_loss" or "accuracy").
        """
        self.classifier = classifier
        self.features = x
        self.labels = y
        self.n_splits = n_splits
        self.score_sample_weights = score_sample_weights
        self.train_sample_weights = train_sample_weights
        self.scoring = scoring

    def compute(self) -> pd.DataFrame:
        """
        Compute the Single Feature Importance.

        :return: Feature importances as a dataframe with "FeatureName", "Mean", and "StandardDeviation" columns.
        """
        if self.train_sample_weights is None:
            self.train_sample_weights = np.ones(self.features.shape[0])
        if self.score_sample_weights is None:
            self.score_sample_weights = np.ones(self.features.shape[0])

        cv_generator = KFold(n_splits=self.n_splits)
        feature_names = self.features.columns
        importances = []

        for feature_name in feature_names:
            scores = []

            for train, test in cv_generator.split(self.features):
                feature_train, label_train, sample_weights_train = (
                    self.features.loc[train, [feature_name]],
                    self.labels.iloc[train],
                    self.train_sample_weights[train],
                )

                feature_test, label_test, sample_weights_test = (
                    self.features.loc[test, [feature_name]],
                    self.labels.iloc[test],
                    self.score_sample_weights[test],
                )

                self.classifier.fit(feature_train, label_train, sample_weight=sample_weights_train)

                if self.scoring == "log_loss":
                    prediction_probability = self.classifier.predict_proba(feature_test)
                    score = -log_loss(
                        label_test,
                        prediction_probability,
                        sample_weight=sample_weights_test,
                        labels=self.classifier.classes_,
                    )
                elif self.scoring == "accuracy":
                    prediction = self.classifier.predict(feature_test)
                    score = accuracy_score(label_test, prediction, sample_weight=sample_weights_test)
                else:
                    raise ValueError(f"'{self.scoring}' method not defined.")

                scores.append(score)

            importances.append({
                "FeatureName": feature_name,
                "Mean": np.mean(scores),
                "StandardDeviation": np.std(scores, ddof=1) * len(scores) ** -0.5,
            })

        return pd.DataFrame(importances)
