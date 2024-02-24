# from feature_importance_strategy import FeatureImportanceStrategy
from RiskLabAI.features.feature_importance.feature_importance_strategy import FeatureImportanceStrategy
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from typing import List, Optional


class FeatureImportanceMDA(FeatureImportanceStrategy):
    """
    Computes the feature importance using the Mean Decrease Accuracy (MDA) method.

    The method shuffles each feature one by one and measures how much the performance 
    (log loss in this context) decreases due to the shuffling.

    .. math::

        \\text{importance}_{j} = \\frac{\\text{score without shuffling} - \\text{score with shuffling}_{j}}
        {\\text{score without shuffling}}

    """

    def __init__(
            self,
            classifier: object,
            x: pd.DataFrame,
            y: pd.Series,
            n_splits: int = 10,
            score_sample_weights: Optional[List[float]] = None,
            train_sample_weights: Optional[List[float]] = None
    ) -> None:
        """
        Initialize the class with parameters.

        :param classifier: The classifier object.
        :param x: The feature data.
        :param y: The target data.
        :param n_splits: Number of splits for cross-validation.
        :param score_sample_weights: Weights for scoring samples.
        :param train_sample_weights: Weights for training samples.
        """
        self.classifier = classifier
        self.x = x
        self.y = y
        self.n_splits = n_splits
        self.score_sample_weights = score_sample_weights
        self.train_sample_weights = train_sample_weights

    def compute(self) -> pd.DataFrame:
        """
        Compute the feature importances.

        :return: Feature importances as a dataframe with "Mean" and "StandardDeviation" columns.
        """
        if self.train_sample_weights is None:
            self.train_sample_weights = np.ones(self.x.shape[0])
        if self.score_sample_weights is None:
            self.score_sample_weights = np.ones(self.x.shape[0])

        cv_generator = KFold(n_splits=self.n_splits)
        initial_scores, shuffled_scores = pd.Series(dtype=float), pd.DataFrame(columns=self.x.columns)

        for i, (train, test) in enumerate(cv_generator.split(self.x)):
            print(f"Fold {i} start ...")

            x_train, y_train, weights_train = self.x.iloc[train, :], self.y.iloc[train], self.train_sample_weights[train]
            x_test, y_test, weights_test = self.x.iloc[test, :], self.y.iloc[test], self.score_sample_weights[test]

            fitted_classifier = self.classifier.fit(X=x_train, y=y_train, sample_weight=weights_train)
            prediction_probability = fitted_classifier.predict_proba(x_test)

            initial_scores.loc[i] = -log_loss(
                y_test,
                prediction_probability,
                labels=self.classifier.classes_,
                sample_weight=weights_test
            )

            for feature in self.x.columns:
                x_test_shuffled = x_test.copy(deep=True)
                np.random.shuffle(x_test_shuffled[feature].values)
                shuffled_proba = fitted_classifier.predict_proba(x_test_shuffled)
                shuffled_scores.loc[i, feature] = -log_loss(y_test, shuffled_proba, labels=self.classifier.classes_)

        importances = (-1 * shuffled_scores).add(initial_scores, axis=0)
        importances /= (-1 * shuffled_scores)

        importances = pd.concat({
            "Mean": importances.mean(),
            "StandardDeviation": importances.std() * importances.shape[0]**-0.5
        }, axis=1)

        return importances
