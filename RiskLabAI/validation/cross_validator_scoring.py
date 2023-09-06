from sklearn.base import ClassifierMixin
from sklearn.metrics import log_loss, accuracy_score
import pandas as pd
import numpy as np
from custom_cv import CrossValidationStrategy  # Replace with the actual import

def calculate_cross_validation_score(
        classifier: ClassifierMixin,
        data: pd.DataFrame,
        labels: pd.Series,
        sample_weights: np.ndarray = None,
        scoring_method: str = 'neg_log_loss',
        cv_strategy: CrossValidationStrategy = None,
        embargo_percent: float = 0.0
) -> np.ndarray:
    """
    Calculate cross-validation scores using a custom or provided cross-validation strategy.

    :param classifier: A classifier model instance that follows the scikit-learn interface.
    :type classifier: ClassifierMixin

    :param data: The data frame containing the features to be used in the classification.
    :type data: pd.DataFrame

    :param labels: The labels corresponding to each row in `data`.
    :type labels: pd.Series

    :param sample_weights: Weights for each sample in `data`.
    :type sample_weights: np.ndarray

    :param scoring_method: The scoring method to use. Supports 'neg_log_loss' and 'accuracy'.
    :type scoring_method: str

    :param cv_strategy: A custom cross-validation strategy that adheres to the `CrossValidationStrategy` interface.
    :type cv_strategy: CrossValidationStrategy

    :param embargo_percent: The percentage of the dataset to hold out as an embargo during purging.
    :type embargo_percent: float

    :return: An array of scores for each fold in the cross-validation.
    :rtype: np.ndarray

    Reference: De Prado, M. (2018) Advances in Financial Machine Learning
    Methodology: page 110, snippet 7.4
    """

    if scoring_method not in ['neg_log_loss', 'accuracy']:
        raise ValueError('Invalid scoring method.')

    if sample_weights is None:
        sample_weights = np.ones(len(data))

    scores = []

    for train_indices, test_indices in cv_strategy.split(data):
        classifier.fit(
            X=data.iloc[train_indices, :],
            y=labels.iloc[train_indices],
            sample_weight=sample_weights[train_indices]
        )

        if scoring_method == 'neg_log_loss':
            predicted_probabilities = classifier.predict_proba(data.iloc[test_indices, :])
            score = -log_loss(
                y_true=labels.iloc[test_indices],
                y_pred=predicted_probabilities,
                sample_weight=sample_weights[test_indices],
                labels=classifier.classes_
            )
        else:
            predicted_labels = classifier.predict(data.iloc[test_indices, :])
            score = accuracy_score(
                y_true=labels.iloc[test_indices],
                y_pred=predicted_labels,
                sample_weight=sample_weights[test_indices]
            )

        scores.append(score)

    return np.array(scores)
