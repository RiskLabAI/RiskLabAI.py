from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.metrics import log_loss, accuracy_score
import pandas as pd
import numpy as np


def calculate_cross_validation_score(
        classifier: ClassifierMixin,
        data: pd.DataFrame,
        labels: pd.Series = None,
        sample_weights: np.ndarray = None,
        scoring_method: str = 'neg_log_loss',
        observation_times: pd.Series = None,
        num_splits: int = None,
        cv_generator: BaseEstimator = None,
        embargo_percent: float = 0.0
) -> np.ndarray:
    """
    Calculate cross-validation scores using the PurgedKFold method or other provided cross-validation generator.

    :param classifier: A classifier model instance that follows the scikit-learn interface.
    :param data: The data frame containing the features to be used in the classification.
    :param labels: The labels corresponding to each row in `data`.
    :param sample_weights: Weights for each sample in `data`.
    :param scoring_method: The scoring method to use. Currently supports 'neg_log_loss' and 'accuracy'.
    :param observation_times: The times corresponding to each observation, used for purging in cross-validation.
    :param num_splits: The number of splits to use in cross-validation if a cross-validation generator is not provided.
    :param cv_generator: A custom cross-validation generator. If `None`, `PurgedKFold` will be used.
    :param embargo_percent: The percentage of the dataset to hold out as an embargo during purging.
    :return: An array of scores for each fold in the cross-validation.

    Reference: De Prado, M. (2018) Advances in Financial Machine Learning
    Methodology: page 110, snippet 7.4
    """

    if scoring_method not in ['neg_log_loss', 'accuracy']:
        raise ValueError('Invalid scoring method.')

    if cv_generator is None:
        cv_generator = PurgedKFold(
            n_splits=num_splits,
            times=observation_times,
            percent_embargo=embargo_percent
        )

    if sample_weights is None:
        sample_weights = np.ones(len(data))

    scores = []

    for train_indices, test_indices in cv_generator.split(data):
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
