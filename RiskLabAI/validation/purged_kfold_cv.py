import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.base import ClassifierMixin, BaseCrossValidator
from sklearn.metrics import log_loss, accuracy_score


def purged_train_times(
        data: pd.Series,  # Times of entire observations
        test: pd.Series  # Times of testing observations
) -> pd.Series:
    """
    Purges test observations in the training set.

    :param data: Times of entire observations
    :param test: Times of testing observations
    :return: Updated training times

    Reference: De Prado, M. (2018) Advances in Financial Machine Learning
    Methodology: page 106, snippet 7.1
    """
    train_times = data.copy(deep=True)

    for start, end in test.iteritems():
        mask = train_times.index.to_series().between(start, end, inclusive=True)
        train_times = train_times.drop(train_times[mask].index)

    return train_times


def embargo_times(
        times: pd.Series,  # Entire observation times
        percent_embargo: float  # Embargo size percentage divided by 100
) -> pd.Series:
    """
    Gets embargo time for each bar.

    :param times: Entire observation times
    :param percent_embargo: Embargo size percentage divided by 100
    :return: Series with embargo times for each bar

    Reference: De Prado, M. (2018) Advances in Financial Machine Learning
    Methodology: page 108, snippet 7.2
    """
    step = int(times.shape[0] * percent_embargo)

    if step == 0:
        return pd.Series(times, index=times)
    else:
        embargo = pd.Series(times[step:], index=times[:-step])
        embargo = embargo.append(pd.Series(times[-1], index=times[-step:]))
        return embargo


class PurgedKFold(KFold):
    """
    Splits the data and performs cross-validation when observations overlap.

    Reference: De Prado, M. (2018) Advances in Financial Machine Learning
    Methodology: page 109, snippet 7.3
    """

    def __init__(
            self,
            n_splits: int = 3,  # The number of KFold splits
            times: pd.Series = None,  # Entire observation times
            percent_embargo: float = 0.0  # Embargo size percentage divided by 100
    ):
        if not isinstance(times, pd.Series):
            raise ValueError('Label Through Dates must be a pandas series')
        super(PurgedKFold, self).__init__(n_splits, shuffle=False, random_state=None)
        self.times = times
        self.percent_embargo = percent_embargo

    def split(
            self,
            data: pd.DataFrame,  # The sample that is going to be split
            labels: pd.Series = None,  # The labels that are going to be split
            groups=None  # Group our labels
    ):
        if (data.index == self.times.index).sum() != len(self.times):
            raise ValueError('data and ThruDateValues must have the same index')

        indices = np.arange(data.shape[0])
        embargo = int(data.shape[0] * self.percent_embargo)

        test_starts = [(i[0], i[-1] + 1) for i in \
                       np.array_split(np.arange(data.shape[0]), self.n_splits)]

        for start, end in test_starts:
            first_test_index = self.times.index[start]
            test_indices = indices[start:end]
            max_test_index = self.times.index.searchsorted(self.times[test_indices].max())
            train_indices = self.times.index.searchsorted(self.times[self.times <= first_test_index].index)

            if max_test_index + embargo < data.shape[0]:
                train_indices = np.concatenate((train_indices, indices[max_test_index + embargo:]))

            yield train_indices, test_indices


def cross_validation_score(
        classifier: ClassifierMixin,  # A classifier model
        data: pd.DataFrame,  # The sample that is going to be split
        labels: pd.Series = None,  # The labels that are going to be split
        sample_weight: np.ndarray = None,  # The sample weights for the classifier
        scoring: str = 'neg_log_loss',  # Scoring type: ['neg_log_loss','accuracy']
        times: pd.Series = None,  # Entire observation times
        n_splits: int = None,  # The number of KFold splits
        cross_validation_generator: BaseCrossValidator = None,
        percent_embargo: float = 0.0  # Embargo size percentage divided by 100
) -> np.array:
    """
    Uses the PurgedKFold class and functions to perform cross-validation.

    :param classifier: A classifier model
    :param data: The sample that is going to be split
    :param labels: The labels that are going to be split
    :param sample_weight: The sample weights for the classifier
    :param scoring: Scoring type: ['neg_log_loss','accuracy']
    :param times: Entire observation times
    :param n_splits: The number of KFold splits
    :param cross_validation_generator: Cross-validation generator instance
    :param percent_embargo: Embargo size percentage divided by 100
    :return: Array with cross-validation scores

    Reference: De Prado, M. (2018) Advances in Financial Machine Learning
    Methodology: page 110, snippet 7.4
    """
    if scoring not in ['neg_log_loss', 'accuracy']:
        raise Exception('wrong scoring method.')

    if cross_validation_generator is None:
        cross_validation_generator = PurgedKFold(n_splits=n_splits, times=times, percent_embargo=percent_embargo)

    if sample_weight is None:
        sample_weight = pd.Series(np.ones(len(data)))

    scores = []

    for train, test in cross_validation_generator.split(data):
        fit = classifier.fit(X=data.iloc[train, :], y=labels.iloc[train],
                             sample_weight=sample_weight[train])

        if scoring == 'neg_log_loss':
            prob = fit.predict_proba(data.iloc[test, :])
            score = -log_loss(labels.iloc[test], prob,
                              sample_weight=sample_weight.iloc[test].values, labels=classifier.classes_)
        else:
            pred = fit.predict(data.iloc[test, :])
            score = accuracy_score(labels.iloc[test], pred, sample_weight=sample_weight.iloc[test].values)

        scores.append(score)

    return np.array(scores)
