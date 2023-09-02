from typing import Dict, Union, Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, BaseCrossValidator
from sklearn.base import ClassifierMixin
from sklearn.metrics import log_loss, accuracy_score

class CrossValidationModel(KFold):
    """
    Implements a KFold cross-validator with purging and embargo based on 
    De Prado's "Advances in Financial Machine Learning" methodology.

    :param n_splits: Number of KFold splits
    :param times: Entire observation times
    :param percent_embargo: Embargo size percentage divided by 100
    """

    def __init__(
        self,
        n_splits: int,
        times: pd.Series,
        percent_embargo: float
    ):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.times = times
        self.percent_embargo = percent_embargo

    def purged_kfold_split(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Splits the data using purged KFold cross-validator with purging and embargo.

        :param data: The sample to be split
        :return: Tuple of training and test indices
        """
        test_starts = {
            ticker: [(i[0], i[-1] + 1) for i in np.array_split(
                np.arange(df.shape[0]), self.n_splits
            )]
            for ticker, df in data.items()
        }

        is_single_asset = len(list(data.keys())) == 1

        for split in range(self.n_splits):
            train_indices = {}
            test_indices = {}

            for ticker, df in data.items():
                all_indices = np.arange(df.shape[0])
                embargo_size = int(df.shape[0] * self.percent_embargo)

                start, end = test_starts[ticker][split]

                first_test_time = self.times[ticker].index[start]
                current_test_indices = all_indices[start:end]
                max_test_time = self.times[ticker][current_test_indices].max()
                max_test_index = self.times[ticker].index.searchsorted(max_test_time)

                left_train_indices = self.times[ticker].index.searchsorted(
                    self.times[ticker][self.times[ticker] <= first_test_time].index
                )

                if max_test_index + embargo_size < df.shape[0]:
                    right_train_indices = all_indices[max_test_index + embargo_size:]
                    all_train_indices = np.concatenate((left_train_indices, right_train_indices))
                else:
                    all_train_indices = left_train_indices

                train_indices[ticker] = np.array(all_train_indices)
                test_indices[ticker] = np.array(current_test_indices)

            if is_single_asset:
                train_indices = train_indices['ASSET']
                test_indices = test_indices['ASSET']

            yield train_indices, test_indices

    @staticmethod
    def purged_train_times(
        data: pd.Series,
        test: pd.Series
    ) -> pd.Series:
        """
        Purges test observations in the training set.
        
        .. math:: \\text{Not applicable.}
        
        :param data: Times of entire observations
        :param test: Times of testing observations
        :return: Purged train times
        """
        train_times = data.copy(deep=True)

        for start, end in test.iteritems():
            mask = ((start <= train_times.index) & (train_times.index <= end)) | \
                   ((start <= train_times) & (train_times <= end)) | \
                   ((train_times.index <= start) & (end <= train_times))

            train_times = train_times.loc[~mask]

        return train_times

    @staticmethod
    def embargo_times(
        times: pd.Series,
        percent_embargo: float
    ) -> pd.Series:
        """
        Gets embargo time for each bar.
        
        .. math:: \\text{Not applicable.}
        
        :param times: Entire observation times
        :param percent_embargo: Embargo size percentage divided by 100
        :return: Series of embargo times
        """
        step = int(times.shape[0] * percent_embargo)

        if step == 0:
            return pd.Series(times, index=times)

        embargo = pd.Series(times[step:], index=times[:-step])
        embargo = embargo.append(pd.Series(times[-1], index=times[-step:]))

        return embargo

    @staticmethod
    def cross_validation_score(
        classifier: ClassifierMixin,
        data: pd.DataFrame,
        labels: Optional[pd.Series] = None,
        sample_weight: Optional[np.ndarray] = None,
        scoring: str = 'neg_log_loss',
        times: Optional[pd.Series] = None,
        n_splits: Optional[int] = None,
        cross_validation_generator: Optional[BaseCrossValidator] = None,
        percent_embargo: float = 0.0
    ) -> np.ndarray:
        """
        Uses the CrossValidationModel to perform cross-validation.
        
        .. math:: \\text{Scoring can be either negative log loss or accuracy.}

        :param classifier: A classifier model
        :param data: The sample to be split
        :param labels: The labels to be split
        :param sample_weight: The sample weights for the classifier
        :param scoring: Scoring type: ['neg_log_loss','accuracy']
        :param times: Entire observation times
        :param n_splits: Number of KFold splits
        :param cross_validation_generator: Cross-validation generator
        :param percent_embargo: Embargo size percentage divided by 100
        :return: Array of scores
        """
        if scoring not in ['neg_log_loss', 'accuracy']:
            raise Exception('Invalid scoring method.')

        if cross_validation_generator is None:
            cross_validation_generator = CrossValidationModel(
                n_splits=n_splits, times=times, percent_embargo=percent_embargo
            )

        if sample_weight is None:
            sample_weight = np.ones(len(data))

        scores = []

        for train, test in cross_validation_generator.purged_kfold_split(data):
            fit = classifier.fit(
                X=data.iloc[train, :],
                y=labels.iloc[train],
                sample_weight=sample_weight[train]
            )

            if scoring == 'neg_log_loss':
                probabilities = fit.predict_proba(data.iloc[test, :])
                score = -log_loss(
                    labels.iloc[test],
                    probabilities,
                    sample_weight=sample_weight[test],
                    labels=classifier.classes_
                )
            else:
                predictions = fit.predict(data.iloc[test, :])
                score = accuracy_score(
                    labels.iloc[test],
                    predictions,
                    sample_weight=sample_weight[test]
                )

            scores.append(score)

        return np.array(scores)
