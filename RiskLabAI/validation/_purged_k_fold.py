from sklearn.model_selection import KFold
import pandas as pd
import numpy as np

class PurgedKFold(KFold):
    """
    Extends the KFold class to prevent information leakage in the presence of overlapping samples.

    Reference: De Prado, M. (2018) Advances in Financial Machine Learning
    Methodology: page 109, snippet 7.3
    """

    def __init__(
            self, 
            n_splits: int = 3,
            times: pd.Series = None, 
            percent_embargo: float = 0.0
    ):
        """
        Initialize the PurgedKFold class with the number of splits, times, and embargo percentage.

        :param int n_splits: Number of KFold splits.
        :param pd.Series times: Series representing the entire observation times.
        :param float percent_embargo: Embargo size as a percentage divided by 100.
        """
        if not isinstance(times, pd.Series):
            raise ValueError('Label through dates must be a pandas Series')

        super().__init__(n_splits, shuffle=False, random_state=None)
        self.times = times
        self.percent_embargo = percent_embargo

    def split(
            self, 
            data: pd.DataFrame, 
            labels: pd.Series = None, 
            groups = None
    ):
        """
        Generate indices to split data into training and test set.

        :param pd.DataFrame data: The sample data that is going to be split.
        :param pd.Series labels: The labels that are going to be split.
        :param groups: Group labels for the samples used while splitting the dataset.
        :yield: Indices for training and test data.
        """
        if (data.index == self.times.index).sum() != len(self.times):
            raise ValueError('Data and through date values must have the same index')

        indices = np.arange(data.shape[0])
        embargo_size = int(data.shape[0] * self.percent_embargo)

        test_starts = [(i[0], i[-1] + 1) for i in np.array_split(np.arange(data.shape[0]), self.n_splits)]

        for start, end in test_starts:
            first_test_index = self.times.index[start]
            test_indices = indices[start:end]
            max_test_time = self.times.iloc[test_indices].max()
            max_test_idx = self.times.index.get_loc(max_test_time)

            train_indices = self.times.index.get_loc(self.times[self.times <= first_test_index].index)

            if max_test_idx + embargo_size < data.shape[0]:
                train_indices = np.concatenate((train_indices, indices[max_test_idx + embargo_size:]))

            yield train_indices, test_indices
