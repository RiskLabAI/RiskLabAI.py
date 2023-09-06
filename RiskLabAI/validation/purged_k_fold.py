from cross_validator_interface import CrossValidator
from typing import Iterator, Tuple, Union
import pandas as pd
import numpy as np

class PurgedKFold(CrossValidator):
    """
    Extends the CrossValidator interface to prevent information leakage in the presence of overlapping samples.

    Reference: De Prado, M. (2018) Advances in Financial Machine Learning
    Methodology: page 109, snippet 7.3

    Methods
    -------
    split(
        data: pd.DataFrame,
        labels: Union[None, pd.Series] = None,
        groups: Union[None, pd.Series] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]
        Yields the indices for the training and test data.
    """

    def __init__(
            self, 
            n_splits: int,
            times: pd.Series, 
            percent_embargo: float
    ):
        """
        Initialize the PurgedKFold class with the number of splits, times, and embargo percentage.

        Parameters
        ----------
        n_splits : int
            Number of KFold splits.
        times : pd.Series
            Series representing the entire observation times.
        percent_embargo : float
            Embargo size as a percentage divided by 100.
        """
        self.n_splits = n_splits
        self.times = times
        self.percent_embargo = percent_embargo

    def split(
            self, 
            data: pd.DataFrame, 
            labels: Union[None, pd.Series] = None,
            groups: Union[None, pd.Series] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        data : pd.DataFrame
            The sample data that is going to be split.
        labels : Union[None, pd.Series], optional
            The labels that are going to be split, by default None
        groups : Union[None, pd.Series], optional
            Group labels for the samples used while splitting the dataset, by default None

        Returns
        -------
        Iterator[Tuple[np.ndarray, np.ndarray]]
            Yields the indices for the training and test data.

        .. note:: 
            This implementation respects time series structures, ensuring that the training set 
            does not include any data "from the future" (i.e., data that would have been unknown
            at the time the model was trained).
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
