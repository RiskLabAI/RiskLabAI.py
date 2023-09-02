import pandas as pd

def remove_overlapping_train_times(
        data: pd.Series, 
        test: pd.Series
) -> pd.Series:
    """
    Remove overlapping time intervals in training data based on testing data intervals.

    Given a series of time intervals in the training data (`data`) and testing data (`test`),
    this function returns a new series containing only the time intervals from the training data
    that do not overlap with any time interval in the testing data.

    :param pd.Series data: The training data time intervals, where the index represents the time.
    :param pd.Series test: The testing data time intervals, where the index represents the time.
    :return: A new Pandas Series containing the filtered training data time intervals.
    :rtype: pd.Series

    .. note:: This assumes that the start and end times in both `data` and `test` series are sorted.
    """
    # Create a copy of the original training times to avoid modifying the input
    filtered_train_times = data.copy(deep=True)

    # Iterate through each testing interval and remove overlapping intervals from the training data
    for start, end in test.iteritems():
        overlapping_indices = filtered_train_times.index.to_series().between(start, end, inclusive=True)
        filtered_train_times.drop(filtered_train_times.index[overlapping_indices], inplace=True)

    return filtered_train_times
