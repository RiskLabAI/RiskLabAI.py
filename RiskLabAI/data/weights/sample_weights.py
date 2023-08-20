import numpy as np
import pandas as pd
from typing import List, Series, DataFrame

def expand_label_for_meta_labeling(
    close_index: pd.DataFrame,
    timestamp: pd.DataFrame,
    molecule: pd.Index
) -> pd.Series:
    """
    Expand label to incorporate meta-labeling.

    This function expands the label to incorporate meta-labeling by taking
    a DataFrame with events, another DataFrame with the return and label of each period,
    and an Index to apply the function on. It then returns a Series with the count
    of events spanning a bar for each molecule.

    :param close_index: DataFrame that has events.
    :param timestamp: DataFrame that has return and label of each period.
    :param molecule: Index that function must apply on it.
    :return: Series with the count of events spanning a bar for each molecule.
    """
    timestamp = timestamp.fillna(close_index[-1])
    timestamp = timestamp[timestamp >= molecule[0]]
    timestamp = timestamp.loc[:timestamp[molecule].max()]
    iloc = close_index.searchsorted(np.array([timestamp.index[0], timestamp.max()]))
    count = pd.Series(0, index=close_index[iloc[0]:iloc[1] + 1])

    for t_in, t_out in timestamp.iteritems():
        count.loc[t_in:t_out] += 1

    return count.loc[molecule[0]:timestamp[molecule].max()]

def calculate_sample_weight(
    timestamp: pd.DataFrame,
    concurrency_events: pd.DataFrame,
    molecule: pd.Index
) -> pd.Series:
    """
    Calculate sample weight using triple barrier method.

    :param timestamp: DataFrame of events start and end for labelling.
    :param concurrency_events: Data frame of concurrent events for each event.
    :param molecule: Index that function must apply on it.
    :return: Series of sample weights.
    """
    weight = pd.Series(index=molecule)

    for t_in, t_out in timestamp.loc[weight.index].iteritems():
        weight.loc[t_in] = (1. / concurrency_events.loc[t_in:t_out]).mean()

    return weight

def create_index_matgrix(
    bar_index: pd.Index,
    timestamp: pd.DataFrame
) -> pd.DataFrame:
    """
    Create an indicator matrix.

    :param bar_index: Index of all data.
    :param timestamp: DataFrame with starting and ending times of events.
    :return: Indicator matrix.
    """
    ind_matrix = pd.DataFrame(0, index=bar_index, columns=range(timestamp.shape[0]))

    for row in timestamp.itertuples():
        t0 = int(row.date)
        t1 = int(row.timestamp)
        ind_matrix.loc[t0:t1, row.Index] = 1

    return ind_matrix

def calculate_average_uniqueness(index_matrix: pd.DataFrame) -> pd.Series:
    """
    Calculate average uniqueness from indicator matrix.

    :param index_matrix: Indicator matrix.
    :return: Series of average uniqueness values.
    """
    concurrency = index_matrix.sum(axis=1)
    uniqueness = index_matrix.div(concurrency, axis=0)
    average_uniqueness = uniqueness[uniqueness > 0].mean()

    return average_uniqueness

def perform_sequential_bootstrap(
    index_matrix: pd.DataFrame,
    sample_length: int
) -> List[int]:
    """
    Perform sequential bootstrap to generate a sample.

    :param index_matrix: Matrix of indicators for events.
    :param sample_length: Number of samples.
    :return: List of indices representing the sample.
    """
    if sample_length is None:
        sample_length = index_matrix.shape[1]

    phi = []

    while len(phi) < sample_length:
        average_uniqueness = pd.Series(dtype=np.float64)

        for i in index_matrix:
            index_matrix_ = index_matrix[phi + [i]]
            average_uniqueness.loc[i] = calculate_average_uniqueness(index_matrix_).iloc[-1]

        prob = average_uniqueness / average_uniqueness.sum()
        phi += [np.random.choice(index_matrix.columns, p=prob)]

    return phi

def calculate_sample_weight_absolute_return(
    timestamp: pd.DataFrame,
    concurrency_events: pd.DataFrame,
    returns: pd.DataFrame,
    molecule: pd.Index
) -> pd.Series:
    """
    Calculate sample weight using absolute returns.

    :param timestamp: DataFrame for events.
    :param concurrency_events: DataFrame that contains number of concurrent events for each event.
    :param returns: DataFrame that contains returns.
    :param molecule: Index for the calculation.
    :return: Series of sample weights.
    """
    return_ = np.log(returns).diff()
    weight = pd.Series(index=molecule)

    for t_in, t_out in timestamp.loc[weight.index].iteritems():
        weight.loc[t_in] = (return_.loc[t_in:t_out] / concurrency_events.loc[t_in:t_out]).sum()

    return weight.abs()

def calculate_time_decay(
    weight: pd.Series,
    clf_last_weight: float = 1.0
) -> pd.Series:
    """
    Calculate time decay on weight.

    :param weight: Weight computed for each event.
    :param clf_last_weight: Weight of oldest observation.
    :return: Series of weights after applying time decay.
    """
    clf_weight = weight.sort_index().cumsum()

    if clf_last_weight >= 0:
        slope = (1. - clf_last_weight) / clf_weight.iloc[-1]
    else:
        slope = 1. / ((clf_last_weight + 1) * clf_weight.iloc[-1])

    const = 1. - slope * clf_weight.iloc[-1]
    clf_weight = const + slope * clf_weight
    clf_weight[clf_weight < 0] = 0

    return clf_weight
