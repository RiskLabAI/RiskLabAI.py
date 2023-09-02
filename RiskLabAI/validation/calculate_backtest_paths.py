from math import comb

def calculate_backtest_paths(
        n_train_splits: int, 
        n_test_splits: int
) -> float:
    """
    Calculate the number of backtest paths for time-series cross-validation.

    This function calculates the number of unique training/testing data split combinations that can be 
    created during time-series cross-validation. This is particularly useful for time-series models where 
    data cannot be randomly split.

    :param n_train_splits: The total number of splits in the training set.
    :type n_train_splits: int

    :param n_test_splits: The total number of splits in the testing set.
    :type n_test_splits: int

    :return: The number of unique backtest paths.
    :rtype: float

    .. math::
        \\text{Number of backtest paths} = \\frac{{C(n_{\\text{train}}, n_{\\text{train}} - n_{\\text{test}}) \\times n_{\\text{test}}}}{n_{\\text{train}}}
    """
    numerator = comb(n_train_splits, n_train_splits - n_test_splits) * n_test_splits
    return float(numerator / n_train_splits)
