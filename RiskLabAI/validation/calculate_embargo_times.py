import pandas as pd

def calculate_embargo_times(
        times: pd.Series, 
        percent_embargo: float
) -> pd.Series:
    """
    Calculate embargo times for each time bar based on a given embargo percentage.

    The function applies the concept of embargo to time bars, ensuring that bars close
    to each other are not used in the training/testing set simultaneously. This is to
    avoid leakage between training and testing data.

    .. math::
        \\text{step} = \\text{int}(\\text{length of time series} \\times \\text{percent_embargo})

    :param pd.Series times: Series representing the entire observation times.
    :param float percent_embargo: Embargo size as a percentage divided by 100.
    :return: A Pandas Series with the embargo times for each bar.
    :rtype: pd.Series

    Reference: De Prado, M. (2018) Advances in Financial Machine Learning
    Methodology: page 108, snippet 7.2
    """
    # Calculate the step size based on the embargo percentage
    step = int(times.shape[0] * percent_embargo)

    # Handle edge case where step size is zero
    if step == 0:
        return pd.Series(times, index=times)

    # Calculate embargo times based on the step size
    embargo_times = pd.Series(times[step:], index=times[:-step])
    embargo_times = embargo_times.append(pd.Series(times[-1], index=times[-step:]))

    return embargo_times
