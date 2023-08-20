import numpy as np
from typing import List


def compute_exponential_weighted_moving_average(
    input_series: np.ndarray, 
    window_length: int
) -> np.ndarray:
    """
    Compute the exponential weighted moving average (EWMA) of a time series array.

    The EWMA is calculated using the formula:

    .. math::
        EWMA_t = \\frac{x_t + (1 - \\alpha) x_{t-1} + (1 - \\alpha)^2 x_{t-2} + \\ldots}{\\omega_t}

    where:

    .. math::
        \\omega_t = 1 + (1 - \\alpha) + (1 - \\alpha)^2 + \\ldots + (1 - \\alpha)^t,
        \\alpha = \\frac{2}{{window\_length + 1}}

    :param input_series: Input time series array.
    :type input_series: np.ndarray
    :param window_length: Window length for the exponential weighted moving average.
    :type window_length: int
    :return: An array containing the computed EWMA values.
    :rtype: np.ndarray
    """

    num_values = input_series.shape[0]
    ewma_output = np.empty(num_values, dtype='float64')
    alpha = 2 / float(window_length + 1)
    multiplier = 1 - alpha
    current_weighted_sum = input_series[0]
    ewma_output[0] = current_weighted_sum

    for i in range(1, num_values):
        current_weighted_sum = current_weighted_sum * multiplier + input_series[i]
        ewma_output[i] = current_weighted_sum / (1 - multiplier ** (i+1))

    return ewma_output
