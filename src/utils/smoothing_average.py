import numpy as np
from typing import List, Tuple

def exponential_weighted_moving_average(
    input_series: np.ndarray,
    window_length: int
) -> np.ndarray:
    """
    Compute the exponential weighted moving average (EWMA) of a time series array.

    The EWMA is calculated using the formula:

    .. math::

        EWMA_t = \\frac{x_t + (1 - \\alpha) * x_{t-1} + (1 - \\alpha)^2 * x_{t-2} + \\cdots}{\\omega_t}

    where :math:`\\omega_t = 1 + (1 - \\alpha) + (1 - \\alpha)^2 + \\cdots + (1 - \\alpha)^t`,
    and :math:`\\alpha = \\frac{2}{\\text{window_length} + 1}`

    Args:
        input_series (np.ndarray): Input time series array.
        window_length (int): Window length for the exponential weighted moving average.

    Returns:
        np.ndarray: An array containing the computed EWMA values.
    """
    num_values = input_series.shape[0]
    ewma_output = np.empty(num_values, dtype='float64')
    omega = 1
    alpha = 2 / float(window_length + 1)
    current_input_value = input_series[0]
    ewma_output[0] = current_input_value

    for i in range(1, num_values):
        omega += (1 - alpha) ** i
        current_input_value = current_input_value * (1 - alpha) + input_series[i]
        ewma_output[i] = current_input_value / omega

    return ewma_output
