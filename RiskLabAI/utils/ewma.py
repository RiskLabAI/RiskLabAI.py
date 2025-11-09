"""
Implements a Numba-jitted Exponentially Weighted Moving Average (EWMA)
that correctly handles the 'adjust=True' logic from pandas.
"""

import numpy as np
from numba import jit, float64, int64
from typing import Union

@jit(nopython=True)
def ewma(array: np.ndarray, window: int) -> np.ndarray:
    r"""
    Calculate the Exponentially Weighted Moving Average (EWMA) of an array.

    This function mimics the pandas `ewm(span=window, adjust=True).mean()`
    method, which is crucial for financial calculations as it does not
    lose the first `window` observations.

    The EWMA is calculated using the formula:
    .. math::
        EWMA_t = \frac{x_t + (1 - \alpha) x_{t-1} + \ldots + (1 - \alpha)^t x_0}
                      {1 + (1 - \alpha) + \ldots + (1 - \alpha)^t}

    where:
    - \(\alpha = 2 / (window + 1)\)
    - The denominator \(\omega_t\) is the sum of weights.

    Parameters
    ----------
    array : np.ndarray
        Input array (e.g., of tick counts or imbalances).
    window : int
        The span of the EWMA.

    Returns
    -------
    np.ndarray
        The array of EWMA values.
    """
    length = array.shape[0]
    result_ewma_array = np.empty(length, dtype=float64)
    if length == 0:
        return result_ewma_array

    alpha = 2.0 / (float(window) + 1.0)
    multiplier = 1.0 - alpha
    
    # Handle the sum of weights (denominator)
    weight_sum = 1.0
    current_weighted_sum = array[0]
    result_ewma_array[0] = current_weighted_sum

    for i in range(1, length):
        weight_sum += multiplier**i
        current_weighted_sum = current_weighted_sum * multiplier + array[i]
        result_ewma_array[i] = current_weighted_sum / weight_sum

    return result_ewma_array