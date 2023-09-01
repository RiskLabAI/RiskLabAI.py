import numpy as np
from numba import jit
from numba import float64
from numba import int64

@jit((float64[:], int64), nopython=False, nogil=True)
def ewma(
    array, 
    window
):
    """
    This function calculate Exponential Weighted Moving Average of array
    :param array: input array
    :param window: window size
    :return: ewma array
    """
    length = array.shape[0]
    result_ewma_array = np.empty(length, dtype=float64)

    α = 2 / (window + 1)
    weight = 1
    current_ewma_term = array[0]
    result_ewma_array[0] = current_ewma_term
    for i in range(1, length):
        weight += (1 - α) ** i
        current_ewma_term = current_ewma_term * (1 - α) + array[i]
        result_ewma_array[i] = current_ewma_term / weight

    return result_ewma_array