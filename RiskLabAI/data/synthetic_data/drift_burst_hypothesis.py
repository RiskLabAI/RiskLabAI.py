import numpy as np
from typing import Tuple

def drift_volatility_burst(
    bubble_length: int,
    a_before: float,
    a_after: float,
    b_before: float,
    b_after: float,
    alpha: float,
    beta: float, 
    explosion_filter_width: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the drift and volatility for a burst scenario.

    The drift and volatility are calculated based on:
    .. math::
        drift = \frac{a_{value}}{denominator^\alpha}
        volatility = \frac{b_{value}}{denominator^\beta}

    where:
    .. math::
        denominator = |step - 0.5|

    :param bubble_length: The length of the bubble.
    :param a_before: 'a' value before the mid-point.
    :param a_after: 'a' value after the mid-point.
    :param b_before: 'b' value before the mid-point.
    :param b_after: 'b' value after the mid-point.
    :param alpha: Exponent for the drift calculation.
    :param beta: Exponent for the volatility calculation.
    :param explosion_filter_width: Width of the area around the explosion that denominators won't exceed. 
    :return: A tuple containing the drift and volatility arrays.
    """
    steps = np.linspace(0, 1, bubble_length)

    # Create two boolean masks identifying the indices where the values are within the specified range
    before_mask = (steps >= (0.5 - explosion_filter_width)) & (steps < 0.5)
    after_mask = (steps > 0.5) & (steps <= (0.5 + explosion_filter_width))

    # Replace the values at these indices with 0.5 - explosion_filter_width
    steps[before_mask] = 0.5 - explosion_filter_width
    steps[after_mask] = 0.5 + explosion_filter_width

    a_values = np.where(steps <= 0.5, a_before, a_after)
    b_values = np.where(steps <= 0.5, b_before, b_after)

    denominators = np.abs(steps - 0.5)
    denominators[steps == 0.5] = np.nan  # Set the denominator to NaN for step == 0.5

    drifts = a_values / denominators ** alpha
    volatilities = b_values / denominators ** beta

    # Fill NaN values with preceding values
    nan_mask = np.isnan(denominators)
    if np.sum(nan_mask) > 0:
        drifts[nan_mask] = 0
        volatilities[nan_mask] = volatilities[np.where(nan_mask)[0][0] - 1]

    return drifts, volatilities
