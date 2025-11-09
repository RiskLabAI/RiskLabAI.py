"""
Implements the Drift-Burst Hypothesis (DBH) model for synthetic data.

This model generates drift and volatility parameters for a bubble scenario,
featuring an "explosion" at the midpoint.
"""

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
    r"""
    Compute the drift and volatility for a DBH burst scenario.

    The model defines drift and volatility as a function of time `t`
    (represented by `steps` from 0 to 1), with an explosion at `t=0.5`.

    .. math::
        drift(t) = \frac{a(t)}{|t - 0.5|^\alpha}
        vol(t) = \frac{b(t)}{|t - 0.5|^\beta}

    To prevent division by zero, an `explosion_filter_width` is used
    to clamp the denominator near `t=0.5`.

    Parameters
    ----------
    bubble_length : int
        The number of timesteps in the bubble.
    a_before : float
        Drift coefficient 'a' for `t < 0.5`.
    a_after : float
        Drift coefficient 'a' for `t > 0.5`.
    b_before : float
        Volatility coefficient 'b' for `t < 0.5`.
    b_after : float
        Volatility coefficient 'b' for `t > 0.5`.
    alpha : float
        Exponent for the drift denominator.
    beta : float
        Exponent for the volatility denominator.
    explosion_filter_width : float, default=0.1
        Width of the "safe" area around the midpoint (0.5)
        to prevent division by zero.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - drifts: The array of drift values.
        - volatilities: The array of volatility values.
    """
    steps = np.linspace(0, 1, bubble_length)

    # Clamp values near the midpoint to avoid division by zero
    before_mask = (steps >= (0.5 - explosion_filter_width)) & (steps < 0.5)
    after_mask = (steps > 0.5) & (steps <= (0.5 + explosion_filter_width))

    steps_filtered = np.copy(steps)
    steps_filtered[before_mask] = 0.5 - explosion_filter_width
    steps_filtered[after_mask] = 0.5 + explosion_filter_width

    # Assign 'a' and 'b' parameters based on position
    a_values = np.where(steps <= 0.5, a_before, a_after)
    b_values = np.where(steps <= 0.5, b_before, b_after)

    # Calculate denominator, with NaN at the exact midpoint
    denominators = np.abs(steps_filtered - 0.5)
    denominators[steps == 0.5] = np.nan

    drifts = a_values / (denominators**alpha)
    volatilities = b_values / (denominators**beta)

    # Handle the NaN at the midpoint by filling with 0 drift
    # and the preceding volatility value.
    nan_mask = np.isnan(denominators)
    if np.any(nan_mask):
        drifts[nan_mask] = 0.0
        # Find first NaN index and fill with previous value
        nan_index = np.where(nan_mask)[0][0]
        if nan_index > 0:
            volatilities[nan_mask] = volatilities[nan_index - 1]
        else:
            # Handle case where midpoint is the first element
            volatilities[nan_mask] = b_before # Fallback

    return drifts, volatilities