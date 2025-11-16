"""
Implements the Bekker-Parkinson volatility estimator, which adjusts
the Parkinson volatility by the Corwin-Schultz spread.

Reference:
    De Prado, M. (2018) Advances in Financial Machine Learning,
    page 286, snippet 19.2.
"""

from math import pi
import pandas as pd
import numpy as np
from .corwin_schultz import (
    beta_estimates, 
    gamma_estimates, 
    _DENOMINATOR
)

def sigma_estimates(beta: pd.Series, gamma: pd.Series) -> pd.Series:
    r"""
    Compute Bekker-Parkinson volatility \(\sigma\) estimates.

    .. math::
        k_2 = \sqrt{8 / \pi}
        d = 3 - 2\sqrt{2}
        \sigma = \frac{(\sqrt{2} - 1) \sqrt{\beta}}{d}
                 + \sqrt{\frac{\gamma}{k_2^2 d}}

    Parameters
    ----------
    beta : pd.Series
        \(\beta\) estimates vector from Corwin-Schultz.
    gamma : pd.Series
        \(\gamma\) estimates vector from Corwin-Schultz.

    Returns
    -------
    pd.Series
        Bekker-Parkinson volatility \(\sigma\) estimates.
    """
    k2 = (8 / pi) ** 0.5

    term1 = (2**0.5 - 1) * (beta**0.5) / _DENOMINATOR
    term2 = (gamma / (k2**2 * _DENOMINATOR)) ** 0.5
    
    # Floor at zero
    sigma = np.maximum(term1 + term2, 0)

    return sigma


def bekker_parkinson_volatility_estimates(
    high_prices: pd.Series, low_prices: pd.Series, window_span: int = 20
) -> pd.Series:
    """
    Compute Bekker-Parkinson volatility estimates from high and low prices.

    This function first calculates the Corwin-Schultz \(\beta\) and \(\gamma\)
    parameters and then uses them to compute the volatility estimates.

    Parameters
    ----------
    high_prices : pd.Series
        Time series of high prices.
    low_prices : pd.Series
        Time series of low prices.
    window_span : int, default=20
        Rolling window span for \(\beta\) estimation.

    Returns
    -------
    pd.Series
        Bekker-Parkinson volatility estimates.
    """
    beta = beta_estimates(high_prices, low_prices, window_span)
    gamma = gamma_estimates(high_prices, low_prices)

    return sigma_estimates(beta, gamma)