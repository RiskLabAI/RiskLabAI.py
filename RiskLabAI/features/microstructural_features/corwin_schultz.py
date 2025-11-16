"""
Implements the Corwin and Schultz (2012) bid-ask spread estimator.

Reference:
    Corwin, S. A., & Schultz, P. (2012). A simple way to estimate
    bid-ask spreads from daily high and low prices.
    The Journal of Finance, 67(2), 719-760.
"""

import numpy as np
import pandas as pd

# Constant 'd' from the paper (denominator)
_DENOMINATOR = 3 - 2 * (2**0.5)


def beta_estimates(
    high_prices: pd.Series, low_prices: pd.Series, window_span: int
) -> pd.Series:
    r"""
    Estimate \(\beta\) (sum of squared high-low log-ratios).

    .. math::
        \beta_t = \sum_{j=t-1}^{t} [\ln(H_j / L_j)]^2

    This is then averaged over the `window_span`.

    Parameters
    ----------
    high_prices : pd.Series
        Time series of high prices.
    low_prices : pd.Series
        Time series of low prices.
    window_span : int
        Rolling window span for averaging.

    Returns
    -------
    pd.Series
        The estimated \(\beta\) vector.
    """
    log_ratios_sq = np.log(high_prices / low_prices) ** 2
    
    # Sum of current and previous day's squared log-ratio
    beta = log_ratios_sq.rolling(window=2).sum()
    
    # Average over the window span
    beta = beta.rolling(window=window_span).mean()
    return beta


def gamma_estimates(high_prices: pd.Series, low_prices: pd.Series) -> pd.Series:
    r"""
    Estimate \(\gamma\) (squared log-ratio of two-day high/low).

    .. math::
        \gamma_t = [\ln(\max(H_t, H_{t-1}) / \min(L_t, L_{t-1}))]^2

    Parameters
    ----------
    high_prices : pd.Series
        Time series of high prices.
    low_prices : pd.Series
        Time series of low prices.

    Returns
    -------
    pd.Series
        The estimated \(\gamma\) vector.
    """
    high_prices_max = high_prices.rolling(window=2).max()
    low_prices_min = low_prices.rolling(window=2).min()
    gamma = np.log(high_prices_max / low_prices_min) ** 2
    return gamma


def alpha_estimates(beta: pd.Series, gamma: pd.Series) -> pd.Series:
    r"""
    Estimate \(\alpha\) from \(\beta\) and \(\gamma\).

    .. math::
        d = 3 - 2\sqrt{2}
        \alpha = \frac{(\sqrt{2} - 1)\sqrt{\beta}}{d}
                 - \sqrt{\frac{\gamma}{d}}

    Parameters
    ----------
    beta : pd.Series
        \(\beta\) estimates vector.
    gamma : pd.Series
        \(\gamma\) estimates vector.

    Returns
    -------
    pd.Series
        The estimated \(\alpha\) vector, floored at 0.
    """
    term1 = ((2**0.5) - 1) * (beta**0.5) / _DENOMINATOR
    term2 = (gamma / _DENOMINATOR) ** 0.5
    
    # Floor at zero
    alpha = np.maximum(term1 - term2, 0)
    return alpha


def corwin_schultz_estimator(
    high_prices: pd.Series, low_prices: pd.Series, window_span: int = 20
) -> pd.Series:
    r"""
    Estimate the bid-ask spread using the Corwin and Schultz (2012) method.

    .. math::
        S = \frac{2(e^\alpha - 1)}{1 + e^\alpha}

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
        The estimated spread vector.
    """
    beta = beta_estimates(high_prices, low_prices, window_span)
    gamma = gamma_estimates(high_prices, low_prices)
    alpha = alpha_estimates(beta, gamma)
    
    # Calculate spread
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return spread