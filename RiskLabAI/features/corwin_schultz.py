import numpy as np
import pandas as pd

def beta_estimates(
    high_prices: pd.Series,
    low_prices: pd.Series,
    window_span: int
) -> pd.Series:
    """
    Estimate β using Corwin and Schultz methodology.

    :param high_prices: High prices vector.
    :param low_prices: Low prices vector.
    :param window_span: Rolling window span.
    :return: Estimated β vector.

    .. note:: Reference: Corwin, S. A., & Schultz, P. (2012). A simple way to estimate bid-ask spreads from daily high and low prices. The Journal of Finance, 67(2), 719-760.
    """
    log_ratios = np.log(high_prices / low_prices) ** 2
    beta = log_ratios.rolling(window=2).sum()
    beta = beta.rolling(window=window_span).mean()
    return beta

def gamma_estimates(
    high_prices: pd.Series,
    low_prices: pd.Series
) -> pd.Series:
    """
    Estimate γ using Corwin and Schultz methodology.

    :param high_prices: High prices vector.
    :param low_prices: Low prices vector.
    :return: Estimated γ vector.

    .. note:: Reference: Corwin, S. A., & Schultz, P. (2012). A simple way to estimate bid-ask spreads from daily high and low prices. The Journal of Finance, 67(2), 719-760.
    """
    high_prices_max = high_prices.rolling(window=2).max()
    low_prices_min = low_prices.rolling(window=2).min()
    gamma = np.log(high_prices_max / low_prices_min)**2
    return gamma

def alpha_estimates(
    beta: pd.Series,
    gamma: pd.Series
) -> pd.Series:
    """
    Estimate α using Corwin and Schultz methodology.

    :param beta: β Estimates vector.
    :param gamma: γ Estimates vector.
    :return: Estimated α vector.

    .. note:: Reference: Corwin, S. A., & Schultz, P. (2012). A simple way to estimate bid-ask spreads from daily high and low prices. The Journal of Finance, 67(2), 719-760.
    """
    denominator = 3 - 2 * 2**0.5
    alpha = (2**0.5 - 1) * (beta**0.5) / denominator
    alpha -= (gamma / denominator)**0.5
    alpha[alpha < 0] = 0
    return alpha

def corwin_schultz_estimator(
    high_prices: pd.Series,
    low_prices: pd.Series,
    window_span: int = 20
) -> pd.Series:
    """
    Estimate spread using Corwin and Schultz methodology.

    :param high_prices: High prices vector.
    :param low_prices: Low prices vector.
    :param window_span: Rolling window span, default is 20.
    :return: Estimated spread vector.

    .. note:: Reference: Corwin, S. A., & Schultz, P. (2012). A simple way to estimate bid-ask spreads from daily high and low prices. The Journal of Finance, 67(2), 719-760.
    """
    beta = beta_estimates(high_prices, low_prices, window_span)
    gamma = gamma_estimates(high_prices, low_prices)
    alpha = alpha_estimates(beta, gamma)
    spread = 2 * (alpha - 1) / (1 + np.exp(alpha))
    return spread
