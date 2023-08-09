import numpy as np
import pandas as pd

def beta_estimates(high_prices: pd.Series, low_prices: pd.Series, window_span: int) -> pd.Series:
    """
    Estimate β using Corwin and Schultz methodology.

    :param high_prices: High prices vector
    :type high_prices: pd.Series
    :param low_prices: Low prices vector
    :type low_prices: pd.Series
    :param window_span: Rolling window span
    :type window_span: int
    :return: Estimated β vector
    :rtype: pd.Series
    """
    log_ratios = np.log(high_prices / low_prices) ** 2
    beta = log_ratios.rolling(window=2).sum()
    beta = beta.rolling(window=window_span).mean()
    return beta

def gamma_estimates(high_prices: pd.Series, low_prices: pd.Series) -> pd.Series:
    """
    Estimate γ using Corwin and Schultz methodology.

    :param high_prices: High prices vector
    :type high_prices: pd.Series
    :param low_prices: Low prices vector
    :type low_prices: pd.Series
    :return: Estimated γ vector
    :rtype: pd.Series
    """
    high_prices_max = high_prices.rolling(window=2).max()
    low_prices_min = low_prices.rolling(window=2).min()
    gamma = np.log(high_prices_max / low_prices_min)**2
    return gamma

def alpha_estimates(beta: pd.Series, gamma: pd.Series) -> pd.Series:
    """
    Estimate α using Corwin and Schultz methodology.

    :param beta: β Estimates vector
    :type beta: pd.Series
    :param gamma: γ Estimates vector
    :type gamma: pd.Series
    :return: Estimated α vector
    :rtype: pd.Series
    """
    denominator = 3 - 2 * 2**0.5
    alpha = (2**0.5 - 1) * (beta**0.5) / denominator
    alpha -= (gamma / denominator)**0.5
    alpha[alpha < 0] = 0
    return alpha

def corwin_schultz_estimator(high_prices: pd.Series, low_prices: pd.Series, window_span: int = 20) -> pd.Series:
    """
    Estimate spread using Corwin and Schultz methodology.

    :param high_prices: High prices vector
    :type high_prices: pd.Series
    :param low_prices: Low prices vector
    :type low_prices: pd.Series
    :param window_span: Rolling window span, default is 20
    :type window_span: int
    :return: Estimated spread vector
    :rtype: pd.Series
    """
    beta = beta_estimates(high_prices, low_prices, window_span)
    gamma = gamma_estimates(high_prices, low_prices)
    alpha = alpha_estimates(beta, gamma)
    
    spread = 2 * (alpha - 1) / (1 + np.exp(alpha))
    return spread
