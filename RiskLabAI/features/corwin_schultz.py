import numpy as np
import pandas as pd

"""
function: Corwin and Schultz β Estimation 
reference: De Prado, M. (18) Advances in Financial Machine Learning
methodology: page 285 snippet 19.1
"""


def beta_estimates(
    high_prices: pd.Series,  # high prices vector
    low_prices: pd.Series,  # low prices vector
    window_span: int  # rolling window span
) -> pd.Series:

    # log_ratios = np.log(high_prices / low_prices) ** 2

    # beta = rolling_sum(log_ratios, 2)
    # beta = rolling_mean(beta, window_span)
    # return beta

    log_ratios = np.log(high_prices / low_prices) ** 2
    beta = log_ratios.rolling(window=2).sum()
    beta = beta.rolling(window=window_span).mean()

    return beta

"""
function: Corwin and Schultz γ Estimation 
reference: De Prado, M. (18) Advances in Financial Machine Learning
methodology: page 285 snippet 19.1
"""


def gamma_estimates(
    high_prices: pd.Series,  # high prices vector
    low_prices: pd.Series,  # low prices vector
) -> pd.Series:

    high_prices_max = high_prices.rolling(window=2).max()
    low_prices_min = low_prices.rolling(window=2).min()
    gamma = np.log(high_prices_max / low_prices_min)**2
    return gamma


"""
function: Corwin and Schultz α Estimation 
reference: De Prado, M. (18) Advances in Financial Machine Learning
methodology: page 285 snippet 19.1
"""


def alpha_estimates(
    beta: pd.Series,  # β Estimates vector
    gamma: pd.Series  # γ Estimates vector
) -> pd.Series:

    denominator = 3 - 2 * 2**0.5
    alpha = (2**0.5 - 1) * (beta**0.5) / denominator
    alpha -= (gamma / denominator)**0.5
    alpha[alpha < 0] = 0  # set negative alphas to 0 (see p.727 of paper)

    return alpha


"""
function: Corwin and Schultz spread estimator 
reference: De Prado, M. (18) Advances in Financial Machine Learning
methodology: page 285 snippet 19.1
"""


def corwin_schultz_estimator(
    high_prices: pd.Series,  # high prices vector
    low_prices: pd.Series,  # low prices vector
    windowSpan: int = 20  # rolling window span
) -> pd.Series:

    # Note: S<0 iif α < 0
    beta = beta_estimates(high_prices, low_prices, windowSpan)
    gamma = gamma_estimates(high_prices, low_prices)
    alpha = alpha_estimates(beta, gamma)
    
    spread = 2 * (alpha - 1) / (1 + np.exp(alpha))

    return spread
    
