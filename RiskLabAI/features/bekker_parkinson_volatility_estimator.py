from math import pi
import numpy as np
from CorwinSchultz import *

"""
    function: Bekker-Parkinson volatility σ Estimation 
    reference: De Prado, M. (18) Advances in Financial Machine Learning
    methodology: page 286 snippet 19.2
"""
def sigma_estimates(
    beta:pd.Series, # β Estimates vector
    gamma:pd.Series # gamma Estimates vector
) -> pd.Series:

    k2 = (8 / pi)**0.5
    denominator = 3 - 2 * (2**0.5)
    sigma = (2**0.5 - 1) * (beta ** 0.5) / denominator
    sigma += (gamma / (k2**2 * denominator)) ** 0.5
    # .< does'nt work with missing values
    sigma[sigma < 0] = 0

    return sigma


"""
    function: Bekker-Parkinson volatility Estimation
    reference: De Prado, M. (18) Advances in Financial Machine Learning
    methodology: page 286 Corwin and Schultz section 
"""
def bekker_parkinson_volatility_estimates(
    highPrices:pd.Series, # high prices vector
    lowPrices:pd.Series, # low prices vector
    windowSpan:int=20 # rolling window span
) -> pd.Series:

    beta = beta_estimates(highPrices, lowPrices, windowSpan)
    gamma = gamma_estimates(highPrices, lowPrices)

    return sigma_estimates(beta, gamma)