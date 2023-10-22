from math import pi
import pandas as pd
from RiskLabAI.features.microstructural_features.corwin_schultz import beta_estimates, gamma_estimates


def sigma_estimates(
    beta: pd.Series,
    gamma: pd.Series
) -> pd.Series:
    """
    Compute Bekker-Parkinson volatility σ estimates.

    This function calculates the Bekker-Parkinson volatility estimates based on the provided
    beta and gamma values. The mathematical formula used is:

    .. math::
        \sigma = \frac{(2^{0.5} - 1) \cdot (\beta ^ {0.5})}{3 - 2 \cdot (2^{0.5})}
                + \left(\frac{\gamma}{\left(\frac{8}{\pi}\right)^{0.5} \cdot (3 - 2 \cdot (2^{0.5}))}\right)^{0.5}

    Negative resulting values are set to 0.

    :param beta: β Estimates vector.
    :param gamma: γ Estimates vector.
    :return: Bekker-Parkinson volatility σ estimates.

    Reference:
        De Prado, M. (2018) Advances in Financial Machine Learning, page 286, snippet 19.2.
    """
    k2 = (8 / pi) ** 0.5
    denominator = 3 - 2 * (2 ** 0.5)

    sigma = (2 ** 0.5 - 1) * (beta ** 0.5) / denominator
    sigma += (gamma / (k2 ** 2 * denominator)) ** 0.5
    sigma[sigma < 0] = 0

    return sigma


def bekker_parkinson_volatility_estimates(
    high_prices: pd.Series,
    low_prices: pd.Series,
    window_span: int = 20
) -> pd.Series:
    """
    Compute Bekker-Parkinson volatility estimates based on high and low prices.

    Utilizes Corwin and Schultz estimation techniques to calculate the Bekker-Parkinson
    volatility. The function first determines the beta and gamma values and then
    uses them to compute the volatility estimates.

    :param high_prices: High prices vector.
    :param low_prices: Low prices vector.
    :param window_span: Rolling window span for beta estimation.
    :return: Bekker-Parkinson volatility estimates.

    Reference:
        De Prado, M. (2018) Advances in Financial Machine Learning, page 286, "Corwin and Schultz" section.
    """
    beta = beta_estimates(high_prices, low_prices, window_span)
    gamma = gamma_estimates(high_prices, low_prices)

    return sigma_estimates(beta, gamma)
