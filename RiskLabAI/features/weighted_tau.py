import scipy.stats as stats
import numpy as np

def calculate_weighted_tau(
    feature_importances: np.ndarray,
    principal_component_ranks: np.ndarray
) -> float:
    """
    Calculate the weighted Kendall's tau (τ) using feature importances and principal component ranks.

    Kendall's tau is a measure of correlation between two rankings. The weighted version of
    Kendall's tau takes into account the weights of the rankings. In this case, the weights
    are the inverse of the principal component ranks.

    :param feature_importances: Vector of feature importances.
    :type feature_importances: np.ndarray
    :param principal_component_ranks: Vector of principal component ranks.
    :type principal_component_ranks: np.ndarray
    :return: Weighted τ value.
    :rtype: float

    .. math::

        \\tau_B = \\frac{(P - Q)}{\\sqrt{(P + Q + T) (P + Q + U)}}

    where:
        - P is the number of concordant pairs
        - Q is the number of discordant pairs
        - T is the number of ties only in the first ranking
        - U is the number of ties only in the second ranking
    """
    return stats.weightedtau(feature_importances, 1 / principal_component_ranks)[0]
