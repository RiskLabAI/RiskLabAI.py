import scipy.stats as Stats
import numpy as np

def weighted_tau(
    feature_importances: np.ndarray,
    principal_component_ranks: np.ndarray
) -> float:
    """
    Calculate weighted τ using feature importances and principal component ranks.

    :param feature_importances: Vector of feature importances
    :type feature_importances: np.ndarray
    :param principal_component_ranks: Vector of principal component ranks
    :type principal_component_ranks: np.ndarray
    :return: Weighted τ value
    :rtype: float
    """
    return Stats.weightedtau(feature_importances, 1 / principal_component_ranks)[0]
