"""
Calculates the weighted Kendall's tau.
"""

import scipy.stats as stats
import numpy as np

def calculate_weighted_tau(
    feature_importances: np.ndarray, principal_component_ranks: np.ndarray
) -> float:
    r"""
    Calculate the weighted Kendall's tau (\(\tau\)).

    This function calculates a weighted correlation between two rankings.
    Here, it's used to compare feature importances to their ranks,
    weighted by the inverse of the rank (giving more weight to
    disagreements at the top).

    Parameters
    ----------
    feature_importances : np.ndarray
        Vector of feature importances.
    principal_component_ranks : np.ndarray
        Vector of ranks (e.g., 1, 2, 3, ...).

    Returns
    -------
    float
        The weighted Kendall's tau correlation coefficient.
    """
    # Weights are the inverse of the rank
    weights = 1.0 / principal_component_ranks
    
    tau, _ = stats.weightedtau(feature_importances, weights)
    return tau