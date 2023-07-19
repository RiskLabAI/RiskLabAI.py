import scipy.stats as Stats
import numpy as np

"""
function: Weighted τ calculation
reference: De Prado, M. (2018) Advances In Financial Machine Learning
methodology: page 121 Orthogonal Features section snippet 8.6
"""
def weighted_tau(
    feature_importances:np.ndarray, # vector of feature importances 
    principal_component_ranks:np.ndarray, # vector of principal component ranks
)->float:

    return Stats.weightedtau(feature_importances, 1 / principal_component_ranks)[0]
