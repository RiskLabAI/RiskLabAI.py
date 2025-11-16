# RiskLabAI/ensemble/bagging_classifier_accuracy.py
"""
Implements the theoretical accuracy of a bagging classifier,
as described in "Advances in Financial Machine Learning" by de Prado (2018),
Chapter 6, Section 6.2, p. 86.
"""
# import numpy as np  <-- Removed unused import
from scipy.stats import binom

def bagging_classifier_accuracy(N: int, p: float) -> float:
    """
    Calculates the theoretical accuracy of a bagging classifier
    assuming independent classifiers.

    The bagging classifier is correct if the majority of
    classifiers are correct.

    Parameters
    ----------
    N : int
        Number of classifiers (must be odd).
    p : float
        Accuracy of a single classifier (0 <= p <= 1).

    Returns
    -------
    float
        The accuracy of the ensemble.
    """
    if N % 2 == 0:
        raise ValueError(f"Number of estimators N must be odd. Got {N}.")
        
    # The majority threshold.
    # e.g., if N=101, k=50. We need 51 or more correct votes.
    k = (N - 1) // 2
    
    # Probability of k or fewer successes (P(X <= k))
    prob_majority_wrong = binom.cdf(k, N, p)
    
    # Probability of more than k successes (P(X > k))
    prob_majority_correct = 1.0 - prob_majority_wrong
    
    return prob_majority_correct