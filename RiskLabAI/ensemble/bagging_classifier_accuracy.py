"""
Calculates the theoretical accuracy of an ensemble of classifiers.
"""

from math import comb, floor
from typing import Union
import numpy as np

def bagging_classifier_accuracy(
    N: int, p: float, k: int = 2
) -> Union[float, np.float64]:
    r"""
    Calculate the accuracy of a bagging classifier.

    This function calculates the probability that the majority vote of
    an ensemble of `N` classifiers, each with an accuracy `p`,
    will be correct.

    The formula computes the sum of the binomial probabilities for a
    correct majority vote:
    .. math::
        P(\text{correct}) = \sum_{i=\lfloor N/k \rfloor + 1}^{N}
                           \binom{N}{i} p^i (1-p)^{N-i}

    Reference:
        De Prado, M. (2018) Advances in financial machine learning.
        Methodology: page 96, "Improved Accuracy" section.

    Parameters
    ----------
    N : int
        Number of independent classifiers. Must be odd (or will be incremented).
    p : float
        Probability of a single classifier being correct (accuracy).
    k : int, default=2
        Number of classes. Assumes `k=2` for binary classification.

    Returns
    -------
    float
        The bagging classifier's overall accuracy.
    """
    if N % 2 == 0:
        # N must be odd to avoid ties for k=2
        N += 1
        
    # Majority threshold
    majority_threshold = floor(N / k) + 1
    
    # Sum the probabilities from the first majority (threshold) to N
    prob_sum = sum(
        comb(N, i) * (p**i) * ((1 - p) ** (N - i))
        for i in range(majority_threshold, N + 1)
    )

    return prob_sum