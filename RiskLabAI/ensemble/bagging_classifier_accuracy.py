from math import comb
import numpy as np

def bagging_classifier_accuracy(
        N: int,
        p: float,
        k: int = 2
) -> float:
    """
    Calculate the accuracy of a bagging classifier.

    The function calculates the accuracy of a bagging classifier based on the given
    parameters and according to the formula:
    
    .. math::
        1 - \sum_{i=0}^{N/k} \binom{N}{i} p^i (1-p)^{N-i}
    
    :param N: Number of independent classifiers.
    :param p: Probability of a classifier labeling a prediction as 1.
    :param k: Number of classes (default is 2).
    :return: Bagging classifier accuracy.

    Reference:
        De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
        Methodology: page 96, "Improved Accuracy" section.
    """

    probability_sum = sum(comb(N, i) * p**i * (1 - p)**(N - i) for i in range(floor(N / k) + 1))
    
    return 1 - probability_sum
