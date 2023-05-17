from math import floor
from scipy.special import comb
import numpy as np


"""
function: Calculates accuracy of bagging classifier
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: page 96 Improved Accuracy section
"""
def bagging_classifier_accuracy(
    N: int,  # number of independent classifers
    p: float,  # The accuracy of a classifier is the probability p of labeling a prediction as 1
    k: int = 2,  # number of classes
) -> float:

    probability_sum = 0
    for i in range(0, int(N / k) + 1):
        probability_sum += comb(N, i) * p**i * (1 - p)**(N - i)
    
    return 1 - probability_sum