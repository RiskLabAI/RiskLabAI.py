from collections import Counter
from math import log2
from typing import Dict, Tuple


def plug_in_entropy_estimator(
    message: str,
    approximate_word_length: int = 1
) -> float:
    """
    Calculate Plug-in Entropy Estimator.

    :param message: Input encoded message
    :type message: str
    :param approximate_word_length: Approximation of word length, default is 1
    :type approximate_word_length: int
    :return: Calculated Plug-in Entropy Estimator
    :rtype: float
    """
    pmf = probability_mass_function(message, approximate_word_length)
    plug_in_entropy_estimator = -sum(
        pmf[key] * log2(pmf[key])
        for key in pmf.keys()
    ) / approximate_word_length

    return plug_in_entropy_estimator

