from collections import Counter
from math import log2
from typing import Dict, Tuple


def probability_mass_function(
    message: str,
    approximate_word_length: int
) -> Dict[str, float]:
    """
    Calculate Probability Mass Function.

    :param message: Input encoded message
    :type message: str
    :param approximate_word_length: Approximation of word length
    :type approximate_word_length: int
    :return: Probability Mass Function
    :rtype: dict
    """
    library = Counter(message[i:i + approximate_word_length]
                     for i in range(len(message) - approximate_word_length + 1))

    denominator = float(len(message) - approximate_word_length)
    probability_mass_function_ = {key: len(library[key]) / denominator
                                 for key in library}

    return probability_mass_function_

