"""
Implements the Probability Mass Function (PMF) calculation.
"""

from collections import Counter
from typing import Dict

def probability_mass_function(
    message: str, approximate_word_length: int
) -> Dict[str, float]:
    """
    Calculate the Probability Mass Function (PMF) of n-grams.

    Parameters
    ----------
    message : str
        Input string.
    approximate_word_length : int
        The length of the "words" or n-grams to analyze (e.g., 1, 2, 3).

    Returns
    -------
    Dict[str, float]
        A dictionary mapping each n-gram to its probability.
    """
    if not message or len(message) < approximate_word_length:
        return {}
        
    # Find all n-grams (words)
    library = Counter(
        message[i : i + approximate_word_length]
        for i in range(len(message) - approximate_word_length + 1)
    )

    # The total number of n-grams
    num_windows = float(len(message) - approximate_word_length + 1)
    if num_windows == 0:
        return {}

    # Calculate probability for each n-gram
    pmf = {
        key: count / num_windows for key, count in library.items()
    }

    return pmf