"""
Implements the Plug-in Entropy estimator.
"""

from math import log2
from typing import Dict
from .pmf import probability_mass_function

def plug_in_entropy_estimator(
    message: str, approximate_word_length: int = 1
) -> float:
    """
    Calculate the Plug-in Entropy Estimator (based on n-gram PMF).

    This is the Shannon entropy applied to the PMF of n-grams
    (words of `approximate_word_length`).

    Parameters
    ----------
    message : str
        Input string.
    approximate_word_length : int, default=1
        The n-gram length.

    Returns
    -------
    float
        The calculated Plug-in Entropy, normalized by word length.
    """
    if not message:
        return 0.0
        
    pmf = probability_mass_function(message, approximate_word_length)
    if not pmf:
        return 0.0

    plug_in_entropy = -sum(
        p * log2(p) for p in pmf.values() if p > 0
    )
    
    # Normalize by word length
    return plug_in_entropy / approximate_word_length