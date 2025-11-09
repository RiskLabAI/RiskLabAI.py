"""
Implements the Shannon Entropy estimator.
"""

from collections import Counter
from math import log2

def shannon_entropy(message: str) -> float:
    """
    Calculate the Shannon Entropy of a message.

    Parameters
    ----------
    message : str
        Input string (e.g., a discretized time series "1-1101...").

    Returns
    -------
    float
        The calculated Shannon Entropy (in bits).
    """
    if not message:
        return 0.0

    character_counts = Counter(message)
    message_length = len(message)

    entropy = -sum(
        (count / message_length) * log2(count / message_length)
        for count in character_counts.values()
    )

    return entropy