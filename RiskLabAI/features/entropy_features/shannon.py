from collections import Counter
from math import log2
from typing import Dict, Tuple


def shannon_entropy(message: str) -> float:
    """
    Calculate Shannon Entropy.

    :param message: Input encoded message
    :type message: str
    :return: Calculated Shannon Entropy
    :rtype: float
    """
    character_counts = Counter(message)
    message_length = len(message)

    entropy = -sum(
        count / message_length * log2(count / message_length)
        for count in character_counts.values()
    )

    return entropy
