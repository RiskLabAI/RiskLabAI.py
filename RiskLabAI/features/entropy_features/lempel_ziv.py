from collections import Counter
from math import log2
from typing import Dict, Tuple


def lempel_ziv_entropy(message: str) -> float:
    """
    Calculate Lempel-Ziv Entropy.

    :param message: Input encoded message
    :type message: str
    :return: Calculated Lempel-Ziv Entropy
    :rtype: float
    """
    library = set()
    message_length = len(message)
    i = 0

    while i < message_length:
        j = i
        while message[i:j + 1] in library and j < message_length:
            j += 1
        library.add(message[i:j + 1])
        i = j + 1

    return len(library) / message_length
