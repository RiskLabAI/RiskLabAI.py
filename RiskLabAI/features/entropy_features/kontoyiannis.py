from collections import Counter
from math import log2
from typing import Dict, Tuple


def longest_match_length(
    message: str,
    i: int,
    n: int
) -> Tuple[int, str]:
    """
    Calculate the length of the longest match.

    :param message: Input encoded message
    :type message: str
    :param i: Index value
    :type i: int
    :param n: Length parameter
    :type n: int
    :return: Tuple containing matched length and substring
    :rtype: tuple
    """
    longest_match = ""
    for l in range(1, n + 1):
        pattern = message[i:i + l + 1]
        for j in range(i - n + 1, i + 1):
            candidate = message[j:j + l + 1]
            if pattern == candidate:
                longest_match = pattern
                break

    return len(longest_match) + 1, longest_match


def kontoyiannis_entropy(
    message: str,
    window: int = None
) -> float:
    """
    Calculate Kontoyiannis Entropy.

    :param message: Input encoded message
    :type message: str
    :param window: Length of expanding window, default is None
    :type window: int or None
    :return: Calculated Kontoyiannis Entropy
    :rtype: float
    """
    output = {"num": 0, "sum": 0, "sub_string": []}
    message_length = len(message)

    if window is None:
        points = range(2, message_length // 2 + 2)
    else:
        window = min(window, message_length // 2)
        points = range(window + 1, message_length - window + 2)

    for i in points:
        n = i if window is None else window
        l, sub_string = longest_match_length(message, i, n)
        output["sum"] += log2(n) / l
        output["sub_string"].append(sub_string)
        output["num"] += 1

    output["h"] = output["sum"] / output["num"]
    output["r"] = 1 - output["h"] / log2(message_length)

    return output["h"]
