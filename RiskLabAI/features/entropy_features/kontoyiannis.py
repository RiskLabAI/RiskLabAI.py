"""
Implements the Kontoyiannis Entropy estimator (LZ-based).
"""

from math import log2
from typing import Tuple, Optional

def longest_match_length(
    message: str, i: int, n: int
) -> Tuple[int, str]:
    """
    Find the length of the longest match for the substring starting at `i`.

    This function searches for the longest match of `message[i:i+l]`
    within the preceding window `message[max(0, i-n):i]`.

    Parameters
    ----------
    message : str
        Input string.
    i : int
        The starting index of the substring to match.
    n : int
        The length of the look-back window.

    Returns
    -------
    Tuple[int, str]
        - The length of the longest match (L_i) + 1.
        - The matched substring.
    """
    longest_match = ""
    # Iterate through possible lengths `l`
    for l in range(1, n + 1):
        pattern = message[i : i + l]
        
        # Stop if pattern goes beyond message length
        if i + l > len(message):
            break
            
        found = False
        # Look back in the window [max(0, i-n), i-1]
        for j in range(max(0, i - n), i):
            candidate = message[j : j + l]
            if pattern == candidate:
                longest_match = pattern
                found = True
                break
        
        # If pattern of length `l` was not found, the
        # longest match was of length `l-1`.
        if not found:
            break

    return len(longest_match) + 1, longest_match


def kontoyiannis_entropy(
    message: str, window: Optional[int] = None
) -> float:
    r"""
    Calculate Kontoyiannis Entropy (an LZ78-based estimator).

    .. math::
        H_k(n) = \frac{1}{\sum_{i} 1} \sum_{i} \frac{\log_2(n_i)}{L_i(n)}

    Reference:
        Kontoyiannis, I. (1998). "Pointwise redundancy in Lempel-Ziv
        parsing."

    Parameters
    ----------
    message : str
        Input string.
    window : int, optional
        If None, uses an expanding window (full lookback, n_i = i).
        If set, uses a rolling window (n_i = window).

    Returns
    -------
    float
        The calculated Kontoyiannis Entropy (H_k).
    """
    message_length = len(message)
    sum_h = 0.0
    num_points = 0

    if window is None:
        # Expanding window: n = i
        points = range(2, message_length)
    else:
        # Rolling window: n = window
        window = min(window, message_length - 1)
        points = range(window, message_length)

    if not points:
        return 0.0

    for i in points:
        n = i if window is None else window
        if n == 0: continue # Avoid log2(0)
            
        l_i, _ = longest_match_length(message, i, n)
        
        sum_h += log2(n) / l_i
        num_points += 1

    if num_points == 0:
        return 0.0

    return sum_h / num_points