"""
Implements the Lempel-Ziv (LZ) Entropy estimator.
"""

def lempel_ziv_entropy(message: str) -> float:
    """
    Calculate the Lempel-Ziv (LZ) complexity as an entropy estimator.

    This function calculates the number of unique substrings encountered
    during a one-pass traversal of the message, normalized by the
    message length.

    Parameters
    ----------
    message : str
        Input string.

    Returns
    -------
    float
        The calculated Lempel-Ziv Entropy.
    """
    if not message:
        return 0.0

    library = set()
    message_length = len(message)
    i = 0

    while i < message_length:
        j = i
        # Find the longest substring starting at `i` that is *not* in the library
        while j < message_length and message[i : j + 1] in library:
            j += 1
        
        # Add the new, unseen substring to the library
        library.add(message[i : j + 1])
        i = j + 1

    return len(library) / message_length