"""
Implements the n-gram count and Probability Mass Function (PMF) calculation.
"""

from collections import Counter


def ngram_counts(message: str, approximate_word_length: int) -> dict[str, int]:
    """
    Count the overlapping n-grams (words) in a message.

    This is the shared counting path used by both the plug-in estimator (via
    :func:`probability_mass_function`) and the bias-corrected estimators, which need
    the raw counts and the sample size rather than the normalized probabilities.

    Parameters
    ----------
    message : str
        Input string.
    approximate_word_length : int
        The length of the "words" or n-grams to analyze (e.g., 1, 2, 3).

    Returns
    -------
    Dict[str, int]
        A dictionary mapping each observed n-gram to its integer count.
    """
    if not message or len(message) < approximate_word_length:
        return {}

    return dict(
        Counter(
            message[i : i + approximate_word_length]
            for i in range(len(message) - approximate_word_length + 1)
        )
    )


def probability_mass_function(
    message: str, approximate_word_length: int
) -> dict[str, float]:
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
    counts = ngram_counts(message, approximate_word_length)

    # The total number of n-grams (windows).
    num_windows = float(sum(counts.values()))
    if num_windows == 0:
        return {}

    return {key: count / num_windows for key, count in counts.items()}
