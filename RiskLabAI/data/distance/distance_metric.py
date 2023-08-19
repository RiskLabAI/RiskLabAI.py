import numpy as np
import scipy.stats as ss
import math
from sklearn.metrics import mutual_info_score


def calculate_variation_of_information(
    x: np.ndarray,
    y: np.ndarray,
    bins: int,
    norm: bool = False
) -> float:
    """
    Calculates Variation of Information.

    :param x: First data array.
    :param y: Second data array.
    :param bins: Number of bins for the histogram.
    :param norm: If True, the result will be normalized.

    :return: Variation of Information.
    
    Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    Methodology: Snippet 3.2, Page 44
    """
    histogram_xy = np.histogram2d(x, y, bins)[0]
    mutual_information = mutual_info_score(None, None, contingency=histogram_xy)
    marginal_x = ss.entropy(np.histogram(x, bins)[0])
    marginal_y = ss.entropy(np.histogram(y, bins)[0])
    variation_xy = marginal_x + marginal_y - 2 * mutual_information

    if norm:
        joint_xy = marginal_x + marginal_y - mutual_information
        variation_xy /= joint_xy

    return variation_xy


def calculate_number_of_bins(
    num_observations: int,
    correlation: float = None
) -> int:
    """
    Calculates the optimal number of bins for discretization.

    :param num_observations: Number of observations.
    :param correlation: Correlation value. If None, the function will use the univariate case.

    :return: Optimal number of bins.

    Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    Methodology: Snippet 3.3, Page 46
    """
    if correlation is None:
        z = (8 + 324 * num_observations + 12 * (36 * num_observations + 729 * num_observations ** 2) ** .5) ** (1 / 3.)
        bins = round(z / 6. + 2. / (3 * z) + 1. / 3)
    else:
        bins = round(2 ** -.5 * (1 + (1 + 24 * num_observations / (1. - correlation ** 2)) ** .5) ** .5)
    return int(bins)


def calculate_variation_of_information_extended(
    x: np.ndarray,
    y: np.ndarray,
    norm: bool = False
) -> float:
    """
    Calculates Variation of Information with calculating number of bins.

    :param x: First data array.
    :param y: Second data array.
    :param norm: If True, the result will be normalized.

    :return: Variation of Information.

    Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    Methodology: Snippet 3.3, Page 46
    """
    num_bins = calculate_number_of_bins(x.shape[0], correlation=np.corrcoef(x, y)[0, 1])
    histogram_xy = np.histogram2d(x, y, num_bins)[0]
    mutual_information = mutual_info_score(None, None, contingency=histogram_xy)
    marginal_x = ss.entropy(np.histogram(x, num_bins)[0])
    marginal_y = ss.entropy(np.histogram(y, num_bins)[0])
    variation_xy = marginal_x + marginal_y - 2 * mutual_information

    if norm:
        joint_xy = marginal_x + marginal_y - mutual_information
        variation_xy /= joint_xy

    return variation_xy


def calculate_mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    norm: bool = False
) -> float:
    """
    Calculates Mutual Information with calculating number of bins.

    :param x: First data array.
    :param y: Second data array.
    :param norm: If True, the result will be normalized.

    :return: Mutual Information.

    Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    Methodology: Snippet 3.4, Page 48
    """
    num_bins = calculate_number_of_bins(x.shape[0], correlation=np.corrcoef(x, y)[0, 1])
    histogram_xy = np.histogram2d(x, y, num_bins)[0]
    mutual_information = mutual_info_score(None, None, contingency=histogram_xy)

    if norm:
        marginal_x = ss.entropy(np.histogram(x, num_bins)[0])
        marginal_y = ss.entropy(np.histogram(y, num_bins)[0])
        mutual_information /= min(marginal_x, marginal_y)

    return mutual_information


def calculate_distance(
    dependence: np.ndarray,
    metric: str = "angular"
) -> np.ndarray:
    """
    Calculates distance from a dependence matrix.

    :param dependence: Dependence matrix.
    :param metric: Metric used to calculate distance. Available options are "angular" and "absolute_angular".

    :return: Distance matrix.

    Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    """
    if metric == "angular":
        distance = ((1 - dependence).round(5) / 2.) ** 0.5
    elif metric == "absolute_angular":
        distance = ((1 - abs(dependence)).round(5) / 2.) ** 0.5
    return distance


def calculate_kullback_leibler_divergence(
    p: np.ndarray,
    q: np.ndarray
) -> float:
    """
    Calculates Kullback-Leibler divergence from two discrete probability distributions defined on the same probability space.

    :param p: First distribution.
    :param q: Second distribution.

    :return: Kullback-Leibler divergence.

    Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    """
    divergence = -(p * np.log(q / p)).sum()
    return divergence


def calculate_cross_entropy(
    p: np.ndarray,
    q: np.ndarray
) -> float:
    """
    Calculates cross-entropy from two discrete probability distributions defined on the same probability space.

    :param p: First distribution.
    :param q: Second distribution.

    :return: Cross-entropy.

    Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    """
    entropy = -(p * np.log(q)).sum()
    return entropy
