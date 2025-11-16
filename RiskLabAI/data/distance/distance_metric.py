"""
Implements information-theoretic and other distance metrics for
financial data analysis.

Includes:
- Variation of Information (VI)
- Mutual Information (MI)
- Optimal number of bins calculation
- Angular distance

Reference:
    De Prado, M. (2020) Advances in financial machine learning.
    John Wiley & Sons, Chapter 3.
"""

import numpy as np
import scipy.stats as ss
from sklearn.metrics import mutual_info_score
from typing import Optional

def calculate_variation_of_information(
    x: np.ndarray, y: np.ndarray, bins: int, norm: bool = False
) -> float:
    """
    Calculate the Variation of Information (VI).

    VI(X, Y) = H(X) + H(Y) - 2 * I(X, Y)

    Reference:
        Snippet 3.2, Page 44.

    Parameters
    ----------
    x : np.ndarray
        First data array.
    y : np.ndarray
        Second data array.
    bins : int
        Number of bins for the 2D histogram.
    norm : bool, default=False
        If True, normalize VI to [0, 1] by dividing by
        the joint entropy H(X, Y).

    Returns
    -------
    float
        The Variation of Information.
    """
    histogram_xy = np.histogram2d(x, y, bins)[0]
    mutual_information = mutual_info_score(
        None, None, contingency=histogram_xy
    )
    
    marginal_x = ss.entropy(histogram_xy.sum(axis=1))
    marginal_y = ss.entropy(histogram_xy.sum(axis=0))

    variation_xy = marginal_x + marginal_y - 2 * mutual_information

    if norm:
        joint_xy = marginal_x + marginal_y - mutual_information
        if joint_xy == 0:
            return 0.0 # Avoid division by zero if entropies are 0
        variation_xy /= joint_xy

    return variation_xy


def calculate_number_of_bins(
    num_observations: int, correlation: Optional[float] = None
) -> int:
    """
    Calculate the optimal number of bins for discretization.

    Uses a formula optimized for either univariate (correlation=None)
    or bivariate (correlation is provided) cases.

    Reference:
        Snippet 3.3, Page 46.

    Parameters
    ----------
    num_observations : int
        Number of observations (N).
    correlation : float, optional
        Correlation between the two series. If None,
        uses the univariate formula.

    Returns
    -------
    int
        The optimal number of bins.
    """
    if correlation is None:
        # Univariate formula
        z = (
            8
            + 324 * num_observations
            + 12 * (36 * num_observations + 729 * num_observations**2) ** 0.5
        ) ** (1 / 3.0)
        bins = round(z / 6.0 + 2.0 / (3 * z) + 1.0 / 3)
    else:
        if np.isclose(correlation, 1.0) or np.isclose(correlation, -1.0):
            # Handle perfect correlation case by setting to almost 1
            correlation = np.sign(correlation) * (1.0 - 1e-10)
            
        if (1.0 - correlation**2) == 0:
            # Handle numerical instability if correlation is still 1
            return calculate_number_of_bins(num_observations, correlation=None)

        # Bivariate formula
        bins = round(
            (2**-0.5)
            * (
                1
                + (
                    1
                    + 24 * num_observations / (1.0 - correlation**2)
                )
                ** 0.5
            )
            ** 0.5
        )
    return int(bins)


def calculate_variation_of_information_extended(
    x: np.ndarray, y: np.ndarray, norm: bool = False
) -> float:
    """
    Calculate Variation of Information with an optimal number of bins.

    This function first calculates the optimal number of bins
    using `calculate_number_of_bins` before computing VI.

    Reference:
        Snippet 3.3, Page 46.

    Parameters
    ----------
    x : np.ndarray
        First data array.
    y : np.ndarray
        Second data array.
    norm : bool, default=False
        If True, normalize the VI.

    Returns
    -------
    float
        The Variation of Information.
    """
    correlation = np.corrcoef(x, y)[0, 1]
    num_bins = calculate_number_of_bins(x.shape[0], correlation=correlation)
    
    return calculate_variation_of_information(x, y, num_bins, norm)


def calculate_mutual_information(
    x: np.ndarray, y: np.ndarray, norm: bool = False
) -> float:
    """
    Calculate Mutual Information (MI) with an optimal number of bins.

    MI(X, Y) = H(X) + H(Y) - H(X, Y)
    Normalized MI = MI(X, Y) / min(H(X), H(Y))

    Reference:
        Snippet 3.4, Page 48.

    Parameters
    ----------
    x : np.ndarray
        First data array.
    y : np.ndarray
        Second data array.
    norm : bool, default=False
        If True, normalize the MI to [0, 1].

    Returns
    -------
    float
        The Mutual Information.
    """
    correlation = np.corrcoef(x, y)[0, 1]
    num_bins = calculate_number_of_bins(x.shape[0], correlation=correlation)
    
    histogram_xy = np.histogram2d(x, y, num_bins)[0]
    mutual_information = mutual_info_score(
        None, None, contingency=histogram_xy
    )

    if norm:
        marginal_x = ss.entropy(histogram_xy.sum(axis=1))
        marginal_y = ss.entropy(histogram_xy.sum(axis=0))

        min_entropy = min(marginal_x, marginal_y)
        if min_entropy == 0:
            return 0.0 # Avoid division by zero
        
        mutual_information /= min_entropy

    return mutual_information


def calculate_distance(
    dependence: np.ndarray, metric: str = "angular"
) -> np.ndarray:
    r"""
    Calculate a distance matrix from a dependence matrix (e.g., correlation).

    Metrics:
    - 'angular': \( D_A(i, j) = \sqrt{0.5 (1 - \rho_{ij})} \)
    - 'absolute_angular': \( D_{AA}(i, j) = \sqrt{0.5 (1 - |\rho_{ij}|)} \)

    Parameters
    ----------
    dependence : np.ndarray
        The dependence (correlation) matrix.
    metric : str, default="angular"
        The metric to use: "angular" or "absolute_angular".

    Returns
    -------
    np.ndarray
        The resulting distance matrix.
    """
    # Clip to handle potential floating point errors
    dependence = np.clip(dependence, -1.0, 1.0)
    
    if metric == "angular":
        distance = ((1 - dependence).round(6) / 2.0) ** 0.5
    elif metric == "absolute_angular":
        distance = ((1 - np.abs(dependence)).round(6) / 2.0) ** 0.5
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return distance


def calculate_kullback_leibler_divergence(
    p: np.ndarray, q: np.ndarray
) -> float:
    """
    Calculate Kullback-Leibler (KL) divergence.

    D_KL(P || Q) = -sum[ p_i * log(q_i / p_i) ]
                 =  sum[ p_i * log(p_i / q_i) ]

    Parameters
    ----------
    p : np.ndarray
        The "true" probability distribution.
    q : np.ndarray
        The "approximating" probability distribution.

    Returns
    -------
    float
        The KL divergence.
    """
    # Ensure probabilities sum to 1
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # Filter for terms where p > 0 and q > 0
    # Where p_i = 0, the term is 0.
    # Where q_i = 0 (and p_i > 0), the term is +inf.
    mask = (p > 0) & (q > 0)
    p_filtered = p[mask]
    q_filtered = q[mask]

    if len(p_filtered) == 0:
        return 0.0 # No overlapping support
        
    # Check if any p_i > 0 corresponds to q_i = 0
    if np.any(p[q == 0] > 0):
        return np.inf

    divergence = -np.sum(p_filtered * np.log(q_filtered / p_filtered))
    return divergence


def calculate_cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
    """
    Calculate the cross-entropy between two distributions.

    H(P, Q) = -sum[ p_i * log(q_i) ]

    Parameters
    ----------
    p : np.ndarray
        The "true" probability distribution.
    q : np.ndarray
        The "approximating" probability distribution.

    Returns
    -------
    float
        The cross-entropy.
    """
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # Filter for terms where p > 0 and q > 0
    # Where p_i = 0, the term is 0.
    # Where q_i = 0 (and p_i > 0), the term is +inf.
    mask = (p > 0) & (q > 0)
    p_filtered = p[mask]
    q_filtered = q[mask]
    
    if len(p_filtered) == 0:
        return 0.0

    # Check if any p_i > 0 corresponds to q_i = 0
    if np.any(p[q == 0] > 0):
        return np.inf

    entropy = -np.sum(p_filtered * np.log(q_filtered))
    return entropy