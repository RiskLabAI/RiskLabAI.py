import numpy as np
import scipy.stats as ss
import sympy
from sympy import symbols, factor


def sharpe_ratio_trials(
        p: float,
        n_run: int
) -> tuple[float, float, float]:
    """
    Simulate trials to calculate the mean, standard deviation, and Sharpe ratio.

    The Sharpe ratio is calculated as follows:

    .. math:: S = \\frac{\\mu}{\\sigma}

    where:
    - \(\\mu\) is the mean of the returns
    - \(\\sigma\) is the standard deviation of the returns

    Args:
        p (float): Probability of success.
        n_run (int): Number of runs.

    Returns:
        tuple[float, float, float]: Tuple containing mean, standard deviation, and Sharpe ratio.
    """
    outcomes = np.random.binomial(n=1, p=p, size=n_run) * 2 - 1
    mean_outcome = np.mean(outcomes)
    std_outcome = np.std(outcomes)
    sharpe_ratio = mean_outcome / std_outcome

    return mean_outcome, std_outcome, sharpe_ratio


def target_sharpe_ratio_symbolic() -> sympy.Add:
    """
    Calculate the target Sharpe ratio using symbolic operations.

    The Sharpe ratio is calculated using the following formula:

    .. math:: S = \\frac{p \\cdot u^2 + (1 - p) \\cdot d^2 - (p \\cdot u + (1 - p) \\cdot d)^2}{\\sigma}

    where:
    - \(p\) is the probability of success
    - \(u\) is the upward movement
    - \(d\) is the downward movement
    - \(\\sigma\) is the standard deviation of the returns

    Returns:
        sympy.Add: Symbolic expression for target Sharpe ratio.
    """
    p, u, d = symbols("p u d")

    m2 = p * u**2 + (1 - p) * d**2
    m1 = p * u + (1 - p) * d
    v = m2 - m1**2

    return factor(v)

import numpy as np


def implied_precision(
        stop_loss: float,
        profit_taking: float,
        frequency: float,
        target_sharpe_ratio: float
) -> float:
    """
    Calculate the implied precision for given parameters.

    The implied precision is calculated as follows:

    .. math::
        a = (f + S^2) * (p - s)^2
        b = (2 * f * s - S^2 * (p - s)) * (p - s)
        c = f * s^2
        precision = (-b + \\sqrt{b^2 - 4 * a * c}) / (2 * a)

    where:
    - \(f\) is the frequency of bets per year
    - \(S\) is the target annual Sharpe ratio
    - \(p\) is the profit-taking threshold
    - \(s\) is the stop-loss threshold

    Args:
        stop_loss (float): Stop-loss threshold.
        profit_taking (float): Profit-taking threshold.
        frequency (float): Number of bets per year.
        target_sharpe_ratio (float): Target annual Sharpe ratio.

    Returns:
        float: Calculated implied precision.
    """
    a = (frequency + target_sharpe_ratio**2) * (profit_taking - stop_loss)**2
    b = (2 * frequency * stop_loss - target_sharpe_ratio**2 * (profit_taking - stop_loss)) * (profit_taking - stop_loss)
    c = frequency * stop_loss**2
    precision = (-b + (b**2 - 4 * a * c)**0.5) / (2 * a)

    return precision


def bin_frequency(
        stop_loss: float,
        profit_taking: float,
        precision: float,
        target_sharpe_ratio: float
) -> float:
    """
    Calculate the number of bets per year needed to achieve a target Sharpe ratio with a certain precision.

    The frequency of bets is calculated as follows:

    .. math::
        frequency = \\frac{S^2 * (p - s)^2 * precision * (1 - precision)}{((p - s) * precision + s)^2}

    where:
    - \(S\) is the target annual Sharpe ratio
    - \(p\) is the profit-taking threshold
    - \(s\) is the stop-loss threshold
    - \(precision\) is the precision rate

    Args:
        stop_loss (float): Stop-loss threshold.
        profit_taking (float): Profit-taking threshold.
        precision (float): Precision rate p.
        target_sharpe_ratio (float): Target annual Sharpe ratio.

    Returns:
        float: Calculated frequency of bets.
    """
    frequency = (
        (target_sharpe_ratio * (profit_taking - stop_loss))**2 * precision * (1 - precision)
    ) / ((profit_taking - stop_loss) * precision + stop_loss)**2

    return frequency

import numpy as np

def binomial_sharpe_ratio(
        stop_loss: float,
        profit_taking: float,
        frequency: float,
        probability: float
) -> float:
    """
    Calculate the Sharpe Ratio for a binary outcome.

    The Sharpe ratio is calculated as follows:

    .. math::
        SR = \\frac{(p - s) * p + s}{(p - s) * \\sqrt{p * (1 - p)}} * \\sqrt{f}

    where:
    - \(p\) is the profit-taking threshold
    - \(s\) is the stop-loss threshold
    - \(f\) is the frequency of bets per year

    Args:
        stop_loss (float): Stop loss threshold.
        profit_taking (float): Profit taking threshold.
        frequency (float): Frequency of bets per year.
        probability (float): Probability of success.

    Returns:
        float: Calculated Sharpe Ratio.
    """
    return (
        ((profit_taking - stop_loss) * probability + stop_loss)
        / ((profit_taking - stop_loss) * (probability * (1 - probability))**0.5)
    ) * frequency**0.5


def mix_gaussians(
        mu1: float,
        mu2: float,
        sigma1: float,
        sigma2: float,
        probability: float,
        n_obs: int
) -> np.ndarray:
    """
    Generate a mixture of Gaussian-distributed bet outcomes.

    Args:
        mu1 (float): Mean of the first Gaussian distribution.
        mu2 (float): Mean of the second Gaussian distribution.
        sigma1 (float): Standard deviation of the first Gaussian distribution.
        sigma2 (float): Standard deviation of the second Gaussian distribution.
        probability (float): Probability of success.
        n_obs (int): Number of observations.

    Returns:
        np.ndarray: Array of generated bet outcomes.
    """
    returns1 = np.random.normal(mu1, sigma1, size=int(n_obs * probability))
    returns2 = np.random.normal(mu2, sigma2, size=int(n_obs) - returns1.shape[0])

    returns = np.append(returns1, returns2, axis=0)
    np.random.shuffle(returns)

    return returns

import numpy as np
import scipy.stats as ss

def failure_probability(
        returns: np.ndarray,
        frequency: float,
        target_sharpe_ratio: float
) -> float:
    """
    Calculate the probability that the strategy may fail.

    Args:
        returns (np.ndarray): Array of returns.
        frequency (float): Number of bets per year.
        target_sharpe_ratio (float): Target annual Sharpe ratio.

    Returns:
        float: Calculated failure probability.
    """
    positive_returns = returns[returns > 0]
    negative_returns = returns[returns <= 0]

    positive_mean = positive_returns.mean()
    negative_mean = negative_returns.mean()
    probability = positive_returns.shape[0] / float(returns.shape[0])

    threshold_probability = implied_precision(
        negative_mean, positive_mean, frequency, target_sharpe_ratio
    )
    risk = ss.norm.cdf(threshold_probability, probability, probability * (1 - probability))

    return risk

def calculate_strategy_risk(
        mu1: float,
        mu2: float,
        sigma1: float,
        sigma2: float,
        probability: float,
        n_obs: int,
        frequency: float,
        target_sharpe_ratio: float
) -> float:
    """
    Calculate the strategy risk in practice.

    Args:
        mu1 (float): Mean of the first Gaussian distribution.
        mu2 (float): Mean of the second Gaussian distribution.
        sigma1 (float): Standard deviation of the first Gaussian distribution.
        sigma2 (float): Standard deviation of the second Gaussian distribution.
        probability (float): Probability of success.
        n_obs (int): Number of observations.
        frequency (float): Number of bets per year.
        target_sharpe_ratio (float): Target annual Sharpe ratio.

    Returns:
        float: Calculated probability of strategy failure.
    """
    returns = mix_gaussians(mu1, mu2, sigma1, sigma2, probability, n_obs)
    probability_fail = failure_probability(returns, frequency, target_sharpe_ratio)
    print("Probability that strategy will fail:", probability_fail)

    return probability_fail
