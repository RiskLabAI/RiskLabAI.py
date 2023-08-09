import numpy as np
from sympy import *
import scipy.stats as ss

def sharpe_ratio_trials(p: float, n_run: int) -> tuple:
    """
    Simulate trials to calculate the mean, standard deviation, and Sharpe ratio.

    :param p: Probability of success.
    :param n_run: Number of runs.
    :return: Tuple containing mean, standard deviation, and Sharpe ratio.
    """
    output = []

    for _ in range(n_run):
        random = np.random.binomial(n=1, p=p)
        x = 1 if random == 1 else -1
        output.append(x)

    mean_output = np.mean(output)
    std_output = np.std(output)
    sharpe_ratio = mean_output / std_output

    return mean_output, std_output, sharpe_ratio

def target_sharpe_ratio_symbolic() -> sympy.Add:
    """
    Calculate the target Sharpe ratio using symbolic operations.

    :return: Symbolic expression for target Sharpe ratio.
    """
    p, u, d = symbols("p u d")

    m2 = p * u**2 + (1 - p) * d**2
    m1 = p * u + (1 - p) * d
    v = m2 - m1**2

    return factor(v)

def implied_precision(stop_loss: float, profit_taking: float, frequency: float, target_sharpe_ratio: float) -> float:
    """
    Calculate the implied precision for given parameters.

    :param stop_loss: Stop loss threshold.
    :param profit_taking: Profit taking threshold.
    :param frequency: Number of bets per year.
    :param target_sharpe_ratio: Target annual Sharpe ratio.
    :return: Calculated implied precision.
    """
    a = (frequency + target_sharpe_ratio**2) * (profit_taking - stop_loss)**2
    b = (2 * frequency * stop_loss - target_sharpe_ratio**2 * (profit_taking - stop_loss)) * (profit_taking - stop_loss)
    c = frequency * stop_loss**2
    precision = (-b + (b**2 - 4 * a * c)**0.5) / (2 * a)

    return precision

def bin_frequency(stop_loss: float, profit_taking: float, precision: float, target_sharpe_ratio: float) -> float:
    """
    Calculate the number of bets per year needed to achieve a target Sharpe ratio with a certain precision.

    :param stop_loss: Stop loss threshold.
    :param profit_taking: Profit taking threshold.
    :param precision: Precision rate p.
    :param target_sharpe_ratio: Target annual Sharpe ratio.
    :return: Calculated frequency of bets.
    """
    frequency = (
        (target_sharpe_ratio * (profit_taking - stop_loss))**2 * precision * (1 - precision)
    ) / ((profit_taking - stop_loss) * precision + stop_loss)**2

    if not np.isclose(binSR(stop_loss, profit_taking, frequency, precision), target_sharpe_ratio):
        return None

    return frequency

def binSR(sl: float, pt: float, frequency: float, p: float) -> float:
    """
    Calculate the Sharpe Ratio function.

    :param sl: Stop loss threshold.
    :param pt: Profit taking threshold.
    :param frequency: Frequency of bets per year.
    :param p: Probability of success.
    :return: Calculated Sharpe Ratio.
    """
    return (
        ((pt - sl) * p + sl)
        / ((pt - sl) * (p * (1 - p))**0.5)
    ) * frequency**0.5

def mixGaussians(
    μ1: float,
    μ2: float,
    σ1: float,
    σ2: float,
    probability: float,
    nObs: int
) -> np.ndarray:
    """
    Generate a mixture of Gaussian-distributed bet outcomes.

    :param μ1: Mean of the first Gaussian distribution.
    :param μ2: Mean of the second Gaussian distribution.
    :param σ1: Standard deviation of the first Gaussian distribution.
    :param σ2: Standard deviation of the second Gaussian distribution.
    :param probability: Probability of success.
    :param nObs: Number of observations.
    :return: Array of generated bet outcomes.
    """
    return1 = np.random.normal(μ1, σ1, size=int(nObs * probability))
    return2 = np.random.normal(μ2, σ2, size=int(nObs) - return1.shape[0])

    returns = np.append(return1, return2, axis=0)
    np.random.shuffle(returns)

    return returns

def failure_probability(returns: np.ndarray, frequency: float, target_sharpe_ratio: float) -> float:
    """
    Calculate the probability that the strategy may fail.

    :param returns: Array of returns.
    :param frequency: Number of bets per year.
    :param target_sharpe_ratio: Target annual Sharpe ratio.
    :return: Calculated failure probability.
    """
    rPositive, rNegative = returns[returns > 0].mean(), returns[returns <= 0].mean()
    p = returns[returns > 0].shape[0] / float(returns.shape[0])
    thresholdP = implied_precision(rNegative, rPositive, frequency, target_sharpe_ratio)
    risk = ss.norm.cdf(thresholdP, p, p * (1 - p))

    return risk

def calculate_strategy_risk(
    μ1: float,
    μ2: float,
    σ1: float,
    σ2: float,
    probability: float,
    nObs: int,
    frequency: float,
    target_sharpe_ratio: float
) -> float:
    """
    Calculate the strategy risk in practice.

    :param μ1: Mean of the first Gaussian distribution.
    :param μ2: Mean of the second Gaussian distribution.
    :param σ1: Standard deviation of the first Gaussian distribution.
    :param σ2: Standard deviation of the second Gaussian distribution.
    :param probability: Probability of success.
    :param nObs: Number of observations.
    :param frequency: Number of bets per year.
    :param target_sharpe_ratio: Target annual Sharpe ratio.
    :return: Calculated probability of strategy failure.
    """
    returns = mixGaussians(μ1, μ2, σ1, σ2, probability, nObs)
    probability_fail = failure_probability(returns, frequency, target_sharpe_ratio)
    print("Probability that strategy will fail:", probability_fail)

    return probability_fail
