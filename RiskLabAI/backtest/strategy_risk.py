"""
Functions for calculating strategy risk metrics, such as implied
precision, binomial Sharpe ratio, and probability of failure.
"""

from typing import Tuple
import numpy as np
import scipy.stats as ss
import sympy
from sympy import symbols, factor

def sharpe_ratio_trials(p: float, n_run: int) -> Tuple[float, float, float]:
    r"""
    Simulate binomial trials to estimate mean, std dev, and Sharpe ratio.

    Each trial is a binomial outcome (1 or -1) with probability `p`.

    .. math::
        \text{outcome} = \begin{cases} 1 & \text{with probability } p \\
                                     -1 & \text{with probability } 1-p
                       \end{cases}
        SR = \frac{\mu}{\sigma}

    Parameters
    ----------
    p : float
        Probability of success (outcome 1).
    n_run : int
        Number of simulation runs (trials).

    Returns
    -------
    Tuple[float, float, float]
        (mean_outcome, std_outcome, sharpe_ratio)
    """
    outcomes = np.random.binomial(n=1, p=p, size=n_run) * 2 - 1
    mean_outcome = np.mean(outcomes)
    std_outcome = np.std(outcomes)
    sharpe_ratio = mean_outcome / std_outcome if std_outcome > 0 else 0.0

    return mean_outcome, std_outcome, sharpe_ratio


def target_sharpe_ratio_symbolic() -> sympy.Expr:
    r"""
    Calculate the variance of a binomial trial symbolically.

    The variance is:
    .. math::
        Var = E[X^2] - (E[X])^2
        Var = (p u^2 + (1 - p) d^2) - (p u + (1 - p) d)^2

    This function simplifies this expression using SymPy.

    Returns
    -------
    sympy.Expr
        The symbolic expression for variance: `p*(1 - p)*(d - u)**2`
    """
    p, u, d = symbols("p u d")

    m2 = p * u**2 + (1 - p) * d**2
    m1 = p * u + (1 - p) * d
    v = m2 - m1**2

    return factor(v)


def implied_precision(
    stop_loss: float,
    profit_taking: float,
    frequency: float,
    target_sharpe_ratio: float,
) -> float:
    r"""
    Calculate the implied precision (probability) to achieve a target SR.

    Solves the quadratic equation for `p` from the binomial SR formula:
    .. math::
        a = (f + S^2) (pt - sl)^2
        b = (2 f sl - S^2 (pt - sl)) (pt - sl)
        c = f sl^2
        p = \frac{-b + \sqrt{b^2 - 4ac}}{2a}

    Where:
    - \(f\) is the frequency (bets per year)
    - \(S\) is the target_sharpe_ratio
    - \(pt\) is the profit_taking
    - \(sl\) is the stop_loss (as a positive value)

    Parameters
    ----------
    stop_loss : float
        Stop-loss threshold (positive value, e.g., 0.02 for 2%).
    profit_taking : float
        Profit-taking threshold (positive value, e.g., 0.04 for 4%).
    frequency : float
        Number of bets per year.
    target_sharpe_ratio : float
        Target annual Sharpe ratio.

    Returns
    -------
    float
        The required precision (probability of success).
    """
    a = (frequency + target_sharpe_ratio**2) * (profit_taking - stop_loss) ** 2
    b = (
        (2 * frequency * stop_loss - target_sharpe_ratio**2 * (profit_taking - stop_loss))
        * (profit_taking - stop_loss)
    )
    c = frequency * stop_loss**2
    
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return np.nan # No real solution

    precision = (-b + np.sqrt(discriminant)) / (2.0 * a)
    return precision


def bin_frequency(
    stop_loss: float,
    profit_taking: float,
    precision: float,
    target_sharpe_ratio: float,
) -> float:
    r"""
    Calculate the frequency needed to achieve a target SR.

    .. math::
        f = \frac{S^2 (pt - sl)^2 p (1 - p)}{((pt - sl) p - sl)^2}
    
    Note: The original formula had `((pt - sl) * p + sl)` in the denominator,
    which corresponds to `sl` being negative (a loss). This
    implementation assumes `stop_loss` is a positive value, so the
    expected return is `(pt - sl) * p - sl * (1-p) = p*pt - p*sl - sl + p*sl = p*pt - sl`.
    Wait, the original `binomial_sharpe_ratio` uses `(pt - sl) * p + sl`.
    This implies `sl` is treated as a negative number.
    Let's assume `stop_loss` is a positive threshold, so `d = -stop_loss`.
    E[R] = p*pt + (1-p)*(-sl) = p*(pt+sl) - sl
    Stdev = (pt - (-sl)) * sqrt(p(1-p)) = (pt+sl)*sqrt(p(1-p))
    SR = (p*(pt+sl) - sl) / ((pt+sl)*sqrt(p(1-p)))
    Let's stick to the formula provided in `binomial_sharpe_ratio`.

    .. math::
        f = \frac{S^2 (pt - sl)^2 p (1-p)}{((pt - sl) p + sl)^2}

    Parameters
    ----------
    stop_loss : float
        Stop-loss threshold (e.g., -0.02 for 2% loss).
    profit_taking : float
        Profit-taking threshold (e.g., 0.04 for 4% gain).
    precision : float
        The precision (probability of a profitable trade).
    target_sharpe_ratio : float
        The target annual Sharpe ratio.

    Returns
    -------
    float
        The required number of bets per year (frequency).
    """
    if precision <= 0 or precision >= 1:
        return np.inf

    numerator = (
        (target_sharpe_ratio * (profit_taking - stop_loss)) ** 2
        * precision
        * (1 - precision)
    )
    denominator = ((profit_taking - stop_loss) * precision + stop_loss) ** 2
    
    if denominator == 0:
        return np.inf

    return numerator / denominator


def binomial_sharpe_ratio(
    stop_loss: float,
    profit_taking: float,
    frequency: float,
    probability: float,
) -> float:
    r"""
    Calculate the Sharpe Ratio for a binary outcome strategy.

    .. math::
        E[R] = p \cdot pt + (1-p) \cdot sl
        \sigma[R] = (pt - sl) \sqrt{p(1-p)}
        SR_{trade} = \frac{E[R]}{\sigma[R]}
        SR_{annual} = SR_{trade} \times \sqrt{f}

    Parameters
    ----------
    stop_loss : float
        Stop-loss outcome (negative value, e.g., -0.02).
    profit_taking : float
        Profit-taking outcome (positive value, e.g., 0.04).
    frequency : float
        Number of bets per year.
    probability : float
        Probability of success (profit_taking).

    Returns
    -------
    float
        The annualized Sharpe Ratio.
    """
    expected_return = (profit_taking * probability) + (stop_loss * (1 - probability))
    
    p = probability
    stdev_return = (profit_taking - stop_loss) * np.sqrt(p * (1 - p))

    if stdev_return == 0:
        return 0.0 if expected_return == 0 else np.inf * np.sign(expected_return)

    sr_trade = expected_return / stdev_return
    sr_annual = sr_trade * np.sqrt(frequency)
    
    return sr_annual


def mix_gaussians(
    mu1: float,
    mu2: float,
    sigma1: float,
    sigma2: float,
    probability: float,
    n_obs: int,
) -> np.ndarray:
    """
    Generate a mixture of two Gaussian-distributed outcomes.

    Parameters
    ----------
    mu1 : float
        Mean of the first Gaussian (e.g., winning trades).
    mu2 : float
        Mean of the second Gaussian (e.g., losing trades).
    sigma1 : float
        Standard deviation of the first Gaussian.
    sigma2 : float
        Standard deviation of the second Gaussian.
    probability : float
        Probability of drawing from the first Gaussian (e.g., prob of win).
    n_obs : int
        Total number of observations (trades) to generate.

    Returns
    -------
    np.ndarray
        Array of generated outcomes, shuffled.
    """
    n_obs1 = int(n_obs * probability)
    n_obs2 = n_obs - n_obs1
    
    returns1 = np.random.normal(mu1, sigma1, size=n_obs1)
    returns2 = np.random.normal(mu2, sigma2, size=n_obs2)

    returns = np.append(returns1, returns2, axis=0)
    np.random.shuffle(returns)

    return returns


def failure_probability(
    returns: np.ndarray, frequency: float, target_sharpe_ratio: float
) -> float:
    """
    Calculate the probability that the strategy may fail.

    This compares the observed precision (from returns) with the
    implied precision required to meet the target SR.

    Parameters
    ----------
    returns : np.ndarray
        Array of returns from a strategy.
    frequency : float
        Number of bets per year (observed).
    target_sharpe_ratio : float
        The target annual Sharpe ratio.

    Returns
    -------
    float
        Calculated failure probability (a Z-score CDF).
    """
    positive_returns = returns[returns > 0]
    negative_returns = returns[returns <= 0]

    if len(positive_returns) == 0 or len(negative_returns) == 0:
        return 0.0 # Cannot calculate

    profit_taking = positive_returns.mean()
    stop_loss = negative_returns.mean() # This will be negative
    
    # Observed precision
    observed_precision = positive_returns.shape[0] / float(returns.shape[0])

    # Required precision
    required_precision = implied_precision(
        abs(stop_loss), profit_taking, frequency, target_sharpe_ratio
    )
    
    if np.isnan(required_precision):
        return 1.0 # Cannot achieve target SR

    # Probability that observed_precision < required_precision
    # This is a test on a proportion
    p_var = observed_precision * (1 - observed_precision)
    if p_var == 0:
        return 0.0 if observed_precision >= required_precision else 1.0
        
    p_std = np.sqrt(p_var / returns.shape[0]) # Std dev of the proportion
    
    z_score = (observed_precision - required_precision) / p_std
    risk = ss.norm.cdf(z_score) # Prob of being <= required_precision

    return risk

def calculate_strategy_risk(
    mu1: float,
    mu2: float,
    sigma1: float,
    sigma2: float,
    probability: float,
    n_obs: int,
    frequency: float,
    target_sharpe_ratio: float,
) -> float:
    """
    Run a simulation to calculate the strategy risk.

    Parameters
    ----------
    mu1 : float
        Mean of winning trades.
    mu2 : float
        Mean of losing trades.
    sigma1 : float
        Standard deviation of winning trades.
    sigma2 : float
        Standard deviation of losing trades.
    probability : float
        Probability of a winning trade.
    n_obs : int
        Number of observations (trades) in the simulation.
    frequency : float
        Number of bets per year.
    target_sharpe_ratio : float
        Target annual Sharpe ratio.

    Returns
    -------
    float
        Calculated probability of strategy failure.
    """
    returns = mix_gaussians(
        mu1, mu2, sigma1, sigma2, probability, n_obs
    )
    probability_fail = failure_probability(
        returns, frequency, target_sharpe_ratio
    )
    
    print(f"Probability that strategy will fail: {probability_fail:.2%}")
    return probability_fail