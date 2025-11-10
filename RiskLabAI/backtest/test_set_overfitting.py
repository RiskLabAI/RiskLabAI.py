"""
Functions for calculating metrics related to test-set overfitting,
including the expected maximum Sharpe Ratio and Type I/II errors.
"""

from typing import List
import numpy as np
import pandas as pd
import scipy.stats as ss
from scipy.stats import norm

def expected_max_sharpe_ratio(
    n_trials: int, mean_sharpe_ratio: float, std_sharpe_ratio: float
) -> float:
    r"""
    Calculate the expected maximum Sharpe Ratio from N trials.

    Uses the formula for the expected maximum of N standard normal variables:
    .. math::
        E[\max(SR)] = \mu + \sigma \left[ (1 - \gamma) \Phi^{-1}(1 - \frac{1}{N}) +
                       \gamma \Phi^{-1}(1 - (N e)^{-1}) \right]

    Where:
    - \(\mu\) is the mean_sharpe_ratio
    - \(\sigma\) is the std_sharpe_ratio
    - \(\gamma\) is the Euler-Mascheroni constant
    - \(\Phi^{-1}\) is the inverse CDF of the standard normal distribution
    - \(N\) is n_trials

    Parameters
    ----------
    n_trials : int
        Number of trials (e.g., number of strategies tested).
    mean_sharpe_ratio : float
        The mean Sharpe Ratio across all trials.
    std_sharpe_ratio : float
        The standard deviation of Sharpe Ratios across all trials.

    Returns
    -------
    float
        The expected maximum Sharpe Ratio.
    """
    if n_trials == 0:
        return 0.0
    if n_trials == 1:
        return mean_sharpe_ratio
        
    euler_gamma = 0.5772156649

    term1 = (1 - euler_gamma) * norm.ppf(1.0 - 1.0 / n_trials)
    term2 = euler_gamma * norm.ppf(1.0 - (n_trials * np.e) ** -1)

    expected_max_sr = mean_sharpe_ratio + std_sharpe_ratio * (term1 + term2)

    return expected_max_sr

def generate_max_sharpe_ratios(
    n_sims: int,
    n_trials_list: List[int],
    std_sharpe_ratio: float,
    mean_sharpe_ratio: float,
) -> pd.DataFrame:
    """
    Generate a DataFrame of maximum Sharpe Ratios from simulations.

    Parameters
    ----------
    n_sims : int
        Number of simulations to run for each `n_trials`.
    n_trials_list : List[int]
        A list of N-trials (e.g., [10, 50, 100] strategies).
    std_sharpe_ratio : float
        The standard deviation of Sharpe Ratios.
    mean_sharpe_ratio : float
        The mean of Sharpe Ratios.

    Returns
    -------
    pd.DataFrame
        A long-format DataFrame with columns ['max_SR', 'n_trials'].
    """
    rng = np.random.default_rng()
    output_list = []

    for n_trials in n_trials_list:
        # Generate all simulations for this n_trials
        sr_sims = rng.normal(
            loc=0.0, scale=1.0, size=(n_sims, n_trials)
        )
        
        # Normalize (z-score) each simulation row
        sr_sims = (sr_sims - sr_sims.mean(axis=1, keepdims=True)) / \
                  sr_sims.std(axis=1, keepdims=True)
        
        # Scale by target mean and std
        sr_sims = mean_sharpe_ratio + sr_sims * std_sharpe_ratio

        # Get max SR for each simulation
        max_sr = sr_sims.max(axis=1)
        
        output_temp = pd.DataFrame({'max_SR': max_sr, 'n_trials': n_trials})
        output_list.append(output_temp)

    return pd.concat(output_list, ignore_index=True)


def mean_std_error(
    n_sims0: int,
    n_sims1: int,
    n_trials: List[int],
    std_sharpe_ratio: float = 1.0,
    mean_sharpe_ratio: float = 0.0,
) -> pd.DataFrame:
    """
    Calculate the mean and standard deviation of the estimation error.

    This function compares the analytical E[max{SR}] with the simulated
    average max{SR} over `n_sims1` repetitions.

    Parameters
    ----------
    n_sims0 : int
        Number of max{SR} simulations used to estimate E[max{SR}] (inner loop).
    n_sims1 : int
        Number of errors on which to compute the std dev (outer loop).
    n_trials : List[int]
        List of numbers of trials (e.g., [10, 20, 50]).
    std_sharpe_ratio : float, default=1.0
        Standard deviation of Sharpe Ratios.
    mean_sharpe_ratio : float, default=0.0
        Mean of Sharpe Ratios.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by 'nTrials' with columns ['meanErr', 'stdErr'].
    """
    # 1. Analytical E[max{SR}]
    expected_sr = pd.Series(
        {
            n: expected_max_sharpe_ratio(n, mean_sharpe_ratio, std_sharpe_ratio)
            for n in n_trials
        },
        name="E[max{SR}]",
    )
    expected_sr.index.name = "nTrials"

    error_list = []
    
    # 2. Run n_sims1 experiments
    for _ in range(int(n_sims1)):
        # 3. Generate simulated max SRs
        simulated_sr = generate_max_sharpe_ratios(
            n_sims=n_sims0,
            n_trials_list=n_trials,
            mean_sharpe_ratio=mean_sharpe_ratio,
            std_sharpe_ratio=std_sharpe_ratio,
        )
        # 4. Average simulated max SRs
        avg_simulated_sr = simulated_sr.groupby("n_trials")["max_SR"].mean()

        # 5. Calculate error
        error = avg_simulated_sr / expected_sr - 1.0
        error_list.append(error.rename('error'))

    all_errors = pd.concat(error_list, axis=1).T

    # 6. Compute mean and std of errors
    output = pd.DataFrame({
        "meanErr": all_errors.mean(),
        "stdErr": all_errors.std(),
    })

    return output


def estimated_sharpe_ratio_z_statistics(
    sharpe_ratio: float,
    t: int,
    true_sharpe_ratio: float = 0.0,
    skew: float = 0.0,
    kurt: int = 3,
) -> float:
    r"""
    Calculate the Z-statistic for an estimated Sharpe Ratio.

    This is the test statistic used in the Probabilistic Sharpe Ratio.

    .. math::
        Z = \frac{(\hat{SR} - SR_0) \sqrt{T - 1}}
                 {\sqrt{1 - S \hat{SR} + \frac{K - 1}{4} \hat{SR}^2}}

    Parameters
    ----------
    sharpe_ratio : float
        Estimated Sharpe Ratio (\(\hat{SR}\)).
    t : int
        Number of observations (T).
    true_sharpe_ratio : float, default=0.0
        The null hypothesis Sharpe Ratio (\(SR_0\)).
    skew : float, default=0.0
        Skewness of returns (S).
    kurt : int, default=3
        Kurtosis of returns (K).

    Returns
    -------
    float
        The calculated Z-statistic.
    """
    denominator = (
        1 - skew * sharpe_ratio + (kurt - 1) / 4.0 * sharpe_ratio**2
    )
    if denominator <= 0:
        return np.nan
        
    z = (sharpe_ratio - true_sharpe_ratio) * np.sqrt(t - 1)
    z /= np.sqrt(denominator)

    return z


def strategy_type1_error_probability(z: float, k: int = 1) -> float:
    r"""
    Calculate the Type I error probability (alpha) for multiple tests.

    This is the probability of at least one false positive (rejecting
    a true null) when conducting `k` independent tests.

    .. math::
        \alpha_k = 1 - (1 - \alpha)^k

    Where \(\alpha = N(-z)\) is the Type I error for a single test.

    Parameters
    ----------
    z : float
        Z-statistic for the significance threshold (e.g., 1.96).
    k : int, default=1
        Number of independent tests.

    Returns
    -------
    float
        The family-wise Type I error rate.
    """
    alpha_single_test = ss.norm.cdf(-z)
    alpha_k = 1 - (1 - alpha_single_test) ** k
    return alpha_k


def theta_for_type2_error(
    sharpe_ratio: float,
    t: int,
    true_sharpe_ratio: float,
    skew: float = 0.0,
    kurt: int = 3,
) -> float:
    r"""
    Calculate the \(\theta\) parameter for Type II error probability.

    .. math::
        \theta = \frac{SR_{True} \sqrt{T - 1}}
                      {\sqrt{1 - S \hat{SR} + \frac{K - 1}{4} \hat{SR}^2}}

    Parameters
    ----------
    sharpe_ratio : float
        The estimated Sharpe Ratio (\(\hat{SR}\)).
    t : int
        Number of observations (T).
    true_sharpe_ratio : float
        The true Sharpe Ratio (\(SR_{True}\)) (the alternative hypothesis).
    skew : float, default=0.0
        Skewness of returns (S).
    kurt : int, default=3
        Kurtosis of returns (K).

    Returns
    -------
    float
        The \(\theta\) parameter.
    """
    denominator = (
        1 - skew * sharpe_ratio + (kurt - 1) / 4.0 * sharpe_ratio**2
    )
    if denominator <= 0:
        return np.nan
        
    theta = true_sharpe_ratio * np.sqrt(t - 1)
    theta /= np.sqrt(denominator)
    return theta


def strategy_type2_error_probability(
    alpha_k: float, k: int, theta: float
) -> float:
    r"""
    Calculate the Type II error probability (beta) for multiple tests.

    This is the probability of failing to reject a false null hypothesis.

    .. math::
        Z_{\alpha} = \Phi^{-1}((1 - \alpha_k)^{1/k})
        \beta = N(Z_{\alpha} - \theta)

    Parameters
    ----------
    alpha_k : float
        The family-wise Type I error rate.
    k : int
        Number of independent tests.
    theta : float
        The \(\theta\) parameter from `theta_for_type2_error`.

    Returns
    -------
    float
        The Type II error probability (\(\beta\)).
    """
    z_alpha = ss.norm.ppf((1 - alpha_k) ** (1.0 / k))
    beta = ss.norm.cdf(z_alpha - theta)
    return beta