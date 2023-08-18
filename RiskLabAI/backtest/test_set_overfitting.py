import numpy as np
import pandas as pd
from scipy.stats import norm

def expected_max_sharpe_ratio(
        n_trials: int,
        mean_sharpe_ratio: float,
        std_sharpe_ratio: float
) -> float:
    """
    Calculate the expected maximum Sharpe Ratio.

    Uses the formula:
    .. math::
        \text{sharpe\_ratio} = (\text{mean\_sharpe\_ratio} - \gamma) \times \Phi^{-1}(1 - \frac{1}{n\_trials}) + 
                               \gamma \times \Phi^{-1}(1 - n\_trials \times e^{-1})

    where:
    - \(\gamma\) is the Euler's gamma constant
    - \(\Phi^{-1}\) is the inverse of the cumulative distribution function of the standard normal distribution

    :param n_trials: Number of trials.
    :param mean_sharpe_ratio: Mean Sharpe Ratio.
    :param std_sharpe_ratio: Standard deviation of Sharpe Ratios.

    :return: Expected maximum Sharpe Ratio.
    """
    euler_gamma_constant = 0.577215664901532860606512090082402431042159336

    sharpe_ratio = ((1 - euler_gamma_constant) * norm.ppf(1 - 1.0 / n_trials) +
                   euler_gamma_constant * norm.ppf(1 - (n_trials * np.e) ** -1))
    sharpe_ratio = mean_sharpe_ratio + std_sharpe_ratio * sharpe_ratio

    return sharpe_ratio

def generate_max_sharpe_ratios(
        n_sims: int,
        n_trials_list: list,
        std_sharpe_ratio: float,
        mean_sharpe_ratio: float
) -> pd.DataFrame:
    """
    Generate maximum Sharpe Ratios from simulations.

    :param n_sims: Number of simulations.
    :param n_trials_list: List of numbers of trials.
    :param std_sharpe_ratio: Standard deviation of Sharpe Ratios.
    :param mean_sharpe_ratio: Mean of Sharpe Ratios.

    :return: DataFrame containing generated maximum Sharpe Ratios.
    """
    rng = np.random.default_rng()
    output = pd.DataFrame()

    for n_trials in n_trials_list:
        sharpe_ratio_sim = pd.DataFrame(rng.randn(n_sims, n_trials))
        sharpe_ratio_sim = sharpe_ratio_sim.sub(sharpe_ratio_sim.mean(axis=1), axis=0)
        sharpe_ratio_sim = sharpe_ratio_sim.div(sharpe_ratio_sim.std(axis=1), axis=0)
        sharpe_ratio_sim = mean_sharpe_ratio + sharpe_ratio_sim * std_sharpe_ratio

        output_temp = sharpe_ratio_sim.max(axis=1).to_frame('max_SR')
        output_temp['n_trials'] = n_trials
        output = output.append(output_temp, ignore_index=True)

    return output


import pandas as pd
from typing import List


def mean_std_error(
        n_sims0: int,
        n_sims1: int,
        n_trials: List[int],
        std_sharpe_ratio: float = 1,
        mean_sharpe_ratio: float = 0
) -> pd.DataFrame:
    """
    Calculate mean and standard deviation of the predicted errors.

    :param n_sims0: Number of max{SR} used to estimate E[max{SR}].
    :param n_sims1: Number of errors on which std is computed.
    :param n_trials: List of numbers of trials.
    :param std_sharpe_ratio: Standard deviation of Sharpe Ratios.
    :param mean_sharpe_ratio: Mean of Sharpe Ratios.

    :return: DataFrame containing mean and standard deviation of errors.
    """
    sharpe_ratio0 = pd.Series({
        i: expected_max_sharpe_ratio(i, mean_sharpe_ratio, std_sharpe_ratio)
        for i in n_trials
    })
    sharpe_ratio0 = sharpe_ratio0.to_frame('E[max{SR}]')
    sharpe_ratio0.index.name = 'nTrials'
    error = pd.DataFrame()

    for i in range(int(n_sims1)):
        sharpe_ratio1 = generate_max_sharpe_ratios(
            n_sims=n_sims0,
            n_trials=n_trials,
            mean_sharpe_ratio=mean_sharpe_ratio,
            std_sharpe_ratio=std_sharpe_ratio
        )
        sharpe_ratio1 = sharpe_ratio1.groupby('nTrials').mean()
        error_temp = sharpe_ratio0.join(sharpe_ratio1).reset_index()
        error_temp['error'] = error_temp['max{SR}'] / error_temp['E[max{SR}]'] - 1.0
        error = error.append(error_temp)

    output = {
        'meanErr': error.groupby('nTrials')['error'].mean(),
        'stdErr': error.groupby('nTrials')['error'].std()
    }
    output = pd.DataFrame.from_dict(output, orient='columns')

    return output


def estimated_sharpe_ratio_z_statistics(
        sharpe_ratio: float,
        t: int,
        true_sharpe_ratio: float = 0,
        skew: float = 0,
        kurt: int = 3
) -> float:
    """
    Calculate z statistics for the estimated Sharpe Ratios.

    Uses the formula:
    .. math::
        z = \frac{(sharpe\_ratio - true\_sharpe\_ratio) \times \sqrt{t - 1}}{\sqrt{1 - skew \times sharpe\_ratio + \frac{kurt - 1}{4} \times sharpe\_ratio^2}}

    :param sharpe_ratio: Estimated Sharpe Ratio.
    :param t: Number of observations.
    :param true_sharpe_ratio: True Sharpe Ratio.
    :param skew: Skewness of returns.
    :param kurt: Kurtosis of returns.

    :return: Calculated z statistics.
    """
    z = (sharpe_ratio - true_sharpe_ratio) * (t - 1)**0.5
    z /= (1 - skew * sharpe_ratio + (kurt - 1) / 4.0 * sharpe_ratio**2)**0.5

    return z

import scipy.stats as ss

def strategy_type1_error_probability(
        z: float,
        k: int = 1
) -> float:
    """
    Calculate type I error probability of strategies.

    .. math::
        \\alpha_k = 1 - (1 - \\alpha)^k

    :param z: Z statistic for the estimated Sharpe Ratios.
    :param k: Number of tests.

    :return: Calculated type I error probability.
    """
    α = ss.norm.cdf(-z)
    α_k = 1 - (1 - α)**k

    return α_k


def theta_for_type2_error(
        sharpe_ratio: float,
        t: int,
        true_sharpe_ratio: float = 0,
        skew: float = 0,
        kurt: int = 3
) -> float:
    """
    Calculate θ parameter for type II error probability.

    .. math::
        \\theta = \\frac{\\text{true\_sharpe\_ratio} \cdot \\sqrt{t - 1}}{\\sqrt{1 - \\text{skew} \cdot \\text{sharpe\_ratio} + \\frac{\\text{kurt} - 1}{4} \cdot \\text{sharpe\_ratio}^2}}

    :param sharpe_ratio: Estimated Sharpe Ratio.
    :param t: Number of observations.
    :param true_sharpe_ratio: True Sharpe Ratio.
    :param skew: Skewness of returns.
    :param kurt: Kurtosis of returns.

    :return: Calculated θ parameter.
    """
    θ = true_sharpe_ratio * (t - 1)**0.5
    θ /= (1 - skew * sharpe_ratio + (kurt - 1) / 4.0 * sharpe_ratio**2)**0.5

    return θ


def strategy_type2_error_probability(
        α_k: float,
        k: int,
        θ: float
) -> float:
    """
    Calculate type II error probability of strategies.

    .. math::
        z = \\text{ss.norm.ppf}((1 - \\alpha_k)^{1.0 / k})
        \\beta = \\text{ss.norm.cdf}(z - \\theta)

    :param α_k: Type I error.
    :param k: Number of tests.
    :param θ: Calculated θ parameter.

    :return: Calculated type II error probability.
    """
    z = ss.norm.ppf((1 - α_k)**(1.0 / k))
    β = ss.norm.cdf(z - θ)

    return β
