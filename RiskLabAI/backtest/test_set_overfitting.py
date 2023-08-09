import numpy as np
import pandas as pd
from scipy.stats import norm
import scipy.stats as ss

def expected_max_sharpe_ratio(nTrials: int, mean_sharpe_ratio: float, std_sharpe_ratio: float) -> float:
    """
    Calculate the expected maximum Sharpe Ratio.

    :param nTrials: Number of trials.
    :param mean_sharpe_ratio: Mean Sharpe Ratio.
    :param std_sharpe_ratio: Standard deviation of Sharpe Ratios.
    :return: Expected maximum Sharpe Ratio.
    """
    emc = 0.577215664901532860606512090082402431042159336  # Euler's gamma constant

    sharpe_ratio = (1 - emc) * norm.ppf(1 - 1.0 / nTrials) + emc * norm.ppf(1 - (nTrials * np.e)**-1)
    sharpe_ratio = mean_sharpe_ratio + std_sharpe_ratio * sharpe_ratio

    return sharpe_ratio

def generated_max_sharpe_ratio(nSims: int, nTrials: list, std_sharpe_ratio: float, mean_sharpe_ratio: float) -> pd.DataFrame:
    """
    Generate maximum Sharpe Ratios from simulations.

    :param nSims: Number of simulations.
    :param nTrials: Array of numbers of trials.
    :param std_sharpe_ratio: Standard deviation of Sharpe Ratios.
    :param mean_sharpe_ratio: Mean of Sharpe Ratios.
    :return: DataFrame containing generated maximum Sharpe Ratios.
    """
    rng = np.random.RandomState()
    output = pd.DataFrame()

    for nTrials_ in nTrials:
        sharpe_ratio = pd.DataFrame(rng.randn(nSims, nTrials_))
        sharpe_ratio = sharpe_ratio.sub(sharpe_ratio.mean(axis=1), axis=0)
        sharpe_ratio = sharpe_ratio.div(sharpe_ratio.std(axis=1), axis=0)
        sharpe_ratio = mean_sharpe_ratio + sharpe_ratio * std_sharpe_ratio

        output_ = sharpe_ratio.max(axis=1).to_frame('max{SR}')
        output_['nTrials'] = nTrials_
        output = output.append(output_, ignore_index=True)

    return output

def mean_std_error(nSims0: int, nSims1: int, nTrials: list, std_sharpe_ratio: float = 1, mean_sharpe_ratio: float = 0) -> pd.DataFrame:
    """
    Calculate mean and standard deviation of the predicted errors.

    :param nSims0: Number of max{SR} used to estimate E[max{SR}].
    :param nSims1: Number of errors on which std is computed.
    :param nTrials: Array of numbers of trials.
    :param std_sharpe_ratio: Standard deviation of Sharpe Ratios.
    :param mean_sharpe_ratio: Mean of Sharpe Ratios.
    :return: DataFrame containing mean and standard deviation of errors.
    """
    sharpe_ratio0 = pd.Series({i: expected_max_sharpe_ratio(i, mean_sharpe_ratio, std_sharpe_ratio) for i in nTrials})
    sharpe_ratio0 = sharpe_ratio0.to_frame('E[max{SR}]')
    sharpe_ratio0.index.name = 'nTrials'
    error = pd.DataFrame()

    for i in range(int(nSims1)):
        sharpe_ratio1 = generated_max_sharpe_ratio(nSims=nSims0, nTrials=nTrials, mean_sharpe_ratio=mean_sharpe_ratio, std_sharpe_ratio=std_sharpe_ratio)
        sharpe_ratio1 = sharpe_ratio1.groupby('nTrials').mean()
        error_ = sharpe_ratio0.join(sharpe_ratio1).reset_index()
        error_['error'] = error_['max{SR}'] / error_['E[max{SR}]'] - 1.0
        error = error.append(error_)

    output = {'meanErr': error.groupby('nTrials')['error'].mean()}
    output['stdErr'] = error.groupby('nTrials')['error'].std()
    output = pd.DataFrame.from_dict(output, orient='columns')

    return output

def estimated_sharpe_ratio_z_statistics(sharpe_ratio: float, t: int, sharpe_ratio_: float = 0, skew: float = 0, kurt: int = 3) -> float:
    """
    Calculate z statistics for the estimated Sharpe Ratios.

    :param sharpe_ratio: Estimated Sharpe Ratio.
    :param t: Number of observations.
    :param sharpe_ratio_: True Sharpe Ratio.
    :param skew: Skewness of returns.
    :param kurt: Kurtosis of returns.
    :return: Calculated z statistics.
    """
    z = (sharpe_ratio - sharpe_ratio_) * (t - 1)**0.5
    z /= (1 - skew * sharpe_ratio + (kurt - 1) / 4.0 * sharpe_ratio**2)**0.5

    return z

def strategy_type1_error_probability(z: float, k: int = 1) -> float:
    """
    Calculate type I error probability of strategies.

    :param z: Z statistic for the estimated Sharpe Ratios.
    :param k: Number of tests.
    :return: Calculated type I error probability.
    """
    α = ss.norm.cdf(-z)
    α_k = 1 - (1 - α)**k

    return α_k

def theta_for_type2_error(sharpe_ratio: float, t: int, sharpe_ratio_: float = 0, skew: float = 0, kurt: int = 3) -> float:
    """
    Calculate θ parameter for type II error probability.

    :param sharpe_ratio: Estimated Sharpe Ratio.
    :param t: Number of observations.
    :param sharpe_ratio_: True Sharpe Ratio.
    :param skew: Skewness of returns.
    :param kurt: Kurtosis of returns.
    :return: Calculated θ parameter.
    """
    θ = sharpe_ratio_ * (t - 1)**.5
    θ /= (1 - skew * sharpe_ratio + (kurt - 1) / 4.0 * sharpe_ratio**2)**0.5

    return θ

def strategy_type2_error_probability(α_k: float, k: int, θ: float) -> float:
    """
    Calculate type II error probability of strategies.

    :param α_k: Type I error.
    :param k: Number of tests.
    :param θ: Calculated θ parameter.
    :return: Calculated type II error probability.
    """
    z = ss.norm.ppf((1 - α_k)**(1.0 / k))
    β = ss.norm.cdf(z - θ)

    return β
