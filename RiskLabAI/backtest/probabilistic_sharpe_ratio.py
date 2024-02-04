import numpy as np
from scipy import stats as ss


def probabilistic_sharpe_ratio(
        observed_sharpe_ratio: float, 
        benchmark_sharpe_ratio: float, 
        number_of_returns: int,                       
        skewness_of_returns: float = 0, 
        kurtosis_of_returns: float = 3,
        return_test_statistic: bool = False,
) -> float:
    """
    Calculates the Probabilistic Sharpe Ratio (PSR) based on observed and benchmark Sharpe ratios.

    The PSR provides a means to test whether a track record would have achieved an observed 
    level of outperformance due to skill or luck. It is calculated using:
    
    .. math::
        \frac{(\hat{SR} - SR^*) \sqrt{T-1}}{\sqrt{1 - S \hat{SR} + \frac{K-1}{4} \hat{SR}^2}}
    
    Where:
    - \(\hat{SR}\) is the observed Sharpe ratio
    - \(SR^*\) is the benchmark Sharpe ratio
    - \(T\) is the number of returns
    - \(S\) is the skewness of returns
    - \(K\) is the kurtosis of returns

    :param observed_sharpe_ratio: The observed Sharpe ratio.
    :param benchmark_sharpe_ratio: The benchmark Sharpe ratio.
    :param number_of_returns: The number of return observations.
    :param skewness_of_returns: The skewness of the returns (default = 0).
    :param kurtosis_of_returns: The kurtosis of the returns (default = 3).
    :param return_test_statistic: Return the test statistic instead of the CDF value.
    :return: The Probabilistic Sharpe Ratio.
    """
    test_value = (
        (observed_sharpe_ratio - benchmark_sharpe_ratio) * np.sqrt(number_of_returns - 1)
    ) / (
        (1 - skewness_of_returns * observed_sharpe_ratio 
        + (kurtosis_of_returns - 1) / 4 * observed_sharpe_ratio ** 2) ** (1 / 2)
    )

    if return_test_statistic:
        return test_value
    
    else:
        return ss.norm.cdf(test_value)

def benchmark_sharpe_ratio(
        sharpe_ratio_estimates: list
) -> float:
    """
    Calculates the Benchmark Sharpe Ratio based on Sharpe ratio estimates.

    The benchmark Sharpe ratio is computed using:
    
    .. math::
        \sigma_{SR} \left[ (1 - \gamma) \Phi^{-1}(1 - \frac{1}{N}) + \gamma \Phi^{-1}(1 - \frac{1}{N} e^{-1}) \right]
    
    Where:
    - \(\sigma_{SR}\) is the standard deviation of Sharpe ratio estimates
    - \(\gamma\) is the Euler's constant
    - \(\Phi^{-1}\) is the inverse of the cumulative distribution function (CDF) of a standard normal distribution
    - \(N\) is the number of Sharpe ratio estimates

    :param sharpe_ratio_estimates: List of Sharpe ratio estimates.
    :return: The Benchmark Sharpe Ratio.
    """
    standard_deviation = np.array(sharpe_ratio_estimates).std()
    benchmark_value = (
        standard_deviation * (
            (1 - np.euler_gamma) * ss.norm.ppf(1 - 1 / len(sharpe_ratio_estimates))
            + np.euler_gamma * ss.norm.ppf(1 - 1 / len(sharpe_ratio_estimates) * np.e ** (-1))
        )
    )
    
    return benchmark_value
