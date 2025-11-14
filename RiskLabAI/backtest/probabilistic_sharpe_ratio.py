"""
Implements the Probabilistic Sharpe Ratio (PSR) and related metrics
as described by Marcos Lopez de Prado.

## TODO:
- [ ] Add a `compute_psr_curve` helper function (as seen in the
      original notebook) that iterates `probabilistic_sharpe_ratio`
      over a range of `observed_sharpe_ratio` values to
      easily plot the PSR curve.
"""

from typing import List
import numpy as np
from scipy import stats as ss

def probabilistic_sharpe_ratio(
    observed_sharpe_ratio: float,
    benchmark_sharpe_ratio: float,
    number_of_returns: int,
    skewness_of_returns: float = 0.0,
    kurtosis_of_returns: float = 3.0,
    return_test_statistic: bool = False,
) -> float:
    r"""
    Calculate the Probabilistic Sharpe Ratio (PSR).

    The PSR estimates the probability that the observed Sharpe ratio (SR)
    is greater than a benchmark SR (e.g., the expected maximum SR from
    multiple trials) given the track record length and return moments.

    The test statistic is:
    .. math::
        Z = \frac{(\hat{SR} - SR^*) \sqrt{T-1}}
                 {\sqrt{1 - S \hat{SR} + \frac{K-1}{4} \hat{SR}^2}}

    PSR = N(Z)

    Where:
    - \(\hat{SR}\) is the observed Sharpe ratio
    - \(SR^*\) is the benchmark Sharpe ratio
    - \(T\) is the number of returns
    - \(S\) is the skewness of returns
    - \(K\) is the kurtosis of returns

    Parameters
    ----------
    observed_sharpe_ratio : float
        The observed Sharpe ratio of the strategy.
    benchmark_sharpe_ratio : float
        The benchmark Sharpe ratio (e.g., E[max(SR)]).
    number_of_returns : int
        The number of return observations (T).
    skewness_of_returns : float, default=0.0
        The skewness of the returns (S).
    kurtosis_of_returns : float, default=3.0
        The kurtosis of the returns (K). Assumes 3 for normal.
    return_test_statistic : bool, default=False
        If True, return the Z-statistic instead of the CDF value (PSR).

    Returns
    -------
    float
        The Probabilistic Sharpe Ratio (if `return_test_statistic` is False)
        or the Z test statistic (if True).

    Example
    -------
    >>> psr = probabilistic_sharpe_ratio(
    ...     observed_sharpe_ratio=2.5,
    ...     benchmark_sharpe_ratio=1.0,
    ...     number_of_returns=252,
    ...     skewness_of_returns=-0.5,
    ...     kurtosis_of_returns=4.0
    ... )
    >>> print(f"PSR: {psr:.2f}")
    PSR: 1.00
    """
    denominator = (
        1
        - skewness_of_returns * observed_sharpe_ratio
        + (kurtosis_of_returns - 1) / 4 * observed_sharpe_ratio**2
    )

    # Handle cases where denominator is non-positive due to extreme inputs
    if denominator <= 0:
        return 0.0 if not return_test_statistic else -np.inf

    test_statistic = (
        (observed_sharpe_ratio - benchmark_sharpe_ratio)
        * np.sqrt(number_of_returns - 1)
    ) / np.sqrt(denominator)

    if return_test_statistic:
        return test_statistic

    return ss.norm.cdf(test_statistic)


def benchmark_sharpe_ratio(sharpe_ratio_estimates: List[float]) -> float:
    r"""
    Calculate the expected maximum Sharpe Ratio (Benchmark SR).

    This is used as the benchmark SR* in the PSR calculation. It represents
    the expected maximum SR one would observe from N independent trials.

    .. math::
        SR^* = \sigma_{SR} \left[ (1 - \gamma) \Phi^{-1}(1 - \frac{1}{N})
               + \gamma \Phi^{-1}(1 - \frac{1}{N} e^{-1}) \right]

    Where:
    - \(\sigma_{SR}\) is the standard deviation of SR estimates
    - \(\gamma\) is the Euler-Mascheroni constant
    - \(\Phi^{-1}\) is the inverse CDF of a standard normal distribution
    - \(N\) is the number of SR estimates (trials)

    Parameters
    ----------
    sharpe_ratio_estimates : List[float]
        A list or array of Sharpe ratio estimates from N different trials.

    Returns
    -------
    float
        The Benchmark Sharpe Ratio (E[max(SR)]).

    Example
    -------
    >>> sr_list = [0.5, 1.2, -0.3, 0.8, 1.5, 0.9]
    >>> bsr = benchmark_sharpe_ratio(sr_list)
    >>> print(f"Benchmark SR: {bsr:.2f}")
    Benchmark SR: 1.03
    """
    n_estimates = len(sharpe_ratio_estimates)
    if n_estimates <= 1:
        return np.mean(sharpe_ratio_estimates) if n_estimates == 1 else 0.0

    standard_deviation = np.array(sharpe_ratio_estimates).std()

    term1 = (1 - np.euler_gamma) * ss.norm.ppf(1 - 1 / n_estimates)
    term2 = np.euler_gamma * ss.norm.ppf(1 - 1 / (n_estimates * np.e))

    benchmark_value = standard_deviation * (term1 + term2)

    return benchmark_value