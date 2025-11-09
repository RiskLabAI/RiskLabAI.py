"""
Tests for probabilistic_sharpe_ratio.py
"""

import numpy as np
import pytest
from scipy.stats import norm
from .probabilistic_sharpe_ratio import (
    probabilistic_sharpe_ratio,
    benchmark_sharpe_ratio,
)

def test_probabilistic_sharpe_ratio_normal():
    """
    Test PSR with normal parameters (skew=0, kurtosis=3).
    """
    # 1. Observed SR matches benchmark -> Z=0, PSR=0.5
    psr = probabilistic_sharpe_ratio(
        observed_sharpe_ratio=1.0,
        benchmark_sharpe_ratio=1.0,
        number_of_returns=252,
    )
    assert np.isclose(psr, 0.5)

    # 2. Observed SR > benchmark -> Z>0, PSR>0.5
    psr = probabilistic_sharpe_ratio(
        observed_sharpe_ratio=1.5,
        benchmark_sharpe_ratio=1.0,
        number_of_returns=252,
    )
    assert psr > 0.5

    # 3. Observed SR < benchmark -> Z<0, PSR<0.5
    psr = probabilistic_sharpe_ratio(
        observed_sharpe_ratio=0.5,
        benchmark_sharpe_ratio=1.0,
        number_of_returns=252,
    )
    assert psr < 0.5

def test_probabilistic_sharpe_ratio_non_normal():
    """
    Test PSR with non-normal parameters.
    """
    # High skew and kurtosis should adjust the denominator
    psr_normal = probabilistic_sharpe_ratio(
        observed_sharpe_ratio=2.0,
        benchmark_sharpe_ratio=1.0,
        number_of_returns=100,
        skewness_of_returns=0.0,
        kurtosis_of_returns=3.0,
    )

    psr_non_normal = probabilistic_sharpe_ratio(
        observed_sharpe_ratio=2.0,
        benchmark_sharpe_ratio=1.0,
        number_of_returns=100,
        skewness_of_returns=-1.0,  # Negative skew
        kurtosis_of_returns=5.0,   # High kurtosis
    )
    
    # Denominator (normal) = 1
    # Denominator (non-normal) = 1 - (-1)*(2) + (5-1)/4 * (2**2) = 1 + 2 + 4 = 7
    # Z-stat (non-normal) will be lower, so PSR will be lower.
    assert psr_non_normal < psr_normal

def test_probabilistic_sharpe_ratio_statistic():
    """
    Test the return_test_statistic flag.
    """
    z_stat = probabilistic_sharpe_ratio(
        observed_sharpe_ratio=1.5,
        benchmark_sharpe_ratio=1.0,
        number_of_returns=100,
        skewness_of_returns=0,
        kurtosis_of_returns=3,
        return_test_statistic=True,
    )
    # Z = (1.5 - 1.0) * sqrt(99) / sqrt(1) = 0.5 * 9.95 = 4.97
    assert np.isclose(z_stat, 0.5 * np.sqrt(99))
    assert np.isclose(norm.cdf(z_stat), 1.0) # Very high prob

def test_benchmark_sharpe_ratio():
    """
    Test the benchmark Sharpe ratio calculation.
    """
    sr_list = [0.5, 1.0, 1.5, 0.8, 1.2]
    n_estimates = 5
    std_dev = np.std(sr_list)
    
    bsr = benchmark_sharpe_ratio(sr_list)
    
    # Manual calculation
    term1 = (1 - np.euler_gamma) * norm.ppf(1 - 1 / n_estimates)
    term2 = np.euler_gamma * norm.ppf(1 - 1 / (n_estimates * np.e))
    expected_bsr = std_dev * (term1 + term2)
    
    assert np.isclose(bsr, expected_bsr)

def test_benchmark_sharpe_ratio_edge_cases():
    """
    Test benchmark_sharpe_ratio with 0 or 1 estimate.
    """
    assert np.isclose(benchmark_sharpe_ratio([]), 0.0)
    assert np.isclose(benchmark_sharpe_ratio([1.5]), 1.5)