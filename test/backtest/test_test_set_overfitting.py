"""
Tests for test_set_overfitting.py
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from RiskLabAI.backtest.test_set_overfitting import (
    expected_max_sharpe_ratio,
    generate_max_sharpe_ratios,
    mean_std_error,
    estimated_sharpe_ratio_z_statistics,
    strategy_type1_error_probability,
    theta_for_type2_error,
    strategy_type2_error_probability,
)

def test_expected_max_sharpe_ratio():
    """Test E[max SR] calculation."""
    # With 1 trial, E[max SR] = mean SR
    assert np.isclose(
        expected_max_sharpe_ratio(1, 0.5, 1.0), 0.5
    )
    
    # With N trials, E[max SR] > mean SR
    assert expected_max_sharpe_ratio(10, 0.5, 1.0) > 0.5
    
    # Test with 0 trials
    assert np.isclose(expected_max_sharpe_ratio(0, 0.5, 1.0), 0.0)

def test_generate_max_sharpe_ratios():
    """Test the simulation of max SRs."""
    n_sims = 100
    n_trials_list = [10, 20]
    df = generate_max_sharpe_ratios(
        n_sims=n_sims,
        n_trials_list=n_trials_list,
        std_sharpe_ratio=1.0,
        mean_sharpe_ratio=0.0
    )
    
    assert df.shape == (n_sims * len(n_trials_list), 2)
    assert df['n_trials'].value_counts()[10] == n_sims
    assert df['n_trials'].value_counts()[20] == n_sims
    
    # E[max SR] for N=20 should be > E[max SR] for N=10
    assert df[df['n_trials'] == 20]['max_SR'].mean() > \
           df[df['n_trials'] == 10]['max_SR'].mean()

def test_mean_std_error():
    """Test the mean_std_error function."""
    df = mean_std_error(
        n_sims0=100,
        n_sims1=10,
        n_trials=[10, 20],
        std_sharpe_ratio=1.0,
        mean_sharpe_ratio=0.0
    )
    
    assert df.shape == (2, 2)
    assert 'meanErr' in df.columns
    assert 'stdErr' in df.columns
    assert 10 in df.index
    assert 20 in df.index

def test_z_statistics_and_errors():
    """Test Z-stat and error probability functions."""
    # 1. Z-stat (standard normal)
    z = estimated_sharpe_ratio_z_statistics(
        sharpe_ratio=1.96, t=1000, true_sharpe_ratio=0
    )
    assert np.isclose(z, 1.96 * np.sqrt(999), atol=0.1)

    # 2. Type 1 Error
    # For z=1.96, alpha should be 0.025 (one-sided)
    alpha_1 = strategy_type1_error_probability(z=1.96, k=1)
    assert np.isclose(alpha_1, 1 - norm.cdf(1.96), atol=1e-4)
    assert np.isclose(alpha_1, 0.025, atol=1e-3)
    
    # For k=2, alpha_k = 1 - (1-0.025)^2 = 0.049375
    alpha_2 = strategy_type1_error_probability(z=1.96, k=2)
    assert np.isclose(alpha_2, 1 - (1 - alpha_1)**2, atol=1e-4)

    # 3. Theta
    theta = theta_for_type2_error(
        sharpe_ratio=1.0, t=100, true_sharpe_ratio=0.5
    )
    # theta = 0.5 * sqrt(99) / 1 = 4.97
    assert np.isclose(theta, 0.5 * np.sqrt(99))

    # 4. Type 2 Error
    beta = strategy_type2_error_probability(alpha_k=alpha_2, k=2, theta=theta)
    # z_alpha = norm.ppf((1 - 0.049375)**0.5) = norm.ppf(0.975) = 1.96
    # beta = norm.cdf(1.96 - 4.97) = norm.cdf(-3.01)
    assert np.isclose(beta, norm.cdf(1.96 - theta), atol=1e-3)