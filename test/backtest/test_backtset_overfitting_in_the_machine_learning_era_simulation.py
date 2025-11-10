"""
Tests for backtest_overfitting_simulation.py
"""

import numpy as np
import pandas as pd
import pytest
from RiskLabAI.backtest.backtest_overfitting_in_the_machine_learning_era_simulation import (
    sharpe_ratio,
    sortino_ratio,
    expected_shortfall,
    financial_features_backtest_overfitting_simulation,
    get_cpu_info,
)

@pytest.fixture
def sample_prices():
    """Fixture for a sample price series."""
    return pd.Series(
        np.cumprod(1 + np.random.normal(0.001, 0.01, 300)),
        index=pd.date_range("2020-01-01", periods=300)
    )

def test_local_metric_functions():
    """Test the locally defined metric functions."""
    returns = pd.Series([0.01, 0.01, 0.01, 0.01])
    # Test Sharpe
    # mean=0.01, std=0 -> SR=0
    assert np.isclose(sharpe_ratio(returns, 0.0), 0.0) 
    
    returns_var = pd.Series([0.02, -0.01, 0.02, -0.01])
    # Test Sortino
    # mean=0.005, rf=0
    # downside_returns = [-0.01, -0.01], std = 0
    # sortino = inf
    assert np.isclose(sortino_ratio(returns_var, 0.0), np.inf)

    # Test ES
    returns_es = pd.Series([-0.1, -0.08, -0.07, -0.05, -0.02, 0.01])
    # 5% VAR is not well-defined, percentile will pick lowest
    # VAR(5%) = -0.1
    # ES = mean(-0.1) = -0.1
    es = expected_shortfall(returns_es, 0.0, confidence_level=0.05)
    assert np.isclose(es, -0.1)

def test_financial_features_generation(sample_prices):
    """Test the financial_features function."""
    features = financial_features_backtest_overfitting_simulation(
        sample_prices, noise_scale=0.0
    )
    
    assert isinstance(features, pd.DataFrame)
    assert features.shape[0] == sample_prices.shape[0]
    # Check for a few expected columns
    assert "FracDiff" in features.columns
    assert "Volatility" in features.columns
    assert "Log MACD Hist" in features.columns
    assert "Kumo Breakout" in features.columns
    
    # Check that noise is applied
    features_noised = financial_features_backtest_overfitting_simulation(
        sample_prices, noise_scale=1.0, random_state=42
    )
    # Volatility should be different
    assert not features["Volatility"].equals(features_noised["Volatility"])

@pytest.mark.skipif(platform.system() == "Windows", reason="lscpu not on Windows")
def test_get_cpu_info():
    """Test the CPU info function (on non-Windows)."""
    info = get_cpu_info()
    assert isinstance(info, dict)
    assert "Model name" in info
    assert "CPU(s)" in info