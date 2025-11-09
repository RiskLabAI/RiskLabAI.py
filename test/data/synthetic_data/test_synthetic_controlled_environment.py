"""
Tests for data/synthetic_data/synthetic_controlled_environment.py
"""

import numpy as np
import pandas as pd
import pytest
from .synthetic_controlled_environment import (
    align_params_length,
    generate_prices_from_regimes,
    parallel_generate_prices,
)

@pytest.fixture
def sample_regimes():
    """Fixture for a simple two-regime model."""
    regimes = {
        "calm": {
            "mu": 0.05, "kappa": 1.0, "theta": 0.04, "xi": 0.1,
            "rho": -0.5, "lam": 0.05, "m": -0.01, "v": 0.02
        },
        "crisis": {
            "mu": -0.1, "kappa": 0.5, "theta": 0.2, "xi": 0.3,
            "rho": -0.8, "lam": 0.2, "m": -0.05, "v": 0.1,
            # Test list-based params
            "v": [0.1, 0.15] # 2-step regime
        }
    }
    # P(calm->calm)=0.9, P(crisis->crisis)=0.8
    transition_matrix = np.array([[0.9, 0.1], [0.2, 0.8]])
    return regimes, transition_matrix

def test_align_params_length():
    """Test the parameter alignment helper."""
    params = {"mu": 0.1, "v": [0.02, 0.03], "xi": 0.5}
    
    aligned_params, max_len = align_params_length(params)
    
    assert max_len == 2
    assert aligned_params["mu"] == [0.1, 0.1]
    assert aligned_params["v"] == [0.02, 0.03]
    assert aligned_params["xi"] == [0.5, 0.5]

def test_generate_prices_from_regimes(sample_regimes):
    """Test the single-path price generation."""
    regimes, tm = sample_regimes
    n_steps = 100
    
    prices, regime_path = generate_prices_from_regimes(
        regimes, tm, total_time=1.0, n_steps=n_steps, random_state=42
    )
    
    assert isinstance(prices, pd.Series)
    assert prices.shape == (n_steps,)
    assert isinstance(regime_path, np.ndarray)
    assert regime_path.shape == (n_steps,)
    
    assert prices.isna().sum() == 0
    assert all(r in regimes for r in regime_path)

def test_parallel_generate_prices(sample_regimes):
    """Test the parallel price generation."""
    regimes, tm = sample_regimes
    n_steps = 50
    n_paths = 4
    
    prices_df, regimes_df = parallel_generate_prices(
        n_paths, regimes, tm, total_time=1.0, 
        n_steps=n_steps, random_state=42, n_jobs=2
    )
    
    assert isinstance(prices_df, pd.DataFrame)
    assert prices_df.shape == (n_steps, n_paths)
    assert isinstance(regimes_df, pd.DataFrame)
    assert regimes_df.shape == (n_steps, n_paths)
    
    # Check that paths are different
    assert not prices_df[0].equals(prices_df[1])