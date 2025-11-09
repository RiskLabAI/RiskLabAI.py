"""
Tests for probability_of_backtest_overfitting.py
"""

import numpy as np
import pytest
from .probability_of_backtest_overfitting import (
    sharpe_ratio,
    performance_evaluation,
    probability_of_backtest_overfitting,
)

@pytest.fixture
def sample_performance_matrix():
    """
    Fixture for a sample performance matrix (T x N).
    T=4, N=3
    Strategy 0: Best (good train, good test)
    Strategy 1: Overfit (good train, bad test)
    Strategy 2: Underfit (bad train, good test)
    """
    matrix = np.array(
        [
            # Train Part 1
            [0.2, 0.2, 0.1],
            [0.2, 0.2, 0.1],
            # Test Part 1
            [0.2, -0.2, 0.2],
            [0.2, -0.2, 0.2],
        ]
    )
    return matrix

def test_sharpe_ratio_numba():
    """Test the Numba-jitted Sharpe ratio."""
    returns = np.array([0.1, 0.1, 0.1, 0.1])
    # std=0, so SR=0
    assert np.isclose(sharpe_ratio(returns, 0.0), 0.0)
    
    returns_var = np.array([0.1, -0.1, 0.1, -0.1])
    # mean=0, std > 0, so SR=0
    assert np.isclose(sharpe_ratio(returns_var, 0.0), 0.0)
    
    returns_pos = np.array([0.1, 0.1, 0.2, 0.0])
    # mean=0.1, std > 0, so SR > 0
    assert sharpe_ratio(returns_pos, 0.0) > 0

def test_performance_evaluation(sample_performance_matrix):
    """Test the performance_evaluation function."""
    train_part = sample_performance_matrix[:2, :] # [0.2, 0.2, 0.1], [0.2, 0.2, 0.1]
    test_part = sample_performance_matrix[2:, :]  # [0.2, -0.2, 0.2], [0.2, -0.2, 0.2]
    n_strat = 3
    
    # Train SRs:
    # S0: mean=0.2, std=0 -> SR=0
    # S1: mean=0.2, std=0 -> SR=0
    # S2: mean=0.1, std=0 -> SR=0
    # `np.argmax` will pick the first one, index 0.
    
    # Test SRs:
    # S0: mean=0.2, std=0 -> SR=0
    # S1: mean=-0.2, std=0 -> SR=0
    # S2: mean=0.2, std=0 -> SR=0
    
    # This is a bad example. Let's add variance.
    train_part = np.array([[0.2, 0.3, 0.1], [0.2, 0.3, 0.1]])
    test_part = np.array([[0.2, -0.2, 0.3], [0.2, -0.2, 0.3]])
    
    # Train SRs (approx, since std=0):
    # S0: ~inf (mean=0.2)
    # S1: ~inf (mean=0.3) -> Best strategy is 1
    # S2: ~inf (mean=0.1)
    
    # Test SRs (approx):
    # S0: ~inf (mean=0.2)
    # S1: ~-inf (mean=-0.2)
    # S2: ~inf (mean=0.3)
    
    # Test Ranks (from lowest to highest): S1, S0, S2
    # Ranks (1-based): [2, 1, 3]
    # Rank of best IS (S1) is 1.
    
    # w_bar = 1 / (3 + 1) = 0.25
    # logit = log(0.25 / 0.75) = log(1/3) < 0
    # is_overfit = True
    
    is_overfit, logit = performance_evaluation(
        train_part, test_part, n_strat, sharpe_ratio, 0.0
    )
    
    assert is_overfit is True
    assert logit < 0

def test_probability_of_backtest_overfitting(sample_performance_matrix):
    """Test the main PBO function."""
    # S=2 (T=4 / 2 = 2 rows per partition)
    # C(2, 1) = 2 combinations
    
    # Combo 1: Train=[0,1], Test=[2,3]
    #   (This is the test from test_performance_evaluation)
    #   Train: [[0.2, 0.2, 0.1], [0.2, 0.2, 0.1]] -> Best S0
    #   Test:  [[0.2, -0.2, 0.2], [0.2, -0.2, 0.2]]
    #   Test SRs: S0(0), S1(0), S2(0).
    #   Ranks: [1, 1, 1]. Rank of S0 is 1.
    #   w_bar = 1 / 4 = 0.25. logit < 0. is_overfit = True
    
    # Combo 2: Train=[2,3], Test=[0,1]
    #   Train: [[0.2, -0.2, 0.2], [0.2, -0.2, 0.2]] -> Best S0
    #   Test:  [[0.2, 0.2, 0.1], [0.2, 0.2, 0.1]]
    #   Test SRs: S0(0), S1(0), S2(0).
    #   Ranks: [1, 1, 1]. Rank of S0 is 1.
    #   w_bar = 1 / 4 = 0.25. logit < 0. is_overfit = True
    
    pbo, logits = probability_of_backtest_overfitting(
        sample_performance_matrix, n_partitions=2, n_jobs=1
    )
    
    # Both combos show overfitting
    assert np.isclose(pbo, 1.0)
    assert len(logits) == 2
    assert np.all(logits < 0)