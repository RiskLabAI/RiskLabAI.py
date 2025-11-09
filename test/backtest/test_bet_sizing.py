"""
Tests for bet_sizing.py
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from .bet_sizing import (
    probability_bet_size,
    average_bet_sizes,
    strategy_bet_sizing,
    betSize,
    TPos,
    getW,
)

def test_probability_bet_size():
    """Test probability_bet_size function."""
    # Prob = 0.5 -> CDF(0) = 0.5 -> 2*0.5 - 1 = 0
    probs = np.array([0, 0, 0])
    sides = np.array([1, -1, 1])
    sizes = probability_bet_size(probs, sides)
    assert np.allclose(sizes, [0, 0, 0])

    # Prob > 0 -> size > 0
    probs_high = np.array([norm.ppf(0.75), norm.ppf(0.75)]) # CDF = 0.75
    sides = np.array([1, -1])
    # Expected: [1 * (2*0.75 - 1), -1 * (2*0.75 - 1)] = [0.5, -0.5]
    sizes_high = probability_bet_size(probs_high, sides)
    assert np.allclose(sizes_high, [0.5, -0.5])

def test_average_bet_sizes_numba():
    """Test the Numba-jitted average_bet_sizes."""
    price_dates = np.arange(10)
    # Bet 1: [0, 5], size = 1.0
    # Bet 2: [2, 7], size = 0.5
    # Bet 3: [8, 9], size = -1.0
    start_dates = np.array([0, 2, 8])
    end_dates = np.array([5, 7, 9])
    bet_sizes = np.array([1.0, 0.5, -1.0])

    avg_sizes = average_bet_sizes(
        price_dates, start_dates, end_dates, bet_sizes
    )

    # Date 0: [0, 5] -> 1.0
    # Date 1: [0, 5] -> 1.0
    # Date 2: [0, 5], [2, 7] -> (1.0 + 0.5) / 2 = 0.75
    # Date 3: [0, 5], [2, 7] -> 0.75
    # Date 4: [0, 5], [2, 7] -> 0.75
    # Date 5: [0, 5], [2, 7] -> 0.75
    # Date 6: [2, 7] -> 0.5
    # Date 7: [2, 7] -> 0.5
    # Date 8: [8, 9] -> -1.0
    # Date 9: [8, 9] -> -1.0
    expected = np.array(
        [1.0, 1.0, 0.75, 0.75, 0.75, 0.75, 0.5, 0.5, -1.0, -1.0]
    )
    assert np.allclose(avg_sizes, expected)

def test_strategy_bet_sizing():
    """Test the strategy_bet_sizing wrapper."""
    price_idx = pd.to_datetime(pd.date_range("2020-01-01", periods=10))
    bet_idx = pd.to_datetime(
        ["2020-01-01", "2020-01-03", "2020-01-09"]
    )
    
    times = pd.Series(
        pd.to_datetime(["2020-01-06", "2020-01-08", "2020-01-10"]),
        index=bet_idx
    )
    sides = pd.Series([1, 1, -1], index=bet_idx)
    # Probs -> CDF(0) = 0.5, CDF(0.674) = 0.75, CDF(0.674) = 0.75
    # Sizes -> 0, 0.5, -0.5
    probs = pd.Series([0.0, norm.ppf(0.75), norm.ppf(0.75)], index=bet_idx)

    avg_sizes = strategy_bet_sizing(price_idx, times, sides, probs)
    
    # Dates:    1  2  3   4   5   6   7   8   9   10
    # Bet 1 (0):  [--------]
    # Bet 2 (0.5):    [----------]
    # Bet 3 (-0.5):                       [----]
    # Active:   1  1  2   2   2   2   1   1   1   1
    # Sizes:
    # 1: 0.0
    # 2: 0.0
    # 3: (0.0 + 0.5) / 2 = 0.25
    # 4: 0.25
    # 5: 0.25
    # 6: 0.25
    # 7: 0.5
    # 8: 0.5
    # 9: (-0.5) / 1 = -0.5
    # 10: (-0.5) / 1 = -0.5
    
    expected_vals = [0, 0, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5, -0.5, -0.5]
    assert np.allclose(avg_sizes.values, expected_vals)

def test_desprado_bet_sizing_snippets():
    """Test snippets 10.4 from de Prado."""
    w = 1.0
    x = 0.5 # divergence
    m = betSize(w, x)
    # m = 0.5 / sqrt(1 + 0.25) = 0.5 / sqrt(1.25) = 0.4472
    assert np.isclose(m, 0.44721359)
    
    # Test getW
    w_calc = getW(x, m)
    assert np.isclose(w, w_calc)
    
    # Test TPos
    pos = TPos(w=1.0, f=10.5, acctualPrice=10.0, maximumPositionSize=100)
    # x = 0.5, m = 0.4472...
    # pos = int(0.4472 * 100) = 44
    assert pos == 44