"""
Tests for the EDGE effective bid-ask spread estimator
(features/microstructural_features/edge.py).

Mirrors the Julia parity testset ("Features - EDGE spread estimator (parity with
Python)") in ``RiskLabAI.jl/test/runtests.jl``: the exact 12-bar OHLC fixture value,
the NaN degenerate cases, and the mismatched-length raise.
"""

import numpy as np
import pytest

from RiskLabAI.features.microstructural_features import edge_estimator

# Same 12-bar OHLC fixture as the Julia parity testset.
OPEN = [
    100.0,
    101.2,
    100.5,
    102.1,
    101.8,
    103.0,
    102.4,
    104.1,
    103.6,
    105.2,
    104.8,
    106.0,
]
HIGH = [
    101.0,
    101.9,
    101.3,
    102.8,
    102.5,
    103.6,
    103.2,
    104.7,
    104.3,
    105.8,
    105.5,
    106.6,
]
LOW = [99.4, 100.6, 99.9, 101.3, 101.0, 102.2, 101.7, 103.4, 102.9, 104.5, 104.0, 105.3]
CLOSE = [
    101.1,
    100.7,
    101.0,
    102.3,
    101.6,
    103.2,
    102.1,
    104.4,
    103.3,
    105.4,
    104.4,
    106.4,
]

EXPECTED = 0.006962635220907141


def test_edge_parity_fixture():
    """The fixture reproduces the pinned Julia parity value to full precision."""
    assert edge_estimator(OPEN, HIGH, LOW, CLOSE) == EXPECTED


def test_edge_sign_flag_matches_on_fixture():
    """sign=True returns the same value on the fixture (squared spread is positive)."""
    assert edge_estimator(OPEN, HIGH, LOW, CLOSE, sign=True) == EXPECTED


def test_edge_nan_fewer_than_three_observations():
    """Fewer than three observations returns NaN."""
    assert np.isnan(edge_estimator([1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]))


def test_edge_nan_flat_no_trade_bars():
    """Flat / no-trade bars (all prices equal) return NaN."""
    flat = [100.0] * 5
    assert np.isnan(edge_estimator(flat, flat, flat, flat))


def test_edge_mismatched_lengths_raise():
    """Mismatched lengths raise ValueError."""
    with pytest.raises(ValueError):
        edge_estimator([1.0, 2.0, 3.0], [1.0, 2.0], [1.0], [1.0])
