"""
Tests for backtest/advanced_bet_sizing.py (Platt calibration, distributionally-robust Kelly).
"""

import numpy as np

from RiskLabAI.backtest.advanced_bet_sizing import (
    PlattCalibrator,
    distributionally_robust_kelly_fraction,
    expected_calibration_error,
    kelly_bet_fraction,
)


def test_kelly_fraction_closed_form():
    assert abs(float(kelly_bet_fraction(0.6, 1.0)) - 0.2) < 1e-12
    assert float(kelly_bet_fraction(0.5, 1.0)) == 0.0  # no edge -> no bet
    assert float(kelly_bet_fraction(0.4, 1.0)) == 0.0  # unfavorable clipped to 0


def test_platt_reduces_ece_under_miscalibration():
    """
    Replication of the Appraisal 18 mechanism: Platt calibration sharply lowers ECE on a miscalibrated
    probability (the high-miscalibration regime where it should help).
    """
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 4000)
    true_p = np.where(y == 1, 0.7, 0.3)
    # miscalibrated: push the logit away from calibrated by a factor (distortion)
    logit = np.log(true_p / (1 - true_p)) * 2.0
    p_hat = 1.0 / (1.0 + np.exp(-logit))
    cal = PlattCalibrator().fit(p_hat, y)
    p_cal = cal.transform(p_hat)
    assert expected_calibration_error(p_cal, y) < expected_calibration_error(p_hat, y)
    assert expected_calibration_error(p_cal, y) < 0.05  # well calibrated after Platt


def test_platt_is_safe_no_op_on_calibrated_input():
    """The verdict's no-op side: on already-calibrated probabilities Platt barely changes ECE."""
    rng = np.random.default_rng(1)
    p_hat = rng.uniform(0.05, 0.95, 6000)
    y = (rng.uniform(size=p_hat.size) < p_hat).astype(int)  # calibrated by construction
    ece_before = expected_calibration_error(p_hat, y)
    p_cal = PlattCalibrator().fit(p_hat, y).transform(p_hat)
    ece_after = expected_calibration_error(p_cal, y)
    assert ece_after <= ece_before + 0.02  # no material degradation (near no-op)


def test_platt_single_class_is_identity():
    p_hat = np.array([0.2, 0.5, 0.8])
    cal = PlattCalibrator().fit(p_hat, np.zeros(3, dtype=int))  # one class only
    assert np.allclose(cal.transform(p_hat), p_hat)


def test_dr_kelly_converges_to_full_kelly_with_data():
    """DR-Kelly underbets full Kelly at small n and converges to it as n grows (Sun-Boyd box collapses)."""
    full = float(kelly_bet_fraction(0.6, 1.0))  # 0.2
    small = float(distributionally_robust_kelly_fraction(0.6, 40))
    large = float(distributionally_robust_kelly_fraction(0.6, 1_000_000))
    assert small < large <= full + 1e-9
    assert abs(large - full) < 0.01  # converged


def test_dr_kelly_lower_drawdown_proxy_than_full_kelly():
    """
    Replication of the held-out mechanism direction: at small n DR-Kelly sizes strictly below full Kelly
    (a smaller fraction is the lever for its lower from-initial drawdown).
    """
    for p in (0.55, 0.62):
        dr = float(distributionally_robust_kelly_fraction(p, 40))
        full = float(kelly_bet_fraction(p, 1.0))
        assert dr < full
        # and DR is never more aggressive than full Kelly
        assert dr <= full + 1e-12


def test_dr_kelly_edge_cases():
    assert (
        float(distributionally_robust_kelly_fraction(0.5, 10)) == 0.0
    )  # no edge after pessimism
    # respects max_fraction clip
    assert (
        float(distributionally_robust_kelly_fraction(0.99, 10_000, max_fraction=0.3))
        <= 0.3
    )
    # vectorized input
    out = distributionally_robust_kelly_fraction(np.array([0.55, 0.6, 0.7]), 100)
    assert out.shape == (3,) and np.all(out >= 0)


def test_ece_perfectly_calibrated_is_low():
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    assert expected_calibration_error(p, y) < 0.02
