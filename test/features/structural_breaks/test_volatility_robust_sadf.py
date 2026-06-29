"""
Tests for features/structural_breaks/volatility_robust_sadf.py (wild-bootstrap SADF/GSADF).
"""

import numpy as np
import pytest

from RiskLabAI.features.structural_breaks.volatility_robust_sadf import (
    volatility_robust_sadf,
)


def _random_walk(t, rng, vol_kind="constant", strength=6.0):
    sigma = np.ones(t)
    if vol_kind == "varbreak":
        sigma[t // 2 :] = strength
    return np.cumsum(rng.standard_normal(t) * sigma)


def _bubble(t, rng, magnitude=1.06):
    y = np.cumsum(rng.standard_normal(t))
    a, b = int(0.4 * t), int(0.4 * t) + int(0.3 * t)
    for k in range(a, b):
        y[k] = magnitude * y[k - 1] + rng.standard_normal()
    return y


def test_holds_size_on_variance_break_null():
    """
    Replication of the Appraisal 26 mechanism: under a non-stationary-volatility (variance-break) random
    walk with NO bubble, the volatility-robust SADF does not spuriously reject at anything like the rate a
    homoskedastic test would (it holds roughly nominal size).
    """
    rejects = 0
    n_sims = 16
    for s in range(n_sims):
        rng = np.random.default_rng(200 + s)
        y = _random_walk(130, rng, vol_kind="varbreak", strength=6.0)
        if volatility_robust_sadf(y, n_bootstrap=49, random_state=s)["reject_sadf"]:
            rejects += 1
    assert (
        rejects / n_sims <= 0.25
    )  # near nominal (small-sim allowance over the 0.05 target)


def test_keeps_power_on_a_clear_bubble():
    """It still flags a genuine explosive episode (power retained)."""
    detections = 0
    for s in range(8):
        rng = np.random.default_rng(300 + s)
        y = _bubble(160, rng, magnitude=1.07)
        if volatility_robust_sadf(y, n_bootstrap=99, random_state=s)["reject_sadf"]:
            detections += 1
    assert detections >= 5  # majority of clear bubbles flagged


def test_returns_expected_keys_and_pvalues_in_range():
    rng = np.random.default_rng(1)
    res = volatility_robust_sadf(_random_walk(150, rng), n_bootstrap=49, random_state=1)
    assert set(res) == {
        "sadf",
        "gsadf",
        "sadf_pvalue",
        "gsadf_pvalue",
        "reject_sadf",
        "reject_gsadf",
    }
    assert 0.0 < res["sadf_pvalue"] <= 1.0 and 0.0 < res["gsadf_pvalue"] <= 1.0


def test_deterministic_with_seed_and_validates_input():
    rng1, rng2 = np.random.default_rng(5), np.random.default_rng(5)
    y1, y2 = _random_walk(140, rng1), _random_walk(140, rng2)
    a = volatility_robust_sadf(y1, n_bootstrap=39, random_state=7)
    b = volatility_robust_sadf(y2, n_bootstrap=39, random_state=7)
    assert a["sadf_pvalue"] == b["sadf_pvalue"]
    with pytest.raises(ValueError):
        volatility_robust_sadf(np.array([1.0, 2.0, 3.0]))  # too short
