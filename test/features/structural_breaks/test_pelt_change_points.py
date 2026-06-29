"""
Tests for features/structural_breaks/pelt_change_points.py (PELT change-point detection).
"""

import numpy as np
import pytest

pytest.importorskip("ruptures")  # PELT requires the optional ruptures dependency

from RiskLabAI.features.structural_breaks.pelt_change_points import (  # noqa: E402
    pelt_change_points,
)


def _near(cps, target, tol=15):
    return any(abs(c - target) <= tol for c in cps)


def test_recovers_multiple_mean_shifts():
    """
    Replication of the Appraisal 26 mechanism: PELT recovers multiple mean change-points (which CUSUM,
    targeting a single shift, misses).
    """
    rng = np.random.default_rng(0)
    x = np.concatenate(
        [
            rng.standard_normal(100),
            rng.standard_normal(100) + 3.0,
            rng.standard_normal(100) - 2.0,
        ]
    )
    cps = pelt_change_points(x, model="normal", penalty_multiplier=3.0, min_size=10)
    assert _near(cps, 100) and _near(cps, 200)


def test_detects_a_pure_variance_change():
    """With the Gaussian cost PELT detects a variance change-point a mean-only test (CUSUM) is blind to."""
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.standard_normal(120), rng.standard_normal(120) * 3.0])
    cps = pelt_change_points(x, model="normal", penalty_multiplier=3.0, min_size=10)
    assert len(cps) >= 1 and _near(cps, 120, tol=20)


def test_does_not_oversegment_no_change_null():
    """
    Over-segmentation control: on a no-change null PELT returns few change-points (it does not buy
    detections with false positives), across seeds.
    """
    false_positive = 0
    n_sims = 20
    for s in range(n_sims):
        rng = np.random.default_rng(500 + s)
        x = rng.standard_normal(240)
        if (
            len(
                pelt_change_points(
                    x, model="normal", penalty_multiplier=3.0, min_size=10
                )
            )
            > 0
        ):
            false_positive += 1
    assert false_positive / n_sims <= 0.20  # low spurious-segmentation rate


def test_single_clean_shift_converges():
    rng = np.random.default_rng(2)
    x = np.concatenate([rng.standard_normal(150), rng.standard_normal(150) + 4.0])
    cps = pelt_change_points(x)
    assert _near(cps, 150) and len(cps) <= 3  # finds the shift, does not over-segment


def test_short_series_returns_empty():
    assert (
        pelt_change_points(np.random.default_rng(3).standard_normal(8), min_size=10)
        == []
    )
