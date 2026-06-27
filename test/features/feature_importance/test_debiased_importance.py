"""
Tests for features/feature_importance/debiased_importance.py (MDI+ and CPI).
"""

import numpy as np
import pandas as pd

from RiskLabAI.features.feature_importance.debiased_importance import (
    conditional_predictive_impact,
    mdi_plus_importance,
)


def _supervised(n, rng, n_noise=3, high_card=False):
    """A binary task: feature 'sig' drives the label; the rest are noise (optionally high-cardinality)."""
    sig = rng.standard_normal(n)
    data = {"sig": sig}
    for j in range(n_noise):
        if high_card and j == 0:
            data[f"noise{j}"] = rng.integers(0, n, n).astype(
                float
            )  # high-cardinality noise
        else:
            data[f"noise{j}"] = rng.standard_normal(n)
    x = pd.DataFrame(data)
    y = pd.Series((sig + 0.3 * rng.standard_normal(n) > 0).astype(int))
    return x, y


def test_mdi_plus_ranks_signal_top():
    """MDI+ ranks the true signal feature above all noise features."""
    rng = np.random.default_rng(0)
    x, y = _supervised(500, rng)
    imp = mdi_plus_importance(x, y)
    assert imp.idxmax() == "sig"


def test_mdi_plus_rejects_high_cardinality_noise():
    """
    Replication of the Appraisal 10 mechanism: MDI+ does not let a high-cardinality noise feature
    out-rank the true signal (the inflation MDI is prone to).
    """
    rng = np.random.default_rng(1)
    x, y = _supervised(800, rng, high_card=True)
    imp = mdi_plus_importance(x, y)
    assert imp["sig"] > imp["noise0"]  # signal beats the high-cardinality noise


def test_mdi_plus_returns_series_over_columns():
    rng = np.random.default_rng(2)
    x, y = _supervised(300, rng)
    imp = mdi_plus_importance(x, y)
    assert list(imp.index) == list(x.columns)
    assert np.all(np.isfinite(imp.to_numpy()))


def test_cpi_significant_for_signal_not_noise():
    """CPI's test rejects for the signal feature and not for pure noise."""
    rng = np.random.default_rng(3)
    x, y = _supervised(500, rng)
    cpi = conditional_predictive_impact(x, y)
    assert cpi.loc["sig", "p_value"] < 0.05
    assert cpi.loc["noise0", "p_value"] > 0.05


def test_cpi_size_calibrated_on_null_features():
    """
    Calibration: across seeds with a single pure-noise feature, CPI rejects the null at roughly the
    nominal rate (allowing the mild finite-sample elevation documented for the method).
    """
    rejects = 0
    n_sims = 40
    for s in range(n_sims):
        rng = np.random.default_rng(200 + s)
        sig = rng.standard_normal(300)
        x = pd.DataFrame({"sig": sig, "null": rng.standard_normal(300)})
        y = pd.Series((sig + 0.3 * rng.standard_normal(300) > 0).astype(int))
        cpi = conditional_predictive_impact(x, y)
        rejects += cpi.loc["null", "p_value"] < 0.05
    assert (
        rejects / n_sims <= 0.20
    )  # near nominal with finite-sample / small-sim allowance


def test_cpi_returns_importance_and_pvalue():
    rng = np.random.default_rng(4)
    x, y = _supervised(300, rng)
    cpi = conditional_predictive_impact(x, y)
    assert set(cpi.columns) == {"importance", "p_value"}
    assert list(cpi.index) == list(x.columns)
