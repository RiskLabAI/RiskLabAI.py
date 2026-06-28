"""
Tests for backtest/validation/leakage_aware_hpo.py (leakage-aware HPO methodology + Deflated-Sharpe gate).
"""

import numpy as np
import pandas as pd
import pytest

from RiskLabAI.backtest.validation.leakage_aware_hpo import (
    deflated_sharpe_gate,
    leakage_aware_hpo,
)


def test_gate_erases_no_edge_series_at_high_trial_count():
    """
    Replication of the Appraisal 20 honest finding: a no-edge out-of-sample return series does NOT pass
    the Deflated-Sharpe gate once it is deflated by a large HPO trial count (tuning yields no edge).
    """
    rng = np.random.default_rng(0)
    no_edge = rng.standard_normal(800) * 0.01  # ~zero Sharpe
    gate = deflated_sharpe_gate(no_edge, n_trials=150, trial_sharpe_std=0.05)
    assert gate["deflated_sharpe"] < 0.5
    assert gate["passes"] is False


def test_gate_benchmark_grows_with_trial_count():
    """More trials raise the expected-maximum-Sharpe bar, so the same series is deflated harder."""
    rng = np.random.default_rng(1)
    r = rng.standard_normal(800) * 0.01 + 0.0005
    few = deflated_sharpe_gate(r, n_trials=2, trial_sharpe_std=0.05)
    many = deflated_sharpe_gate(r, n_trials=200, trial_sharpe_std=0.05)
    assert many["benchmark_sharpe"] > few["benchmark_sharpe"]
    assert many["deflated_sharpe"] <= few["deflated_sharpe"] + 1e-9


def test_gate_passes_a_strong_edge_with_single_trial():
    """A genuinely strong OOS Sharpe with no multiplicity (1 trial) clears the gate."""
    rng = np.random.default_rng(2)
    strong = rng.standard_normal(1000) * 0.01 + 0.006  # high Sharpe
    gate = deflated_sharpe_gate(strong, n_trials=1, trial_sharpe_std=0.05)
    assert gate["benchmark_sharpe"] == 0.0  # E[max] of one trial is the mean (0)
    assert gate["deflated_sharpe"] > 0.9 and gate["passes"]


def _purged_dataset(n=240, seed=0):
    """Synthetic classification with a learnable signal, indexed by event start/end times for purging."""
    rng = np.random.default_rng(seed)
    feat = rng.standard_normal((n, 4))
    y = (feat[:, 0] + 0.3 * rng.standard_normal(n) > 0).astype(int)
    start = pd.date_range("2020-01-01", periods=n, freq="D")
    x = pd.DataFrame(feat, index=start, columns=[f"f{i}" for i in range(4)])
    y = pd.Series(y, index=start)
    times = pd.Series(
        start + pd.Timedelta(days=3), index=start
    )  # 3-day label horizon (overlap)
    return x, y, times


def test_leakage_aware_hpo_runs_and_returns_valid_result():
    """leakage_aware_hpo runs Optuna under PurgedKFold and returns a coherent result."""
    optuna = pytest.importorskip("optuna")  # noqa: F841
    from sklearn.ensemble import RandomForestClassifier

    x, y, times = _purged_dataset()

    def suggest(trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 20, 60, step=20),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
        }

    def factory(params):
        return RandomForestClassifier(random_state=0, n_jobs=1, **params)

    result = leakage_aware_hpo(
        suggest,
        factory,
        x,
        y,
        times,
        n_trials=6,
        sampler="tpe",
        n_splits=4,
        embargo=0.0,
        random_state=0,
    )
    assert set(result) >= {"best_params", "best_score", "n_trials", "trial_scores"}
    assert result["n_trials"] == 6
    assert len(result["trial_scores"]) >= 1
    assert 0.0 <= result["best_score"] <= 1.0
    assert "n_estimators" in result["best_params"]


def test_leakage_aware_hpo_rejects_bad_sampler():
    pytest.importorskip("optuna")
    from sklearn.ensemble import RandomForestClassifier

    x, y, times = _purged_dataset(n=120)
    with pytest.raises(ValueError):
        leakage_aware_hpo(
            lambda t: {"max_depth": t.suggest_int("max_depth", 2, 4)},
            lambda p: RandomForestClassifier(**p),
            x,
            y,
            times,
            n_trials=2,
            sampler="bogus",
        )
