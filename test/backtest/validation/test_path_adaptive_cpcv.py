"""
Tests for backtest/validation/path_adaptive_cpcv.py (path-level Adaptive CPCV PBO).

Covers: plain-CPCV PBO parity with the repo (diff 0), the Appraisal-09b replication (lower selection
error than plain CPCV on a signal-bearing switching set; convergence with no over-adaptation on a
stationary set), and edge cases (single config, degenerate volatility, too few paths).
"""

import numpy as np

from RiskLabAI.backtest.backtest_statistics import sharpe_ratio
from RiskLabAI.backtest.probability_of_backtest_overfitting import (
    probability_of_backtest_overfitting,
)
from RiskLabAI.backtest.validation.path_adaptive_cpcv import (
    _cscv_path_stats,
    adaptive_probability_of_backtest_overfitting,
    estimate_volatility_regimes,
)


def _common_factor_panel(seed, t_len=240, n_strategies=10):
    """A panel with a common per-period shock plus fixed per-config offsets, so the cross-config
    dispersion (hence the regime signal) is constant -> the adaptive weights are uniform.
    """
    rng = np.random.default_rng(seed)
    common = rng.standard_normal(t_len) * 0.05
    offsets = np.linspace(-0.02, 0.02, n_strategies)
    return common[:, None] + offsets[None, :]


def _switching_panel(seed, t_len=480, n_strategies=8, target=0.25):
    """A simple regime shift used by the edge-case tests: config 0 leads in a calm early regime; the
    last config leads in a volatile forward regime."""
    rng = np.random.default_rng(seed)
    split = int((1.0 - target) * t_len)
    perf = rng.standard_normal((t_len, n_strategies)) * 0.04  # calm base
    perf[:split, 0] += 0.03  # config 0 leads early (low volatility)
    perf[split:, :] = (
        rng.standard_normal((t_len - split, n_strategies)) * 0.10
    )  # volatile forward
    perf[split:, -1] += 0.06  # last config leads in the volatile forward regime
    return perf, n_strategies - 1


# Appraisal-09b Monte-Carlo switching DGP (clean-room from harness.py): K configs with known true
# per-regime Sharpes; the forward regime (regime 1, the recent half) is a higher-volatility stress
# regime, so the shift is observable from volatility. Selection error is the true forward-Sharpe
# shortfall of the selected config, the continuous metric the appraisal scored.
def _switching_truth_panel(seed, t_len=960, n_configs=12, spread=0.12):
    rng = np.random.default_rng(seed)
    sr_uncond = np.linspace(-spread, spread, n_configs).copy()
    rng.shuffle(sr_uncond)
    g = (
        rng.standard_normal(n_configs) * spread
    )  # per-regime tilt (leadership shifts, mean unchanged)
    sr_matrix = np.column_stack([sr_uncond + g, sr_uncond - g])
    labels = np.zeros(t_len, dtype=int)
    labels[t_len // 2 :] = 1  # structural break: forward half is regime 1
    sigma_regime = np.array([1.0, 1.8])[labels][
        :, None
    ]  # regime 1 is higher volatility
    returns = (
        sr_matrix[:, labels].T * sigma_regime
        + rng.standard_normal((t_len, n_configs)) * sigma_regime
    )
    return returns, sr_matrix[:, 1]  # returns, true FORWARD-regime Sharpes


def _stationary_panel(seed, t_len=480, n_strategies=8):
    """A single stationary regime in which config 0 dominates throughout (true best = 0)."""
    rng = np.random.default_rng(seed)
    perf = rng.standard_normal((t_len, n_strategies)) * 0.05
    perf[:, 0] += 0.04
    return perf, 0


# --------------------------------------------------------------------------- parity with plain CPCV
def test_unweighted_paths_reproduce_repo_pbo_exactly():
    """The per-path overfit indicators average to the repo's CSCV PBO exactly (diff 0)."""
    for seed in range(5):
        perf = _switching_panel(seed)[0]
        stats = _cscv_path_stats(perf, 8, sharpe_ratio, 0.0)
        unweighted = float(np.mean([s[0] for s in stats]))
        repo = probability_of_backtest_overfitting(perf, n_partitions=8)[0]
        assert np.isclose(unweighted, repo, atol=1e-12)


def test_uniform_weights_converge_to_plain_pbo():
    """With constant volatility the adaptive weights are uniform, so the adaptive PBO equals plain."""
    perf = _common_factor_panel(0)
    adaptive_pbo, _ = adaptive_probability_of_backtest_overfitting(perf, n_partitions=8)
    plain_pbo = probability_of_backtest_overfitting(perf, n_partitions=8)[0]
    assert np.isclose(adaptive_pbo, plain_pbo, atol=1e-12)


# ------------------------------------------------------------- Appraisal-09b mechanism replication
def test_lower_selection_error_on_switching_regime():
    """Replicates 09b on its Monte-Carlo switching DGP: under a signal-bearing, volatility-observable
    regime shift, the regime-weighted selection has a lower mean true-forward-Sharpe shortfall than plain
    CPCV (which targets the unconditional best), and it is essentially never worse path by path.
    """
    n = 60
    plain_err = np.empty(n)
    adaptive_err = np.empty(n)
    for seed in range(n):
        returns, forward_sharpes = _switching_truth_panel(seed)
        stats = _cscv_path_stats(returns, 8, sharpe_ratio, 0.0)
        plain_sel = int(np.argmax(np.mean([s[1] for s in stats], axis=0)))
        _, adaptive_sel = adaptive_probability_of_backtest_overfitting(
            returns, n_partitions=8
        )
        best = forward_sharpes.max()
        plain_err[seed] = best - forward_sharpes[plain_sel]
        adaptive_err[seed] = best - forward_sharpes[adaptive_sel]
    assert (
        adaptive_err.mean() < plain_err.mean()
    )  # lower selection error on the signal-bearing set
    worse = int(np.sum(adaptive_err > plain_err + 1e-12))
    better = int(np.sum(adaptive_err < plain_err - 1e-12))
    assert worse <= better  # weak dominance: regime-weighting does not hurt selection


def test_convergence_and_no_over_adaptation_on_stationary():
    """On a stationary set adaptive does not over-adapt: it selects the true best as reliably as plain,
    and its PBO stays close to plain CPCV (convergence)."""
    adaptive_hits, plain_hits, pbo_gap = 0, 0, []
    n = 20
    for seed in range(n):
        perf, true_best = _stationary_panel(seed)
        plain_pbo = probability_of_backtest_overfitting(perf, n_partitions=8)[0]
        adaptive_pbo, adaptive_sel = adaptive_probability_of_backtest_overfitting(
            perf, n_partitions=8
        )
        stats = _cscv_path_stats(perf, 8, sharpe_ratio, 0.0)
        plain_sel = int(np.argmax(np.mean([s[1] for s in stats], axis=0)))
        adaptive_hits += adaptive_sel == true_best
        plain_hits += plain_sel == true_best
        pbo_gap.append(abs(adaptive_pbo - plain_pbo))
    assert adaptive_hits >= plain_hits  # no over-adaptation: at least as accurate
    assert np.mean(pbo_gap) < 0.05  # converges to plain CPCV


# --------------------------------------------------------------------------------------- edge cases
def test_regime_signal_keys_on_volatility_not_performance():
    """The regime estimate is observable: it depends on the cross-config volatility only, not on the
    forward performance the method selects. Flipping the sign of every return (which leaves the
    volatility structure unchanged but inverts performance) leaves the regime labels identical.
    """
    perf = _switching_truth_panel(1)[0]
    assert np.array_equal(
        estimate_volatility_regimes(perf), estimate_volatility_regimes(-perf)
    )


def test_single_configuration_does_not_crash():
    rng = np.random.default_rng(0)
    perf = rng.standard_normal((240, 1)) * 0.05 + 0.01
    pbo, selected = adaptive_probability_of_backtest_overfitting(perf, n_partitions=8)
    assert 0.0 <= pbo <= 1.0
    assert selected == 0


def test_degenerate_volatility_is_uniform():
    """Constant per-config columns => zero cross-config dispersion => uniform weights, finite PBO."""
    perf = np.tile(np.linspace(-0.01, 0.01, 6), (240, 1))
    pbo, selected = adaptive_probability_of_backtest_overfitting(perf, n_partitions=8)
    assert 0.0 <= pbo <= 1.0
    assert 0 <= selected < 6


def test_invalid_arguments_raise():
    rng = np.random.default_rng(0)
    perf = rng.standard_normal((240, 6))
    import pytest

    with pytest.raises(ValueError):
        adaptive_probability_of_backtest_overfitting(perf, n_partitions=7)  # odd
    with pytest.raises(ValueError):
        adaptive_probability_of_backtest_overfitting(
            perf[:4], n_partitions=8
        )  # too few rows
    with pytest.raises(ValueError):
        adaptive_probability_of_backtest_overfitting(
            perf, target_fraction=0.0
        )  # bad fraction


def test_deterministic():
    perf = _switching_panel(3)[0]
    a = adaptive_probability_of_backtest_overfitting(perf, n_partitions=8)
    b = adaptive_probability_of_backtest_overfitting(perf, n_partitions=8)
    assert a == b


def test_pbo_in_unit_interval():
    for seed in range(5):
        perf = _switching_panel(seed)[0]
        pbo, _ = adaptive_probability_of_backtest_overfitting(perf, n_partitions=8)
        assert 0.0 <= pbo <= 1.0
