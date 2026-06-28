"""
Tests for backtest/robust_statistics.py (Conditional Expected Drawdown; Ledoit-Wolf Sharpe-diff test).
"""

import numpy as np

from RiskLabAI.backtest.robust_statistics import (
    _drawdown_series,
    _wealth_from_returns,
    conditional_expected_drawdown,
    sharpe_difference_test,
)


def _max_drawdown(returns):
    return float(_drawdown_series(_wealth_from_returns(returns)).max())


def _heavy_track(rng, n=126, scale=0.01, df=3):
    return rng.standard_t(df, size=n) * scale


def test_ced_in_unit_interval_and_monotone_in_alpha():
    rng = np.random.default_rng(0)
    r = rng.standard_normal(2000) * 0.01
    ced_90 = conditional_expected_drawdown(r, horizon=60, alpha=0.90)
    ced_95 = conditional_expected_drawdown(r, horizon=60, alpha=0.95)
    assert 0.0 <= ced_90 < 1.0
    # a deeper tail (higher alpha) cannot be smaller than a shallower one
    assert ced_95 >= ced_90 - 1e-9


def test_ced_lower_estimator_variance_than_maxdrawdown():
    """
    Replication of the Appraisal 22 mechanism: on short heavy-tailed tracks CED (a tail mean of the
    max-drawdown distribution) has lower across-seed estimator variance than max-drawdown (a single
    extreme order statistic).
    """
    ced_vals, mdd_vals = [], []
    for s in range(160):
        rng = np.random.default_rng(s)
        track = _heavy_track(rng, n=126, df=3)
        ced_vals.append(conditional_expected_drawdown(track, horizon=40, alpha=0.90))
        mdd_vals.append(_max_drawdown(track))
    cv_ced = np.std(ced_vals) / np.mean(ced_vals)
    cv_mdd = np.std(mdd_vals) / np.mean(mdd_vals)
    assert cv_ced < cv_mdd  # CED is the more stable estimator


def test_ced_edge_cases():
    # series shorter than the horizon falls back to the whole-track max drawdown (finite, in [0,1))
    short = conditional_expected_drawdown(
        np.array([0.01, -0.02, 0.03, -0.01]), horizon=50
    )
    assert 0.0 <= short < 1.0
    # a monotonically rising track has no drawdown
    assert conditional_expected_drawdown(np.full(200, 0.001), horizon=20) == 0.0
    import pytest

    with pytest.raises(ValueError):
        conditional_expected_drawdown(np.zeros(100), horizon=10, alpha=1.5)


def _ar_pair(rng, n, phi, scale=0.01):
    """Two AR(1) return series with the SAME population Sharpe (the null SR_A = SR_B)."""
    eps_a = rng.standard_normal(n + 50)
    eps_b = rng.standard_normal(n + 50)
    a = np.zeros(n + 50)
    b = np.zeros(n + 50)
    for t in range(1, n + 50):
        a[t] = phi * a[t - 1] + eps_a[t]
        b[t] = phi * b[t - 1] + eps_b[t]
    return a[50:] * scale, b[50:] * scale  # same DGP -> equal Sharpe (both ~0)


def test_lw_holds_size_where_naive_overrejects_under_dependence():
    """
    Replication of the held-out mechanism: under AR(0.3) the naive Sharpe-difference test over-rejects
    the (true) null while the Ledoit-Wolf test holds roughly nominal size.
    """
    n_sims = 80
    naive_rej = lw_rej = 0
    for s in range(n_sims):
        rng = np.random.default_rng(1000 + s)
        a, b = _ar_pair(rng, n=250, phi=0.5)
        naive_rej += sharpe_difference_test(a, b, method="naive")["reject"]
        lw_rej += sharpe_difference_test(
            a, b, method="ledoit_wolf", n_boot=199, random_state=s
        )["reject"]
    naive_size = naive_rej / n_sims
    lw_size = lw_rej / n_sims
    assert (
        naive_size > lw_size
    )  # naive over-rejects relative to LW under serial dependence
    assert (
        lw_size <= 0.20
    )  # LW stays near nominal (small-sim allowance over the 0.05 target)


def test_naive_and_lw_converge_on_iid():
    """On i.i.d. returns the two tests give similar point estimates and neither systematically rejects."""
    rng = np.random.default_rng(7)
    a, b = _ar_pair(rng, n=400, phi=0.0)
    naive = sharpe_difference_test(a, b, method="naive")
    lw = sharpe_difference_test(a, b, method="ledoit_wolf", n_boot=300)
    assert abs(naive["delta"] - lw["delta"]) < 1e-9  # same point estimate
    assert (
        abs(naive["se"] - lw["se"]) < 0.02
    )  # HAC ~ iid variance when there is no dependence


def test_sharpe_difference_test_detects_real_gap_and_validates_input():
    import pytest

    rng = np.random.default_rng(3)
    strong = rng.standard_normal(3000) * 0.01 + 0.004  # clear positive edge
    flat = rng.standard_normal(3000) * 0.01
    res = sharpe_difference_test(strong, flat, method="ledoit_wolf", n_boot=300)
    assert res["delta"] > 0 and res["reject"]
    with pytest.raises(ValueError):
        sharpe_difference_test(np.zeros(10), np.zeros(9))
    with pytest.raises(ValueError):
        sharpe_difference_test(np.zeros(10), np.zeros(10), method="bogus")
