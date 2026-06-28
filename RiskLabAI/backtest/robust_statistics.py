r"""
Robust backtest statistics: Conditional Expected Drawdown (CED) and the Ledoit-Wolf Sharpe-difference test.

de Prado reports maximum drawdown and time-under-water and compares Sharpe ratios directly
(`backtest_statistics.py`). Maximum drawdown is a single extreme order statistic - high variance, not
coherent, not attributable - and the naive Sharpe-difference comparison assumes i.i.d. returns, so it
over-rejects under the serial dependence real returns carry. Two admitted upgrades sit alongside those
baselines:

- **Conditional Expected Drawdown (CED, Goldberg-Mahmoud 2017)**: the tail mean (CVaR) of the
  maximum-drawdown distribution at a fixed horizon, a lower-variance, coherent, factor-attributable
  drawdown-risk measure; and
- **the Ledoit-Wolf (2008) bootstrap Sharpe-difference test**: a HAC-studentized statistic calibrated by
  a studentized circular block bootstrap, which holds nominal size under serial dependence where the
  naive test inflates.

Admitted in Appraisal 22 (CONTRIBUTIONS_LEDGER 2026-06-27). Scope tags, verbatim from the verdict:

    CED: prefer CED over max-drawdown as a drawdown-risk statistic: lower estimator variance and better
    ranking of true drawdown risk, most of all on short/heavy-tailed tracks; converges to max-DD on
    benign returns; factor-attributable.

    Ledoit-Wolf Sharpe-difference test: prefer it over the naive test when comparing two Sharpes under
    serial dependence / heavy tails (it holds nominal size where the naive test inflates ~3x); converges
    on i.i.d. returns.

Held-out confirmed (appraisals/22_results, HELDOUT.md): on the sealed short-sample x heavy-tail corner CED
has lower estimator variance (CV 0.30-0.32 vs max-drawdown 0.40-0.42) and better rank-recovery (Spearman
0.42-0.78 vs 0.31-0.69); the LW test holds nominal size (0.056) under AR(0.3) where the naive test rejects
0.161 (~3x), and converges on i.i.d. returns. CDaR@0.95 was NOT admitted (no variance/rank advantage over
max-drawdown as a descriptive statistic). Evidence and caveats: appraisals/22_verdict.md.

References
----------
Goldberg, L. R. and Mahmoud, O. (2017) Drawdown: from practice to theory and back again.
    Quantitative Finance, 17(5), 745-761. (Conditional Expected Drawdown.)
Ledoit, O. and Wolf, M. (2008) Robust performance hypothesis testing with the Sharpe ratio.
    Journal of Empirical Finance, 15(5), 850-859.
"""

from __future__ import annotations

from math import erfc, sqrt

import numpy as np


# --------------------------------------------------------------------------------------------------
# Conditional Expected Drawdown
# --------------------------------------------------------------------------------------------------
def _wealth_from_returns(returns: np.ndarray) -> np.ndarray:
    return np.cumprod(1.0 + np.asarray(returns, dtype=float))


def _drawdown_series(wealth: np.ndarray) -> np.ndarray:
    w = np.asarray(wealth, dtype=float)
    return 1.0 - w / np.maximum.accumulate(w)


def _cvar_upper(values: np.ndarray, alpha: float) -> float:
    """Empirical upper-tail CVaR: the mean of the worst ``1 - alpha`` fraction of ``values``."""
    v = np.sort(np.asarray(values, dtype=float))
    n = v.size
    if n == 0:
        return 0.0
    k = (1.0 - alpha) * n
    if k <= 0:
        return float(v[-1])
    n_full = int(np.floor(k))
    total = v[n - n_full :].sum() if n_full > 0 else 0.0
    frac = k - n_full
    if frac > 0 and n - n_full - 1 >= 0:
        total += frac * v[n - n_full - 1]
    return float(total / k)


def _rolling_max_drawdowns(wealth: np.ndarray, horizon: int) -> np.ndarray:
    w = np.asarray(wealth, dtype=float)
    n = w.size
    if n < horizon or horizon < 2:
        d = _drawdown_series(w)
        return np.array([float(d.max()) if d.size else 0.0], dtype=float)
    windows = np.lib.stride_tricks.sliding_window_view(w, horizon)
    high_water = np.maximum.accumulate(windows, axis=1)
    return (1.0 - windows / high_water).max(axis=1)


def conditional_expected_drawdown(
    returns: np.ndarray,
    horizon: int,
    alpha: float = 0.90,
) -> float:
    r"""
    Conditional Expected Drawdown (CED; Goldberg-Mahmoud 2017): the CVaR of the maximum-drawdown
    distribution at a fixed horizon.

    From a single return track it is estimated by the upper-tail (CVaR at level ``alpha``) mean of the
    maximum drawdowns within overlapping windows of length ``horizon``. Because it is a tail MEAN of the
    max-drawdown distribution rather than the single most-extreme realization, it has lower estimator
    variance and ranks true drawdown risk better than max-drawdown, especially on short / heavy-tailed
    tracks, and it converges to agree with max-drawdown on benign returns. It is coherent and
    factor-attributable.

    Prefer CED over max-drawdown as a drawdown-risk statistic (see the module docstring for the full
    verbatim scope tag and appraisals/22_verdict.md). The de Prado max-drawdown / time-under-water
    statistics are unchanged.

    Parameters
    ----------
    returns : np.ndarray
        Per-period simple returns of the track.
    horizon : int
        The rolling-window length (the drawdown-measurement horizon, in periods).
    alpha : float, default 0.90
        The CVaR tail level (the mean is taken over the worst ``1 - alpha`` fraction of window maxima).

    Returns
    -------
    float
        The conditional expected drawdown in ``[0, 1)``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> r = rng.standard_normal(2000) * 0.01
    >>> ced = conditional_expected_drawdown(r, horizon=60)
    >>> 0.0 <= ced < 1.0
    True
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    wealth = _wealth_from_returns(returns)
    return _cvar_upper(_rolling_max_drawdowns(wealth, int(horizon)), alpha)


# --------------------------------------------------------------------------------------------------
# Sharpe-difference tests
# --------------------------------------------------------------------------------------------------
def _sharpe_influence(r_a: np.ndarray, r_b: np.ndarray):
    """Influence-function series of the Sharpe difference SR_A - SR_B and the point estimate."""
    a = np.asarray(r_a, dtype=float)
    b = np.asarray(r_b, dtype=float)
    mu_a, mu_b = a.mean(), b.mean()
    g_a, g_b = (a * a).mean(), (b * b).mean()
    s_a = np.sqrt(max(g_a - mu_a * mu_a, 1e-300))
    s_b = np.sqrt(max(g_b - mu_b * mu_b, 1e-300))
    delta = mu_a / s_a - mu_b / s_b
    ga0, ga1 = g_a / s_a**3, -mu_a / (2.0 * s_a**3)
    gb0, gb1 = g_b / s_b**3, -mu_b / (2.0 * s_b**3)
    influence = (
        ga0 * (a - mu_a)
        + ga1 * (a * a - g_a)
        - (gb0 * (b - mu_b) + gb1 * (b * b - g_b))
    )
    return float(delta), influence


def _bartlett_hac_var(influence: np.ndarray, bandwidth: int) -> float:
    """Bartlett-kernel HAC variance of the mean of the influence series (= Var(delta_hat))."""
    x = np.asarray(influence, dtype=float)
    x = x - x.mean()
    t = x.size
    if t < 2:
        return float("nan")
    s = np.dot(x, x) / t
    h = max(0, int(bandwidth))
    for j in range(1, h + 1):
        if j >= t:
            break
        s += 2.0 * (1.0 - j / (h + 1.0)) * (np.dot(x[j:], x[:-j]) / t)
    return float(max(s, 1e-300) / t)


def _iid_var(influence: np.ndarray) -> float:
    x = np.asarray(influence, dtype=float)
    t = x.size
    if t < 2:
        return float("nan")
    return float(max(x.var(), 1e-300) / t)


def _circular_block_indices(
    t: int, block_len: int, rng: np.random.Generator
) -> np.ndarray:
    n_blocks = int(np.ceil(t / block_len))
    starts = rng.integers(0, t, size=n_blocks)
    offsets = np.arange(block_len)
    return ((starts[:, None] + offsets[None, :]) % t).reshape(-1)[:t]


def sharpe_difference_test(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    method: str = "ledoit_wolf",
    block_length: int | None = None,
    n_boot: int = 1000,
    bandwidth: int | None = None,
    random_state: int | None = 0,
) -> dict:
    r"""
    Test H0: SR_A = SR_B (equal Sharpe ratios), naive or Ledoit-Wolf robust.

    ``method="naive"`` is the influence-function delta-method z-test using the i.i.d. (lag-0) variance of
    the Sharpe-difference influence function and the normal cutoff - it ignores serial dependence and
    over-rejects under it. ``method="ledoit_wolf"`` (Ledoit-Wolf 2008) studentizes the same Sharpe
    difference by a Bartlett-kernel HAC standard error and calibrates the two-sided p-value by a
    studentized circular block bootstrap that preserves serial dependence, so it holds nominal size where
    the naive test inflates (~3x under AR), and converges to the naive test on i.i.d. returns.

    Prefer the Ledoit-Wolf test when comparing two Sharpes under serial dependence / heavy tails (see the
    module docstring for the full verbatim scope tag and appraisals/22_verdict.md). The Sharpe ratio uses
    the population standard deviation, matching ``backtest_statistics.sharpe_ratio``.

    Parameters
    ----------
    returns_a, returns_b : np.ndarray
        The two equal-length per-period return series being compared.
    method : {"ledoit_wolf", "naive"}, default "ledoit_wolf"
        The test variant.
    block_length : int, optional
        Circular-block length for the LW bootstrap (also the default HAC bandwidth). Defaults to
        ``ceil(T ** (1/3))``. Ignored for ``method="naive"``.
    n_boot : int, default 1000
        Bootstrap resamples for the LW p-value. Ignored for ``method="naive"``.
    bandwidth : int, optional
        HAC (Bartlett) bandwidth; defaults to ``block_length``. Ignored for ``method="naive"``.
    random_state : int, optional
        Seed for the LW bootstrap.

    Returns
    -------
    dict
        ``delta`` (SR_A - SR_B), ``se``, ``stat``, ``pvalue``, and ``reject`` (at the 5% level).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> a = rng.standard_normal(1000) * 0.01 + 0.001
    >>> b = rng.standard_normal(1000) * 0.01
    >>> res = sharpe_difference_test(a, b, method="ledoit_wolf", n_boot=200)
    >>> set(res) == {"delta", "se", "stat", "pvalue", "reject"}
    True
    """
    if method not in ("ledoit_wolf", "naive"):
        raise ValueError("method must be 'ledoit_wolf' or 'naive'")
    a = np.asarray(returns_a, dtype=float)
    b = np.asarray(returns_b, dtype=float)
    if a.size != b.size:
        raise ValueError("returns_a and returns_b must have the same length")
    delta, influence = _sharpe_influence(a, b)

    if method == "naive":
        se = np.sqrt(_iid_var(influence))
        z = delta / se if se > 0 else 0.0
        pvalue = float(erfc(abs(z) / sqrt(2.0)))
        return {
            "delta": delta,
            "se": float(se),
            "stat": float(z),
            "pvalue": pvalue,
            "reject": bool(pvalue < 0.05),
        }

    t = a.size
    if block_length is None:
        block_length = max(1, int(np.ceil(t ** (1.0 / 3.0))))
    bw = block_length if bandwidth is None else bandwidth
    se_hat = np.sqrt(_bartlett_hac_var(influence, bw))
    z_hat = abs(delta / se_hat) if se_hat > 0 else 0.0
    rng = np.random.default_rng(random_state)
    count = 0
    for _ in range(n_boot):
        idx = _circular_block_indices(t, block_length, rng)
        delta_b, influence_b = _sharpe_influence(a[idx], b[idx])
        se_b = np.sqrt(_bartlett_hac_var(influence_b, bw))
        if se_b <= 0:
            continue
        if abs((delta_b - delta) / se_b) >= z_hat:
            count += 1
    pvalue = (count + 1.0) / (n_boot + 1.0)
    return {
        "delta": delta,
        "se": float(se_hat),
        "stat": float(z_hat),
        "pvalue": float(pvalue),
        "reject": bool(pvalue < 0.05),
    }
