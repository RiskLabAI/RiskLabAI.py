"""
Implements the EDGE effective bid-ask spread estimator
(Ardia, Guidotti & Kroencke 2024).

Admitted in Appraisal 03 (CONTRIBUTIONS_LEDGER row 1, 2026-06-26). Regime tag,
verbatim from the verdict:

    Prefer EDGE over Roll and Abdi-Ranaldo for low-frequency spread estimation in
    all regimes; over Corwin-Schultz at small spreads (the edge narrows at very high
    illiquidity and very large spreads).

EDGE pools the open, high, low and close prices and corrects for discrete,
infrequently traded data, giving lower bias and variance than the close-to-close
Roll (1984) estimator and the two-day high-low Corwin-Schultz (2012) estimator, and
never returning an invalid (negative) point estimate. The result is a proportional
spread (0.01 is a 1% spread). Real-data confirmation is a logged follow-up pending an
adequate public intraday / quote dataset (DECISION_LOG 2026-06-26). Evidence and
caveats: appraisals/03_verdict.md (03_results/RESULTS.md + HELDOUT.md).

Reference:
    Ardia, D., Guidotti, E., & Kroencke, T. A. (2024). Efficient estimation of
    bid-ask spreads from open, high, low, and close prices. Journal of Financial
    Economics, 161, 103916.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

_ArrayLike = Sequence[float] | np.ndarray


def _lag_nan(values: np.ndarray) -> np.ndarray:
    """Shift by one period; the first element (no predecessor) becomes NaN."""
    lagged = np.empty_like(values)
    lagged[0] = np.nan
    lagged[1:] = values[:-1]
    return lagged


def edge_estimator(
    open: _ArrayLike,
    high: _ArrayLike,
    low: _ArrayLike,
    close: _ArrayLike,
    sign: bool = False,
) -> float:
    r"""
    Estimate the effective bid-ask spread using the EDGE estimator
    (Ardia, Guidotti & Kroencke 2024).

    The estimator forms two unbiased squared-spread moment conditions from the log
    open/high/low/close prices and a tick-rule trade indicator, then combines them
    with the GMM-optimal, variance-weighted average:

    .. math::
        \hat{S}^2 = \frac{v_2 \, e_1 + v_1 \, e_2}{v_1 + v_2}, \qquad
        \hat{S} = \sqrt{|\hat{S}^2|}

    where :math:`e_k` and :math:`v_k` are the mean and variance of the two estimators
    (equal weight if the total variance is not positive).

    Parameters
    ----------
    open, high, low, close : array-like
        Four equal-length sequences of bar open, high, low and close prices
        (arrays of bar prices, not a DataFrame).
    sign : bool, default=False
        If True, return a signed estimate (negative when the estimated squared
        spread is negative); otherwise the spread magnitude is returned.

    Returns
    -------
    float
        The estimated proportional spread (0.01 is a 1% spread), or ``NaN`` when the
        estimate is undefined (fewer than three observations, fewer than two traded
        periods, or a degenerate / zero-variance input).

    Raises
    ------
    ValueError
        If the open, high, low and close sequences are not all the same length.

    Examples
    --------
    >>> o = [100.0, 101.2, 100.5, 102.1, 101.8, 103.0]
    >>> h = [101.0, 101.9, 101.3, 102.8, 102.5, 103.6]
    >>> l = [99.4, 100.6, 99.9, 101.3, 101.0, 102.2]
    >>> c = [101.1, 100.7, 101.0, 102.3, 101.6, 103.2]
    >>> round(edge_estimator(o, h, l, c), 6)  # doctest: +SKIP
    0.011176
    """
    open_prices = np.asarray(open, dtype=float)
    high_prices = np.asarray(high, dtype=float)
    low_prices = np.asarray(low, dtype=float)
    close_prices = np.asarray(close, dtype=float)

    n = open_prices.shape[0]
    if not (
        high_prices.shape[0] == n
        and low_prices.shape[0] == n
        and close_prices.shape[0] == n
    ):
        raise ValueError("open, high, low, and close must have the same length.")
    if n < 3:
        return float("nan")

    o = np.log(open_prices)
    h = np.log(high_prices)
    ll = np.log(low_prices)
    c = np.log(close_prices)
    m = (h + ll) / 2.0

    h1 = _lag_nan(h)
    l1 = _lag_nan(ll)
    c1 = _lag_nan(c)
    m1 = _lag_nan(m)

    # Log-returns; r1's first element is masked to align with the lagged quantities.
    r1 = m - o
    r1[0] = np.nan
    r2 = o - m1
    r3 = m - c1
    r4 = c1 - m1
    r5 = o - c1

    # Trade indicator: the bar traded if its range is non-zero or the low differs
    # from the previous close. NaN where any required input is missing.
    valid_tau = ~(np.isnan(h) | np.isnan(ll) | np.isnan(c1))
    tau = np.full(n, np.nan)
    tau[valid_tau] = (
        (h[valid_tau] != ll[valid_tau]) | (ll[valid_tau] != c1[valid_tau])
    ).astype(float)

    def _indicator(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """(tau and a != b) as 0/1, NaN where tau or any required input is missing."""
        out = np.full(n, np.nan)
        valid = ~(np.isnan(tau) | np.isnan(a) | np.isnan(b))
        out[valid] = ((tau[valid] == 1.0) & (a[valid] != b[valid])).astype(float)
        return out

    po1 = _indicator(o, h)
    po2 = _indicator(o, ll)
    pc1 = _indicator(c1, h1)
    pc2 = _indicator(c1, l1)

    pt = np.nanmean(tau)
    po = np.nanmean(po1) + np.nanmean(po2)
    pc = np.nanmean(pc1) + np.nanmean(pc2)

    if np.nansum(tau) < 2 or po == 0.0 or pc == 0.0:
        return float("nan")

    # De-meaned log-returns, weighted by the trade indicator and trade probability.
    d1 = r1 - (np.nanmean(r1) / pt) * tau
    d3 = r3 - (np.nanmean(r3) / pt) * tau
    d5 = r5 - (np.nanmean(r5) / pt) * tau

    # Two unbiased squared-spread estimators, from the open and the previous close.
    x1 = (-4.0 / po) * d1 * r2 + (-4.0 / pc) * d3 * r4
    x2 = (-4.0 / po) * d1 * r5 + (-4.0 / pc) * d5 * r4

    e1 = np.nanmean(x1)
    e2 = np.nanmean(x2)
    v1 = np.nanmean(x1 * x1) - e1 * e1
    v2 = np.nanmean(x2 * x2) - e2 * e2

    # Variance-weighted (GMM-optimal) average of the two estimators; equal weight if
    # the total variance is not positive.
    total_variance = v1 + v2
    if total_variance > 0.0:
        squared_spread = (v2 * e1 + v1 * e2) / total_variance
    else:
        squared_spread = (e1 + e2) / 2.0

    spread = float(np.sqrt(np.abs(squared_spread)))
    if sign and squared_spread < 0.0:
        spread = -spread
    return spread
