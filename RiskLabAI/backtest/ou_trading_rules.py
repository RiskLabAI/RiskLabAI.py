r"""
Closed-form optimal Ornstein-Uhlenbeck trading rules (Lipton-Lopez de Prado 2020).

de Prado finds optimal profit-take / stop-loss thresholds for a mean-reverting (OU) process by a
Monte-Carlo grid search (AFML ch.13): simulate many OU paths and, for each (profit-take, stop-loss) pair
on a grid, score the realized trade outcomes and pick the grid argmax. That grid carries simulation
noise, costs compute, and is limited by the grid resolution. The Lipton-Lopez de Prado closed form gives
the exact optimum for the OU model from first-passage theory, with no simulation noise and negligible
compute.

For an OU deviation ``Y`` with ``dY = -theta Y dt + sigma dW`` entered at deviation ``y0 = -entry_gap``,
with an upper barrier ``b = y0 + profit_take`` (gain ``+profit_take``) and a lower barrier
``a = y0 - stop_loss`` (gain ``-stop_loss``), the probability of hitting the upper barrier first is
``u(y0) = [S(y0) - S(a)] / [S(b) - S(a)]`` with scale function ``S(x) = int_0^x exp(theta s^2 / sigma^2) ds``
(proportional to the imaginary error function), and the mean first-exit time follows the Karlin-Taylor
Green's-function form. The per-trade gain is two-valued, so ``E[gain] = profit_take * u - stop_loss * (1-u)
- cost``, and the objective is the expected net return per unit time ``E[gain] / E[tau]`` (the well-posed
optimal-trading-rule objective; the raw per-trade Sharpe is degenerate, maximized by profit_take -> 0).

Admitted in Appraisal 23 (CONTRIBUTIONS_LEDGER 2026-06-27). Scope tag, verbatim from the verdict:

    prefer the closed-form OU rule over the Monte-Carlo PT/SL grid whenever the OU model is used - it
    gives the exact optimum with no simulation noise at a fraction of the compute; both are only as good
    as the OU assumption, so check the OU fit at decision time.

Held-out confirmed (appraisals/23_results, HELDOUT.md): on the sealed OU corner the closed form agrees
with the Monte-Carlo grid optimum within grid resolution (the grid pick's regret on the exact surface
stays ~0.0002-0.0008 and tightens as the grid refines) at a compute speed-up that grows with resolution
(3.4x -> 16x), and it degrades in step with the grid off-model (no false out-of-model win). The advantage
is exactness and speed under the OU model, not robustness to misspecification, so check the OU fit at
decision time. The de Prado Monte-Carlo PT/SL grid is left unchanged. Evidence: appraisals/23_verdict.md.

References
----------
Lipton, A. and Lopez de Prado, M. (2020) A closed-form solution for optimal mean-reverting trading
    strategies. SSRN / Risk.
Lopez de Prado, M. (2018) Advances in Financial Machine Learning, ch. 13 (the Monte-Carlo PT/SL grid).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import erfi


def theta_from_half_life(half_life: float) -> float:
    """Continuous mean-reversion speed ``theta`` from a half-life in time units."""
    return float(np.log(2.0) / float(half_life))


def stationary_std(theta: float, sigma: float) -> float:
    """Stationary standard deviation of the OU deviation: ``sigma / sqrt(2 theta)``."""
    return float(sigma) / np.sqrt(2.0 * float(theta))


def _scale_relative(x: float, a: float, b: float, c: float) -> float:
    """``exp(-cK)(S(x) - S(a))`` with ``K = max(a^2, b^2)`` (overflow-safe), S the OU scale function."""
    big = max(a * a, b * b)
    rc = np.sqrt(c)
    val = (
        (np.sqrt(np.pi) / (2.0 * rc)) * (erfi(rc * x) - erfi(rc * a)) * np.exp(-c * big)
    )
    if np.isfinite(val):
        return float(val)
    s = np.linspace(a, x, 2000)
    return float(np.trapz(np.exp(c * (s * s - big)), s))


def hit_upper_probability(
    entry_gap: float, profit_take: float, stop_loss: float, theta: float, sigma: float
) -> float:
    r"""
    Probability the OU deviation hits the upper barrier (profit-take) before the lower (stop-loss).

    Entered at ``y0 = -entry_gap``, barriers ``a = y0 - stop_loss`` and ``b = y0 + profit_take``;
    ``u = [S(y0) - S(a)] / [S(b) - S(a)]`` in an overflow-safe max-subtracted form.
    """
    c = theta / (sigma * sigma)
    y0 = -float(entry_gap)
    a, b = y0 - stop_loss, y0 + profit_take
    if b <= a:
        return float("nan")
    num = _scale_relative(y0, a, b, c)
    den = _scale_relative(b, a, b, c)
    if den <= 0:
        return float("nan")
    return float(np.clip(num / den, 0.0, 1.0))


def mean_exit_time(
    entry_gap: float, profit_take: float, stop_loss: float, theta: float, sigma: float
) -> float:
    r"""
    Closed-form OU mean first-exit time ``E[tau]`` from the two-barrier interval, via the Karlin-Taylor
    Green's function ``v(x) = int_a^b G(x, y) m(y) dy`` (speed density ``m``), evaluated by quadrature.
    """
    c = theta / (sigma * sigma)
    y0 = -float(entry_gap)
    a, b = y0 - stop_loss, y0 + profit_take
    if b <= a:
        return float("nan")
    rc = np.sqrt(c)
    pref = np.sqrt(np.pi) / (2.0 * rc)
    ys = np.linspace(a, b, 400)
    erfi_y, erfi_a, erfi_b, erfi_y0 = (
        erfi(rc * ys),
        float(erfi(rc * a)),
        float(erfi(rc * b)),
        float(erfi(rc * y0)),
    )
    if not (np.all(np.isfinite(erfi_y)) and np.isfinite(erfi_b)):
        return _mean_exit_time_quadrature(
            entry_gap, profit_take, stop_loss, theta, sigma
        )
    s_ya = pref * (erfi_y - erfi_a)
    s_by = pref * (erfi_b - erfi_y)
    s_ba = pref * (erfi_b - erfi_a)
    s_y0a = pref * (erfi_y0 - erfi_a)
    s_by0 = pref * (erfi_b - erfi_y0)
    green = np.where(ys <= y0, s_ya * s_by0, s_y0a * s_by) / s_ba
    speed = (2.0 / (sigma * sigma)) * np.exp(-c * ys * ys)
    return float(np.trapz(green * speed, ys))


def _mean_exit_time_quadrature(
    entry_gap: float, profit_take: float, stop_loss: float, theta: float, sigma: float
) -> float:
    """Overflow-safe ``E[tau]`` for very wide barriers (max-subtracted scale-function quadrature)."""
    c = theta / (sigma * sigma)
    y0 = -float(entry_gap)
    a, b = y0 - stop_loss, y0 + profit_take
    big = max(a * a, b * b)

    def s_hat(x):
        s = np.linspace(a, x, 3000)
        return np.trapz(np.exp(c * (s * s - big)), s)

    s_hat_b, s_hat_y0 = s_hat(b), s_hat(y0)
    ys = np.linspace(a, b, 3000)
    s_ya = np.array([s_hat(y) for y in ys])
    s_by = s_hat_b - s_ya
    green = np.where(ys <= y0, s_ya * (s_hat_b - s_hat_y0), s_hat_y0 * s_by) / s_hat_b
    speed = (2.0 / (sigma * sigma)) * np.exp(c * (big - ys * ys))
    return float(np.trapz(green * speed, ys))


def ou_rule_metrics(
    profit_take: float,
    stop_loss: float,
    theta: float,
    sigma: float,
    entry_gap: float,
    cost: float = 0.0,
) -> dict:
    r"""
    Exact OU per-trade metrics for a rule entered ``entry_gap`` from the mean (all thresholds in the same
    units as the OU deviation; time in model time units).

    Returns ``hit_probability`` (``u``), ``expected_gain`` (net mean per-trade gain), ``std_gain``,
    ``expected_holding_time`` (``E[tau]``), ``return_rate`` (``expected_gain / E[tau]``, the optimization
    objective), and ``time_scaled_sharpe`` (``expected_gain / (std_gain sqrt(E[tau]))``, a secondary
    metric that is degenerate as ``u -> 1``).
    """
    bad = dict(
        hit_probability=np.nan,
        expected_gain=-np.inf,
        std_gain=np.nan,
        expected_holding_time=np.nan,
        return_rate=-np.inf,
        time_scaled_sharpe=-np.inf,
    )
    if profit_take <= 0 or stop_loss <= 0:
        return bad
    u = hit_upper_probability(entry_gap, profit_take, stop_loss, theta, sigma)
    if not np.isfinite(u):
        return bad
    q = 1.0 - u
    expected_gain = profit_take * u - stop_loss * q - cost
    std_gain = (profit_take + stop_loss) * np.sqrt(max(u * q, 0.0))
    e_tau = mean_exit_time(entry_gap, profit_take, stop_loss, theta, sigma)
    if not np.isfinite(e_tau) or e_tau <= 0:
        return bad
    return dict(
        hit_probability=float(u),
        expected_gain=float(expected_gain),
        std_gain=float(std_gain),
        expected_holding_time=float(e_tau),
        return_rate=float(expected_gain / e_tau),
        time_scaled_sharpe=(
            float(expected_gain / (std_gain * np.sqrt(e_tau)))
            if std_gain > 0
            else -np.inf
        ),
    )


def optimal_ou_trading_rule(
    theta: float,
    sigma: float,
    entry_gap: float,
    cost: float = 0.0,
    bounds: tuple = (0.25, 4.0),
) -> dict:
    r"""
    Exact optimal OU profit-take / stop-loss by maximizing the closed-form expected net return per unit
    time (no simulation).

    Multi-start L-BFGS-B on the negative ``return_rate`` objective over the ``(profit_take, stop_loss)``
    box. Prefer this over the Monte-Carlo PT/SL grid whenever the OU model is used: it gives the exact
    optimum with no simulation noise at a fraction of the compute. Both are only as good as the OU
    assumption, so check the OU fit at decision time (see :func:`fit_ornstein_uhlenbeck` and the module
    docstring for the verbatim scope tag and appraisals/23_verdict.md).

    Parameters
    ----------
    theta : float
        OU mean-reversion speed.
    sigma : float
        OU instantaneous volatility.
    entry_gap : float
        Deviation from the mean at entry (the trade is opened ``entry_gap`` below the mean).
    cost : float, default 0.0
        Per-trade transaction cost in the same units as the thresholds.
    bounds : tuple, default (0.25, 4.0)
        ``(low, high)`` box for both the profit-take and the stop-loss.

    Returns
    -------
    dict
        ``profit_take``, ``stop_loss``, ``return_rate`` (the maximized objective), and the full
        :func:`ou_rule_metrics` of the optimum.

    Examples
    --------
    >>> rule = optimal_ou_trading_rule(theta=0.1, sigma=0.1, entry_gap=1.0)
    >>> rule["profit_take"] > 0 and rule["stop_loss"] > 0
    True
    """
    lo, hi = bounds

    def negative_objective(x):
        v = ou_rule_metrics(x[0], x[1], theta, sigma, entry_gap, cost)["return_rate"]
        return -v if np.isfinite(v) else 1e9

    span = hi - lo
    starts = [
        (lo + 0.15 * span, lo + 0.2 * span),
        (lo + 0.1 * span, lo + 0.6 * span),
        (lo + 0.3 * span, lo + 0.4 * span),
    ]
    best = None
    for start in starts:
        res = minimize(
            negative_objective, x0=start, method="L-BFGS-B", bounds=[(lo, hi), (lo, hi)]
        )
        if best is None or res.fun < best.fun:
            best = res
    profit_take, stop_loss = float(best.x[0]), float(best.x[1])
    metrics = ou_rule_metrics(profit_take, stop_loss, theta, sigma, entry_gap, cost)
    return {
        "profit_take": profit_take,
        "stop_loss": stop_loss,
        "return_rate": -float(best.fun),
        **metrics,
    }


def fit_ornstein_uhlenbeck(series: np.ndarray, dt: float = 1.0) -> dict:
    r"""
    Fit an OU process by AR(1) regression ``x_t = a + b x_{t-1} + e`` (the OU goodness-of-fit check).

    Returns ``theta`` (mean-reversion speed), ``sigma`` (per-unit-time noise), ``mu`` (long-run mean),
    ``rho`` (AR(1) coefficient), ``half_life``, ``stationary_std``, and ``r2`` (the AR(1) fit quality, a
    simple OU goodness-of-fit proxy). Check ``r2`` / the half-life before trusting the closed-form rule:
    both the closed form and the Monte-Carlo grid are only as good as the OU assumption.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> x = np.zeros(2000)
    >>> for t in range(1, 2000):
    ...     x[t] = 0.9 * x[t - 1] + rng.standard_normal() * 0.1
    >>> fit = fit_ornstein_uhlenbeck(x)
    >>> fit["theta"] > 0 and 0.0 <= fit["r2"] <= 1.0
    True
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    x0, x1 = x[:-1], x[1:]
    design = np.vstack([np.ones_like(x0), x0]).T
    coef, *_ = np.linalg.lstsq(design, x1, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    residual = x1 - (a + b * x0)
    b = min(max(b, 1e-6), 0.999999)
    theta = -np.log(b) / dt
    mu = a / (1.0 - b)
    var_eps = float(np.var(residual, ddof=2))
    sigma = np.sqrt(var_eps * 2.0 * theta / (1.0 - b * b))
    ss_tot = float(np.var(x1, ddof=1)) * (len(x1) - 1)
    r2 = 1.0 - float(np.sum(residual**2)) / ss_tot if ss_tot > 0 else 0.0
    return dict(
        theta=float(theta),
        sigma=float(sigma),
        mu=float(mu),
        rho=float(b),
        half_life=float(np.log(2.0) / theta),
        stationary_std=stationary_std(theta, sigma),
        r2=float(r2),
    )
