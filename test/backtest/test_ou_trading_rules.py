"""
Tests for backtest/ou_trading_rules.py (closed-form OU trading rules).
"""

import numpy as np

from RiskLabAI.backtest.ou_trading_rules import (
    fit_ornstein_uhlenbeck,
    hit_upper_probability,
    mean_exit_time,
    optimal_ou_trading_rule,
    ou_rule_metrics,
    theta_from_half_life,
)


def _simulate_ou(theta, sigma, n_paths, n_steps, dt, rng, y_start=0.0):
    """Exact-discretization OU paths starting at y_start (deviation from the mean)."""
    rho = np.exp(-theta * dt)
    sd = sigma * np.sqrt((1.0 - rho**2) / (2.0 * theta))
    y = np.full(n_paths, float(y_start))
    paths = np.empty((n_paths, n_steps))
    for t in range(n_steps):
        y = rho * y + sd * rng.standard_normal(n_paths)
        paths[:, t] = y
    return paths


def _mc_return_rate(paths, entry_gap, profit_take, stop_loss, cost, dt):
    """Empirical expected net return per unit time for a (PT, SL) rule on simulated OU paths."""
    y0 = -entry_gap
    upper, lower = y0 + profit_take, y0 - stop_loss
    n_paths, n_steps = paths.shape
    gains, times = [], []
    for p in range(n_paths):
        path = paths[p]
        hit = np.where((path >= upper) | (path <= lower))[0]
        if hit.size == 0:
            continue
        k = hit[0]
        gains.append(profit_take if path[k] >= upper else -stop_loss)
        times.append((k + 1) * dt)
    if not gains:
        return -np.inf
    return (np.mean(gains) - cost) / np.mean(times)


def test_hit_probability_and_exit_time_sane():
    theta, sigma = 0.1, 0.1
    u = hit_upper_probability(
        entry_gap=1.0, profit_take=0.5, stop_loss=0.5, theta=theta, sigma=sigma
    )
    assert 0.0 <= u <= 1.0
    # entered below the mean, a symmetric barrier set is more likely to revert UP (hit upper) than down
    assert u > 0.5
    tau = mean_exit_time(
        entry_gap=1.0, profit_take=0.5, stop_loss=0.5, theta=theta, sigma=sigma
    )
    assert tau > 0 and np.isfinite(tau)


def test_closed_form_matches_monte_carlo_grid_within_resolution():
    """
    Replication of the Appraisal 23 mechanism: where the price is OU, the closed-form optimal profit-take
    agrees with the Monte-Carlo grid argmax within the grid spacing.
    """
    theta, sigma, gap, cost, dt = theta_from_half_life(10.0), 0.10, 1.0, 0.0, 1.0
    rule = optimal_ou_trading_rule(theta, sigma, gap, cost, bounds=(0.25, 4.0))
    rng = np.random.default_rng(0)
    paths = _simulate_ou(
        theta, sigma, n_paths=8000, n_steps=400, dt=dt, rng=rng, y_start=-gap
    )
    grid = np.round(np.arange(0.25, 4.01, 0.25), 3)
    best_pt, best_val = None, -np.inf
    sl_fixed = rule["stop_loss"]
    for pt in grid:
        val = _mc_return_rate(paths, gap, pt, sl_fixed, cost, dt)
        if val > best_val:
            best_val, best_pt = val, pt
    # closed-form PT agrees with the MC grid argmax within one grid spacing (0.25)
    assert abs(rule["profit_take"] - best_pt) <= 0.25 + 1e-9


def test_optimum_agrees_with_grid_on_exact_surface():
    """The continuous optimizer's objective agrees with the best coarse grid point on the exact surface
    (the closed-form optimum matches the grid argmax; a tiny gap is local-optimizer imprecision).
    """
    theta, sigma, gap = 0.1, 0.1, 1.0
    rule = optimal_ou_trading_rule(theta, sigma, gap, cost=0.0, bounds=(0.25, 4.0))
    grid = np.arange(0.25, 4.01, 0.5)
    grid_best = max(
        ou_rule_metrics(pt, sl, theta, sigma, gap, 0.0)["return_rate"]
        for pt in grid
        for sl in grid
    )
    assert abs(rule["return_rate"] - grid_best) < 1e-3


def test_metrics_reject_nonpositive_thresholds():
    m = ou_rule_metrics(
        profit_take=0.0, stop_loss=0.5, theta=0.1, sigma=0.1, entry_gap=1.0
    )
    assert m["return_rate"] == -np.inf
    m2 = ou_rule_metrics(
        profit_take=0.5, stop_loss=-0.1, theta=0.1, sigma=0.1, entry_gap=1.0
    )
    assert m2["return_rate"] == -np.inf


def test_fit_recovers_known_ou():
    """fit_ornstein_uhlenbeck recovers the half-life of a simulated OU within tolerance."""
    rng = np.random.default_rng(5)
    theta_true = theta_from_half_life(20.0)
    path = _simulate_ou(
        theta_true, 0.1, n_paths=1, n_steps=6000, dt=1.0, rng=rng
    ).ravel()
    fit = fit_ornstein_uhlenbeck(path, dt=1.0)
    assert abs(fit["half_life"] - 20.0) < 6.0  # recovered within a reasonable band
    assert 0.0 <= fit["r2"] <= 1.0
    assert fit["theta"] > 0
