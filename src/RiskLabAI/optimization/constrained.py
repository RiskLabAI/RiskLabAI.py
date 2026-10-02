"""Checked convex portfolio problems using optional CVXPY solvers."""

import numpy as np

from RiskLabAI.utils._validation import real_array, real_scalar

__all__ = ["constrained_minimum_variance", "robust_mean_variance"]


def _inputs(covariance, expected_returns, caps, return_floor):
    covariance = real_array(covariance, "covariance", 2)
    n = len(covariance)
    magnitude = max(float(np.max(np.abs(covariance))), np.finfo(float).tiny)
    if covariance.shape != (n, n) or not np.allclose(
        covariance / magnitude, covariance.T / magnitude, atol=1e-12, rtol=1e-12
    ):
        raise ValueError("covariance must be symmetric and square.")
    if np.linalg.eigvalsh(covariance / magnitude).min() < -1e-12 * np.linalg.norm(
        covariance / magnitude, 2
    ):
        raise ValueError("covariance must be positive semidefinite.")
    mean = (
        np.zeros(n)
        if expected_returns is None
        else real_array(expected_returns, "expected_returns", 1)
    )
    cap = np.ones(n) if caps is None else real_array(caps, "caps", 1)
    if mean.shape != (n,) or cap.shape != (n,) or np.any(cap < 0) or cap.sum() < 1:
        raise ValueError(
            "Means/caps must match covariance and caps must allow full investment."
        )
    floor = None if return_floor is None else real_scalar(return_floor, "return_floor")
    if floor is not None and expected_returns is None:
        raise ValueError("A return floor requires expected_returns.")
    return covariance, mean, cap, floor


def _solve(
    covariance,
    mean,
    cap,
    floor,
    *,
    loading=None,
    radius=0.0,
    risk_aversion=1.0,
    minimum_variance=False,
):
    try:
        import cvxpy as cp
    except ImportError as error:
        raise ImportError("This method requires the optional cvxpy package.") from error
    w = cp.Variable(len(mean))
    constraints = [cp.sum(w) == 1, w >= 0, w <= cap]
    if floor is not None:
        constraints.append(mean @ w >= floor)
    variance = cp.quad_form(w, cp.psd_wrap(covariance))
    if minimum_variance:
        objective = variance
        scale = max(np.max(np.abs(covariance)), np.finfo(float).tiny)
    else:
        penalty = 0 if radius == 0 else radius * cp.norm(loading.T @ w, 2)
        objective = risk_aversion * variance / 2 - mean @ w + penalty
        scale = max(
            np.max(np.abs(mean)),
            risk_aversion * np.max(np.abs(covariance)),
            radius * np.linalg.norm(loading),
            np.finfo(float).tiny,
        )
    problem = cp.Problem(cp.Minimize(objective / scale), constraints)
    try:
        problem.solve(
            solver="CLARABEL", tol_gap_abs=1e-10, tol_feas=1e-10, tol_gap_rel=1e-10
        )
    except cp.error.SolverError as error:
        raise RuntimeError("Portfolio solver failed.") from error
    if problem.status in (cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE):
        raise ValueError("Portfolio constraints are infeasible.")
    if problem.status != cp.OPTIMAL or w.value is None:
        raise RuntimeError(f"Portfolio optimization is not verified: {problem.status}")
    weights = np.asarray(w.value)
    if (
        not np.all(np.isfinite(weights))
        or abs(weights.sum() - 1) > 1e-7
        or np.any(weights < -1e-7)
        or np.any(weights > cap + 1e-7)
        or (floor is not None and mean @ weights < floor - 1e-7 * max(1, abs(floor)))
    ):
        raise RuntimeError("Portfolio solver returned an infeasible result.")
    value = float(weights @ covariance @ weights)
    expected = float(mean @ weights)
    worst_mean = (
        expected
        if loading is None
        else expected - radius * float(np.linalg.norm(loading.T @ weights))
    )
    return {
        "weights": weights,
        "variance": value,
        "expected_return": expected,
        "worst_case_expected_return": worst_mean,
        "objective": (
            value if minimum_variance else worst_mean - risk_aversion * value / 2
        ),
        "solver_status": problem.status,
    }


def constrained_minimum_variance(
    covariance, *, expected_returns=None, return_floor=None, caps=None
):
    """Minimize variance over fully invested nonnegative capped weights.

    An optional return floor uses the supplied expected returns. Covariance
    must be PSD within numerical eigensolver tolerance; it is not repaired.
    Requires optional CVXPY with Clarabel. Returns weights and diagnostics.
    """
    covariance, mean, cap, floor = _inputs(
        covariance, expected_returns, caps, return_floor
    )
    return _solve(covariance, mean, cap, floor, minimum_variance=True)


def robust_mean_variance(
    expected_returns,
    covariance,
    uncertainty_loading,
    *,
    radius,
    risk_aversion,
    caps=None,
    return_floor=None,
):
    """Maximize the worst mean minus half risk_aversion times variance.

    Means range over mu + L*u, ||u||_2 <= radius. Covariance is fixed.
    The optional return floor constrains nominal, not worst-case, return.
    All uncertainty inputs are supplied, not inferred or calibrated here.
    """
    covariance, mean, cap, floor = _inputs(
        covariance, expected_returns, caps, return_floor
    )
    loading = real_array(uncertainty_loading, "uncertainty_loading", 2)
    if loading.shape[0] != len(mean):
        raise ValueError("uncertainty_loading must have one row per asset.")
    radius = real_scalar(radius, "radius", minimum=0)
    risk_aversion = real_scalar(risk_aversion, "risk_aversion", minimum=0)
    return _solve(
        covariance,
        mean,
        cap,
        floor,
        loading=loading,
        radius=radius,
        risk_aversion=risk_aversion,
    )
