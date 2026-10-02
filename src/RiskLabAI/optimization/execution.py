"""Discrete single-asset execution with constant linear market impact."""

import numpy as np


def almgren_chriss_execution(
    inventory: float,
    horizon: float,
    intervals: int,
    *,
    volatility: float,
    temporary_impact: float,
    permanent_impact: float = 0.0,
    risk_aversion: float = 0.0,
) -> dict[str, np.ndarray | float]:
    """Minimize expected execution cost plus risk_aversion times variance.

    This is the discrete Almgren--Chriss single-asset sell model with zero
    drift and spread, constant volatility, and linear temporary/permanent
    impact. ``inventory`` is a nonnegative quantity, ``horizon`` is positive,
    and ``intervals`` is a positive integer. All other inputs are finite and
    nonnegative, with strictly positive temporary impact.

    With tau = horizon / intervals, the effective temporary coefficient
    eta = temporary_impact - permanent_impact * tau / 2 must be positive.
    This is the strict-convexity condition of the discrete model.
    Volatility is price per square-root time; temporary impact is price
    times time per quantity; permanent impact is price per quantity.
    Risk aversion has units of inverse monetary cost.

    Return ``times`` and ``inventory`` arrays of length intervals + 1,
    ``trades`` of length intervals, and scalar ``expected_cost`` and
    ``cost_variance``. For post-trade inventory x[k] and trades n[k], cost
    is permanent_impact * inventory**2 / 2 + eta / tau * sum(n[k]**2),
    and variance is volatility**2 * tau * sum(x[k]**2), k = 1,...,N.
    Thus one-interval liquidation has zero modeled variance. Zero risk
    aversion or volatility gives constant-size trades. Numerical underflow
    may round very small remaining inventories to zero.

    Invalid inputs or nonfinite derived costs raise ValueError.
    """
    values = {}
    for name, value in (
        ("inventory", inventory),
        ("horizon", horizon),
        ("volatility", volatility),
        ("temporary_impact", temporary_impact),
        ("permanent_impact", permanent_impact),
        ("risk_aversion", risk_aversion),
    ):
        raw = np.asarray(value)
        if raw.ndim != 0 or raw.dtype.kind not in "iuf":
            raise ValueError(f"{name} must be a real numeric scalar.")
        number = float(raw)
        if not np.isfinite(number) or number < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
        values[name] = number
    if (
        isinstance(intervals, (bool, np.bool_))
        or not isinstance(intervals, (int, np.integer))
        or intervals <= 0
    ):
        raise ValueError("intervals must be a positive integer.")
    quantity = values["inventory"]
    duration = values["horizon"]
    sigma = values["volatility"]
    eta = values["temporary_impact"]
    gamma = values["permanent_impact"]
    risk = values["risk_aversion"]
    if duration == 0 or eta == 0:
        raise ValueError("horizon and temporary_impact must be positive.")
    tau = duration / intervals
    effective_eta = eta - 0.5 * gamma * tau
    if tau == 0 or not np.isfinite(effective_eta) or effective_eta <= 0:
        raise ValueError(
            "The time step and effective temporary impact must be positive."
        )
    index = np.arange(intervals + 1, dtype=float)
    if quantity == 0 or risk == 0 or sigma == 0:
        remaining = quantity * (1 - index / intervals)
    else:
        log_z = (
            np.log(tau)
            + np.log(sigma)
            + 0.5 * (np.log(risk) - np.log(effective_eta))
            - np.log(2.0)
        )
        step = (
            2 * np.arcsinh(np.exp(log_z)) if log_z < 350 else 2 * (log_z + np.log(2.0))
        )
        if step == 0:
            remaining = quantity * (1 - index / intervals)
        else:
            remaining = (
                quantity
                * np.exp(-step * index)
                * (-np.expm1(-2 * step * (intervals - index)))
                / (-np.expm1(-2 * step * intervals))
            )
    remaining[0], remaining[-1] = quantity, 0.0
    trades = -np.diff(remaining)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        expected_cost = 0.5 * gamma * np.float64(
            quantity
        ) ** 2 + effective_eta / tau * np.dot(trades, trades)
        variance = np.float64(sigma) ** 2 * tau * np.dot(remaining[1:], remaining[1:])
    if not np.isfinite(expected_cost) or not np.isfinite(variance):
        raise ValueError("Execution costs exceed the supported floating-point range.")
    return {
        "times": np.linspace(0, duration, intervals + 1),
        "inventory": remaining,
        "trades": trades,
        "expected_cost": float(expected_cost),
        "cost_variance": float(variance),
    }
