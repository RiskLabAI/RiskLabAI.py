"""Exponential Hawkes events and VPIN from supplied completed volume buckets."""

import numpy as np
from scipy.optimize import minimize

from RiskLabAI.utils._validation import positive_integer, real_array, real_scalar

__all__ = [
    "hawkes_intensity",
    "hawkes_integrated_intensity",
    "hawkes_log_likelihood",
    "simulate_hawkes",
    "fit_hawkes",
    "volume_synchronized_pin",
]


def _parameters(mu, alpha, beta):
    mu = real_scalar(mu, "mu", minimum=0, strict=True)
    alpha = real_scalar(alpha, "alpha", minimum=0)
    beta = real_scalar(beta, "beta", minimum=0, strict=True)
    if alpha >= beta:
        raise ValueError("Stationarity requires alpha < beta.")
    return mu, alpha, beta


def _events(events, horizon):
    events = real_array(events, "events", 1, empty=True)
    if np.any(events < 0) or np.any(events > horizon) or np.any(np.diff(events) <= 0):
        raise ValueError("Events must be strictly increasing in [0, horizon].")
    return events


def hawkes_intensity(times, events, mu, alpha, beta):
    """Return predictable intensity using only events strictly before each time.

    The model has empty prehistory at zero and kernel alpha*exp(-beta*t).
    All query times and event times must be finite and nonnegative.
    """
    mu, alpha, beta = _parameters(mu, alpha, beta)
    times = real_array(times, "times", 1, empty=True)
    events = _events(events, np.inf)
    if np.any(times < 0):
        raise ValueError("times must be nonnegative.")
    return np.array(
        [mu + alpha * np.exp(-beta * (t - events[events < t])).sum() for t in times]
    )


def hawkes_integrated_intensity(events, horizon, mu, alpha, beta):
    """Integrate empty-history intensity over [0, horizon]."""
    mu, alpha, beta = _parameters(mu, alpha, beta)
    horizon = real_scalar(horizon, "horizon", minimum=0, strict=True)
    events = _events(events, horizon)
    value = mu * horizon + alpha / beta * (-np.expm1(-beta * (horizon - events))).sum()
    if not np.isfinite(value):
        raise ValueError("Integrated intensity is outside floating-point range.")
    return float(value)


def hawkes_log_likelihood(events, horizon, mu, alpha, beta):
    """Compute the exact event log likelihood with an O(n) exponential recurrence."""
    mu, alpha, beta = _parameters(mu, alpha, beta)
    horizon = real_scalar(horizon, "horizon", minimum=0, strict=True)
    events = _events(events, horizon)
    excitation = 0.0
    total = 0.0
    previous = 0.0
    for index, event in enumerate(events):
        excitation = np.exp(-beta * (event - previous)) * (
            excitation + (1.0 if index else 0.0)
        )
        total += np.log(mu + alpha * excitation)
        previous = event
    return float(total - hawkes_integrated_intensity(events, horizon, mu, alpha, beta))


def simulate_hawkes(horizon, mu, alpha, beta, *, random_state=None, max_events=100000):
    """Simulate empty-history events by adaptive thinning with a local RNG.

    Raise RuntimeError at max_events instead of returning a truncated path.
    """
    mu, alpha, beta = _parameters(mu, alpha, beta)
    horizon = real_scalar(horizon, "horizon", minimum=0, strict=True)
    max_events = positive_integer(max_events, "max_events")
    rng = np.random.default_rng(random_state)
    events = []
    time, excitation = 0.0, 0.0
    while True:
        bound = mu + excitation
        wait = rng.exponential(1 / bound)
        next_time = time + wait
        if next_time > horizon:
            break
        if next_time <= time:
            raise RuntimeError("Time increments are below floating-point resolution.")
        excitation *= np.exp(-beta * wait)
        time = next_time
        if rng.random() * bound <= mu + excitation:
            if len(events) >= max_events:
                raise RuntimeError("max_events reached before simulation completed.")
            events.append(time)
            excitation += alpha
    return np.asarray(events)


def fit_hawkes(events, horizon, *, initial=None):
    """Fit mu, alpha, beta by constrained likelihood with three fixed starts.

    Empty histories have no positive-baseline MLE and are rejected. Return
    optimizer diagnostics; a successful numerical fit is not an assertion
    of parameter identification or a globally optimal likelihood.
    """
    horizon = real_scalar(horizon, "horizon", minimum=0, strict=True)
    events = _events(events, horizon)
    if len(events) < 2:
        raise ValueError("At least two events are required to fit three parameters.")
    scaled = events / horizon
    n = len(events)
    starts = [(n * (1 - ratio), ratio * n, n) for ratio in (0.1, 0.5, 0.8)]
    if initial is not None:
        initial = real_array(initial, "initial", 1)
        if initial.shape != (3,):
            raise ValueError("initial must contain mu, alpha and beta.")
        starts = [tuple(x * horizon for x in _parameters(*initial))]

    def objective(x):
        baseline, branching, decay = x
        return -hawkes_log_likelihood(scaled, 1.0, baseline, branching * decay, decay)

    results = [
        minimize(
            objective,
            [mu, alpha / beta, beta],
            method="L-BFGS-B",
            bounds=[(1e-9, None), (0, 1 - 1e-8), (1e-9, None)],
        )
        for mu, alpha, beta in starts
    ]
    valid = [r for r in results if r.success and np.isfinite(r.fun)]
    if not valid:
        raise RuntimeError("Hawkes likelihood optimization failed.")
    best = min(valid, key=lambda r: r.fun)
    mu, branching, beta = best.x
    parameters = (mu / horizon, branching * beta / horizon, beta / horizon)
    return {
        "mu": float(parameters[0]),
        "alpha": float(parameters[1]),
        "beta": float(parameters[2]),
        "log_likelihood": hawkes_log_likelihood(events, horizon, *parameters),
        "branching_ratio": float(branching),
        "converged_starts": len(valid),
        "message": str(best.message),
    }


def volume_synchronized_pin(buy_volume, sell_volume, *, bucket_volume, window):
    """Compute rolling VPIN for equal-volume completed, preclassified buckets.

    Return n-window+1 values, aligned to the final bucket of each window.
    Partial buckets and implicit trade classification are not accepted.
    """
    buy = real_array(buy_volume, "buy_volume", 1)
    sell = real_array(sell_volume, "sell_volume", 1)
    volume = real_scalar(bucket_volume, "bucket_volume", minimum=0, strict=True)
    window = positive_integer(window, "window")
    if (
        sell.shape != buy.shape
        or np.any(buy < 0)
        or np.any(sell < 0)
        or not np.allclose((buy + sell) / volume, 1, atol=1e-12, rtol=1e-12)
    ):
        raise ValueError(
            "Buy and sell volumes must be nonnegative completed equal-volume buckets."
        )
    if window > len(buy):
        raise ValueError("window cannot exceed the number of completed buckets.")
    return np.convolve(
        np.abs(buy - sell) / volume, np.full(window, 1 / window), mode="valid"
    )
