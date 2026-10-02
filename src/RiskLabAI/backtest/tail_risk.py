"""Peaks-over-threshold risk and likelihood tests of quantile exceptions."""

import numpy as np
from scipy.special import xlogy
from scipy.stats import chi2, genpareto

from RiskLabAI.utils._validation import real_array, real_scalar

__all__ = [
    "generalized_pareto_tail_risk",
    "fit_pot_tail_risk",
    "quantile_exception_tests",
]


def generalized_pareto_tail_risk(
    threshold, exceedance_probability, shape, scale, probability
):
    """Return a loss quantile and upper-tail expectation from a POT model.

    probability is the unconditional quantile level and must exceed
    1-exceedance_probability. Expected shortfall is infinite for shape >= 1.
    The threshold and scale use the same loss units. No time-series filter
    or threshold-selection rule is implied.
    """
    threshold = real_scalar(threshold, "threshold")
    rate = real_scalar(
        exceedance_probability, "exceedance_probability", minimum=0, strict=True
    )
    shape = real_scalar(shape, "shape")
    scale = real_scalar(scale, "scale", minimum=0, strict=True)
    probability = real_scalar(probability, "probability")
    if rate > 1 or not 1 - rate < probability < 1:
        raise ValueError(
            "Require 0 < exceedance_probability <= 1 and 1-rate < probability < 1."
        )
    excess = float(genpareto.isf((1 - probability) / rate, shape, scale=scale))
    quantile = threshold + excess
    shortfall = (
        np.inf if shape >= 1 else quantile + (scale + shape * excess) / (1 - shape)
    )
    if not np.isfinite(quantile) or (shape < 1 and not np.isfinite(shortfall)):
        raise ValueError("Tail risk is outside the supported floating-point range.")
    return {"value_at_risk": float(quantile), "expected_shortfall": float(shortfall)}


def fit_pot_tail_risk(losses, *, threshold, probability):
    """Fit a zero-location generalized Pareto to strictly positive excesses.

    The caller must choose the threshold without using held-out outcomes.
    Return estimated parameters, exceedance count and risk measures.
    SciPy's numerical MLE does not guarantee a globally optimal fit.
    """
    losses = real_array(losses, "losses", 1)
    threshold = real_scalar(threshold, "threshold")
    excess = losses[losses > threshold] - threshold
    if len(excess) < 2 or np.ptp(excess) == 0:
        raise ValueError("At least two distinct positive excesses are required.")
    shape, location, scale = genpareto.fit(excess, floc=0)
    if (
        location != 0
        or not np.isfinite(genpareto.logpdf(excess, shape, scale=scale)).all()
    ):
        raise RuntimeError(
            "The fitted Pareto model fails its support/likelihood check."
        )
    result = generalized_pareto_tail_risk(
        threshold, len(excess) / len(losses), shape, scale, probability
    )
    return {
        **result,
        "shape": float(shape),
        "scale": float(scale),
        "threshold": threshold,
        "exceedance_count": len(excess),
        "sample_count": len(losses),
        "exceedance_probability": len(excess) / len(losses),
    }


def _bernoulli_log_likelihood(zeros, ones, probability):
    return float(xlogy(ones, probability) + xlogy(zeros, 1 - probability))


def quantile_exception_tests(exceptions, *, exception_probability):
    """Return Kupiec coverage and Christoffersen independence/coverage tests.

    exceptions is a nonempty binary vector, with one indicating a loss above
    its forecast quantile. P-values use asymptotic chi-square reference laws.
    Coverage uses all observations; transition likelihoods condition on the
    first observation. If either previous-state transition row is absent,
    independence and conditional coverage are unidentified and return None.
    These are tests of quantile forecasts, not expected-shortfall forecasts.
    """
    raw = np.asarray(exceptions)
    if (
        raw.ndim != 1
        or raw.size == 0
        or raw.dtype.kind not in "biuf"
        or not np.all((raw == 0) | (raw == 1))
    ):
        raise ValueError("exceptions must be a nonempty binary vector.")
    hits = raw.astype(int)
    probability = real_scalar(exception_probability, "exception_probability")
    if not 0 < probability < 1:
        raise ValueError("exception_probability must lie in (0, 1).")
    ones, total = int(hits.sum()), len(hits)
    lr_coverage = max(
        0.0,
        2
        * (
            _bernoulli_log_likelihood(total - ones, ones, ones / total)
            - _bernoulli_log_likelihood(total - ones, ones, probability)
        ),
    )
    transitions = np.zeros((2, 2), dtype=int)
    np.add.at(transitions, (hits[:-1], hits[1:]), 1)
    rows = transitions.sum(axis=1)
    lr_independence = None
    if np.all(rows > 0):
        conditional = sum(
            _bernoulli_log_likelihood(
                transitions[i, 0], transitions[i, 1], transitions[i, 1] / rows[i]
            )
            for i in range(2)
        )
        transition_ones = transitions[:, 1].sum()
        pooled = _bernoulli_log_likelihood(
            transitions[:, 0].sum(), transition_ones, transition_ones / (total - 1)
        )
        lr_independence = max(0.0, 2 * (conditional - pooled))
    combined = None if lr_independence is None else lr_coverage + lr_independence
    return {
        "coverage_statistic": lr_coverage,
        "coverage_pvalue": float(chi2.sf(lr_coverage, 1)),
        "independence_statistic": lr_independence,
        "independence_pvalue": (
            None if lr_independence is None else float(chi2.sf(lr_independence, 1))
        ),
        "conditional_coverage_statistic": combined,
        "conditional_coverage_pvalue": (
            None if combined is None else float(chi2.sf(combined, 2))
        ),
        "transitions": transitions,
        "exception_count": ones,
        "sample_count": total,
    }
