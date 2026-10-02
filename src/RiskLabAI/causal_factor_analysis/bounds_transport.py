"""Bounded-outcome identification and explicitly supported standardization."""

import numpy as np

from RiskLabAI.utils._validation import probability_rows, real_array, real_scalar
from .treatment_effects import backdoor_adjusted_average_treatment_effect

__all__ = ["manski_ate_bounds", "transported_ate"]


def manski_ate_bounds(outcomes, treatment, *, lower, upper):
    """Return worst-case finite-population ATE bounds under consistency.

    Treatment is binary and every potential outcome must lie in the supplied
    [lower, upper]. No treatment-exchangeability assumption is made. Bounds
    concern this empirical population; no sampling confidence interval is
    supplied. All-treated and all-control populations are allowed.
    """
    y = real_array(outcomes, "outcomes", 1)
    a = np.asarray(treatment)
    lower, upper = real_scalar(lower, "lower"), real_scalar(upper, "upper")
    if (
        a.shape != y.shape
        or a.dtype.kind not in "biuf"
        or not np.all((a == 0) | (a == 1))
        or lower > upper
        or np.any(y < lower)
        or np.any(y > upper)
    ):
        raise ValueError(
            "Require binary matching treatment and outcomes within ordered bounds."
        )
    lower_effect = np.where(a == 1, y - upper, lower - y)
    upper_effect = np.where(a == 1, y - lower, upper - y)
    return {
        "lower": float(lower_effect.mean()),
        "upper": float(upper_effect.mean()),
        "treated_fraction": float(a.mean()),
        "sample_count": len(y),
    }


def transported_ate(
    treated_means,
    control_means,
    target_probabilities,
    source_treated_counts,
    source_control_counts,
):
    """Standardize supported stratum effects to the target population.

    Identification requires within-source treatment exchangeability,
    consistency, shared treatment definitions, and transportability of
    conditional potential-outcome means. These scientific assumptions cannot
    be verified numerically here. Every target-positive stratum needs source
    observations of both treatments. Counts may be nonnegative real weights.
    Zero-target-mass strata contribute nothing, but supplied means stay finite.
    """
    treated = real_array(treated_means, "treated_means", 1)
    control = real_array(control_means, "control_means", 1)
    probabilities = probability_rows(target_probabilities, "target_probabilities", 1)
    count1 = real_array(source_treated_counts, "source_treated_counts", 1)
    count0 = real_array(source_control_counts, "source_control_counts", 1)
    if (
        any(
            array.shape != treated.shape
            for array in (control, probabilities, count1, count0)
        )
        or np.any(count1 < 0)
        or np.any(count0 < 0)
    ):
        raise ValueError("Stratum arrays must match with nonnegative source counts.")
    if np.any((probabilities > 0) & ((count1 == 0) | (count0 == 0))):
        raise ValueError("Target-positive strata lack source treatment support.")
    return backdoor_adjusted_average_treatment_effect(treated, control, probabilities)
