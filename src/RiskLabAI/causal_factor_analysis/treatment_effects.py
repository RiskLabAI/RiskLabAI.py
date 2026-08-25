"""Treatment-effect estimands from *Causal Factor Investing* (2023).

The functions are independent implementations of Equations 2-9 and the
linear instrumental-variable ratio in Section 4.3.2.4.  They operate on
already identified interventional means, conditional means, probabilities,
or covariances.  They do not infer a causal graph from data and do not claim
that the supplied adjustment variables or instrument are valid.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import NamedTuple, Sequence

import numpy as np

__all__ = [
    "DifferenceInDifferencesEstimate",
    "TreatmentEffectDecomposition",
    "average_treatment_effect",
    "backdoor_adjusted_average_treatment_effect",
    "backdoor_adjusted_expectation",
    "difference_in_differences",
    "frontdoor_adjusted_probability",
    "linear_instrumental_variable_effect",
    "randomized_mean_difference",
    "treatment_effect_decomposition",
]


class TreatmentEffectDecomposition(NamedTuple):
    """Observed mean difference decomposed into ATT and selection bias."""

    observed_difference: float
    average_treatment_effect_on_treated: float
    sample_selection_bias: float


class DifferenceInDifferencesEstimate(NamedTuple):
    """Two-group, two-period difference-in-differences decomposition."""

    treated_change: float
    control_change: float
    estimate: float


_PROBABILITY_TOLERANCE = 1.0e-12


def _as_finite_real(name: str, value: Real) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a representable real scalar") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _as_finite_vector(name: str, values: Sequence[Real]) -> np.ndarray:
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must be a one-dimensional numeric sequence"
        ) from error
    if raw.ndim != 1 or raw.size == 0 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a nonempty one-dimensional numeric sequence")
    try:
        result = np.array(raw, dtype=float, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must contain representable real values") from error
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _as_probability_vector(name: str, values: Sequence[Real]) -> np.ndarray:
    result = _as_finite_vector(name, values)
    if np.any(result < 0.0) or np.any(result > 1.0):
        raise ValueError(f"{name} must contain probabilities in [0, 1]")
    if not math.isclose(
        float(np.sum(result)),
        1.0,
        rel_tol=_PROBABILITY_TOLERANCE,
        abs_tol=_PROBABILITY_TOLERANCE,
    ):
        raise ValueError(f"{name} must sum to one")
    return result


def average_treatment_effect(
    treated_interventional_mean: Real,
    control_interventional_mean: Real,
) -> float:
    """Return ``E[Y|do(X=x1)] - E[Y|do(X=x0)]`` (Equation 2)."""

    treated = _as_finite_real(
        "treated_interventional_mean", treated_interventional_mean
    )
    control = _as_finite_real(
        "control_interventional_mean", control_interventional_mean
    )
    return _as_finite_real("average_treatment_effect", treated - control)


def treatment_effect_decomposition(
    observed_treated_mean: Real,
    observed_control_mean: Real,
    counterfactual_control_mean_for_treated: Real,
) -> TreatmentEffectDecomposition:
    """Decompose the observed difference into ATT and selection bias.

    This is Equation 3 under consistency: the treated group's observed mean
    is its treated potential-outcome mean.
    """

    treated = _as_finite_real("observed_treated_mean", observed_treated_mean)
    control = _as_finite_real("observed_control_mean", observed_control_mean)
    counterfactual = _as_finite_real(
        "counterfactual_control_mean_for_treated",
        counterfactual_control_mean_for_treated,
    )
    observed_difference = _as_finite_real("observed_difference", treated - control)
    effect_on_treated = _as_finite_real(
        "average_treatment_effect_on_treated", treated - counterfactual
    )
    selection_bias = _as_finite_real("sample_selection_bias", counterfactual - control)
    return TreatmentEffectDecomposition(
        observed_difference=observed_difference,
        average_treatment_effect_on_treated=effect_on_treated,
        sample_selection_bias=selection_bias,
    )


def randomized_mean_difference(
    treated_outcomes: Sequence[Real],
    control_outcomes: Sequence[Real],
) -> float:
    """Return the randomized-study difference in sample means.

    Under the random-assignment identities in Equations 4-6, this is the
    finite-sample estimator of the average treatment effect.
    """

    treated = _as_finite_vector("treated_outcomes", treated_outcomes)
    control = _as_finite_vector("control_outcomes", control_outcomes)
    return _as_finite_real(
        "randomized_mean_difference",
        float(np.mean(treated) - np.mean(control)),
    )


def difference_in_differences(
    treated_before: Real,
    treated_after: Real,
    control_before: Real,
    control_after: Real,
) -> DifferenceInDifferencesEstimate:
    """Return the two-group, two-period difference-in-differences estimate."""

    treated_change = _as_finite_real(
        "treated_change",
        _as_finite_real("treated_after", treated_after)
        - _as_finite_real("treated_before", treated_before),
    )
    control_change = _as_finite_real(
        "control_change",
        _as_finite_real("control_after", control_after)
        - _as_finite_real("control_before", control_before),
    )
    estimate = _as_finite_real(
        "difference_in_differences", treated_change - control_change
    )
    return DifferenceInDifferencesEstimate(
        treated_change=treated_change,
        control_change=control_change,
        estimate=estimate,
    )


def backdoor_adjusted_expectation(
    conditional_outcome_means: Sequence[Real],
    adjustment_probabilities: Sequence[Real],
) -> float:
    """Standardize conditional outcome means over an adjustment distribution.

    For an indicator outcome, this is the discrete probability adjustment in
    Equation 8.  For a general outcome, it is the expectation form used by
    Equation 7.
    """

    means = _as_finite_vector("conditional_outcome_means", conditional_outcome_means)
    probabilities = _as_probability_vector(
        "adjustment_probabilities", adjustment_probabilities
    )
    if means.shape != probabilities.shape:
        raise ValueError(
            "conditional_outcome_means and adjustment_probabilities must have "
            "the same length"
        )
    return _as_finite_real(
        "backdoor_adjusted_expectation", float(np.dot(means, probabilities))
    )


def backdoor_adjusted_average_treatment_effect(
    treated_conditional_means: Sequence[Real],
    control_conditional_means: Sequence[Real],
    adjustment_probabilities: Sequence[Real],
) -> float:
    """Return the standardized average treatment effect in Equation 7."""

    treated = _as_finite_vector("treated_conditional_means", treated_conditional_means)
    control = _as_finite_vector("control_conditional_means", control_conditional_means)
    probabilities = _as_probability_vector(
        "adjustment_probabilities", adjustment_probabilities
    )
    if treated.shape != control.shape or treated.shape != probabilities.shape:
        raise ValueError(
            "treated_conditional_means, control_conditional_means, and "
            "adjustment_probabilities must have the same length"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        contrasts = treated - control
        result = float(np.dot(contrasts, probabilities))
    return _as_finite_real("backdoor_adjusted_average_treatment_effect", result)


def frontdoor_adjusted_probability(
    mediator_probabilities_given_treatment: Sequence[Real],
    outcome_probabilities_given_mediator_and_treatment: Sequence[Sequence[Real]],
    treatment_probabilities: Sequence[Real],
) -> float:
    """Return the discrete front-door adjustment in Equation 9.

    Rows of ``outcome_probabilities_given_mediator_and_treatment`` correspond
    to mediator states; columns correspond to treatment states.  Each entry is
    the conditional probability of one fixed outcome event.
    """

    mediator_probabilities = _as_probability_vector(
        "mediator_probabilities_given_treatment",
        mediator_probabilities_given_treatment,
    )
    treatment_weights = _as_probability_vector(
        "treatment_probabilities", treatment_probabilities
    )
    try:
        raw = np.asarray(outcome_probabilities_given_mediator_and_treatment)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "outcome_probabilities_given_mediator_and_treatment must be a "
            "two-dimensional numeric array"
        ) from error
    if raw.ndim != 2 or raw.dtype.kind not in "iuf":
        raise ValueError(
            "outcome_probabilities_given_mediator_and_treatment must be a "
            "two-dimensional numeric array"
        )
    try:
        conditional = np.array(raw, dtype=float, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "outcome_probabilities_given_mediator_and_treatment must contain "
            "representable real values"
        ) from error
    if not np.all(np.isfinite(conditional)):
        raise ValueError(
            "outcome_probabilities_given_mediator_and_treatment must contain "
            "only finite values"
        )
    if np.any(conditional < 0.0) or np.any(conditional > 1.0):
        raise ValueError(
            "outcome_probabilities_given_mediator_and_treatment must contain "
            "probabilities in [0, 1]"
        )
    expected_shape = (mediator_probabilities.size, treatment_weights.size)
    if conditional.shape != expected_shape:
        raise ValueError(
            "outcome_probabilities_given_mediator_and_treatment must have shape "
            "(number of mediator states, number of treatment states)"
        )
    probability = float(
        np.dot(mediator_probabilities, np.dot(conditional, treatment_weights))
    )
    probability = _as_finite_real("frontdoor_adjusted_probability", probability)
    if probability < 0.0 and probability >= -_PROBABILITY_TOLERANCE:
        return 0.0
    if probability > 1.0 and probability <= 1.0 + _PROBABILITY_TOLERANCE:
        return 1.0
    return probability


def linear_instrumental_variable_effect(
    outcome_instrument_covariance: Real,
    treatment_instrument_covariance: Real,
) -> float:
    """Return the linear IV covariance ratio from Section 4.3.2.4.

    The caller must establish the graphical and substantive IV premises.  A
    zero treatment-instrument covariance violates the ratio's relevance
    requirement and is rejected.
    """

    numerator = _as_finite_real(
        "outcome_instrument_covariance", outcome_instrument_covariance
    )
    denominator = _as_finite_real(
        "treatment_instrument_covariance", treatment_instrument_covariance
    )
    if denominator == 0.0:
        raise ValueError("treatment_instrument_covariance must be nonzero")
    result = numerator / denominator
    if not math.isfinite(result):
        raise ValueError("the instrumental-variable effect must be finite")
    return result
