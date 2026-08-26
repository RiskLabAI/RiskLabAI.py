"""False-discovery probabilities under latent specification search.

This module implements the identified analytical results in Lopez de Prado
and Fabozzi (2026).  Family-level false discovery treats the whole searched
family as null only when every candidate is null.  Selection-level false
discovery instead asks whether the selected maximum itself came from a null
candidate.  The two estimands are intentionally exposed by different
functions and result records.

The empirical parameter-fitting procedure is not included because its data,
optimizer, initialization, and model-selection contract are not fully
specified by the source.  The likelihood and search-adjustment mappings are
complete without inventing those missing choices.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import IntegrationWarning, quad
from scipy.special import log_ndtr, logsumexp, ndtr

__all__ = [
    "FDRComparisonEvidence",
    "FDRNonIdentificationWitness",
    "GaussianSearchAdjustedFDR",
    "GaussianTrialMixture",
    "MaxSelectionFamilyErrors",
    "SelectionLevelProbabilityEvidence",
    "compare_single_and_family_fdr",
    "conditional_upper_tail_probability",
    "family_level_false_discovery_rate",
    "fdr_nonidentification_witness",
    "gaussian_max_selection_cdf",
    "gaussian_max_selection_log_density",
    "gaussian_max_selection_log_likelihood",
    "gaussian_search_adjusted_false_discovery_rate",
    "gaussian_trial_mixture_cdf",
    "max_selection_family_errors",
    "max_selection_null_probability",
    "maximum_mixture_cdf",
    "single_trial_false_discovery_rate",
]


@dataclass(frozen=True, slots=True)
class MaxSelectionFamilyErrors:
    """Family error probabilities induced by selecting a maximum."""

    n_trials: int
    trial_null_probability: float
    trial_type_i_error: float
    trial_type_ii_error: float
    family_null_probability: float
    family_type_i_error: float
    family_type_ii_error: float


@dataclass(frozen=True, slots=True)
class FDRComparisonEvidence:
    """Log-domain evidence for the source's identification conditions."""

    n_trials: int
    single_trial_false_discovery_rate: float
    family_level_false_discovery_rate: float
    log_observed_type_i_inflation: float
    log_required_type_i_inflation: float
    equation_13_log_gap: float
    identification_condition_holds: bool
    single_trial_upper_bounds_family: bool


@dataclass(frozen=True, slots=True)
class GaussianTrialMixture:
    """Two-component Gaussian model for one candidate specification.

    The null mean is fixed at zero, as in Equations 33-36.  Boundary mixture
    weights are valid for distribution evaluation, although methods that
    condition on a non-null searched family require a strictly interior
    ``trial_null_probability``.
    """

    trial_null_probability: float
    null_standard_deviation: float
    alternative_mean: float
    alternative_standard_deviation: float

    def __post_init__(self) -> None:
        probability = _as_probability(
            "trial_null_probability", self.trial_null_probability
        )
        null_scale = _as_positive_real(
            "null_standard_deviation", self.null_standard_deviation
        )
        alternative_mean = _as_finite_real("alternative_mean", self.alternative_mean)
        alternative_scale = _as_positive_real(
            "alternative_standard_deviation", self.alternative_standard_deviation
        )
        object.__setattr__(self, "trial_null_probability", probability)
        object.__setattr__(self, "null_standard_deviation", null_scale)
        object.__setattr__(self, "alternative_mean", alternative_mean)
        object.__setattr__(self, "alternative_standard_deviation", alternative_scale)


@dataclass(frozen=True, slots=True)
class GaussianSearchAdjustedFDR:
    """Per-threshold and aggregate Gaussian search-adjustment evidence."""

    n_trials: int
    thresholds: NDArray[np.float64]
    trial_type_i_errors: NDArray[np.float64]
    trial_type_ii_errors: NDArray[np.float64]
    family_type_i_errors: NDArray[np.float64]
    family_type_ii_errors: NDArray[np.float64]
    mean_family_type_i_error: float
    mean_family_type_ii_error: float
    family_null_probability: float
    false_discovery_rate: float

    def __post_init__(self) -> None:
        array_fields = (
            "thresholds",
            "trial_type_i_errors",
            "trial_type_ii_errors",
            "family_type_i_errors",
            "family_type_ii_errors",
        )
        lengths: set[int] = set()
        for field_name in array_fields:
            values = _as_finite_vector(field_name, getattr(self, field_name))
            lengths.add(values.size)
            object.__setattr__(self, field_name, _readonly_copy(values))
        if lengths == {0}:
            raise ValueError("threshold evidence arrays must be nonempty")
        if len(lengths) != 1:
            raise ValueError("threshold evidence arrays must have equal lengths")


@dataclass(frozen=True, slots=True)
class FDRNonIdentificationWitness:
    """Two-trial equal-component construction proving nonidentification."""

    n_trials: int
    target_false_discovery_rate: float
    trial_null_probability: float
    observable_cdf_at_threshold: float
    latent_trial_cdf_at_threshold: float
    family_null_probability: float
    family_type_i_error: float
    family_type_ii_error: float
    family_false_discovery_rate: float
    reconstructed_observable_cdf_at_threshold: float


@dataclass(frozen=True, slots=True)
class SelectionLevelProbabilityEvidence:
    """Probability evidence that a selected maximum is a null candidate."""

    joint_null_and_selection_probability: float
    selection_probability: float
    conditional_null_probability: float
    quadrature_absolute_error: float


def _as_finite_real(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a representable real scalar") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _as_probability(name: str, value: Real, *, strict: bool = False) -> float:
    result = _as_finite_real(name, value)
    valid = 0.0 < result < 1.0 if strict else 0.0 <= result <= 1.0
    if not valid:
        interval = "strictly between zero and one" if strict else "between zero and one"
        raise ValueError(f"{name} must be {interval}")
    return result


def _as_positive_real(name: str, value: Real) -> float:
    result = _as_finite_real(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _as_nonnegative_real(name: str, value: Real) -> float:
    result = _as_finite_real(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _as_positive_integer(name: str, value: Integral) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _as_finite_vector(name: str, values: ArrayLike) -> NDArray[np.float64]:
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a one-dimensional real array") from error
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if raw.dtype.kind in {"b", "c"}:
        raise ValueError(f"{name} must contain real, non-boolean values")
    if raw.dtype.kind not in {"i", "u", "f"}:
        converted = np.empty(raw.size, dtype=np.float64)
        for index, value in enumerate(raw):
            converted[index] = _as_finite_real(f"{name}[{index}]", value)
    else:
        try:
            converted = raw.astype(np.float64, copy=True)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{name} must contain real values") from error
    if not np.all(np.isfinite(converted)):
        raise ValueError(f"{name} must contain only finite values")
    return converted


def _readonly_copy(values: NDArray[np.float64]) -> NDArray[np.float64]:
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _log_probability(value: float) -> float:
    return -math.inf if value == 0.0 else math.log(value)


def _log_complement(value: float) -> float:
    return -math.inf if value == 1.0 else math.log1p(-value)


def _log_one_minus_exp(log_value: float) -> float:
    """Return ``log(1-exp(log_value))`` for ``log_value <= 0``."""

    if log_value == -math.inf:
        return 0.0
    if log_value == 0.0:
        return -math.inf
    if log_value < -math.log(2.0):
        return math.log1p(-math.exp(log_value))
    return math.log(-math.expm1(log_value))


def _probability_power(probability: float, n_trials: int) -> float:
    if probability == 0.0:
        return 0.0
    if probability == 1.0:
        return 1.0
    return math.exp(n_trials * math.log(probability))


def _one_minus_probability_power(probability: float, n_trials: int) -> float:
    if probability == 0.0:
        return 1.0
    if probability == 1.0:
        return 0.0
    return -math.expm1(n_trials * math.log(probability))


def _posterior_from_log_terms(log_null: float, log_alternative: float) -> float:
    if log_null == -math.inf and log_alternative == -math.inf:
        raise ValueError("the discovery probability must be positive")
    if log_null == -math.inf:
        return 0.0
    if log_alternative == -math.inf:
        return 1.0
    log_denominator = float(logsumexp((log_null, log_alternative)))
    return math.exp(log_null - log_denominator)


def _log_weighted_pair(
    weight: float, first_log_value: float, second_log_value: float
) -> float:
    if weight == 0.0:
        return second_log_value
    if weight == 1.0:
        return first_log_value
    return float(
        np.logaddexp(
            math.log(weight) + first_log_value,
            math.log1p(-weight) + second_log_value,
        )
    )


def single_trial_false_discovery_rate(
    type_i_error: Real,
    type_ii_error: Real,
    trial_null_probability: Real,
) -> float:
    r"""Return the single-trial posterior FDR from Equation 6.

    The three probabilities may include their boundaries.  A configuration
    with zero probability of a discovery is rejected because the posterior is
    then undefined.
    """

    alpha = _as_probability("type_i_error", type_i_error)
    beta = _as_probability("type_ii_error", type_ii_error)
    pi_zero = _as_probability("trial_null_probability", trial_null_probability)
    log_null = _log_probability(alpha) + _log_probability(pi_zero)
    log_alternative = _log_complement(beta) + _log_complement(pi_zero)
    return _posterior_from_log_terms(log_null, log_alternative)


def family_level_false_discovery_rate(
    family_type_i_error: Real,
    family_type_ii_error: Real,
    trial_null_probability: Real,
    n_trials: Integral,
) -> float:
    r"""Return the family-level search-adjusted FDR from Equations 9-10."""

    alpha_family = _as_probability("family_type_i_error", family_type_i_error)
    beta_family = _as_probability("family_type_ii_error", family_type_ii_error)
    pi_zero = _as_probability("trial_null_probability", trial_null_probability)
    trials = _as_positive_integer("n_trials", n_trials)
    log_family_null = trials * _log_probability(pi_zero)
    log_family_alternative = _log_one_minus_exp(log_family_null)
    log_null = _log_probability(alpha_family) + log_family_null
    log_alternative = _log_complement(beta_family) + log_family_alternative
    return _posterior_from_log_terms(log_null, log_alternative)


def max_selection_family_errors(
    trial_type_i_error: Real,
    trial_type_ii_error: Real,
    trial_null_probability: Real,
    n_trials: Integral,
) -> MaxSelectionFamilyErrors:
    r"""Return maximum-selection family errors from Equations 17-18."""

    alpha = _as_probability("trial_type_i_error", trial_type_i_error)
    beta = _as_probability("trial_type_ii_error", trial_type_ii_error)
    pi_zero = _as_probability(
        "trial_null_probability", trial_null_probability, strict=True
    )
    trials = _as_positive_integer("n_trials", n_trials)

    family_null = _probability_power(pi_zero, trials)
    family_alpha = _one_minus_probability_power(1.0 - alpha, trials)
    first_base = math.fsum((pi_zero * (1.0 - alpha), (1.0 - pi_zero) * beta))
    second_base = pi_zero * (1.0 - alpha)
    log_denominator = _log_one_minus_exp(trials * math.log(pi_zero))

    if first_base == 0.0 or first_base == second_base:
        family_beta = 0.0
    else:
        log_first_power = trials * math.log(first_base)
        if second_base == 0.0:
            log_numerator = log_first_power
        else:
            log_ratio = trials * (math.log(second_base) - math.log(first_base))
            log_numerator = log_first_power + _log_one_minus_exp(log_ratio)
        family_beta = math.exp(log_numerator - log_denominator)
        family_beta = min(1.0, max(0.0, family_beta))

    return MaxSelectionFamilyErrors(
        n_trials=trials,
        trial_null_probability=pi_zero,
        trial_type_i_error=alpha,
        trial_type_ii_error=beta,
        family_null_probability=family_null,
        family_type_i_error=family_alpha,
        family_type_ii_error=family_beta,
    )


def compare_single_and_family_fdr(
    trial_type_i_error: Real,
    trial_type_ii_error: Real,
    trial_null_probability: Real,
    n_trials: Integral,
    *,
    family_type_i_error: Real,
    family_type_ii_error: Real,
    relative_tolerance: Real = 1e-12,
    absolute_tolerance: Real = 1e-15,
) -> FDRComparisonEvidence:
    r"""Evaluate the equality and upper-bound conditions in Equations 13-15."""

    alpha = _as_probability("trial_type_i_error", trial_type_i_error, strict=True)
    beta = _as_probability("trial_type_ii_error", trial_type_ii_error, strict=True)
    pi_zero = _as_probability(
        "trial_null_probability", trial_null_probability, strict=True
    )
    trials = _as_positive_integer("n_trials", n_trials)
    alpha_family = _as_probability(
        "family_type_i_error", family_type_i_error, strict=True
    )
    beta_family = _as_probability(
        "family_type_ii_error", family_type_ii_error, strict=True
    )
    relative = _as_nonnegative_real("relative_tolerance", relative_tolerance)
    absolute = _as_nonnegative_real("absolute_tolerance", absolute_tolerance)

    log_observed = math.log(alpha_family) - math.log(alpha)
    log_family_null = trials * math.log(pi_zero)
    log_required = (
        math.log1p(-beta_family)
        - math.log1p(-beta)
        + _log_one_minus_exp(log_family_null)
        - (trials - 1) * math.log(pi_zero)
        - math.log1p(-pi_zero)
    )
    gap = log_observed - log_required
    scale_tolerance = max(
        absolute, relative * max(abs(log_observed), abs(log_required))
    )
    equality = math.isclose(
        log_observed, log_required, rel_tol=relative, abs_tol=absolute
    )

    return FDRComparisonEvidence(
        n_trials=trials,
        single_trial_false_discovery_rate=single_trial_false_discovery_rate(
            alpha, beta, pi_zero
        ),
        family_level_false_discovery_rate=family_level_false_discovery_rate(
            alpha_family, beta_family, pi_zero, trials
        ),
        log_observed_type_i_inflation=log_observed,
        log_required_type_i_inflation=log_required,
        equation_13_log_gap=gap,
        identification_condition_holds=equality,
        single_trial_upper_bounds_family=gap <= scale_tolerance,
    )


def maximum_mixture_cdf(
    null_cdf: Real,
    alternative_cdf: Real,
    trial_null_probability: Real,
    n_trials: Integral,
) -> float:
    r"""Return the CDF of the maximum in Equation 16 from component CDFs."""

    null_value = _as_probability("null_cdf", null_cdf)
    alternative_value = _as_probability("alternative_cdf", alternative_cdf)
    pi_zero = _as_probability("trial_null_probability", trial_null_probability)
    trials = _as_positive_integer("n_trials", n_trials)
    mixture_value = math.fsum(
        (pi_zero * null_value, (1.0 - pi_zero) * alternative_value)
    )
    mixture_value = min(1.0, max(0.0, mixture_value))
    return _probability_power(mixture_value, trials)


def conditional_upper_tail_probability(
    cdf_at_value: Real, cdf_at_threshold: Real
) -> float:
    r"""Return ``P[X >= x | X >= c]`` from Equation 26 for ``x >= c``."""

    at_value = _as_probability("cdf_at_value", cdf_at_value)
    at_threshold = _as_probability("cdf_at_threshold", cdf_at_threshold)
    if at_threshold == 1.0:
        raise ValueError("cdf_at_threshold must be less than one")
    if at_value < at_threshold:
        raise ValueError("cdf_at_value must not be below cdf_at_threshold")
    return (1.0 - at_value) / (1.0 - at_threshold)


def gaussian_trial_mixture_cdf(model: GaussianTrialMixture, x: Real) -> float:
    r"""Return the Gaussian trial-mixture CDF in Equations 33-35."""

    if not isinstance(model, GaussianTrialMixture):
        raise TypeError("model must be a GaussianTrialMixture")
    value = _as_finite_real("x", x)
    null_cdf = float(ndtr(value / model.null_standard_deviation))
    alternative_cdf = float(
        ndtr((value - model.alternative_mean) / model.alternative_standard_deviation)
    )
    result = math.fsum(
        (
            model.trial_null_probability * null_cdf,
            (1.0 - model.trial_null_probability) * alternative_cdf,
        )
    )
    return min(1.0, max(0.0, result))


def gaussian_max_selection_cdf(
    model: GaussianTrialMixture, x: Real, n_trials: Integral
) -> float:
    r"""Return the selected Gaussian maximum CDF in Equation 35."""

    trials = _as_positive_integer("n_trials", n_trials)
    return _probability_power(gaussian_trial_mixture_cdf(model, x), trials)


def _gaussian_component_logs(
    model: GaussianTrialMixture, x: float
) -> tuple[float, float, float, float]:
    null_z = x / model.null_standard_deviation
    alternative_z = (x - model.alternative_mean) / model.alternative_standard_deviation
    null_log_cdf = float(log_ndtr(null_z))
    alternative_log_cdf = float(log_ndtr(alternative_z))
    log_two_pi = math.log(2.0 * math.pi)
    null_log_density = (
        -0.5 * null_z * null_z
        - math.log(model.null_standard_deviation)
        - 0.5 * log_two_pi
    )
    alternative_log_density = (
        -0.5 * alternative_z * alternative_z
        - math.log(model.alternative_standard_deviation)
        - 0.5 * log_two_pi
    )
    return (
        null_log_cdf,
        alternative_log_cdf,
        null_log_density,
        alternative_log_density,
    )


def gaussian_max_selection_log_density(
    model: GaussianTrialMixture, x: Real, n_trials: Integral
) -> float:
    r"""Return the log density of the selected maximum from Equation 36."""

    if not isinstance(model, GaussianTrialMixture):
        raise TypeError("model must be a GaussianTrialMixture")
    value = _as_finite_real("x", x)
    trials = _as_positive_integer("n_trials", n_trials)
    (
        null_log_cdf,
        alternative_log_cdf,
        null_log_density,
        alternative_log_density,
    ) = _gaussian_component_logs(model, value)
    log_mixture_cdf = _log_weighted_pair(
        model.trial_null_probability, null_log_cdf, alternative_log_cdf
    )
    log_mixture_density = _log_weighted_pair(
        model.trial_null_probability,
        null_log_density,
        alternative_log_density,
    )
    return math.log(trials) + (trials - 1) * log_mixture_cdf + log_mixture_density


def gaussian_max_selection_log_likelihood(
    model: GaussianTrialMixture,
    observations: ArrayLike,
    n_trials: Integral,
) -> float:
    r"""Return the Equation 37 log-likelihood for selected maxima."""

    if not isinstance(model, GaussianTrialMixture):
        raise TypeError("model must be a GaussianTrialMixture")
    values = _as_finite_vector("observations", observations)
    if values.size == 0:
        raise ValueError("observations must be nonempty")
    trials = _as_positive_integer("n_trials", n_trials)
    terms = [
        gaussian_max_selection_log_density(model, value, trials) for value in values
    ]
    result = math.fsum(terms)
    if math.isnan(result) or result == math.inf:
        raise ValueError("log-likelihood could not be evaluated")
    return result


def gaussian_search_adjusted_false_discovery_rate(
    model: GaussianTrialMixture,
    thresholds: ArrayLike,
    n_trials: Integral,
) -> GaussianSearchAdjustedFDR:
    r"""Apply Equations 38-40 to a vector of Gaussian thresholds."""

    if not isinstance(model, GaussianTrialMixture):
        raise TypeError("model must be a GaussianTrialMixture")
    threshold_values = _as_finite_vector("thresholds", thresholds)
    if threshold_values.size == 0:
        raise ValueError("thresholds must be nonempty")
    trials = _as_positive_integer("n_trials", n_trials)

    null_z = threshold_values / model.null_standard_deviation
    alternative_z = (
        threshold_values - model.alternative_mean
    ) / model.alternative_standard_deviation
    trial_alpha = np.asarray(ndtr(-null_z), dtype=np.float64)
    trial_beta = np.asarray(ndtr(alternative_z), dtype=np.float64)
    family_alpha = np.empty_like(trial_alpha)
    family_beta = np.empty_like(trial_beta)
    for index, (alpha, beta) in enumerate(zip(trial_alpha, trial_beta, strict=True)):
        evidence = max_selection_family_errors(
            alpha, beta, model.trial_null_probability, trials
        )
        family_alpha[index] = evidence.family_type_i_error
        family_beta[index] = evidence.family_type_ii_error

    mean_family_alpha = float(np.mean(family_alpha))
    mean_family_beta = float(np.mean(family_beta))
    family_null = _probability_power(model.trial_null_probability, trials)
    false_discovery_rate = family_level_false_discovery_rate(
        mean_family_alpha,
        mean_family_beta,
        model.trial_null_probability,
        trials,
    )
    return GaussianSearchAdjustedFDR(
        n_trials=trials,
        thresholds=threshold_values,
        trial_type_i_errors=trial_alpha,
        trial_type_ii_errors=trial_beta,
        family_type_i_errors=family_alpha,
        family_type_ii_errors=family_beta,
        mean_family_type_i_error=mean_family_alpha,
        mean_family_type_ii_error=mean_family_beta,
        family_null_probability=family_null,
        false_discovery_rate=false_discovery_rate,
    )


def fdr_nonidentification_witness(
    observable_cdf_at_threshold: Real, target_false_discovery_rate: Real
) -> FDRNonIdentificationWitness:
    r"""Return the exact two-trial witness from Equations 41-50.

    The construction belongs to the unrestricted model class in which the
    null and alternative component CDFs may be equal.  It does not assert
    nonidentification under a separated parametric mixture restriction.
    """

    observable_cdf = _as_probability(
        "observable_cdf_at_threshold", observable_cdf_at_threshold, strict=True
    )
    target = _as_probability(
        "target_false_discovery_rate", target_false_discovery_rate, strict=True
    )
    trials = 2
    trial_null = math.sqrt(target)
    latent_cdf = math.sqrt(observable_cdf)
    family_null = trial_null**trials
    family_alpha = 1.0 - observable_cdf
    family_beta = observable_cdf
    family_fdr = family_level_false_discovery_rate(
        family_alpha, family_beta, trial_null, trials
    )
    reconstructed = latent_cdf**trials
    return FDRNonIdentificationWitness(
        n_trials=trials,
        target_false_discovery_rate=target,
        trial_null_probability=trial_null,
        observable_cdf_at_threshold=observable_cdf,
        latent_trial_cdf_at_threshold=latent_cdf,
        family_null_probability=family_null,
        family_type_i_error=family_alpha,
        family_type_ii_error=family_beta,
        family_false_discovery_rate=family_fdr,
        reconstructed_observable_cdf_at_threshold=reconstructed,
    )


def max_selection_null_probability(
    threshold: Real,
    trial_null_probability: Real,
    n_trials: Integral,
    *,
    null_density: Callable[[float], Real],
    mixture_cdf: Callable[[float], Real],
    absolute_tolerance: Real = 1e-12,
    relative_tolerance: Real = 1e-10,
    max_subintervals: Integral = 200,
) -> SelectionLevelProbabilityEvidence:
    r"""Return the selected-winner null probability from footnote 25.

    The numerator is the joint probability that a null candidate wins and
    the selected maximum exceeds ``threshold``.  Adaptive quadrature is
    performed only over the caller-supplied null density and trial-mixture
    CDF; neither callback is treated as an unverified probability oracle.
    """

    threshold_value = _as_finite_real("threshold", threshold)
    pi_zero = _as_probability("trial_null_probability", trial_null_probability)
    trials = _as_positive_integer("n_trials", n_trials)
    if not callable(null_density):
        raise TypeError("null_density must be callable")
    if not callable(mixture_cdf):
        raise TypeError("mixture_cdf must be callable")
    absolute = _as_positive_real("absolute_tolerance", absolute_tolerance)
    relative = _as_positive_real("relative_tolerance", relative_tolerance)
    subdivisions = _as_positive_integer("max_subintervals", max_subintervals)

    def checked_cdf(value: float) -> float:
        return _as_probability("mixture_cdf return value", mixture_cdf(value))

    mixture_at_threshold = checked_cdf(threshold_value)
    selection_probability = _one_minus_probability_power(mixture_at_threshold, trials)
    if selection_probability == 0.0:
        raise ValueError("the selection probability must be positive")

    if pi_zero == 0.0:
        return SelectionLevelProbabilityEvidence(
            joint_null_and_selection_probability=0.0,
            selection_probability=selection_probability,
            conditional_null_probability=0.0,
            quadrature_absolute_error=0.0,
        )

    def integrand(value: float) -> float:
        density = _as_finite_real("null_density return value", null_density(value))
        if density < 0.0:
            raise ValueError("null_density return value must be nonnegative")
        cdf_value = checked_cdf(value)
        return density * _probability_power(cdf_value, trials - 1)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", IntegrationWarning)
            integral, error = quad(
                integrand,
                threshold_value,
                np.inf,
                epsabs=absolute,
                epsrel=relative,
                limit=subdivisions,
            )
    except IntegrationWarning as warning:
        raise ValueError("selection-level quadrature did not converge") from warning
    if not math.isfinite(integral) or not math.isfinite(error) or integral < 0.0:
        raise ValueError("selection-level quadrature returned invalid evidence")

    scale = trials * pi_zero
    joint_probability = scale * integral
    absolute_error = scale * error
    comparison_tolerance = max(
        absolute,
        relative * selection_probability,
        8.0 * np.finfo(np.float64).eps,
    )
    if joint_probability > selection_probability + comparison_tolerance:
        raise ValueError("callbacks imply a null probability above one")
    joint_probability = min(selection_probability, max(0.0, joint_probability))
    conditional = joint_probability / selection_probability
    return SelectionLevelProbabilityEvidence(
        joint_null_and_selection_probability=joint_probability,
        selection_probability=selection_probability,
        conditional_null_probability=conditional,
        quadrature_absolute_error=absolute_error,
    )
