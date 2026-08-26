"""Mathematical tests for false-discovery adjustment under search."""

from __future__ import annotations

import itertools
import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from RiskLabAI.causal_factor_analysis.search_adjusted_fdr import (
    FDRComparisonEvidence,
    FDRNonIdentificationWitness,
    GaussianSearchAdjustedFDR,
    GaussianTrialMixture,
    MaxSelectionFamilyErrors,
    SelectionLevelProbabilityEvidence,
    compare_single_and_family_fdr,
    conditional_upper_tail_probability,
    family_level_false_discovery_rate,
    fdr_nonidentification_witness,
    gaussian_max_selection_cdf,
    gaussian_max_selection_log_density,
    gaussian_max_selection_log_likelihood,
    gaussian_search_adjusted_false_discovery_rate,
    gaussian_trial_mixture_cdf,
    max_selection_family_errors,
    max_selection_null_probability,
    maximum_mixture_cdf,
    single_trial_false_discovery_rate,
)

PUBLIC_NAMES = {
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
}


def _normal_cdf(value: float) -> float:
    return 0.5 * math.erfc(-value / math.sqrt(2.0))


def _normal_density(value: float) -> float:
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def _enumerated_family_beta(
    alpha: float, beta: float, pi_zero: float, n_trials: int
) -> float:
    numerator = 0.0
    for alternatives in range(1, n_trials + 1):
        probability_of_state = (
            math.comb(n_trials, alternatives)
            * pi_zero ** (n_trials - alternatives)
            * (1.0 - pi_zero) ** alternatives
        )
        no_discovery = (1.0 - alpha) ** (n_trials - alternatives) * (beta**alternatives)
        numerator += probability_of_state * no_discovery
    return numerator / (1.0 - pi_zero**n_trials)


def _enumerated_maximum_cdf(
    null_cdf: float, alternative_cdf: float, pi_zero: float, n_trials: int
) -> float:
    result = 0.0
    for alternatives in range(n_trials + 1):
        result += (
            math.comb(n_trials, alternatives)
            * pi_zero ** (n_trials - alternatives)
            * (1.0 - pi_zero) ** alternatives
            * null_cdf ** (n_trials - alternatives)
            * alternative_cdf**alternatives
        )
    return result


def test_module_exposes_exact_frozen_fdr_family():
    from RiskLabAI.causal_factor_analysis import search_adjusted_fdr

    assert set(search_adjusted_fdr.__all__) == PUBLIC_NAMES
    assert len(search_adjusted_fdr.__all__) == 19


def test_single_trial_fdr_matches_bayes_table_and_published_case_b():
    alpha = 0.024997895148220373
    beta = 0.9515427737332771
    pi_zero = 0.15
    expected = alpha * pi_zero / (alpha * pi_zero + (1.0 - beta) * (1.0 - pi_zero))
    result = single_trial_false_discovery_rate(alpha, beta, pi_zero)
    assert result == pytest.approx(expected, rel=2e-15)
    assert result == pytest.approx(0.08344067427559432, rel=2e-15)


@pytest.mark.parametrize(
    "alpha,beta,pi_zero,expected",
    [
        (0.0, 0.2, 0.7, 0.0),
        (0.2, 1.0, 0.7, 1.0),
        (0.4, 0.6, 0.5, 0.5),
        (1.0, 0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0, 1.0),
    ],
)
def test_single_trial_fdr_boundary_identities(alpha, beta, pi_zero, expected):
    assert single_trial_false_discovery_rate(alpha, beta, pi_zero) == expected


def test_single_trial_fdr_remains_stable_for_tiny_probabilities():
    result = single_trial_false_discovery_rate(1e-300, 0.5, 0.5)
    assert result == pytest.approx(2e-300, rel=1e-12, abs=0.0)


def test_family_level_fdr_matches_published_case_a():
    alpha_family = 0.22365361940347483
    beta_family = 0.7531960884251256
    result = family_level_false_discovery_rate(alpha_family, beta_family, 0.95, 10)
    assert result == pytest.approx(0.5748603608683863, rel=2e-15)


def test_family_level_fdr_reduces_to_single_trial_at_one_trial():
    for alpha, beta, pi_zero in (
        (0.05, 0.2, 0.7),
        (0.4, 0.6, 0.5),
        (0.9, 0.1, 0.02),
    ):
        assert family_level_false_discovery_rate(alpha, beta, pi_zero, 1) == (
            pytest.approx(single_trial_false_discovery_rate(alpha, beta, pi_zero))
        )


def test_family_level_fdr_uses_all_null_family_prior():
    alpha_family, beta_family, pi_zero, n_trials = 0.3, 0.4, 0.8, 3
    family_prior = pi_zero**n_trials
    expected = (
        alpha_family
        * family_prior
        / (alpha_family * family_prior + (1.0 - beta_family) * (1.0 - family_prior))
    )
    assert family_level_false_discovery_rate(
        alpha_family, beta_family, pi_zero, n_trials
    ) == pytest.approx(expected, rel=2e-15)


def test_maximum_selection_errors_match_published_examples():
    case_a = max_selection_family_errors(
        0.024997895148220373, 0.9515427737332771, 0.95, 10
    )
    sensitivity = max_selection_family_errors(0.05, 0.90, 0.90, 5)

    assert case_a.family_type_i_error == pytest.approx(0.22365361940347483, rel=2e-15)
    assert case_a.family_type_ii_error == pytest.approx(0.7531960884251256, rel=2e-15)
    assert case_a.family_null_probability == pytest.approx(
        0.5987369392383787, rel=2e-15
    )
    assert sensitivity.family_type_i_error == pytest.approx(
        0.22621906250000023, rel=2e-15
    )
    assert sensitivity.family_type_ii_error == pytest.approx(
        0.7245771630882027, rel=2e-15
    )
    assert sensitivity.family_null_probability == pytest.approx(
        0.5904900000000001, rel=2e-15
    )
    assert family_level_false_discovery_rate(
        sensitivity.family_type_i_error,
        sensitivity.family_type_ii_error,
        0.90,
        5,
    ) == pytest.approx(0.5421963202650197, rel=2e-15)


def test_maximum_selection_errors_match_binomial_state_enumeration():
    rng = np.random.default_rng(99117)
    for _ in range(100):
        alpha, beta, pi_zero = rng.uniform(1e-4, 1.0 - 1e-4, size=3)
        n_trials = int(rng.integers(1, 9))
        result = max_selection_family_errors(alpha, beta, pi_zero, n_trials)
        assert result.family_type_i_error == pytest.approx(
            1.0 - (1.0 - alpha) ** n_trials, rel=2e-13, abs=2e-15
        )
        assert result.family_type_ii_error == pytest.approx(
            _enumerated_family_beta(alpha, beta, pi_zero, n_trials),
            rel=2e-13,
            abs=2e-15,
        )


def test_maximum_selection_errors_one_trial_identity_and_record_fields():
    result = max_selection_family_errors(0.12, 0.37, 0.61, 1)
    assert result == MaxSelectionFamilyErrors(
        n_trials=1,
        trial_null_probability=0.61,
        trial_type_i_error=0.12,
        trial_type_ii_error=0.37,
        family_null_probability=0.61,
        family_type_i_error=pytest.approx(0.12),
        family_type_ii_error=pytest.approx(0.37),
    )


def test_maximum_selection_errors_are_stable_near_boundaries():
    result = max_selection_family_errors(1e-14, 1.0 - 1e-14, 0.999999, 100_000)
    assert result.family_type_i_error == pytest.approx(1e-9, rel=5e-8)
    assert 0.0 <= result.family_type_ii_error <= 1.0
    assert result.family_null_probability == pytest.approx(
        math.exp(100_000 * math.log(0.999999)), rel=2e-15
    )


def test_equation_13_equality_and_both_inequality_directions():
    alpha, beta, pi_zero, n_trials = 0.05, 0.2, 0.8, 2
    beta_family = 0.3
    required_ratio = (
        (1.0 - beta_family)
        / (1.0 - beta)
        * (1.0 - pi_zero**n_trials)
        / (pi_zero ** (n_trials - 1) * (1.0 - pi_zero))
    )
    equality_alpha = alpha * required_ratio
    equal = compare_single_and_family_fdr(
        alpha,
        beta,
        pi_zero,
        n_trials,
        family_type_i_error=equality_alpha,
        family_type_ii_error=beta_family,
    )
    lower = compare_single_and_family_fdr(
        alpha,
        beta,
        pi_zero,
        n_trials,
        family_type_i_error=equality_alpha * 0.8,
        family_type_ii_error=beta_family,
    )
    upper = compare_single_and_family_fdr(
        alpha,
        beta,
        pi_zero,
        n_trials,
        family_type_i_error=equality_alpha * 1.2,
        family_type_ii_error=beta_family,
    )

    assert isinstance(equal, FDRComparisonEvidence)
    assert equal.equation_13_log_gap == pytest.approx(0.0, abs=2e-15)
    assert equal.identification_condition_holds is True
    assert equal.single_trial_upper_bounds_family is True
    assert equal.single_trial_false_discovery_rate == pytest.approx(
        equal.family_level_false_discovery_rate, rel=2e-15
    )
    assert lower.equation_13_log_gap < 0.0
    assert lower.single_trial_upper_bounds_family is True
    assert (
        lower.family_level_false_discovery_rate
        < lower.single_trial_false_discovery_rate
    )
    assert upper.equation_13_log_gap > 0.0
    assert upper.single_trial_upper_bounds_family is False
    assert (
        upper.family_level_false_discovery_rate
        > upper.single_trial_false_discovery_rate
    )


def test_maximum_mixture_cdf_matches_latent_state_enumeration():
    for null_cdf, alternative_cdf, pi_zero, n_trials in (
        (0.4, 0.8, 0.7, 1),
        (0.4, 0.8, 0.7, 4),
        (0.99, 0.2, 0.01, 8),
    ):
        expected = _enumerated_maximum_cdf(null_cdf, alternative_cdf, pi_zero, n_trials)
        assert maximum_mixture_cdf(
            null_cdf, alternative_cdf, pi_zero, n_trials
        ) == pytest.approx(expected, rel=3e-15, abs=1e-16)


def test_maximum_mixture_cdf_boundary_identities():
    assert maximum_mixture_cdf(0.2, 0.7, 1.0, 3) == pytest.approx(0.2**3)
    assert maximum_mixture_cdf(0.2, 0.7, 0.0, 3) == pytest.approx(0.7**3)
    assert maximum_mixture_cdf(0.0, 0.0, 0.4, 5) == 0.0
    assert maximum_mixture_cdf(1.0, 1.0, 0.4, 5) == 1.0


def test_conditional_upper_tail_matches_uniform_and_exponential_oracles():
    assert conditional_upper_tail_probability(0.8, 0.3) == pytest.approx(2.0 / 7.0)
    threshold, value = 0.4, 1.7
    exponential_result = conditional_upper_tail_probability(
        1.0 - math.exp(-value), 1.0 - math.exp(-threshold)
    )
    assert exponential_result == pytest.approx(
        math.exp(-(value - threshold)), rel=2e-15
    )
    assert conditional_upper_tail_probability(0.3, 0.3) == 1.0
    assert conditional_upper_tail_probability(1.0, 0.3) == 0.0


def test_gaussian_trial_mixture_matches_erfc_oracle():
    model = GaussianTrialMixture(0.75, 1.0, 2.0, 0.5)
    for value in (-3.0, 0.0, 1.25, 4.0):
        expected = 0.75 * _normal_cdf(value) + 0.25 * _normal_cdf((value - 2.0) / 0.5)
        assert gaussian_trial_mixture_cdf(model, value) == pytest.approx(
            expected, rel=2e-15, abs=1e-16
        )


def test_gaussian_maximum_cdf_and_log_density_match_direct_formula():
    model = GaussianTrialMixture(0.65, 1.2, 1.4, 0.8)
    value, n_trials = 0.7, 5
    null_z = value / 1.2
    alternative_z = (value - 1.4) / 0.8
    mixture_cdf = 0.65 * _normal_cdf(null_z) + 0.35 * _normal_cdf(alternative_z)
    mixture_density = 0.65 * _normal_density(null_z) / 1.2 + 0.35 * (
        _normal_density(alternative_z) / 0.8
    )
    expected_density = n_trials * mixture_cdf ** (n_trials - 1) * mixture_density

    assert gaussian_max_selection_cdf(model, value, n_trials) == pytest.approx(
        mixture_cdf**n_trials, rel=2e-15
    )
    assert gaussian_max_selection_log_density(model, value, n_trials) == pytest.approx(
        math.log(expected_density), rel=2e-15, abs=2e-15
    )


def test_gaussian_log_density_matches_numerical_cdf_derivative():
    model = GaussianTrialMixture(0.4, 0.9, 1.2, 1.7)
    n_trials = 4
    for value in (-1.0, 0.5, 2.0):
        step = 1e-5
        derivative = (
            gaussian_max_selection_cdf(model, value + step, n_trials)
            - gaussian_max_selection_cdf(model, value - step, n_trials)
        ) / (2.0 * step)
        density = math.exp(gaussian_max_selection_log_density(model, value, n_trials))
        assert density == pytest.approx(derivative, rel=2e-9, abs=2e-12)


def test_gaussian_log_density_is_stable_when_cdf_underflows():
    model = GaussianTrialMixture(0.5, 1.0, 3.0, 2.0)
    assert gaussian_max_selection_cdf(model, -40.0, 10) == 0.0
    log_density = gaussian_max_selection_log_density(model, -40.0, 10)
    assert math.isfinite(log_density)
    assert log_density < -1_000.0


def test_gaussian_log_likelihood_is_direct_sum_and_permutation_invariant():
    model = GaussianTrialMixture(0.7, 1.1, 1.8, 0.9)
    observations = np.array([-0.2, 0.7, 1.3, 2.8])
    expected = math.fsum(
        gaussian_max_selection_log_density(model, value, 6) for value in observations
    )
    actual = gaussian_max_selection_log_likelihood(model, observations, 6)
    reversed_actual = gaussian_max_selection_log_likelihood(
        model, observations[::-1], 6
    )
    assert actual == pytest.approx(expected, rel=2e-15)
    assert reversed_actual == pytest.approx(actual, rel=2e-15)


def test_gaussian_search_adjustment_matches_direct_equations_38_to_40():
    model = GaussianTrialMixture(0.82, 1.1, 0.6, 0.7)
    thresholds = np.array([-0.2, 0.4, 1.0, 1.6])
    n_trials = 7
    result = gaussian_search_adjusted_false_discovery_rate(model, thresholds, n_trials)

    expected_alpha = np.array([1.0 - _normal_cdf(value / 1.1) for value in thresholds])
    expected_beta = np.array([_normal_cdf((value - 0.6) / 0.7) for value in thresholds])
    expected_family = [
        max_selection_family_errors(alpha, beta, 0.82, n_trials)
        for alpha, beta in zip(expected_alpha, expected_beta, strict=True)
    ]
    expected_family_alpha = np.array(
        [evidence.family_type_i_error for evidence in expected_family]
    )
    expected_family_beta = np.array(
        [evidence.family_type_ii_error for evidence in expected_family]
    )
    expected_fdr = family_level_false_discovery_rate(
        float(np.mean(expected_family_alpha)),
        float(np.mean(expected_family_beta)),
        0.82,
        n_trials,
    )

    assert isinstance(result, GaussianSearchAdjustedFDR)
    assert result.n_trials == n_trials
    assert result.thresholds == pytest.approx(thresholds)
    assert result.trial_type_i_errors == pytest.approx(expected_alpha, rel=2e-15)
    assert result.trial_type_ii_errors == pytest.approx(expected_beta, rel=2e-15)
    assert result.family_type_i_errors == pytest.approx(
        expected_family_alpha, rel=2e-15
    )
    assert result.family_type_ii_errors == pytest.approx(
        expected_family_beta, rel=2e-15
    )
    assert result.mean_family_type_i_error == pytest.approx(
        float(np.mean(expected_family_alpha)), rel=2e-15
    )
    assert result.mean_family_type_ii_error == pytest.approx(
        float(np.mean(expected_family_beta)), rel=2e-15
    )
    assert result.family_null_probability == pytest.approx(0.82**n_trials)
    assert result.false_discovery_rate == pytest.approx(expected_fdr, rel=2e-15)


def test_constant_threshold_calibration_reduces_to_one_family_error_pair():
    model = GaussianTrialMixture(0.75, 1.0, 0.5, 1.4)
    result = gaussian_search_adjusted_false_discovery_rate(model, [0.8, 0.8, 0.8], 4)
    direct = max_selection_family_errors(
        result.trial_type_i_errors[0],
        result.trial_type_ii_errors[0],
        0.75,
        4,
    )
    assert np.all(result.family_type_i_errors == result.family_type_i_errors[0])
    assert np.all(result.family_type_ii_errors == result.family_type_ii_errors[0])
    assert result.mean_family_type_i_error == pytest.approx(direct.family_type_i_error)
    assert result.mean_family_type_ii_error == pytest.approx(
        direct.family_type_ii_error
    )


@pytest.mark.parametrize(
    "observable_cdf,target",
    [(0.2, 0.1), (0.63, 0.4), (0.999, 0.95)],
)
def test_nonidentification_witness_reconstructs_observable_and_target(
    observable_cdf, target
):
    result = fdr_nonidentification_witness(observable_cdf, target)
    assert isinstance(result, FDRNonIdentificationWitness)
    assert result.n_trials == 2
    assert result.trial_null_probability == pytest.approx(math.sqrt(target))
    assert result.latent_trial_cdf_at_threshold == pytest.approx(
        math.sqrt(observable_cdf)
    )
    assert result.family_null_probability == pytest.approx(target)
    assert result.family_type_i_error == pytest.approx(1.0 - observable_cdf)
    assert result.family_type_ii_error == pytest.approx(observable_cdf)
    assert result.family_false_discovery_rate == pytest.approx(target, rel=2e-15)
    assert result.reconstructed_observable_cdf_at_threshold == pytest.approx(
        observable_cdf, rel=2e-15
    )


def test_selection_level_probability_is_pi_zero_for_identical_components():
    pi_zero, n_trials, threshold = 0.37, 4, 0.7

    def density(value: float) -> float:
        return math.exp(-value) if value >= 0.0 else 0.0

    def cdf(value: float) -> float:
        return 1.0 - math.exp(-value) if value >= 0.0 else 0.0

    result = max_selection_null_probability(
        threshold,
        pi_zero,
        n_trials,
        null_density=density,
        mixture_cdf=cdf,
    )
    expected_selection = 1.0 - cdf(threshold) ** n_trials
    assert isinstance(result, SelectionLevelProbabilityEvidence)
    assert result.selection_probability == pytest.approx(expected_selection, rel=2e-15)
    assert result.joint_null_and_selection_probability == pytest.approx(
        pi_zero * expected_selection, rel=2e-12
    )
    assert result.conditional_null_probability == pytest.approx(pi_zero, rel=2e-12)
    assert result.quadrature_absolute_error >= 0.0


def test_selection_level_one_trial_matches_bayes_tail_oracle():
    pi_zero, threshold = 0.6, 0.5

    def null_density(value: float) -> float:
        return math.exp(-value) if value >= 0.0 else 0.0

    def mixture_cdf(value: float) -> float:
        if value < 0.0:
            return 0.0
        return pi_zero * (1.0 - math.exp(-value)) + (1.0 - pi_zero) * (
            1.0 - math.exp(-2.0 * value)
        )

    result = max_selection_null_probability(
        threshold,
        pi_zero,
        1,
        null_density=null_density,
        mixture_cdf=mixture_cdf,
    )
    null_tail = math.exp(-threshold)
    alternative_tail = math.exp(-2.0 * threshold)
    expected_selection = pi_zero * null_tail + (1.0 - pi_zero) * alternative_tail
    expected_posterior = pi_zero * null_tail / expected_selection
    assert result.selection_probability == pytest.approx(expected_selection, rel=2e-15)
    assert result.conditional_null_probability == pytest.approx(
        expected_posterior, rel=2e-12
    )


def test_selection_level_zero_null_prior_avoids_unnecessary_quadrature():
    result = max_selection_null_probability(
        0.0,
        0.0,
        3,
        null_density=lambda _: (_ for _ in ()).throw(AssertionError()),
        mixture_cdf=lambda value: 0.5 if value == 0.0 else 1.0,
    )
    assert result.joint_null_and_selection_probability == 0.0
    assert result.conditional_null_probability == 0.0
    assert result.quadrature_absolute_error == 0.0


def test_result_records_are_frozen_slotted_and_arrays_are_read_only():
    family = max_selection_family_errors(0.05, 0.2, 0.7, 3)
    gaussian = gaussian_search_adjusted_false_discovery_rate(
        GaussianTrialMixture(0.7, 1.0, 1.0, 1.0), [0.0, 1.0], 3
    )
    with pytest.raises(FrozenInstanceError):
        family.n_trials = 4
    with pytest.raises(ValueError, match="read-only"):
        gaussian.thresholds[0] = 9.0
    assert not hasattr(family, "__dict__")
    assert not hasattr(gaussian, "__dict__")
    for values in (
        gaussian.thresholds,
        gaussian.trial_type_i_errors,
        gaussian.trial_type_ii_errors,
        gaussian.family_type_i_errors,
        gaussian.family_type_ii_errors,
    ):
        assert values.flags.writeable is False


@pytest.mark.parametrize(
    "invalid", [True, -0.1, 1.1, math.nan, math.inf, -math.inf, 1 + 2j]
)
def test_invalid_probabilities_are_rejected(invalid):
    calls = (
        lambda: single_trial_false_discovery_rate(invalid, 0.2, 0.7),
        lambda: family_level_false_discovery_rate(0.2, invalid, 0.7, 2),
        lambda: maximum_mixture_cdf(0.2, 0.3, invalid, 2),
        lambda: GaussianTrialMixture(invalid, 1.0, 0.0, 1.0),
    )
    for call in calls:
        with pytest.raises(ValueError, match="real scalar|finite|between"):
            call()


@pytest.mark.parametrize("invalid", [True, 0, -1, 2.0, 1.5])
def test_invalid_trial_counts_are_rejected(invalid):
    calls = (
        lambda: family_level_false_discovery_rate(0.2, 0.3, 0.7, invalid),
        lambda: max_selection_family_errors(0.2, 0.3, 0.7, invalid),
        lambda: maximum_mixture_cdf(0.2, 0.3, 0.7, invalid),
        lambda: gaussian_max_selection_cdf(
            GaussianTrialMixture(0.7, 1.0, 0.0, 1.0), 0.0, invalid
        ),
    )
    for call in calls:
        with pytest.raises(ValueError, match="integer|positive"):
            call()


def test_undefined_discovery_posteriors_are_rejected():
    with pytest.raises(ValueError, match="discovery probability"):
        single_trial_false_discovery_rate(0.0, 1.0, 0.5)
    with pytest.raises(ValueError, match="discovery probability"):
        family_level_false_discovery_rate(0.0, 1.0, 0.5, 3)


@pytest.mark.parametrize("pi_zero", [0.0, 1.0])
def test_maximum_family_error_requires_interior_trial_prior(pi_zero):
    with pytest.raises(ValueError, match="strictly"):
        max_selection_family_errors(0.1, 0.2, pi_zero, 2)


def test_comparison_requires_interior_probabilities_and_valid_tolerances():
    with pytest.raises(ValueError, match="strictly"):
        compare_single_and_family_fdr(
            0.0,
            0.2,
            0.7,
            2,
            family_type_i_error=0.1,
            family_type_ii_error=0.2,
        )
    with pytest.raises(ValueError, match="nonnegative"):
        compare_single_and_family_fdr(
            0.1,
            0.2,
            0.7,
            2,
            family_type_i_error=0.1,
            family_type_ii_error=0.2,
            relative_tolerance=-1.0,
        )


def test_conditional_tail_rejects_invalid_order_or_empty_condition():
    with pytest.raises(ValueError, match="below"):
        conditional_upper_tail_probability(0.2, 0.3)
    with pytest.raises(ValueError, match="less than one"):
        conditional_upper_tail_probability(1.0, 1.0)


@pytest.mark.parametrize(
    "arguments",
    [
        (0.5, 0.0, 0.0, 1.0),
        (0.5, -1.0, 0.0, 1.0),
        (0.5, 1.0, 0.0, 0.0),
        (0.5, 1.0, math.nan, 1.0),
    ],
)
def test_gaussian_model_rejects_invalid_parameters(arguments):
    with pytest.raises(ValueError, match="positive|finite"):
        GaussianTrialMixture(*arguments)


@pytest.mark.parametrize(
    "values,match",
    [
        ([], "nonempty"),
        ([[1.0, 2.0]], "one-dimensional"),
        ([1.0, math.nan], "finite"),
        ([True, False], "non-boolean"),
        ([1.0 + 2.0j], "non-boolean"),
    ],
)
def test_gaussian_likelihood_rejects_invalid_observations(values, match):
    model = GaussianTrialMixture(0.5, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match=match):
        gaussian_max_selection_log_likelihood(model, values, 2)


def test_gaussian_calibration_rejects_empty_or_boundary_mixture_prior():
    with pytest.raises(ValueError, match="nonempty"):
        gaussian_search_adjusted_false_discovery_rate(
            GaussianTrialMixture(0.5, 1.0, 1.0, 1.0), [], 2
        )
    with pytest.raises(ValueError, match="strictly"):
        gaussian_search_adjusted_false_discovery_rate(
            GaussianTrialMixture(1.0, 1.0, 1.0, 1.0), [0.0], 2
        )


@pytest.mark.parametrize("observable,target", [(0.0, 0.3), (1.0, 0.3), (0.4, 0.0)])
def test_nonidentification_witness_requires_interior_values(observable, target):
    with pytest.raises(ValueError, match="strictly"):
        fdr_nonidentification_witness(observable, target)


def test_selection_level_validation_fails_closed():
    def valid_density(value: float) -> float:
        return math.exp(-value) if value >= 0.0 else 0.0

    def valid_cdf(value: float) -> float:
        return 1.0 - math.exp(-value) if value >= 0.0 else 0.0

    with pytest.raises(TypeError, match="null_density"):
        max_selection_null_probability(
            0.0, 0.5, 2, null_density=1.0, mixture_cdf=valid_cdf
        )
    with pytest.raises(TypeError, match="mixture_cdf"):
        max_selection_null_probability(
            0.0, 0.5, 2, null_density=valid_density, mixture_cdf=1.0
        )
    with pytest.raises(ValueError, match="selection probability"):
        max_selection_null_probability(
            0.0,
            0.5,
            2,
            null_density=valid_density,
            mixture_cdf=lambda _: 1.0,
        )
    with pytest.raises(ValueError, match="between"):
        max_selection_null_probability(
            0.0,
            0.5,
            2,
            null_density=valid_density,
            mixture_cdf=lambda _: 1.1,
        )
    with pytest.raises(ValueError, match="nonnegative"):
        max_selection_null_probability(
            0.0,
            0.5,
            2,
            null_density=lambda _: -1.0,
            mixture_cdf=valid_cdf,
        )
    with pytest.raises(ValueError, match="positive"):
        max_selection_null_probability(
            0.0,
            0.5,
            2,
            null_density=valid_density,
            mixture_cdf=valid_cdf,
            absolute_tolerance=0.0,
        )


def test_gaussian_distribution_boundaries_are_valid_for_evaluation():
    pure_null = GaussianTrialMixture(1.0, 2.0, 7.0, 3.0)
    pure_alternative = GaussianTrialMixture(0.0, 2.0, 7.0, 3.0)
    for value in (-1.0, 0.0, 2.0):
        assert gaussian_trial_mixture_cdf(pure_null, value) == pytest.approx(
            _normal_cdf(value / 2.0), rel=2e-15
        )
        assert gaussian_trial_mixture_cdf(pure_alternative, value) == pytest.approx(
            _normal_cdf((value - 7.0) / 3.0), rel=2e-15
        )


def test_randomized_gaussian_maximum_cdf_is_monotone_and_bounded():
    rng = np.random.default_rng(3101)
    for _ in range(25):
        model = GaussianTrialMixture(
            float(rng.uniform()),
            float(rng.uniform(0.2, 3.0)),
            float(rng.normal()),
            float(rng.uniform(0.2, 3.0)),
        )
        n_trials = int(rng.integers(1, 20))
        values = np.linspace(-8.0, 8.0, 101)
        cdf_values = np.array(
            [gaussian_max_selection_cdf(model, value, n_trials) for value in values]
        )
        assert np.all(cdf_values >= 0.0)
        assert np.all(cdf_values <= 1.0)
        assert np.all(np.diff(cdf_values) >= -2e-16)


def test_record_types_are_distinct_for_family_and_selection_estimands():
    family = max_selection_family_errors(0.05, 0.2, 0.7, 3)
    selection = max_selection_null_probability(
        0.0,
        0.7,
        3,
        null_density=lambda value: math.exp(-value) if value >= 0.0 else 0.0,
        mixture_cdf=lambda value: 1.0 - math.exp(-value) if value >= 0.0 else 0.0,
    )
    assert isinstance(family, MaxSelectionFamilyErrors)
    assert isinstance(selection, SelectionLevelProbabilityEvidence)
    assert not set(family.__slots__).intersection(
        {"conditional_null_probability", "selection_probability"}
    )


def test_complete_four_state_bayes_table_matches_single_trial_formula():
    alpha, beta, pi_zero = 0.08, 0.25, 0.6
    states = list(itertools.product((False, True), repeat=2))
    assert len(states) == 4
    false_discovery_mass = pi_zero * alpha
    discovery_mass = false_discovery_mass + (1.0 - pi_zero) * (1.0 - beta)
    assert single_trial_false_discovery_rate(alpha, beta, pi_zero) == pytest.approx(
        false_discovery_mass / discovery_mass, rel=2e-15
    )
