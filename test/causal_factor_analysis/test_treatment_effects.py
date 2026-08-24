"""Exact tests for the 2023 book's treatment-effect estimands."""

from __future__ import annotations

import math

import numpy as np
import pytest

from RiskLabAI.causal_factor_analysis import (
    DifferenceInDifferencesEstimate,
    TreatmentEffectDecomposition,
    average_treatment_effect,
    backdoor_adjusted_average_treatment_effect,
    backdoor_adjusted_expectation,
    difference_in_differences,
    frontdoor_adjusted_probability,
    linear_instrumental_variable_effect,
    randomized_mean_difference,
    treatment_effect_decomposition,
)


def test_equation_2_average_treatment_effect_is_exact_mean_contrast():
    assert average_treatment_effect(7.25, 2.0) == 5.25
    assert average_treatment_effect(-2, -5) == 3.0


def test_equation_3_decomposition_closes_exactly():
    result = treatment_effect_decomposition(
        observed_treated_mean=10.0,
        observed_control_mean=4.0,
        counterfactual_control_mean_for_treated=6.0,
    )
    assert result == TreatmentEffectDecomposition(
        observed_difference=6.0,
        average_treatment_effect_on_treated=4.0,
        sample_selection_bias=2.0,
    )
    assert result.observed_difference == (
        result.average_treatment_effect_on_treated + result.sample_selection_bias
    )


def test_equations_4_to_6_randomized_difference_uses_both_groups():
    treated = np.array([4.0, 6.0, 8.0])
    control = np.array([1.0, 3.0, 5.0])
    treated_before = treated.copy()
    control_before = control.copy()
    assert randomized_mean_difference(treated, control) == 3.0
    np.testing.assert_array_equal(treated, treated_before)
    np.testing.assert_array_equal(control, control_before)


def test_two_group_two_period_difference_in_differences_is_exact():
    result = difference_in_differences(
        treated_before=10.0,
        treated_after=15.0,
        control_before=20.0,
        control_after=22.0,
    )
    assert result == DifferenceInDifferencesEstimate(
        treated_change=5.0,
        control_change=2.0,
        estimate=3.0,
    )


def test_equation_8_backdoor_standardization_matches_hand_sum():
    result = backdoor_adjusted_expectation(
        conditional_outcome_means=(0.1, 0.8),
        adjustment_probabilities=(0.75, 0.25),
    )
    assert result == pytest.approx(0.1 * 0.75 + 0.8 * 0.25, abs=1.0e-15)


def test_equation_7_backdoor_ate_matches_stratum_contrasts():
    result = backdoor_adjusted_average_treatment_effect(
        treated_conditional_means=(0.4, 0.9),
        control_conditional_means=(0.1, 0.2),
        adjustment_probabilities=(0.75, 0.25),
    )
    assert result == pytest.approx(
        (0.4 - 0.1) * 0.75 + (0.9 - 0.2) * 0.25,
        abs=1.0e-15,
    )


def test_equation_9_frontdoor_probability_matches_nested_hand_sum():
    result = frontdoor_adjusted_probability(
        mediator_probabilities_given_treatment=(0.25, 0.75),
        outcome_probabilities_given_mediator_and_treatment=(
            (0.1, 0.5),
            (0.4, 0.8),
        ),
        treatment_probabilities=(0.6, 0.4),
    )
    expected = 0.25 * (0.1 * 0.6 + 0.5 * 0.4) + 0.75 * (0.4 * 0.6 + 0.8 * 0.4)
    assert result == pytest.approx(expected, abs=1.0e-15)


def test_linear_instrumental_variable_ratio_is_exact():
    assert linear_instrumental_variable_effect(0.9, 0.3) == pytest.approx(3.0)
    assert linear_instrumental_variable_effect(-2.0, 0.5) == -4.0


@pytest.mark.parametrize(
    "function,args",
    [
        (average_treatment_effect, (True, 0.0)),
        (average_treatment_effect, (math.inf, 0.0)),
        (treatment_effect_decomposition, (1.0, 0.0, math.nan)),
        (randomized_mean_difference, ((), (1.0,))),
        (randomized_mean_difference, (((1.0, 2.0),), (1.0,))),
        (randomized_mean_difference, (("1",), (1.0,))),
        (difference_in_differences, (0.0, 1.0, 0.0, math.inf)),
        (backdoor_adjusted_expectation, ((1.0,), (0.4, 0.6))),
        (backdoor_adjusted_expectation, ((1.0, 2.0), (0.4, 0.4))),
        (
            backdoor_adjusted_average_treatment_effect,
            ((1.0,), (0.0, 1.0), (1.0,)),
        ),
        (
            frontdoor_adjusted_probability,
            ((1.0,), ((0.2, 0.3),), (1.0,)),
        ),
        (
            frontdoor_adjusted_probability,
            ((1.0,), ((1.2,),), (1.0,)),
        ),
        (linear_instrumental_variable_effect, (1.0, 0.0)),
    ],
)
def test_estimands_reject_malformed_or_undefined_inputs(function, args):
    with pytest.raises(ValueError):
        function(*args)


def test_probability_tolerance_accepts_only_roundoff_scale_error():
    assert backdoor_adjusted_expectation(
        (2.0, 4.0),
        (0.5, 0.5 + 5.0e-13),
    ) == pytest.approx(3.0 + 2.0e-12, abs=1.0e-14)
    with pytest.raises(ValueError, match="sum to one"):
        backdoor_adjusted_expectation((2.0, 4.0), (0.5, 0.50001))


@pytest.mark.parametrize(
    "function,args",
    [
        (average_treatment_effect, (np.finfo(float).max, -np.finfo(float).max)),
        (
            treatment_effect_decomposition,
            (np.finfo(float).max, -np.finfo(float).max, 0.0),
        ),
        (
            difference_in_differences,
            (-np.finfo(float).max, np.finfo(float).max, 0.0, 0.0),
        ),
        (
            backdoor_adjusted_average_treatment_effect,
            ((np.finfo(float).max,), (-np.finfo(float).max,), (1.0,)),
        ),
    ],
)
def test_finite_inputs_that_overflow_the_estimand_fail_closed(function, args):
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError):
            function(*args)


def test_public_result_records_are_tuple_immutable_without_instance_dicts():
    treatment = treatment_effect_decomposition(3.0, 1.0, 2.0)
    did = difference_in_differences(1.0, 3.0, 4.0, 5.0)
    for result in (treatment, did):
        assert not hasattr(result, "__dict__")
        with pytest.raises(AttributeError):
            result.estimate = 0.0
