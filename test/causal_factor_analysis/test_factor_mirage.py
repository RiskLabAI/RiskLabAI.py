"""Tests for the published factor-mirage analytical formulas."""

from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import FrozenInstanceError
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from RiskLabAI.causal_factor_analysis import (
    ColliderCoefficients,
    ColliderDiagnostics,
    StrategyPerformance,
    collider_factor_return,
    collider_forecast_return,
    collider_model_diagnostics,
    collider_overcontrolled_coefficients,
    confounder_factor_return,
    confounder_forecast_return,
    confounder_undercontrolled_coefficient,
)


@pytest.mark.parametrize(
    "beta,gamma,delta",
    [
        (1.0, 3.0, 1.0),
        (-0.7, 1.2, -0.4),
        (0.0, -2.0, 3.0),
        (2.5, 0.0, -8.0),
    ],
)
def test_confounder_coefficient_matches_equation_2(beta, gamma, delta):
    expected = beta + gamma * delta / (1.0 + delta**2)
    actual = confounder_undercontrolled_coefficient(beta, gamma, delta)
    assert actual == pytest.approx(expected, rel=1e-14, abs=1e-14)
    assert actual - beta == pytest.approx(
        gamma * delta / (1.0 + delta**2), rel=1e-14, abs=1e-14
    )


def test_confounder_zero_links_remove_omitted_variable_bias():
    assert confounder_undercontrolled_coefficient(2.0, 0.0, 4.0) == 2.0
    assert confounder_undercontrolled_coefficient(2.0, 4.0, 0.0) == 2.0


def test_mirage_confounder_example_matches_equations_5_to_9():
    coefficient = confounder_undercontrolled_coefficient(1.0, 3.0, 1.0)
    factor = confounder_factor_return(1.0, 1.0, 1.0, 3.0, 1.0)
    forecast = confounder_forecast_return(1.0, 3.0, 1.0)

    assert coefficient == 2.5
    assert factor == StrategyPerformance(correct=16.0, misspecified=6.25)
    assert forecast.correct == pytest.approx(17.0, abs=1e-14)
    assert forecast.misspecified == pytest.approx(12.5, abs=1e-14)


@pytest.mark.parametrize(
    "beta,gamma,delta",
    [(1.0, 3.0, 1.0), (-0.5, 0.75, 2.0), (2.0, -1.0, -0.25)],
)
def test_static_confounder_forecast_penalty_identity(beta, gamma, delta):
    result = confounder_forecast_return(beta, gamma, delta)
    expected_gap = gamma**2 / (1.0 + delta**2)
    assert result.correct - result.misspecified == pytest.approx(
        expected_gap, rel=1e-12, abs=1e-14
    )
    assert result.correct >= 0.0
    assert result.misspecified >= 0.0


def test_confounder_shift_can_create_systematic_losses():
    factor = confounder_factor_return(
        1.0,
        1.0,
        1.0,
        3.0,
        1.0,
        delta_realized=-1.0,
    )
    forecast = confounder_forecast_return(1.0, 3.0, 1.0, delta_realized=-1.0)

    assert factor == StrategyPerformance(correct=16.0, misspecified=-1.25)
    assert forecast.correct == pytest.approx(5.0, abs=1e-14)
    assert forecast.misspecified == pytest.approx(-2.5, abs=1e-14)


def test_confounder_shift_loss_has_exact_zero_boundaries():
    # beta=1, gamma=3 has roots (-3 +/- sqrt(5))/2.
    root = (-3.0 + math.sqrt(5.0)) / 2.0
    factor = confounder_factor_return(2.0, 0.5, 1.0, 3.0, root, delta_realized=1.0)
    forecast = confounder_forecast_return(1.0, 3.0, root, delta_realized=1.0)
    assert factor.misspecified == pytest.approx(0.0, abs=2e-15)
    assert forecast.misspecified == pytest.approx(0.0, abs=2e-15)
    assert (
        confounder_factor_return(
            0.0, 0.5, 1.0, 3.0, 1.0, delta_realized=-1.0
        ).misspecified
        == 0.0
    )


def test_omitted_realized_delta_is_the_constant_parameter_case():
    static_factor = confounder_factor_return(0.7, -0.2, 0.8, 1.1, -0.6)
    shifted_factor = confounder_factor_return(
        0.7, -0.2, 0.8, 1.1, -0.6, delta_realized=-0.6
    )
    static_forecast = confounder_forecast_return(0.8, 1.1, -0.6)
    shifted_forecast = confounder_forecast_return(0.8, 1.1, -0.6, delta_realized=-0.6)
    assert shifted_factor == static_factor
    assert shifted_forecast == static_forecast


@pytest.mark.parametrize(
    "beta,gamma,delta",
    [(0.5, 1.0, 0.8), (1.0, -2.0, 0.25), (-0.7, 0.4, -1.2)],
)
def test_collider_coefficients_match_equations_11_and_12(beta, gamma, delta):
    result = collider_overcontrolled_coefficients(beta, gamma, delta)
    assert result.beta_hat == pytest.approx(
        (beta - delta * gamma) / (1.0 + gamma**2), rel=1e-14, abs=1e-14
    )
    assert result.theta_hat == pytest.approx(
        gamma / (1.0 + gamma**2), rel=1e-14, abs=1e-14
    )


def test_collider_unbiased_condition_has_the_correct_sign():
    beta, gamma = 0.7, 1.5
    unbiased = collider_overcontrolled_coefficients(beta, gamma, -beta * gamma)
    delta_zero = collider_overcontrolled_coefficients(beta, gamma, 0.0)
    gamma_zero = collider_overcontrolled_coefficients(beta, 0.0, 4.0)

    assert unbiased.beta_hat == pytest.approx(beta, rel=1e-14, abs=1e-14)
    assert unbiased.theta_hat != 0.0
    assert delta_zero.beta_hat != pytest.approx(beta)
    assert gamma_zero == ColliderCoefficients(beta_hat=beta, theta_hat=0.0)


def test_conditioning_back_over_x_recovers_the_causal_expectation():
    beta, gamma, delta, x = 0.8, 1.2, -0.4, 2.5
    coefficients = collider_overcontrolled_coefficients(beta, gamma, delta)
    expected_proxy = (beta * gamma + delta) * x
    conditional_prediction = (
        coefficients.beta_hat * x + coefficients.theta_hat * expected_proxy
    )
    assert conditional_prediction == pytest.approx(beta * x, rel=1e-14, abs=1e-14)


def test_mirage_collider_factor_example_is_negative():
    result = collider_factor_return(1.0, 1.0, 1.0, 1.0, 3.0)
    assert result == StrategyPerformance(correct=1.0, misspecified=-0.5)


def test_mirage_collider_forecast_example_is_negative():
    result = collider_forecast_return(1.0, 1.0, 2.0)
    assert result == StrategyPerformance(correct=1.0, misspecified=-0.5)


def test_collider_loss_boundaries_are_exact():
    assert collider_forecast_return(0.0, 2.0, 3.0).misspecified == 0.0
    assert collider_forecast_return(2.0, 1.0, 2.0).misspecified == 0.0
    assert collider_factor_return(0.0, 4.0, 1.0, 2.0, 3.0).misspecified == 0.0
    assert collider_factor_return(1.0, 2.0, 0.0, 2.0, 3.0).misspecified == 0.0


def test_zero_collider_link_makes_both_strategies_equal():
    factor = collider_factor_return(1.25, -8.0, 0.6, 0.0, 3.0)
    forecast = collider_forecast_return(0.6, 0.0, 3.0)
    assert factor.misspecified == pytest.approx(factor.correct, abs=1e-15)
    assert forecast.misspecified == pytest.approx(forecast.correct, abs=1e-15)


def test_collider_diagnostics_match_appendix_d():
    result = collider_model_diagnostics(1.0, 1.0, 2.0, 5)
    assert result.residual_variance == 0.5
    assert result.outcome_variance == pytest.approx(2.0, abs=1e-14)
    assert result.correct_r_squared == 0.5
    assert result.overcontrolled_r_squared == 0.75
    assert result.correct_adjusted_r_squared == pytest.approx(1.0 / 3.0)
    assert result.overcontrolled_adjusted_r_squared == 0.5
    assert result.correct_beta_variance == 0.2
    assert result.overcontrolled_beta_variance == pytest.approx(11.0 / 20.0)
    assert result.collider_coefficient_variance == 0.05
    assert result.correct_beta_t_statistic == pytest.approx(math.sqrt(5.0))
    assert result.overcontrolled_beta_t_statistic == pytest.approx(
        -math.sqrt(5.0 / 11.0)
    )
    assert result.collider_t_statistic == pytest.approx(math.sqrt(5.0))
    assert result.adjusted_r_squared_prefers_overcontrolled is True
    assert result.absolute_beta_t_prefers_overcontrolled is False


def test_ordinary_r_squared_is_strict_only_for_nonzero_gamma():
    equal = collider_model_diagnostics(0.8, 0.0, 10.0, 50)
    larger = collider_model_diagnostics(0.8, -0.2, 10.0, 50)
    assert equal.overcontrolled_r_squared == equal.correct_r_squared
    assert larger.overcontrolled_r_squared > larger.correct_r_squared


def test_small_nonzero_links_do_not_cancel_out_of_r_squared():
    beta_only = collider_model_diagnostics(1e-8, 0.0, 0.0, 20)
    gamma_only = collider_model_diagnostics(0.0, 1e-8, 0.0, 20)
    assert beta_only.correct_r_squared == pytest.approx(1e-16, rel=1e-15)
    assert gamma_only.overcontrolled_r_squared == pytest.approx(1e-16, rel=1e-15)


def test_adjusted_r_squared_threshold_is_strict_and_sign_invariant():
    sample_size = 7
    threshold = 1.0 / math.sqrt(sample_size - 3)
    at_boundary = collider_model_diagnostics(1.0, threshold, 0.0, sample_size)
    below = collider_model_diagnostics(1.0, 0.49, 0.0, sample_size)
    above = collider_model_diagnostics(1.0, -0.51, 0.0, sample_size)

    assert at_boundary.correct_adjusted_r_squared == pytest.approx(
        at_boundary.overcontrolled_adjusted_r_squared, abs=1e-15
    )
    assert at_boundary.adjusted_r_squared_prefers_overcontrolled is False
    assert below.adjusted_r_squared_prefers_overcontrolled is False
    assert above.adjusted_r_squared_prefers_overcontrolled is True


@pytest.mark.parametrize(
    "beta,gamma,delta,n_observations",
    [(0.3, 0.8, -0.2, 20), (-1.0, 1.5, 0.4, 100), (0.0, -2.0, 3.0, 8)],
)
def test_expected_t_statistics_match_appendix_d(beta, gamma, delta, n_observations):
    result = collider_model_diagnostics(beta, gamma, delta, n_observations)
    denominator = math.sqrt((beta * gamma + delta) ** 2 + gamma**2 + 1.0)
    expected_overcontrolled = (
        math.sqrt(n_observations) * (beta - delta * gamma) / denominator
    )
    assert result.correct_beta_t_statistic == pytest.approx(
        math.sqrt(n_observations) * beta, rel=1e-13, abs=1e-14
    )
    assert result.overcontrolled_beta_t_statistic == pytest.approx(
        expected_overcontrolled, rel=1e-13, abs=1e-14
    )
    assert result.collider_t_statistic == pytest.approx(
        math.sqrt(n_observations) * gamma, rel=1e-13, abs=1e-14
    )
    assert result.absolute_beta_t_prefers_overcontrolled is (
        abs(beta - delta * gamma) / denominator > abs(beta)
    )


@pytest.mark.parametrize("sample_size", [True, 4.0, 3, 0, -4])
def test_invalid_sample_sizes_are_rejected(sample_size):
    match = "integer" if isinstance(sample_size, (bool, float)) else "greater than 3"
    with pytest.raises(ValueError, match=match):
        collider_model_diagnostics(1.0, 1.0, 1.0, sample_size)


def test_numpy_integer_sample_size_is_accepted():
    result = collider_model_diagnostics(1.0, 1.0, 1.0, np.int64(5))
    assert isinstance(result, ColliderDiagnostics)


@pytest.mark.parametrize(
    "invalid",
    [True, 1.0 + 2.0j, np.nan, np.inf, -np.inf, [1.0], np.array([1.0])],
)
def test_invalid_real_scalars_are_rejected(invalid):
    calls = (
        lambda: confounder_undercontrolled_coefficient(invalid, 1.0, 1.0),
        lambda: confounder_factor_return(invalid, 1.0, 1.0, 1.0, 1.0),
        lambda: confounder_forecast_return(invalid, 1.0, 1.0),
        lambda: collider_overcontrolled_coefficients(invalid, 1.0, 1.0),
        lambda: collider_factor_return(invalid, 1.0, 1.0, 1.0, 1.0),
        lambda: collider_forecast_return(invalid, 1.0, 1.0),
        lambda: collider_model_diagnostics(invalid, 1.0, 1.0, 10),
    )
    for call in calls:
        with pytest.raises(ValueError, match="real scalar|finite"):
            call()


def test_extreme_finite_scaling_avoids_spurious_overflow():
    coefficient = confounder_undercontrolled_coefficient(0.0, 1e308, 1e308)
    factor = confounder_factor_return(1e200, 0.0, 1e-200, 0.0, 1.0)
    collider = collider_overcontrolled_coefficients(1.0, 1e308, 1.0)

    assert coefficient == pytest.approx(1.0, rel=1e-15)
    assert factor.correct == pytest.approx(1.0, rel=1e-15)
    assert factor.misspecified == pytest.approx(1.0, rel=1e-15)
    assert collider.beta_hat == pytest.approx(-1e-308, rel=1e-15, abs=0.0)
    assert collider.theta_hat == pytest.approx(1e-308, rel=1e-15, abs=0.0)


def test_extreme_collider_scaling_preserves_representable_results():
    coefficients = collider_overcontrolled_coefficients(1e308, 1e163, 0.0)
    diagnostics = collider_model_diagnostics(1e154, 1e200, 0.0, 4)
    assert coefficients.beta_hat == pytest.approx(1e-18, rel=1e-14, abs=0.0)
    assert diagnostics.overcontrolled_beta_variance == pytest.approx(
        2.5e-93, rel=1e-14, abs=0.0
    )


def test_confounder_composites_preserve_latent_scaled_coefficients():
    maximum = sys.float_info.max
    finite_large = confounder_factor_return(1e-308, 0.0, maximum, maximum, 1.0)
    latent_factor = confounder_factor_return(1e308, 0.0, 0.0, 1e-200, 1e200)
    latent_forecast = confounder_forecast_return(0.0, 1e-200, 1.0, delta_realized=1e308)

    assert finite_large.correct == pytest.approx(3.2317006071310996, rel=1e-15)
    assert finite_large.misspecified == pytest.approx(7.271326366044974, rel=1e-15)
    assert latent_factor.misspecified == pytest.approx(1e-184, rel=1e-14)
    assert latent_forecast.misspecified == pytest.approx(5e-93, rel=1e-14)


def test_collider_composites_preserve_latent_scaled_coefficients():
    latent_factor = collider_factor_return(1e208, 0.0, 1e-54, 1e200, 0.0)
    latent_forecast = collider_forecast_return(1e75, 1e200, 0.0)
    tiny_prefactor = collider_factor_return(1e-308, 1e308, 1e-308, 1.0, 0.0)

    assert latent_factor.correct == pytest.approx(1e308, rel=1e-14)
    assert latent_factor.misspecified == pytest.approx(1e-92, rel=1e-14)
    assert latent_forecast.correct == pytest.approx(1e150, rel=1e-14)
    assert latent_forecast.misspecified == pytest.approx(1e-250, rel=1e-14)
    assert tiny_prefactor.misspecified == pytest.approx(5e-309, rel=1e-14)


def test_expected_t_statistic_preserves_latent_scaled_ratio():
    diagnostics = collider_model_diagnostics(0.0, 1e-200, 1e-200, 10**200)
    assert diagnostics.overcontrolled_beta_t_statistic == pytest.approx(
        -1e-300, rel=1e-14, abs=0.0
    )
    assert diagnostics.absolute_beta_t_prefers_overcontrolled is True


def test_loss_sign_is_preserved_at_ill_conditioned_boundaries():
    confounder = confounder_undercontrolled_coefficient(1.0, 1e10, -1e-10)
    collider = collider_forecast_return(1.0, -1e-10, -1e10)
    assert confounder == pytest.approx(-3.642219731549774e-17, rel=1e-14)
    assert collider.misspecified == pytest.approx(-3.643219731549774e-17, rel=1e-14)
    assert confounder < 0.0
    assert collider.misspecified < 0.0


def test_adjusted_r_squared_boolean_uses_the_exact_strict_inequality():
    gamma = 1.0 / math.sqrt(3.0)
    diagnostics = collider_model_diagnostics(0.0, gamma, 0.0, 6)
    expected = Fraction.from_float(gamma) ** 2 * 3 > 1
    assert diagnostics.adjusted_r_squared_prefers_overcontrolled is expected


def test_unrepresentable_result_raises_stable_value_error():
    with pytest.raises(ValueError, match="cannot be represented"):
        collider_forecast_return(1e308, 0.0, 0.0)


def test_result_records_are_frozen():
    performance = collider_forecast_return(1.0, 1.0, 2.0)
    coefficients = collider_overcontrolled_coefficients(1.0, 1.0, 2.0)
    diagnostics = collider_model_diagnostics(1.0, 1.0, 2.0, 5)
    with pytest.raises(FrozenInstanceError):
        performance.correct = 0.0
    with pytest.raises(FrozenInstanceError):
        coefficients.beta_hat = 0.0
    with pytest.raises(FrozenInstanceError):
        diagnostics.correct_r_squared = 0.0


def test_public_import_is_isolated_and_source_conflicted_api_is_absent():
    repository_root = Path(__file__).resolve().parents[2]
    program = """
import sys
import RiskLabAI

module = RiskLabAI.causal_factor_analysis
required = {
    'confounder_undercontrolled_coefficient',
    'confounder_factor_return',
    'confounder_forecast_return',
    'collider_overcontrolled_coefficients',
    'collider_factor_return',
    'collider_forecast_return',
    'collider_model_diagnostics',
}
missing = required.difference(module.__all__)
if missing:
    raise SystemExit(f'missing public names: {sorted(missing)}')
for forbidden in ('collider_shift_return', 'collider_p_value'):
    if hasattr(module, forbidden):
        raise SystemExit(f'source-conflicted or unsupported API exposed: {forbidden}')

blocked = {
    'RiskLabAI.causal',
    'RiskLabAI.optimization',
    'RiskLabAI.backtest',
    'RiskLabAI.risk_control',
}
loaded = blocked.intersection(sys.modules)
if loaded:
    raise SystemExit(f'unexpected imports: {sorted(loaded)}')
"""
    subprocess.run(
        [sys.executable, "-c", program],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
