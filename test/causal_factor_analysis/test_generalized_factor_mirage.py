"""Independent tests for the general-variance factor-mirage coefficients."""

from __future__ import annotations

import inspect
from fractions import Fraction

import numpy as np
import pytest

from RiskLabAI.causal_factor_analysis.factor_mirage import (
    ColliderCoefficients,
    collider_overcontrolled_coefficients,
    confounder_undercontrolled_coefficient,
    generalized_collider_overcontrolled_coefficients,
    generalized_confounder_undercontrolled_coefficient,
)


def _fraction(value: float) -> Fraction:
    return Fraction.from_float(float(value))


def _confounder_fraction_oracle(
    beta: float,
    gamma: float,
    delta: float,
    confounder_variance: float,
    exposure_noise_variance: float,
) -> float:
    beta_q = _fraction(beta)
    gamma_q = _fraction(gamma)
    delta_q = _fraction(delta)
    confounder_q = _fraction(confounder_variance)
    noise_q = _fraction(exposure_noise_variance)
    exposure_variance = delta_q**2 * confounder_q + noise_q
    covariance = beta_q * exposure_variance + gamma_q * delta_q * confounder_q
    return float(covariance / exposure_variance)


def _collider_normal_equation_oracle(
    beta: float,
    gamma: float,
    delta: float,
    outcome_noise_variance: float,
    collider_noise_variance: float,
) -> np.ndarray:
    collider_loading = gamma * beta + delta
    regressor_covariance = np.array(
        [
            [1.0, collider_loading],
            [
                collider_loading,
                collider_loading**2
                + gamma**2 * outcome_noise_variance
                + collider_noise_variance,
            ],
        ]
    )
    outcome_covariance = np.array(
        [
            beta,
            collider_loading * beta + gamma * outcome_noise_variance,
        ]
    )
    return np.linalg.solve(regressor_covariance, outcome_covariance)


def test_released_standardized_signatures_are_unchanged():
    confounder = inspect.signature(confounder_undercontrolled_coefficient)
    collider = inspect.signature(collider_overcontrolled_coefficients)

    assert tuple(confounder.parameters) == ("beta", "gamma", "delta")
    assert tuple(collider.parameters) == ("beta", "gamma", "delta")
    assert all(
        parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        for parameter in confounder.parameters.values()
    )
    assert all(
        parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        for parameter in collider.parameters.values()
    )


@pytest.mark.parametrize(
    (
        "beta",
        "gamma",
        "delta",
        "confounder_variance",
        "exposure_noise_variance",
    ),
    [
        (0.75, -1.25, 0.5, 2.0, 3.0),
        (-0.4, 2.5, -0.75, 0.25, 4.0),
        (1.5, 0.0, 8.0, 3.0, 0.5),
        (0.0, -3.0, 0.0, 2.0, 7.0),
    ],
)
def test_generalized_confounder_matches_covariance_ratio(
    beta,
    gamma,
    delta,
    confounder_variance,
    exposure_noise_variance,
):
    expected = _confounder_fraction_oracle(
        beta,
        gamma,
        delta,
        confounder_variance,
        exposure_noise_variance,
    )
    actual = generalized_confounder_undercontrolled_coefficient(
        beta,
        gamma,
        delta,
        confounder_variance=confounder_variance,
        exposure_noise_variance=exposure_noise_variance,
    )
    assert actual == expected


@pytest.mark.parametrize(
    "beta,gamma,delta",
    [(1.0, 3.0, 1.0), (-0.7, 1.2, -0.4), (2.5, 0.0, -8.0)],
)
def test_generalized_confounder_unit_variances_equal_released_result(
    beta, gamma, delta
):
    assert generalized_confounder_undercontrolled_coefficient(
        beta,
        gamma,
        delta,
        confounder_variance=1.0,
        exposure_noise_variance=1.0,
    ) == confounder_undercontrolled_coefficient(beta, gamma, delta)


def test_generalized_confounder_is_invariant_to_common_variance_units():
    baseline = generalized_confounder_undercontrolled_coefficient(
        0.8,
        -1.4,
        0.6,
        confounder_variance=2.5,
        exposure_noise_variance=0.75,
    )
    rescaled = generalized_confounder_undercontrolled_coefficient(
        0.8,
        -1.4,
        0.6,
        confounder_variance=25.0,
        exposure_noise_variance=7.5,
    )
    assert rescaled == baseline


@pytest.mark.parametrize(
    (
        "beta",
        "gamma",
        "delta",
        "outcome_noise_variance",
        "collider_noise_variance",
    ),
    [
        (0.8, 1.2, -0.4, 2.0, 3.0),
        (-0.5, 0.75, 2.0, 0.25, 4.0),
        (2.0, -1.0, -0.25, 3.5, 0.5),
        (1.25, 0.0, 7.0, 5.0, 2.0),
    ],
)
def test_generalized_collider_matches_independent_normal_equations(
    beta,
    gamma,
    delta,
    outcome_noise_variance,
    collider_noise_variance,
):
    expected = _collider_normal_equation_oracle(
        beta,
        gamma,
        delta,
        outcome_noise_variance,
        collider_noise_variance,
    )
    actual = generalized_collider_overcontrolled_coefficients(
        beta,
        gamma,
        delta,
        outcome_noise_variance=outcome_noise_variance,
        collider_noise_variance=collider_noise_variance,
    )
    assert actual.beta_hat == pytest.approx(expected[0], rel=2e-14, abs=2e-14)
    assert actual.theta_hat == pytest.approx(expected[1], rel=2e-14, abs=2e-14)


@pytest.mark.parametrize(
    "beta,gamma,delta",
    [(0.5, 1.0, 0.8), (1.0, -2.0, 0.25), (-0.7, 0.4, -1.2)],
)
def test_generalized_collider_unit_variances_equal_released_result(beta, gamma, delta):
    assert generalized_collider_overcontrolled_coefficients(
        beta,
        gamma,
        delta,
        outcome_noise_variance=1.0,
        collider_noise_variance=1.0,
    ) == collider_overcontrolled_coefficients(beta, gamma, delta)


def test_generalized_collider_is_invariant_to_common_variance_units():
    baseline = generalized_collider_overcontrolled_coefficients(
        0.8,
        1.2,
        -0.4,
        outcome_noise_variance=2.0,
        collider_noise_variance=3.0,
    )
    rescaled = generalized_collider_overcontrolled_coefficients(
        0.8,
        1.2,
        -0.4,
        outcome_noise_variance=20.0,
        collider_noise_variance=30.0,
    )
    assert rescaled == baseline


def test_zero_collider_link_has_the_expected_general_variance_limit():
    result = generalized_collider_overcontrolled_coefficients(
        0.7,
        0.0,
        4.0,
        outcome_noise_variance=8.0,
        collider_noise_variance=0.5,
    )
    assert result == ColliderCoefficients(beta_hat=0.7, theta_hat=0.0)


@pytest.mark.parametrize(
    "function,keywords",
    [
        (
            generalized_confounder_undercontrolled_coefficient,
            {"confounder_variance": 0.0, "exposure_noise_variance": 1.0},
        ),
        (
            generalized_confounder_undercontrolled_coefficient,
            {"confounder_variance": 1.0, "exposure_noise_variance": -1.0},
        ),
        (
            generalized_collider_overcontrolled_coefficients,
            {"outcome_noise_variance": 0.0, "collider_noise_variance": 1.0},
        ),
        (
            generalized_collider_overcontrolled_coefficients,
            {"outcome_noise_variance": 1.0, "collider_noise_variance": -1.0},
        ),
    ],
)
def test_generalized_coefficients_reject_nonpositive_variances(function, keywords):
    with pytest.raises(ValueError, match="strictly positive"):
        function(0.5, 1.0, -0.25, **keywords)


@pytest.mark.parametrize("invalid", [True, np.nan, np.inf, 1.0 + 2.0j, "1"])
def test_generalized_coefficients_reject_invalid_scalars(invalid):
    with pytest.raises(ValueError):
        generalized_confounder_undercontrolled_coefficient(
            invalid,
            1.0,
            0.5,
            confounder_variance=1.0,
            exposure_noise_variance=1.0,
        )
    with pytest.raises(ValueError):
        generalized_collider_overcontrolled_coefficients(
            0.5,
            1.0,
            0.5,
            outcome_noise_variance=invalid,
            collider_noise_variance=1.0,
        )


def test_generalized_variances_are_required_keyword_only_arguments():
    with pytest.raises(TypeError):
        generalized_confounder_undercontrolled_coefficient(1.0, 2.0, 3.0)
    with pytest.raises(TypeError):
        generalized_collider_overcontrolled_coefficients(1.0, 2.0, 3.0)
    with pytest.raises(TypeError):
        generalized_confounder_undercontrolled_coefficient(1.0, 2.0, 3.0, 1.0, 1.0)
