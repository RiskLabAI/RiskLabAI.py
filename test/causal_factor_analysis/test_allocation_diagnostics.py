"""Independent tests for generic allocation-misspecification diagnostics."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from RiskLabAI.causal_factor_analysis.allocation_diagnostics import (
    AllocationMisspecificationDiagnostics,
    allocation_misspecification_diagnostics,
)


def _kkt_weights(
    covariance: np.ndarray,
    factor_exposures: np.ndarray,
    target_exposures: np.ndarray,
) -> np.ndarray:
    """Solve the first-order conditions independently as one block system."""
    n_assets, n_factors = factor_exposures.shape
    system = np.block(
        [
            [2.0 * covariance, factor_exposures],
            [factor_exposures.T, np.zeros((n_factors, n_factors))],
        ]
    )
    right_hand_side = np.concatenate(
        [np.zeros(n_assets, dtype=np.float64), target_exposures]
    )
    return np.linalg.solve(system, right_hand_side)[:n_assets]


def _example_inputs():
    covariance = np.diag([1.0, 2.0, 4.0])
    reference_factor_exposures = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    reference_target_exposures = np.array([0.0, 1.0])
    misspecified_factor_exposures = np.array([[1.0], [1.0], [-2.0]])
    misspecified_target_exposures = np.array([1.0])
    return (
        covariance,
        reference_factor_exposures,
        reference_target_exposures,
        misspecified_factor_exposures,
        misspecified_target_exposures,
    )


def test_diagnostics_match_independent_kkt_and_matrix_oracles():
    inputs = _example_inputs()
    (
        covariance,
        reference_exposures,
        reference_target,
        misspecified_exposures,
        misspecified_target,
    ) = inputs
    result = allocation_misspecification_diagnostics(*inputs)

    reference_weights = _kkt_weights(covariance, reference_exposures, reference_target)
    misspecified_weights = _kkt_weights(
        covariance, misspecified_exposures, misspecified_target
    )
    reference_achieved = reference_exposures.T @ reference_weights
    misspecified_reference = reference_exposures.T @ misspecified_weights
    misspecified_model_achieved = misspecified_exposures.T @ misspecified_weights

    assert isinstance(result, AllocationMisspecificationDiagnostics)
    np.testing.assert_allclose(result.reference_weights, reference_weights, atol=2e-15)
    np.testing.assert_allclose(
        result.misspecified_weights, misspecified_weights, atol=2e-15
    )
    np.testing.assert_allclose(
        result.reference_target_exposures, reference_target, atol=0.0
    )
    np.testing.assert_allclose(
        result.reference_achieved_exposures, reference_achieved, atol=2e-15
    )
    np.testing.assert_allclose(
        result.misspecified_reference_exposures,
        misspecified_reference,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        result.misspecified_model_target_exposures,
        misspecified_target,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.misspecified_model_achieved_exposures,
        misspecified_model_achieved,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        result.reference_exposure_error,
        misspecified_reference - reference_target,
        atol=2e-15,
    )
    np.testing.assert_array_equal(
        result.weight_sign_reversals, np.array([True, False, True])
    )
    assert result.reference_variance == pytest.approx(
        reference_weights @ covariance @ reference_weights, abs=2e-15
    )
    assert result.misspecified_variance == pytest.approx(
        misspecified_weights @ covariance @ misspecified_weights, abs=2e-15
    )


def test_each_allocation_satisfies_only_its_declared_constraints():
    result = allocation_misspecification_diagnostics(*_example_inputs())

    np.testing.assert_allclose(
        result.reference_achieved_exposures,
        result.reference_target_exposures,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        result.misspecified_model_achieved_exposures,
        result.misspecified_model_target_exposures,
        atol=2e-15,
    )
    assert not np.allclose(
        result.misspecified_reference_exposures,
        result.reference_target_exposures,
        atol=1e-12,
    )
    np.testing.assert_allclose(result.reference_exposure_error, [0.2, -1.0])


def test_sign_reversal_is_strict_and_zero_is_not_negative():
    covariance = np.eye(2)
    exposures = np.eye(2)

    strict = allocation_misspecification_diagnostics(
        covariance, exposures, [1.0, 0.0], exposures, [-1.0, 0.0]
    )
    zero_boundary = allocation_misspecification_diagnostics(
        covariance, exposures, [0.0, 1.0], exposures, [-1.0, 1.0]
    )

    np.testing.assert_array_equal(strict.weight_sign_reversals, [True, False])
    np.testing.assert_array_equal(zero_boundary.weight_sign_reversals, [False, False])


def test_covariance_unit_change_preserves_weights_and_scales_variances():
    inputs = _example_inputs()
    baseline = allocation_misspecification_diagnostics(*inputs)
    rescaled = allocation_misspecification_diagnostics(7.5 * inputs[0], *inputs[1:])

    np.testing.assert_allclose(rescaled.reference_weights, baseline.reference_weights)
    np.testing.assert_allclose(
        rescaled.misspecified_weights, baseline.misspecified_weights
    )
    assert rescaled.reference_variance == pytest.approx(
        7.5 * baseline.reference_variance, rel=2e-15
    )
    assert rescaled.misspecified_variance == pytest.approx(
        7.5 * baseline.misspecified_variance, rel=2e-15
    )


def test_result_arrays_are_independent_read_only_snapshots():
    inputs = tuple(value.copy() for value in _example_inputs())
    result = allocation_misspecification_diagnostics(*inputs)
    snapshots = {
        name: getattr(result, name).copy()
        for name in (
            "reference_weights",
            "misspecified_weights",
            "reference_target_exposures",
            "reference_achieved_exposures",
            "misspecified_reference_exposures",
            "misspecified_model_target_exposures",
            "misspecified_model_achieved_exposures",
            "reference_exposure_error",
            "weight_sign_reversals",
        )
    }

    for value in inputs:
        value[...] = 99.0

    for name, expected in snapshots.items():
        observed = getattr(result, name)
        np.testing.assert_array_equal(observed, expected)
        assert not observed.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            observed[0] = observed[0]


def test_result_dataclass_is_frozen_and_array_equality_is_not_implicit():
    result = allocation_misspecification_diagnostics(*_example_inputs())
    duplicate = allocation_misspecification_diagnostics(*_example_inputs())

    with pytest.raises(FrozenInstanceError):
        result.reference_variance = 0.0
    assert result is not duplicate
    assert result != duplicate


@pytest.mark.parametrize(
    "replacement_index,replacement",
    [
        (0, np.array([[True, False], [False, True]])),
        (0, np.array([[1.0, 0.0], [0.0, np.nan]])),
        (0, np.array([[1.0, 1.0], [1.0, 1.0]])),
        (1, np.array([[1.0 + 1.0j, 0.0], [0.0, 1.0], [1.0, 1.0]])),
        (1, np.ones((2, 2))),
        (2, np.ones((2, 1))),
        (3, np.ones((3, 4))),
        (4, np.ones((2,))),
    ],
)
def test_invalid_inputs_fail_at_the_public_boundary(replacement_index, replacement):
    inputs = list(_example_inputs())
    inputs[replacement_index] = replacement
    with pytest.raises(ValueError):
        allocation_misspecification_diagnostics(*inputs)


def test_redundant_exposure_constraints_are_rejected():
    covariance, reference, target, misspecified, misspecified_target = _example_inputs()
    redundant_reference = np.column_stack([reference[:, 0], reference[:, 0]])

    with pytest.raises(ValueError, match="full column rank"):
        allocation_misspecification_diagnostics(
            covariance,
            redundant_reference,
            target,
            misspecified,
            misspecified_target,
        )


def test_input_arrays_are_not_mutated():
    inputs = _example_inputs()
    snapshots = tuple(value.copy() for value in inputs)
    allocation_misspecification_diagnostics(*inputs)
    for observed, expected in zip(inputs, snapshots):
        np.testing.assert_array_equal(observed, expected)
