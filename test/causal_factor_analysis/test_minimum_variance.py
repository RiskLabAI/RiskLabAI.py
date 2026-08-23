"""Source-locked tests for target-exposure minimum-variance allocation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from RiskLabAI.causal_factor_analysis import minimum_variance_factor_weights


def test_eq4_solution_satisfies_exposure_and_kkt_conditions():
    covariance = np.diag([1.0, 2.0, 4.0])
    exposures = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    targets = np.array([0.0, 1.0])

    weights = minimum_variance_factor_weights(covariance, exposures, targets)

    expected_weights = np.array([-2.0, 5.0, 2.0]) / 7.0
    expected_lagrange = np.array([4.0, -20.0]) / 7.0
    np.testing.assert_allclose(weights, expected_weights, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(exposures.T @ weights, targets, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(
        2.0 * covariance @ weights + exposures @ expected_lagrange,
        np.zeros(3),
        rtol=0.0,
        atol=1e-14,
    )
    assert weights @ covariance @ weights == pytest.approx(
        10.0 / 7.0, rel=1e-14, abs=1e-14
    )


def test_eq4_solution_is_strict_minimum_over_feasible_perturbations():
    covariance = np.diag([1.0, 2.0, 4.0])
    exposures = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    targets = np.array([0.0, 1.0])
    weights = minimum_variance_factor_weights(covariance, exposures, targets)
    null_direction = np.array([-1.0, -1.0, 1.0])
    np.testing.assert_allclose(
        exposures.T @ null_direction, np.zeros(2), rtol=0.0, atol=0.0
    )
    optimum = weights @ covariance @ weights

    for scale in (-2.0, -0.5, 0.5, 2.0):
        candidate = weights + scale * null_direction
        np.testing.assert_allclose(
            exposures.T @ candidate, targets, rtol=0.0, atol=1e-14
        )
        assert candidate @ covariance @ candidate > optimum


def test_eq5_square_solution_is_covariance_independent():
    exposures = np.array([[1.0, 1.0], [1.5, 1.0]])
    targets = np.array([0.0, 1.0])
    expected = np.linalg.solve(exposures.T, targets)
    covariance_matrices = (
        np.array([[1.0, 0.2], [0.2, 2.0]]),
        np.array([[5.0, -1.0], [-1.0, 0.5]]),
    )

    np.testing.assert_allclose(expected, np.array([3.0, -2.0]), rtol=0.0, atol=1e-14)
    for covariance in covariance_matrices:
        weights = minimum_variance_factor_weights(covariance, exposures, targets)
        np.testing.assert_allclose(weights, expected, rtol=0.0, atol=1e-13)


def test_eq6_single_factor_solution():
    covariance = np.array([[2.0, 0.5], [0.5, 1.0]])
    exposures = np.array([[1.0], [1.0]])
    targets = np.array([1.0])

    weights = minimum_variance_factor_weights(covariance, exposures, targets)

    np.testing.assert_allclose(weights, np.array([0.25, 0.75]), rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(exposures.T @ weights, targets, rtol=0.0, atol=1e-14)
    assert weights @ covariance @ weights == pytest.approx(
        7.0 / 8.0, rel=1e-14, abs=1e-14
    )


def test_one_asset_one_factor_solution():
    weights = minimum_variance_factor_weights(
        np.array([[2.5]]), np.array([[-4.0]]), np.array([3.0])
    )
    np.testing.assert_allclose(weights, np.array([-0.75]), rtol=0.0, atol=0.0)


def test_near_collinear_square_constraints_remain_feasible():
    covariance = np.eye(2)
    exposures = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-9]])
    targets = np.array([1.0, 0.0])

    weights = minimum_variance_factor_weights(covariance, exposures, targets)
    direct_solution = np.linalg.solve(exposures.T, targets)

    np.testing.assert_allclose(weights, direct_solution, rtol=5e-8, atol=0.0)
    np.testing.assert_allclose(exposures.T @ weights, targets, rtol=0.0, atol=1e-12)


def test_extreme_finite_exposure_scale_does_not_overflow():
    covariance = np.array([[1.0]])
    exposures = np.array([[1e308]])
    targets = np.array([1.0])

    weights = minimum_variance_factor_weights(covariance, exposures, targets)

    np.testing.assert_allclose(weights, np.array([1e-308]), rtol=1e-15, atol=0.0)
    np.testing.assert_allclose(exposures.T @ weights, targets, rtol=1e-15, atol=0.0)


def test_extreme_factor_unit_differences_do_not_create_false_rank_deficiency():
    covariance = np.eye(2)
    exposures = np.diag([1e-200, 1e200])
    targets = np.array([1e-200, 1e200])

    weights = minimum_variance_factor_weights(covariance, exposures, targets)

    np.testing.assert_allclose(weights, np.ones(2), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(exposures.T @ weights, targets, rtol=1e-15, atol=0.0)


def test_feasible_target_normalization_does_not_overflow():
    covariance = np.eye(2)
    exposures = np.array([[1e-308], [1e-308]])
    targets = np.array([2.0])

    weights = minimum_variance_factor_weights(covariance, exposures, targets)

    np.testing.assert_allclose(weights, np.array([1e308, 1e308]), rtol=1e-15, atol=0.0)
    np.testing.assert_allclose(exposures.T @ weights, targets, rtol=1e-15, atol=0.0)


def test_unrepresentable_nonzero_scaled_target_is_rejected():
    smallest_positive = np.nextafter(0.0, 1.0)
    with pytest.raises(ValueError, match="represented.*numerical precision"):
        minimum_variance_factor_weights(
            np.array([[1.0]]),
            np.array([[1e308]]),
            np.array([smallest_positive]),
        )


def test_representable_tiny_target_is_not_treated_as_zero():
    weights = minimum_variance_factor_weights(
        np.array([[1.0]]), np.array([[1.0]]), np.array([1e-20])
    )
    np.testing.assert_allclose(weights, np.array([1e-20]), rtol=1e-15, atol=0.0)


@pytest.mark.parametrize("factor_scale", [1.0, 1e20])
def test_numerically_unresolved_constraints_are_rejected_independently_of_units(
    factor_scale,
):
    exposures = factor_scale * np.array([[1.0, 1.0], [1.0, 1.0 + 1e-9]])
    targets = factor_scale * np.array([1e-20, 0.0])

    with pytest.raises(ValueError, match="resolved.*numerical precision"):
        minimum_variance_factor_weights(np.eye(2), exposures, targets)


def test_extreme_finite_covariance_scale_does_not_overflow_when_symmetrized():
    weights = minimum_variance_factor_weights(
        np.array([[1e308]]), np.array([[1.0]]), np.array([1.0])
    )
    np.testing.assert_allclose(weights, np.array([1.0]), rtol=0.0, atol=0.0)


def test_smallest_positive_covariance_is_normalized_before_symmetrization():
    smallest_positive = np.nextafter(0.0, 1.0)
    weights = minimum_variance_factor_weights(
        np.array([[smallest_positive]]), np.array([[1.0]]), np.array([1.0])
    )
    np.testing.assert_allclose(weights, np.array([1.0]), rtol=0.0, atol=0.0)


def test_zero_targets_return_zero_weights():
    covariance = np.diag([1.0, 2.0, 3.0])
    exposures = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    weights = minimum_variance_factor_weights(covariance, exposures, np.zeros(2))
    np.testing.assert_array_equal(weights, np.zeros(3))


@pytest.mark.parametrize("covariance_scale", [1e-12, 1.0, 1e12])
def test_covariance_scale_does_not_change_weights(covariance_scale):
    covariance = covariance_scale * np.array(
        [[2.0, 0.25, 0.1], [0.25, 1.5, -0.2], [0.1, -0.2, 1.0]]
    )
    exposures = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    targets = np.array([0.2, -0.4])

    baseline = minimum_variance_factor_weights(
        covariance / covariance_scale, exposures, targets
    )
    scaled = minimum_variance_factor_weights(covariance, exposures, targets)
    np.testing.assert_allclose(scaled, baseline, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("target_scale", [-3.0, 0.25, 4.0])
def test_weights_are_homogeneous_in_targets(target_scale):
    covariance = np.diag([1.0, 2.0, 4.0])
    exposures = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    targets = np.array([0.5, -0.25])
    baseline = minimum_variance_factor_weights(covariance, exposures, targets)

    scaled = minimum_variance_factor_weights(
        covariance, exposures, target_scale * targets
    )
    np.testing.assert_allclose(scaled, target_scale * baseline, rtol=1e-12, atol=1e-13)


def test_factor_unit_changes_leave_weights_unchanged():
    covariance = np.diag([1.0, 2.0, 4.0])
    exposures = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    targets = np.array([0.5, -0.25])
    factor_scales = np.array([100.0, 0.01])
    baseline = minimum_variance_factor_weights(covariance, exposures, targets)

    rescaled = minimum_variance_factor_weights(
        covariance,
        exposures * factor_scales,
        targets * factor_scales,
    )
    np.testing.assert_allclose(rescaled, baseline, rtol=1e-12, atol=1e-13)


def test_asset_and_factor_permutations_are_equivariant():
    covariance = np.array([[2.0, 0.2, 0.1], [0.2, 1.5, -0.1], [0.1, -0.1, 1.0]])
    exposures = np.array([[1.0, 0.0], [0.2, 1.0], [1.0, -0.5]])
    targets = np.array([0.3, -0.2])
    baseline = minimum_variance_factor_weights(covariance, exposures, targets)

    asset_order = np.array([2, 0, 1])
    factor_order = np.array([1, 0])
    permuted = minimum_variance_factor_weights(
        covariance[np.ix_(asset_order, asset_order)],
        exposures[asset_order][:, factor_order],
        targets[factor_order],
    )
    restored = np.empty_like(permuted)
    restored[asset_order] = permuted
    np.testing.assert_allclose(restored, baseline, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("covariance_scale", [1e-200, 1.0, 1e200])
def test_tiny_covariance_asymmetry_is_symmetrized(covariance_scale):
    covariance = covariance_scale * np.array([[2.0, 0.3 + 5e-13], [0.3 - 5e-13, 1.0]])
    exposures = np.array([[1.0], [1.0]])
    targets = np.array([1.0])
    symmetric = (covariance + covariance.T) / 2.0

    weights = minimum_variance_factor_weights(covariance, exposures, targets)
    expected = minimum_variance_factor_weights(symmetric, exposures, targets)
    np.testing.assert_allclose(weights, expected, rtol=0.0, atol=1e-14)


@pytest.mark.parametrize("covariance_scale", [1e-200, 1.0, 1e200])
def test_material_covariance_asymmetry_is_rejected(covariance_scale):
    covariance = covariance_scale * np.array([[2.0, 0.3 + 1e-5], [0.3 - 1e-5, 1.0]])
    with pytest.raises(ValueError, match="symmetric"):
        minimum_variance_factor_weights(covariance, np.ones((2, 1)), np.ones(1))


def test_asymmetry_is_measured_relative_to_covariance_scale():
    covariance = np.array([[1e-20, 4e-13], [-4e-13, 1e-20]])
    with pytest.raises(ValueError, match="symmetric"):
        minimum_variance_factor_weights(covariance, np.ones((2, 1)), np.ones(1))


@pytest.mark.parametrize(
    "covariance",
    [
        np.array([[1.0, 1.0], [1.0, 1.0]]),
        np.array([[1.0, 2.0], [2.0, 1.0]]),
    ],
)
def test_non_positive_definite_covariance_is_rejected(covariance):
    with pytest.raises(ValueError, match="positive definite"):
        minimum_variance_factor_weights(covariance, np.ones((2, 1)), np.ones(1))


def test_rank_deficient_exposures_are_rejected():
    exposures = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    with pytest.raises(ValueError, match="full column rank"):
        minimum_variance_factor_weights(np.eye(3), exposures, np.array([1.0, 2.0]))


@pytest.mark.parametrize(
    "covariance, exposures, targets, match",
    [
        (np.ones(2), np.ones((2, 1)), np.ones(1), "2-D square"),
        (np.ones((2, 3)), np.ones((2, 1)), np.ones(1), "square"),
        (np.empty((0, 0)), np.empty((0, 1)), np.ones(1), "non-empty square"),
        (np.eye(2), np.ones(2), np.ones(1), "2-D"),
        (np.eye(2), np.ones((3, 1)), np.ones(1), "rows"),
        (np.eye(2), np.empty((2, 0)), np.empty(0), "at least one"),
        (np.eye(2), np.ones((2, 3)), np.ones(3), "more columns than rows"),
        (np.eye(2), np.ones((2, 1)), np.array([[1.0]]), "1-D"),
        (np.eye(2), np.ones((2, 1)), np.ones(2), "length"),
    ],
)
def test_invalid_shapes_are_rejected(covariance, exposures, targets, match):
    with pytest.raises(ValueError, match=match):
        minimum_variance_factor_weights(covariance, exposures, targets)


@pytest.mark.parametrize("input_name", ["covariance", "exposures", "targets"])
def test_nonfinite_inputs_are_rejected(input_name):
    covariance = np.eye(2)
    exposures = np.ones((2, 1))
    targets = np.ones(1)
    if input_name == "covariance":
        covariance[0, 0] = np.nan
    elif input_name == "exposures":
        exposures[0, 0] = np.inf
    else:
        targets[0] = -np.inf

    with pytest.raises(ValueError, match="finite"):
        minimum_variance_factor_weights(covariance, exposures, targets)


def test_values_outside_float64_range_raise_stable_value_error():
    with pytest.raises(ValueError, match="numeric array"):
        minimum_variance_factor_weights([[10**400]], [[1.0]], [1.0])


@pytest.mark.parametrize("input_name", ["covariance", "exposures", "targets"])
def test_complex_inputs_are_rejected(input_name):
    covariance = np.eye(2).astype(complex)
    exposures = np.ones((2, 1), dtype=complex)
    targets = np.ones(1, dtype=complex)
    if input_name != "covariance":
        covariance = covariance.real
    if input_name != "exposures":
        exposures = exposures.real
    if input_name != "targets":
        targets = targets.real

    with pytest.raises(ValueError, match="real-valued"):
        minimum_variance_factor_weights(covariance, exposures, targets)


def test_inputs_are_not_mutated_and_output_is_fresh_float64():
    covariance = np.array([[2, 1], [1, 3]], dtype=np.int64)
    exposures = np.array([[1], [2]], dtype=np.int64)
    targets = np.array([1], dtype=np.int64)
    originals = (covariance.copy(), exposures.copy(), targets.copy())

    first = minimum_variance_factor_weights(covariance, exposures, targets)
    second = minimum_variance_factor_weights(covariance, exposures, targets)

    for actual, original in zip((covariance, exposures, targets), originals):
        np.testing.assert_array_equal(actual, original)
    assert first.dtype == np.float64
    assert first.shape == (2,)
    assert np.all(np.isfinite(first))
    assert not np.shares_memory(first, second)


def test_explicit_matrix_inverse_is_not_used(monkeypatch):
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("explicit matrix inverse must not be used")

    monkeypatch.setattr(np.linalg, "inv", fail_if_called)
    weights = minimum_variance_factor_weights(np.eye(2), np.ones((2, 1)), np.ones(1))
    np.testing.assert_allclose(weights, np.array([0.5, 0.5]), rtol=0.0, atol=1e-14)


def test_final_factor_exposure_feasibility_is_checked(monkeypatch):
    original_qr = np.linalg.qr

    def zero_orthonormal_basis(*args, **kwargs):
        basis, triangular_factor = original_qr(*args, **kwargs)
        return np.zeros_like(basis), triangular_factor

    monkeypatch.setattr(np.linalg, "qr", zero_orthonormal_basis)
    with pytest.raises(ValueError, match="constraints.*numerical precision"):
        minimum_variance_factor_weights(np.eye(2), np.ones((2, 1)), np.ones(1))


def test_public_import_is_isolated_from_other_model_namespaces():
    repository_root = Path(__file__).resolve().parents[2]
    program = """
import sys
import RiskLabAI

if 'causal_factor_analysis' not in RiskLabAI.__all__:
    raise SystemExit('root public surface omits causal_factor_analysis')
module = RiskLabAI.causal_factor_analysis
if not hasattr(module, 'minimum_variance_factor_weights'):
    raise SystemExit('root lazy import did not resolve the optimizer')

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
