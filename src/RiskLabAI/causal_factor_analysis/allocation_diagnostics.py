r"""Diagnostics for allocations built from a misspecified exposure system.

The reference and misspecified portfolios each solve the released
minimum-variance equality-constrained problem. The diagnostic then evaluates
the misspecified weights against the reference exposure matrix, which makes
risk-attribution errors explicit without assuming how either exposure system
was estimated.

This generic comparison implements the unambiguous allocation procedure in
López de Prado, Lipton, and Zoonekynd (2025). It does not instantiate either
paper-specific collider appendix because those printed coefficient systems are
mutually inconsistent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .optimizer import minimum_variance_factor_weights

__all__ = [
    "AllocationMisspecificationDiagnostics",
    "allocation_misspecification_diagnostics",
]


def _as_finite_real_array(values: ArrayLike, name: str) -> NDArray[np.float64]:
    """Return an independent finite real ``float64`` array."""
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a numeric array") from exc
    if raw.dtype.kind == "b" or (
        raw.dtype.kind == "O"
        and any(isinstance(value, (bool, np.bool_)) for value in raw.flat)
    ):
        raise ValueError(f"{name} must not contain boolean values")
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued")
    try:
        result = np.array(values, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a numeric array") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _readonly_float_vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    result = _as_finite_real_array(values, name)
    if result.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    result.setflags(write=False)
    return result


def _readonly_bool_vector(values: ArrayLike, name: str) -> NDArray[np.bool_]:
    try:
        result = np.array(values, dtype=np.bool_, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a boolean vector") from exc
    if result.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    result.setflags(write=False)
    return result


def _as_nonnegative_finite_scalar(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a nonnegative finite scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a nonnegative finite scalar") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a nonnegative finite scalar")
    return result


@dataclass(frozen=True, slots=True, eq=False)
class AllocationMisspecificationDiagnostics:
    """Immutable evidence comparing reference and misspecified allocations.

    Every array is an independent, read-only snapshot. The reference exposure
    error is the exposure produced by the misspecified weights in the reference
    system minus the requested reference target.
    """

    reference_weights: NDArray[np.float64]
    misspecified_weights: NDArray[np.float64]
    reference_target_exposures: NDArray[np.float64]
    reference_achieved_exposures: NDArray[np.float64]
    misspecified_reference_exposures: NDArray[np.float64]
    misspecified_model_target_exposures: NDArray[np.float64]
    misspecified_model_achieved_exposures: NDArray[np.float64]
    reference_exposure_error: NDArray[np.float64]
    weight_sign_reversals: NDArray[np.bool_]
    reference_variance: float
    misspecified_variance: float

    def __post_init__(self) -> None:
        float_vectors = (
            "reference_weights",
            "misspecified_weights",
            "reference_target_exposures",
            "reference_achieved_exposures",
            "misspecified_reference_exposures",
            "misspecified_model_target_exposures",
            "misspecified_model_achieved_exposures",
            "reference_exposure_error",
        )
        for field_name in float_vectors:
            object.__setattr__(
                self,
                field_name,
                _readonly_float_vector(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "weight_sign_reversals",
            _readonly_bool_vector(self.weight_sign_reversals, "weight_sign_reversals"),
        )
        object.__setattr__(
            self,
            "reference_variance",
            _as_nonnegative_finite_scalar(
                self.reference_variance, "reference_variance"
            ),
        )
        object.__setattr__(
            self,
            "misspecified_variance",
            _as_nonnegative_finite_scalar(
                self.misspecified_variance, "misspecified_variance"
            ),
        )


def _validate_shapes(
    covariance: NDArray[np.float64],
    reference_factor_exposures: NDArray[np.float64],
    reference_target_exposures: NDArray[np.float64],
    misspecified_factor_exposures: NDArray[np.float64],
    misspecified_target_exposures: NDArray[np.float64],
) -> None:
    if covariance.ndim != 2 or covariance.shape[0] == 0:
        raise ValueError("covariance must be a non-empty 2-D square matrix")
    n_assets, covariance_columns = covariance.shape
    if covariance_columns != n_assets:
        raise ValueError("covariance must be a non-empty 2-D square matrix")
    for name, exposures, targets in (
        (
            "reference",
            reference_factor_exposures,
            reference_target_exposures,
        ),
        (
            "misspecified",
            misspecified_factor_exposures,
            misspecified_target_exposures,
        ),
    ):
        if exposures.ndim != 2:
            raise ValueError(f"{name}_factor_exposures must be a 2-D matrix")
        if exposures.shape[0] != n_assets:
            raise ValueError(
                f"{name}_factor_exposures must have the same number of rows "
                "as covariance"
            )
        n_factors = exposures.shape[1]
        if n_factors == 0:
            raise ValueError(
                f"{name}_factor_exposures must contain at least one factor column"
            )
        if n_factors > n_assets:
            raise ValueError(
                f"{name}_factor_exposures cannot have more columns than rows"
            )
        if targets.ndim != 1:
            raise ValueError(f"{name}_target_exposures must be a 1-D vector")
        if targets.shape[0] != n_factors:
            raise ValueError(
                f"{name}_target_exposures length must equal the number of "
                f"{name} factor columns"
            )


def _portfolio_variance(
    covariance: NDArray[np.float64],
    weights: NDArray[np.float64],
    name: str,
) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        result = float(weights @ covariance @ weights)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} cannot be represented as a nonnegative finite float")
    return result


def allocation_misspecification_diagnostics(
    covariance: ArrayLike,
    reference_factor_exposures: ArrayLike,
    reference_target_exposures: ArrayLike,
    misspecified_factor_exposures: ArrayLike,
    misspecified_target_exposures: ArrayLike,
) -> AllocationMisspecificationDiagnostics:
    r"""Compare constrained allocations from two exposure systems.

    ``reference_factor_exposures`` and ``misspecified_factor_exposures`` must
    have the same number of asset rows, but they may contain different numbers
    of factor columns. Each allocation is solved against its corresponding
    target. The misspecified weights are then evaluated against the reference
    exposure matrix and covariance.

    A weight is marked as sign-reversed exactly when its reference and
    misspecified values have a strictly negative product; zero weights are not
    reversals.
    """

    covariance_array = _as_finite_real_array(covariance, "covariance")
    reference_exposure_array = _as_finite_real_array(
        reference_factor_exposures, "reference_factor_exposures"
    )
    reference_target_array = _as_finite_real_array(
        reference_target_exposures, "reference_target_exposures"
    )
    misspecified_exposure_array = _as_finite_real_array(
        misspecified_factor_exposures, "misspecified_factor_exposures"
    )
    misspecified_target_array = _as_finite_real_array(
        misspecified_target_exposures, "misspecified_target_exposures"
    )
    _validate_shapes(
        covariance_array,
        reference_exposure_array,
        reference_target_array,
        misspecified_exposure_array,
        misspecified_target_array,
    )

    reference_weights = minimum_variance_factor_weights(
        covariance_array, reference_exposure_array, reference_target_array
    )
    misspecified_weights = minimum_variance_factor_weights(
        covariance_array, misspecified_exposure_array, misspecified_target_array
    )

    with np.errstate(over="ignore", invalid="ignore"):
        reference_achieved_exposures = reference_exposure_array.T @ reference_weights
        misspecified_reference_exposures = (
            reference_exposure_array.T @ misspecified_weights
        )
        misspecified_model_achieved_exposures = (
            misspecified_exposure_array.T @ misspecified_weights
        )
        reference_exposure_error = (
            misspecified_reference_exposures - reference_target_array
        )

    weight_sign_reversals = (
        (reference_weights != 0.0)
        & (misspecified_weights != 0.0)
        & (np.signbit(reference_weights) != np.signbit(misspecified_weights))
    )

    return AllocationMisspecificationDiagnostics(
        reference_weights=reference_weights,
        misspecified_weights=misspecified_weights,
        reference_target_exposures=reference_target_array,
        reference_achieved_exposures=reference_achieved_exposures,
        misspecified_reference_exposures=misspecified_reference_exposures,
        misspecified_model_target_exposures=misspecified_target_array,
        misspecified_model_achieved_exposures=(misspecified_model_achieved_exposures),
        reference_exposure_error=reference_exposure_error,
        weight_sign_reversals=weight_sign_reversals,
        reference_variance=_portfolio_variance(
            covariance_array, reference_weights, "reference_variance"
        ),
        misspecified_variance=_portfolio_variance(
            covariance_array, misspecified_weights, "misspecified_variance"
        ),
    )
