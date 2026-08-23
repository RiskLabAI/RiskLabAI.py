r"""Minimum-variance allocation subject to target factor exposures.

The optimization problem is

.. math::

    \min_{\omega}\; \omega^\mathsf{T} V \omega
    \quad\text{subject to}\quad
    B^\mathsf{T}\omega = c,

where ``V`` is an asset covariance matrix, ``B`` contains asset-by-factor
exposures, and ``c`` is the requested factor-exposure vector. Equations (3)-(6)
and (A-1)-(A-7) of López de Prado, Lipton, and Zoonekynd (2025) give

.. math::

    \omega^* = V^{-1}B(B^\mathsf{T}V^{-1}B)^{-1}c.

This implementation uses linear solves rather than forming either inverse. It
implements only the published allocation problem; factor-exposure estimation
and causal-model selection are separate responsibilities.

References
----------
López de Prado, M., Lipton, A., and Zoonekynd, V. (2025). "Causal Factor
Analysis Is a Necessary Condition for Investment Efficiency." Equations
(3)-(6) and Appendix A. SSRN 5131050.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["minimum_variance_factor_weights"]

_SYMMETRY_RTOL = 1e-10
_SYMMETRY_ATOL = 1e-12
_FEASIBILITY_RTOL = 1e-10
_FEASIBILITY_ATOL = 1e-12


def _as_real_array(values: ArrayLike, name: str) -> NDArray[np.float64]:
    """Convert an input to a real ``float64`` array with a stable error."""
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a numeric array") from exc
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued")
    try:
        return np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a numeric array") from exc


def minimum_variance_factor_weights(
    covariance: ArrayLike,
    factor_exposures: ArrayLike,
    target_exposures: ArrayLike,
) -> NDArray[np.float64]:
    r"""Compute minimum-variance weights with prescribed factor exposures.

    Parameters
    ----------
    covariance : array-like of shape (n_assets, n_assets)
        Symmetric positive-definite asset covariance matrix :math:`V`.
    factor_exposures : array-like of shape (n_assets, n_factors)
        Full-column-rank asset-by-factor exposure matrix :math:`B`.
    target_exposures : array-like of shape (n_factors,)
        Requested factor exposures :math:`c`.

    Returns
    -------
    numpy.ndarray
        One-dimensional ``float64`` vector of optimal asset weights.

    Raises
    ------
    ValueError
        If an input has the wrong shape, contains a non-finite or complex value,
        if ``covariance`` is not symmetric positive definite, or if
        ``factor_exposures`` is not full column rank or its constraints cannot
        be resolved to numerical precision.

    Notes
    -----
    The published inverse formula is evaluated by Cholesky-whitening the
    exposures and solving the resulting minimum-norm problem with reduced QR.
    This avoids both explicit inverses and the condition-number squaring of a
    normal-equation Gram matrix. Covariance and constraint units are normalized
    by positive scalars without changing the optimizer, and the requested
    exposures are verified before return. No diagonal jitter or pseudoinverse
    is applied: singular covariance matrices and redundant factor constraints
    are outside the stated assumptions of the model.

    Examples
    --------
    >>> covariance = np.diag([1.0, 2.0, 4.0])
    >>> exposures = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    >>> targets = np.array([0.0, 1.0])
    >>> minimum_variance_factor_weights(covariance, exposures, targets)
    array([-0.28571429,  0.71428571,  0.28571429])
    """
    covariance_array = _as_real_array(covariance, "covariance")
    exposure_array = _as_real_array(factor_exposures, "factor_exposures")
    target_array = _as_real_array(target_exposures, "target_exposures")

    if covariance_array.ndim != 2:
        raise ValueError("covariance must be a 2-D square matrix")
    n_assets, covariance_columns = covariance_array.shape
    if n_assets == 0 or covariance_columns != n_assets:
        raise ValueError("covariance must be a non-empty square matrix")

    if exposure_array.ndim != 2:
        raise ValueError("factor_exposures must be a 2-D matrix")
    if exposure_array.shape[0] != n_assets:
        raise ValueError(
            "factor_exposures must have the same number of rows as covariance"
        )
    n_factors = exposure_array.shape[1]
    if n_factors == 0:
        raise ValueError("factor_exposures must contain at least one factor column")
    if n_factors > n_assets:
        raise ValueError("factor_exposures cannot have more columns than rows")

    if target_array.ndim != 1:
        raise ValueError("target_exposures must be a 1-D vector")
    if target_array.shape[0] != n_factors:
        raise ValueError(
            "target_exposures length must equal the number of factor columns"
        )

    if not np.all(np.isfinite(covariance_array)):
        raise ValueError("covariance must contain only finite values")
    if not np.all(np.isfinite(exposure_array)):
        raise ValueError("factor_exposures must contain only finite values")
    if not np.all(np.isfinite(target_array)):
        raise ValueError("target_exposures must contain only finite values")

    covariance_scale = float(np.max(np.abs(covariance_array)))
    if covariance_scale == 0.0:
        raise ValueError("covariance must be positive definite")
    scaled_covariance = covariance_array / covariance_scale
    if not np.allclose(
        scaled_covariance,
        scaled_covariance.T,
        rtol=_SYMMETRY_RTOL,
        atol=_SYMMETRY_ATOL,
    ):
        raise ValueError("covariance must be symmetric")
    scaled_covariance = 0.5 * scaled_covariance + 0.5 * scaled_covariance.T

    try:
        covariance_cholesky = np.linalg.cholesky(scaled_covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance must be positive definite") from exc

    maximum_exposures = np.max(np.abs(exposure_array), axis=0)
    if np.any(maximum_exposures == 0.0):
        raise ValueError("factor_exposures must have full column rank")
    target_scales = np.abs(target_array) / (np.finfo(np.float64).max / 2.0)
    constraint_scales = np.maximum(maximum_exposures, target_scales)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        scaled_exposures = exposure_array / constraint_scales
        scaled_targets = target_array / constraint_scales
    target_underflow = (target_array != 0.0) & (scaled_targets == 0.0)
    if (
        np.any(target_underflow)
        or not np.all(np.isfinite(scaled_exposures))
        or not np.all(np.isfinite(scaled_targets))
    ):
        raise ValueError(
            "factor constraints could not be represented at numerical precision"
        )

    try:
        whitened_exposures = np.linalg.solve(covariance_cholesky, scaled_exposures)
        if not np.all(np.isfinite(whitened_exposures)):
            raise ValueError("factor_exposures produce non-finite whitened values")
        if np.linalg.matrix_rank(whitened_exposures) != n_factors:
            raise ValueError("factor_exposures must have full column rank")

        orthonormal_basis, triangular_factor = np.linalg.qr(
            whitened_exposures, mode="reduced"
        )
        transformed_targets = np.linalg.solve(triangular_factor.T, scaled_targets)
        whitened_weights = orthonormal_basis @ transformed_targets
        weights = np.linalg.solve(covariance_cholesky.T, whitened_weights)
    except np.linalg.LinAlgError as exc:
        raise ValueError("factor_exposures must have full column rank") from exc

    weights = np.asarray(weights, dtype=np.float64).reshape(n_assets)
    if not np.all(np.isfinite(weights)):
        raise ValueError("the allocation produced non-finite weights")

    with np.errstate(over="ignore", invalid="ignore"):
        achieved_exposures = scaled_exposures.T @ weights
    absolute_residuals = np.abs(achieved_exposures - scaled_targets)
    feasibility_tolerances = np.where(
        scaled_targets != 0.0,
        _FEASIBILITY_RTOL * np.abs(scaled_targets),
        _FEASIBILITY_ATOL,
    )
    if not np.all(np.isfinite(achieved_exposures)) or np.any(
        absolute_residuals > feasibility_tolerances
    ):
        raise ValueError(
            "factor constraints could not be resolved to numerical precision"
        )

    return weights.copy()
