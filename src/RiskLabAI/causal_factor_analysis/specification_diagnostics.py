"""Specification diagnostics from the 2023 causal-factor experiments.

This module independently implements the three linear-Gaussian structural
experiments in Sections 7.1-7.3 and their Appendix procedures.  Population
functions expose exact analytical regression targets; experiment functions
generate seeded samples and fit both specifications with NumPy only.
"""

from __future__ import annotations

import math
from numbers import Integral
from typing import NamedTuple, Tuple

import numpy as np

__all__ = [
    "PopulationRegressionDiagnostics",
    "RegressionDiagnostics",
    "SpecificationExperimentResult",
    "SpecificationPopulationResult",
    "collider_population_diagnostics",
    "collider_specification_experiment",
    "confounded_mediator_population_diagnostics",
    "confounded_mediator_specification_experiment",
    "fork_population_diagnostics",
    "fork_specification_experiment",
]


class PopulationRegressionDiagnostics(NamedTuple):
    """Exact population coefficients and explained-variance fraction."""

    coefficient_names: Tuple[str, ...]
    coefficients: Tuple[float, ...]
    r_squared: float


class RegressionDiagnostics(NamedTuple):
    """Classical least-squares diagnostics for one generated sample."""

    coefficient_names: Tuple[str, ...]
    coefficients: Tuple[float, ...]
    standard_errors: Tuple[float, ...]
    t_statistics: Tuple[float, ...]
    r_squared: float
    adjusted_r_squared: float
    residual_variance: float
    n_observations: int


class SpecificationPopulationResult(NamedTuple):
    """Exact population targets for unconditioned and conditioned models."""

    structure: str
    conditioning_variable_role: str
    unconditioned: PopulationRegressionDiagnostics
    conditioned: PopulationRegressionDiagnostics


class SpecificationExperimentResult(NamedTuple):
    """Sample diagnostics for unconditioned and conditioned regressions."""

    structure: str
    conditioning_variable_role: str
    unconditioned: RegressionDiagnostics
    conditioned: RegressionDiagnostics


_MAX_SAMPLE_SIZE = 1_000_000


def _as_sample_size(value: Integral) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError("sample_size must be an integer")
    result = int(value)
    if result <= 3:
        raise ValueError("sample_size must be greater than 3")
    if result > _MAX_SAMPLE_SIZE:
        raise ValueError(f"sample_size must not exceed {_MAX_SAMPLE_SIZE}")
    return result


def _as_seed(value: Integral) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError("seed must be an integer")
    result = int(value)
    if result < 0 or result > 2**32 - 1:
        raise ValueError("seed must be between 0 and 2**32 - 1")
    return result


def _population_regression(
    names: Tuple[str, ...], coefficients: Tuple[float, ...], r_squared: float
) -> PopulationRegressionDiagnostics:
    return PopulationRegressionDiagnostics(
        coefficient_names=names,
        coefficients=coefficients,
        r_squared=r_squared,
    )


def _ols_diagnostics(
    outcome: np.ndarray,
    regressors: Tuple[Tuple[str, np.ndarray], ...],
) -> RegressionDiagnostics:
    n_observations = int(outcome.size)
    design = np.column_stack(
        (np.ones(n_observations, dtype=float), *(values for _, values in regressors))
    )
    n_parameters = int(design.shape[1])
    if n_observations <= n_parameters:
        raise ValueError("the regression requires positive residual degrees of freedom")
    if np.linalg.matrix_rank(design) != n_parameters:
        raise ValueError("the generated regression design is rank deficient")

    coefficients, _, _, _ = np.linalg.lstsq(design, outcome, rcond=None)
    fitted = design @ coefficients
    residuals = outcome - fitted
    residual_sum_squares = float(residuals @ residuals)
    centered = outcome - float(np.mean(outcome))
    total_sum_squares = float(centered @ centered)
    if not math.isfinite(total_sum_squares) or total_sum_squares <= 0.0:
        raise ValueError("the generated outcome must have positive finite variance")

    residual_degrees_of_freedom = n_observations - n_parameters
    residual_variance = residual_sum_squares / residual_degrees_of_freedom
    gram_inverse = np.linalg.inv(design.T @ design)
    coefficient_variances = np.diag(gram_inverse) * residual_variance
    if np.any(coefficient_variances < -1.0e-12):
        raise ValueError("the coefficient variance calculation is invalid")
    standard_errors = np.sqrt(np.maximum(coefficient_variances, 0.0))
    t_statistics = np.divide(
        coefficients,
        standard_errors,
        out=np.full(coefficients.shape, np.nan, dtype=float),
        where=standard_errors > 0.0,
    )
    if not np.all(np.isfinite(coefficients)) or not np.all(
        np.isfinite(standard_errors)
    ):
        raise ValueError("the regression diagnostics must be finite")

    r_squared = 1.0 - residual_sum_squares / total_sum_squares
    if r_squared < 0.0 and r_squared >= -1.0e-12:
        r_squared = 0.0
    if r_squared > 1.0 and r_squared <= 1.0 + 1.0e-12:
        r_squared = 1.0
    adjusted_r_squared = 1.0 - (1.0 - r_squared) * (
        (n_observations - 1) / residual_degrees_of_freedom
    )
    names = ("intercept", *(name for name, _ in regressors))
    return RegressionDiagnostics(
        coefficient_names=names,
        coefficients=tuple(float(value) for value in coefficients),
        standard_errors=tuple(float(value) for value in standard_errors),
        t_statistics=tuple(float(value) for value in t_statistics),
        r_squared=float(r_squared),
        adjusted_r_squared=float(adjusted_r_squared),
        residual_variance=float(residual_variance),
        n_observations=n_observations,
    )


def fork_population_diagnostics() -> SpecificationPopulationResult:
    """Return exact targets for Equations 20-24 (the confounder fork)."""

    return SpecificationPopulationResult(
        structure="fork",
        conditioning_variable_role="confounder",
        unconditioned=_population_regression(("intercept", "X"), (0.0, 0.5), 0.25),
        conditioned=_population_regression(
            ("intercept", "X", "Z"), (0.0, 0.0, 1.0), 0.5
        ),
    )


def collider_population_diagnostics() -> SpecificationPopulationResult:
    """Return exact targets for Equations 27-33 (the collider)."""

    return SpecificationPopulationResult(
        structure="collider",
        conditioning_variable_role="collider",
        unconditioned=_population_regression(("intercept", "X"), (0.0, 0.0), 0.0),
        conditioned=_population_regression(
            ("intercept", "X", "Z"), (0.0, -0.5, 0.5), 0.5
        ),
    )


def confounded_mediator_population_diagnostics() -> SpecificationPopulationResult:
    """Return exact targets for Equations 37-42 (the confounded mediator)."""

    return SpecificationPopulationResult(
        structure="confounded_mediator",
        conditioning_variable_role="confounded_mediator",
        unconditioned=_population_regression(("intercept", "X"), (0.0, 1.0), 1.0 / 7.0),
        conditioned=_population_regression(
            ("intercept", "X", "Z"),
            (0.0, -0.5, 1.5),
            11.0 / 14.0,
        ),
    )


def fork_specification_experiment(
    sample_size: Integral = 5_000,
    seed: Integral = 0,
) -> SpecificationExperimentResult:
    """Run the seeded fork experiment from Section 7.1 and the Appendix."""

    n_observations = _as_sample_size(sample_size)
    random_state = np.random.RandomState(_as_seed(seed))
    z = random_state.standard_normal(n_observations)
    x = z + random_state.standard_normal(n_observations)
    y = z + random_state.standard_normal(n_observations)
    return SpecificationExperimentResult(
        structure="fork",
        conditioning_variable_role="confounder",
        unconditioned=_ols_diagnostics(y, (("X", x),)),
        conditioned=_ols_diagnostics(y, (("X", x), ("Z", z))),
    )


def collider_specification_experiment(
    sample_size: Integral = 5_000,
    seed: Integral = 0,
) -> SpecificationExperimentResult:
    """Run the seeded collider experiment from Section 7.2 and the Appendix."""

    n_observations = _as_sample_size(sample_size)
    random_state = np.random.RandomState(_as_seed(seed))
    x = random_state.standard_normal(n_observations)
    y = random_state.standard_normal(n_observations)
    z = x + y + random_state.standard_normal(n_observations)
    return SpecificationExperimentResult(
        structure="collider",
        conditioning_variable_role="collider",
        unconditioned=_ols_diagnostics(y, (("X", x),)),
        conditioned=_ols_diagnostics(y, (("X", x), ("Z", z))),
    )


def confounded_mediator_specification_experiment(
    sample_size: Integral = 5_000,
    seed: Integral = 0,
) -> SpecificationExperimentResult:
    """Run the seeded confounded-mediator experiment from Section 7.3."""

    n_observations = _as_sample_size(sample_size)
    random_state = np.random.RandomState(_as_seed(seed))
    x = random_state.standard_normal(n_observations)
    w = random_state.standard_normal(n_observations)
    z = x + w + random_state.standard_normal(n_observations)
    y = z + w + random_state.standard_normal(n_observations)
    return SpecificationExperimentResult(
        structure="confounded_mediator",
        conditioning_variable_role="confounded_mediator",
        unconditioned=_ols_diagnostics(y, (("X", x),)),
        conditioned=_ols_diagnostics(y, (("X", x), ("Z", z))),
    )
