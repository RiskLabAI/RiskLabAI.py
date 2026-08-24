"""Analytical and source-example tests for the book's specification experiments."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from RiskLabAI.causal_factor_analysis import (
    PopulationRegressionDiagnostics,
    RegressionDiagnostics,
    SpecificationExperimentResult,
    SpecificationPopulationResult,
    collider_population_diagnostics,
    collider_specification_experiment,
    confounder_undercontrolled_coefficient,
    confounded_mediator_population_diagnostics,
    confounded_mediator_specification_experiment,
    fork_population_diagnostics,
    fork_specification_experiment,
)


def _independent_population_regression(
    predictor_covariance,
    predictor_outcome_covariance,
    outcome_variance,
):
    covariance = np.asarray(predictor_covariance, dtype=float)
    cross_covariance = np.asarray(predictor_outcome_covariance, dtype=float)
    slopes = np.linalg.solve(covariance, cross_covariance)
    r_squared = float(cross_covariance @ slopes / outcome_variance)
    return tuple(float(value) for value in slopes), r_squared


def test_equation_16_and_appendix_equations_43_to_53_close_analytically():
    beta, gamma, delta = 1.25, -0.75, 2.0
    covariance_v_x = 1.0
    variance_x = 1.0 + delta**2
    conditional_v_coefficient = covariance_v_x / variance_x
    conditional_v_variance = 1.0 - covariance_v_x**2 / variance_x
    assert conditional_v_coefficient == 1.0 / (1.0 + delta**2)
    assert conditional_v_variance == delta**2 / (1.0 + delta**2)
    expected = beta + gamma * delta / (1.0 + delta**2)
    assert confounder_undercontrolled_coefficient(beta, gamma, delta) == pytest.approx(
        expected, rel=1.0e-15, abs=1.0e-15
    )


def test_appendix_equations_54_to_61_give_exact_conditional_gaussian_moments():
    covariance_xy = np.eye(2)
    covariance_xy_z = np.array([[1.0], [1.0]])
    variance_z = np.array([[3.0]])
    conditional_mean_coefficient = covariance_xy_z @ np.linalg.inv(variance_z)
    conditional_covariance = (
        covariance_xy - conditional_mean_coefficient @ covariance_xy_z.T
    )
    np.testing.assert_allclose(
        conditional_mean_coefficient,
        np.array([[1.0 / 3.0], [1.0 / 3.0]]),
        rtol=0.0,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        conditional_covariance,
        np.array([[2.0 / 3.0, -1.0 / 3.0], [-1.0 / 3.0, 2.0 / 3.0]]),
        rtol=0.0,
        atol=1.0e-15,
    )


def test_fork_population_targets_follow_from_covariance_oracle():
    unconditioned_slopes, unconditioned_r2 = _independent_population_regression(
        ((2.0,),),
        (1.0,),
        2.0,
    )
    conditioned_slopes, conditioned_r2 = _independent_population_regression(
        ((2.0, 1.0), (1.0, 1.0)),
        (1.0, 1.0),
        2.0,
    )
    result = fork_population_diagnostics()
    assert result == SpecificationPopulationResult(
        structure="fork",
        conditioning_variable_role="confounder",
        unconditioned=PopulationRegressionDiagnostics(
            ("intercept", "X"),
            (0.0, *unconditioned_slopes),
            unconditioned_r2,
        ),
        conditioned=PopulationRegressionDiagnostics(
            ("intercept", "X", "Z"),
            (0.0, *conditioned_slopes),
            conditioned_r2,
        ),
    )


def test_collider_population_targets_and_equation_31_are_exact():
    conditional_covariance = 0.0 - 1.0 * (1.0 / 3.0) * 1.0
    assert conditional_covariance == pytest.approx(-1.0 / 3.0)
    unconditioned_slopes, unconditioned_r2 = _independent_population_regression(
        ((1.0,),),
        (0.0,),
        1.0,
    )
    conditioned_slopes, conditioned_r2 = _independent_population_regression(
        ((1.0, 1.0), (1.0, 3.0)),
        (0.0, 1.0),
        1.0,
    )
    result = collider_population_diagnostics()
    assert result.unconditioned.coefficients == (0.0, *unconditioned_slopes)
    assert result.unconditioned.r_squared == unconditioned_r2
    assert result.conditioned.coefficients == (0.0, *conditioned_slopes)
    assert result.conditioned.r_squared == conditioned_r2
    assert result.conditioned.coefficients[1:] == (-0.5, 0.5)


def test_confounded_mediator_population_targets_follow_from_covariance_oracle():
    unconditioned_slopes, unconditioned_r2 = _independent_population_regression(
        ((1.0,),),
        (1.0,),
        7.0,
    )
    conditioned_slopes, conditioned_r2 = _independent_population_regression(
        ((1.0, 1.0), (1.0, 3.0)),
        (1.0, 4.0),
        7.0,
    )
    result = confounded_mediator_population_diagnostics()
    assert result.unconditioned.coefficients == (0.0, *unconditioned_slopes)
    assert result.unconditioned.r_squared == unconditioned_r2
    assert result.conditioned.coefficients == (0.0, *conditioned_slopes)
    assert result.conditioned.r_squared == conditioned_r2
    assert result.unconditioned.r_squared == 1.0 / 7.0
    assert result.conditioned.r_squared == 11.0 / 14.0


@pytest.mark.parametrize(
    "function,expected_unconditioned,expected_conditioned,expected_r_squared",
    [
        (
            fork_specification_experiment,
            (0.008987566774752393, 0.4963512366313372),
            (0.0053778294681722984, 0.0007202300104839902, 0.9957125101775881),
            (0.2470257605084264, 0.49480398566467254),
        ),
        (
            collider_specification_experiment,
            (-0.022119251900918794, 0.00150355284008128),
            (-0.013760210010524129, -0.4962958394242535, 0.4988282874959081),
            (2.2405222220855947e-06, 0.49918868068004596),
        ),
        (
            confounded_mediator_specification_experiment,
            (-0.022225744034636837, 1.0055025493825809),
            (0.002741745742420054, -0.4813665679581204, 1.4899423082054921),
            (0.14399915797065876, 0.7840010726224431),
        ),
    ],
)
def test_seed_zero_source_experiments_reproduce_reported_specification_patterns(
    function,
    expected_unconditioned,
    expected_conditioned,
    expected_r_squared,
):
    result = function()
    assert result.unconditioned.coefficients == pytest.approx(
        expected_unconditioned, rel=1.0e-12, abs=1.0e-12
    )
    assert result.conditioned.coefficients == pytest.approx(
        expected_conditioned, rel=1.0e-12, abs=1.0e-12
    )
    assert result.unconditioned.r_squared == pytest.approx(
        expected_r_squared[0], rel=1.0e-12, abs=1.0e-12
    )
    assert result.conditioned.r_squared == pytest.approx(
        expected_r_squared[1], rel=1.0e-12, abs=1.0e-12
    )
    assert result.unconditioned.n_observations == 5_000
    assert result.conditioned.n_observations == 5_000


def test_seeded_experiments_are_reproducible_and_seed_sensitive():
    first = fork_specification_experiment(sample_size=200, seed=17)
    second = fork_specification_experiment(sample_size=200, seed=17)
    different = fork_specification_experiment(sample_size=200, seed=18)
    assert first == second
    assert first != different


@pytest.mark.parametrize(
    "function",
    [
        fork_specification_experiment,
        collider_specification_experiment,
        confounded_mediator_specification_experiment,
    ],
)
@pytest.mark.parametrize(
    "sample_size,seed",
    [
        (3, 0),
        (True, 0),
        (4.5, 0),
        (1_000_001, 0),
        (10, True),
        (10, -1),
        (10, 2**32),
    ],
)
def test_experiments_reject_invalid_resource_and_seed_inputs(
    function,
    sample_size,
    seed,
):
    with pytest.raises(ValueError):
        function(sample_size=sample_size, seed=seed)


def test_public_diagnostic_records_are_immutable_tuple_records():
    records = (
        fork_population_diagnostics(),
        fork_population_diagnostics().conditioned,
        fork_specification_experiment(sample_size=20, seed=3),
        fork_specification_experiment(sample_size=20, seed=3).conditioned,
    )
    assert isinstance(records[0], SpecificationPopulationResult)
    assert isinstance(records[1], PopulationRegressionDiagnostics)
    assert isinstance(records[2], SpecificationExperimentResult)
    assert isinstance(records[3], RegressionDiagnostics)
    for record in records:
        assert not hasattr(record, "__dict__")
        with pytest.raises(AttributeError):
            record.structure = "changed"


def test_module_is_python_39_grammar_and_uses_only_numpy_plus_standard_library():
    source_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "RiskLabAI"
        / "causal_factor_analysis"
        / "specification_diagnostics.py"
    )
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, feature_version=(3, 9))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    assert roots == {"__future__", "math", "numbers", "typing", "numpy"}
