from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from RiskLabAI.causal_factor_analysis import (
    GaussianTrialMixture,
    allocation_misspecification_diagnostics,
    conditional_upper_tail_probability,
    fdr_nonidentification_witness,
    gaussian_max_selection_cdf,
    gaussian_max_selection_log_density,
    generalized_collider_overcontrolled_coefficients,
    generalized_confounder_undercontrolled_coefficient,
    max_selection_family_errors,
    max_selection_null_probability,
    maximum_mixture_cdf,
    single_trial_false_discovery_rate,
)

_FIXTURE_PATH = Path(__file__).with_name("fixtures") / "additive_numeric_parity.tsv"


def _load_cases() -> list[dict[str, str]]:
    with _FIXTURE_PATH.open(encoding="utf-8", newline="") as stream:
        cases = list(csv.DictReader(stream, delimiter="\t"))
    assert [case["case_id"] for case in cases] == [
        "mirage_confounder_general",
        "mirage_collider_general",
        "fdr_single_source_b",
        "fdr_max_source_a",
        "fdr_max_sensitivity",
        "maximum_mixture_simple",
        "conditional_tail_simple",
        "gaussian_identical_cdf",
        "gaussian_identical_log_density",
        "nonidentification_simple",
        "selection_exponential",
        "allocation_two_asset",
    ]
    return cases


def _number(case: dict[str, str], name: str) -> float:
    value = case[name]
    assert value
    return float(value)


@pytest.mark.parametrize("case", _load_cases(), ids=lambda case: case["case_id"])
def test_additive_numeric_cross_language_fixture(case: dict[str, str]) -> None:
    method = case["method"]
    if method == "generalized_confounder":
        actual = generalized_confounder_undercontrolled_coefficient(
            _number(case, "arg1"),
            _number(case, "arg2"),
            _number(case, "arg3"),
            confounder_variance=_number(case, "arg4"),
            exposure_noise_variance=_number(case, "arg5"),
        )
        assert actual == pytest.approx(_number(case, "expected1"), abs=1e-15)
    elif method == "generalized_collider":
        actual = generalized_collider_overcontrolled_coefficients(
            _number(case, "arg1"),
            _number(case, "arg2"),
            _number(case, "arg3"),
            outcome_noise_variance=_number(case, "arg4"),
            collider_noise_variance=_number(case, "arg5"),
        )
        assert actual.beta_hat == pytest.approx(_number(case, "expected1"), abs=1e-15)
        assert actual.theta_hat == pytest.approx(_number(case, "expected2"), abs=1e-15)
    elif method == "single_trial_fdr":
        actual = single_trial_false_discovery_rate(
            _number(case, "arg1"),
            _number(case, "arg2"),
            _number(case, "arg3"),
        )
        assert actual == pytest.approx(_number(case, "expected1"), abs=2e-15)
    elif method == "max_selection_errors":
        actual = max_selection_family_errors(
            _number(case, "arg1"),
            _number(case, "arg2"),
            _number(case, "arg3"),
            int(_number(case, "arg4")),
        )
        assert actual.family_null_probability == pytest.approx(
            _number(case, "expected1"), abs=2e-15
        )
        assert actual.family_type_i_error == pytest.approx(
            _number(case, "expected2"), abs=2e-15
        )
        assert actual.family_type_ii_error == pytest.approx(
            _number(case, "expected3"), abs=2e-14
        )
    elif method == "maximum_mixture_cdf":
        actual = maximum_mixture_cdf(
            _number(case, "arg1"),
            _number(case, "arg2"),
            _number(case, "arg3"),
            int(_number(case, "arg4")),
        )
        assert actual == pytest.approx(_number(case, "expected1"), abs=1e-15)
    elif method == "conditional_upper_tail":
        actual = conditional_upper_tail_probability(
            _number(case, "arg1"), _number(case, "arg2")
        )
        assert actual == pytest.approx(_number(case, "expected1"), abs=1e-15)
    elif method in {"gaussian_max_cdf", "gaussian_max_log_density"}:
        model = GaussianTrialMixture(
            _number(case, "arg1"),
            _number(case, "arg2"),
            _number(case, "arg3"),
            _number(case, "arg4"),
        )
        evaluator = (
            gaussian_max_selection_cdf
            if method == "gaussian_max_cdf"
            else gaussian_max_selection_log_density
        )
        actual = evaluator(model, _number(case, "arg5"), int(_number(case, "arg6")))
        assert actual == pytest.approx(_number(case, "expected1"), abs=2e-15)
    elif method == "nonidentification_witness":
        actual = fdr_nonidentification_witness(
            _number(case, "arg1"), _number(case, "arg2")
        )
        assert actual.trial_null_probability == pytest.approx(
            _number(case, "expected1"), abs=1e-15
        )
        assert actual.latent_trial_cdf_at_threshold == pytest.approx(
            _number(case, "expected2"), abs=1e-15
        )
        assert actual.family_false_discovery_rate == pytest.approx(
            _number(case, "expected3"), abs=1e-15
        )
    elif method == "selection_exponential":
        threshold = _number(case, "arg1")

        def density(value: float) -> float:
            return math.exp(-value) if value >= 0.0 else 0.0

        def distribution(value: float) -> float:
            return 1.0 - math.exp(-value) if value > 0.0 else 0.0

        actual = max_selection_null_probability(
            threshold,
            _number(case, "arg2"),
            int(_number(case, "arg3")),
            null_density=density,
            mixture_cdf=distribution,
        )
        assert actual.selection_probability == pytest.approx(
            _number(case, "expected1"), abs=2e-14
        )
        assert actual.joint_null_and_selection_probability == pytest.approx(
            _number(case, "expected2"), abs=2e-11
        )
        assert actual.conditional_null_probability == pytest.approx(
            _number(case, "expected3"), abs=2e-11
        )
    elif method == "allocation_two_asset":
        actual = allocation_misspecification_diagnostics(
            np.eye(2),
            np.array([[1.0], [1.0]]),
            np.array([1.0]),
            np.array([[1.0], [-1.0]]),
            np.array([1.0]),
        )
        np.testing.assert_allclose(
            actual.reference_weights,
            [_number(case, "expected1"), _number(case, "expected1")],
            atol=1e-14,
        )
        np.testing.assert_allclose(
            actual.misspecified_weights,
            [_number(case, "expected1"), _number(case, "expected2")],
            atol=1e-14,
        )
        np.testing.assert_allclose(
            actual.reference_exposure_error,
            [_number(case, "expected3")],
            atol=1e-14,
        )
        assert actual.weight_sign_reversals.tolist() == [False, True]
    else:
        raise AssertionError(f"unhandled parity method: {method}")
