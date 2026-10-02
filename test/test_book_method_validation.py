"""Invalid inputs and degenerate cases must fail explicitly."""

import numpy as np
import pytest

from RiskLabAI.features.entropy_features.ambiguity import (
    probability_ambiguity,
    normal_prior_bin_probabilities,
)
from RiskLabAI.features.microstructural_features.order_flow import (
    hawkes_log_likelihood,
    fit_hawkes,
    volume_synchronized_pin,
    simulate_hawkes,
)
from RiskLabAI.backtest.tail_risk import (
    generalized_pareto_tail_risk,
    fit_pot_tail_risk,
    quantile_exception_tests,
)
from RiskLabAI.optimization.decision_focused import spo_plus_loss
from RiskLabAI.causal_factor_analysis.bounds_transport import (
    manski_ate_bounds,
    transported_ate,
)
from RiskLabAI.causal_factor_analysis.interference import exposure_mean_contrast
from RiskLabAI.causal_factor_analysis.policy_evaluation import evaluate_policy


@pytest.mark.parametrize(
    "call",
    [
        lambda: probability_ambiguity([[0.2, 0.2]]),
        lambda: probability_ambiguity([[-1, 2]]),
        lambda: probability_ambiguity([[0.5, 0.5]], prior_weights=[0.5, 0.5]),
        lambda: probability_ambiguity([[0.5, 0.5]], bin_width=1),
        lambda: probability_ambiguity([[True, False]]),
        lambda: normal_prior_bin_probabilities([0], [0]),
        lambda: normal_prior_bin_probabilities([0], [1], edges=[1, 0]),
        lambda: normal_prior_bin_probabilities([0, 1], [1]),
        lambda: hawkes_log_likelihood([0.5], 1, 1, 1, 1),
        lambda: hawkes_log_likelihood([0.5], 1, 0, 0, 1),
        lambda: hawkes_log_likelihood([1.5], 1, 1, 0, 1),
        lambda: hawkes_log_likelihood([0.7, 0.5], 1, 1, 0, 1),
        lambda: hawkes_log_likelihood([np.nan], 1, 1, 0, 1),
        lambda: fit_hawkes([], 1),
        lambda: volume_synchronized_pin([1], [0], bucket_volume=1, window=2),
        lambda: volume_synchronized_pin([1], [0], bucket_volume=1, window=True),
        lambda: volume_synchronized_pin([-1], [2], bucket_volume=1, window=1),
        lambda: generalized_pareto_tail_risk(0, 1.1, 0, 1, 0.9),
        lambda: generalized_pareto_tail_risk(0, 0.1, 0, -1, 0.99),
        lambda: fit_pot_tail_risk([1, 1], threshold=1, probability=0.99),
        lambda: quantile_exception_tests([], exception_probability=0.1),
        lambda: quantile_exception_tests([0, 0.5], exception_probability=0.1),
        lambda: quantile_exception_tests([0, 1], exception_probability=1),
        lambda: spo_plus_loss([1], [1, 2], [[1, 2]]),
        lambda: spo_plus_loss([1], [2], []),
        lambda: manski_ate_bounds([2], [1], lower=0, upper=1),
        lambda: manski_ate_bounds([0.5], [2], lower=0, upper=1),
        lambda: transported_ate([1], [0], [0.5], [1], [1]),
        lambda: transported_ate([1], [0], [1], [-1], [1]),
        lambda: exposure_mean_contrast([1], [0], [[1, 0]], first=0, second=1),
        lambda: exposure_mean_contrast([1], [0], [[0.5, 0.5]], first=0, second=0),
        lambda: exposure_mean_contrast([1], [0.0], [[0.5, 0.5]], first=0, second=1),
        lambda: exposure_mean_contrast([1], [0], [[0.6, 0.6]], first=0, second=1),
        lambda: evaluate_policy([0], [1], [[0.5, 0.5]], [[0.4, 0.4]], [[1, 2]]),
        lambda: evaluate_policy([2], [1], [[0.5, 0.5]], [[0.5, 0.5]], [[1, 2]]),
        lambda: evaluate_policy([True], [1], [[0.5, 0.5]], [[0.5, 0.5]], [[1, 2]]),
    ],
)
def test_invalid_contract_inputs(call):
    with pytest.raises(ValueError):
        call()


def test_hawkes_cap_does_not_silently_truncate():
    with pytest.raises(RuntimeError, match="max_events"):
        simulate_hawkes(100, 100, 0, 1, random_state=1, max_events=2)


def test_spo_first_vertex_tie_and_identical_costs():
    vertices = [[0, 0], [1, 0], [0, 1]]
    result = spo_plus_loss([0, 0], [0, 0], vertices)
    assert result["true_vertex"] == result["transformed_vertex"] == 0
    assert result["loss"] == 0


def test_manski_all_treated_and_constant_outcomes():
    assert manski_ate_bounds([1, 1], [1, 1], lower=0, upper=1)["lower"] == 0
    result = manski_ate_bounds([2, 2], [0, 1], lower=2, upper=2)
    assert result["lower"] == result["upper"] == 0


def test_unobserved_policy_actions_have_explicit_zero_weight():
    result = evaluate_policy(
        [0, 0], [1, 2], [[0.5, 0.5]] * 2, [[0, 1]] * 2, [[1, 3]] * 2
    )
    assert result["ipw_value"] == 0
    assert result["effective_sample_size"] == 0
    assert result["dr_value"] == 3
