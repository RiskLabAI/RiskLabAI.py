"""Small, deterministic examples for the frozen causal-factor API."""

import numpy as np
import RiskLabAI

from RiskLabAI.causal_factor_analysis import (
    CausalDAG,
    average_treatment_effect,
    check_backdoor_adjustment_set,
    classify_treatment_outcome_role,
    generalized_confounder_undercontrolled_coefficient,
    max_selection_family_errors,
    minimum_variance_factor_weights,
)

assert RiskLabAI.__version__ == "3.2.0"


covariance = np.diag([1.0, 2.0, 4.0])
factor_exposures = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
target_exposures = np.array([0.0, 1.0])
weights = minimum_variance_factor_weights(
    covariance, factor_exposures, target_exposures
)
np.testing.assert_allclose(weights, [-2.0 / 7.0, 5.0 / 7.0, 2.0 / 7.0])

effect = average_treatment_effect(3.5, 1.25)
assert effect == 2.25

dag = CausalDAG(
    nodes=("T", "U", "Y"),
    directed_edges=(("U", "T"), ("U", "Y"), ("T", "Y")),
    observed_nodes=("T", "U", "Y"),
)
assert not check_backdoor_adjustment_set(dag, "T", "Y").admissible
assert check_backdoor_adjustment_set(dag, "T", "Y", ("U",)).admissible
role = classify_treatment_outcome_role(dag, "T", "Y", "U")
assert role.role.value == "confounder"

generalized_coefficient = generalized_confounder_undercontrolled_coefficient(
    2.0,
    3.0,
    4.0,
    confounder_variance=5.0,
    exposure_noise_variance=6.0,
)
assert generalized_coefficient == 116.0 / 43.0

family_errors = max_selection_family_errors(
    0.024997895148220373,
    0.9515427737332771,
    0.95,
    10,
)
assert np.isclose(family_errors.family_type_i_error, 0.22365361940347483)

print("minimum-variance weights:", weights)
print("average treatment effect:", effect)
print("back-door adjustment set: ('U',)")
print("one-hop role for U:", role.role.value)
print("general-variance confounder coefficient:", generalized_coefficient)
print("ten-trial family Type-I error:", family_errors.family_type_i_error)
