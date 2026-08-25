"""Small, deterministic examples for the frozen causal-factor API."""

import numpy as np
import RiskLabAI

from RiskLabAI.causal_factor_analysis import (
    CausalDAG,
    average_treatment_effect,
    check_backdoor_adjustment_set,
    minimum_variance_factor_weights,
)

assert RiskLabAI.__version__ == "3.0.0"


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

print("minimum-variance weights:", weights)
print("average treatment effect:", effect)
print("back-door adjustment set: ('U',)")
