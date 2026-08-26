"""Independent tests for deterministic structural-model evaluation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from RiskLabAI.causal_factor_analysis.graph_identification import CausalDAG
from RiskLabAI.causal_factor_analysis.structural_models import (
    StructuralCausalModelResult,
    evaluate_structural_causal_model,
)


def _chain():
    return CausalDAG(("X", "Y", "Z"), (("X", "Y"), ("Y", "Z")), ("X", "Y", "Z"))


def test_chain_matches_hand_structural_equations():
    graph = _chain()
    exogenous = {
        "X": np.array([1.0, 2.0, 3.0]),
        "Y": np.array([0.5, -0.5, 1.0]),
        "Z": np.array([2.0, 2.0, 2.0]),
    }
    mechanisms = {
        "X": lambda parents, noise: noise,
        "Y": lambda parents, noise: 2.0 * parents[:, 0] + noise,
        "Z": lambda parents, noise: parents[:, 0] ** 2 + noise,
    }

    result = evaluate_structural_causal_model(graph, mechanisms, exogenous)

    x = exogenous["X"]
    y = 2.0 * x + exogenous["Y"]
    z = y**2 + exogenous["Z"]
    assert result.node_order == ("X", "Y", "Z")
    np.testing.assert_allclose(result.values, np.column_stack((x, y, z)))


def test_lexical_topological_and_parent_order_are_deterministic():
    graph = CausalDAG(
        ("D", "C", "B", "A"),
        (("B", "D"), ("A", "D")),
        ("D", "C", "B", "A"),
    )
    parent_inputs = []

    def d_mechanism(parents, noise):
        parent_inputs.append(parents.copy())
        return parents[:, 0] + 10.0 * parents[:, 1] + noise

    result = evaluate_structural_causal_model(
        graph,
        {
            "D": d_mechanism,
            "C": lambda parents, noise: noise,
            "B": lambda parents, noise: noise,
            "A": lambda parents, noise: noise,
        },
        {
            "D": [0.0, 0.0],
            "C": [30.0, 31.0],
            "B": [20.0, 21.0],
            "A": [1.0, 2.0],
        },
    )

    assert result.node_order == ("A", "B", "C", "D")
    np.testing.assert_array_equal(parent_inputs[0], [[1.0, 20.0], [2.0, 21.0]])
    np.testing.assert_array_equal(result.values[:, 3], [201.0, 212.0])


def test_mechanisms_receive_protected_independent_snapshots():
    graph = _chain()
    captured = []

    def protected_root(parents, noise):
        captured.extend((parents, noise))
        assert parents.shape == (2, 0)
        assert not parents.flags.writeable
        assert not noise.flags.writeable
        with pytest.raises(ValueError):
            noise.setflags(write=True)
        return noise

    result = evaluate_structural_causal_model(
        graph,
        {
            "X": protected_root,
            "Y": lambda parents, noise: parents[:, 0] + noise,
            "Z": lambda parents, noise: parents[:, 0] + noise,
        },
        {"X": [1.0, 2.0], "Y": [0.0, 0.0], "Z": [0.0, 0.0]},
    )

    assert captured
    assert not result.values.flags.writeable
    with pytest.raises(ValueError):
        result.values.setflags(write=True)
    with pytest.raises(ValueError):
        result.values[0, 0] = -1.0


def test_inputs_and_mechanism_outputs_are_snapshotted():
    source = np.array([1.0, 2.0])
    output = source.copy()

    def root(parents, noise):
        return output

    graph = CausalDAG(("X",), (), ("X",))
    result = evaluate_structural_causal_model(graph, {"X": root}, {"X": source})
    source[:] = 10.0
    output[:] = 20.0
    np.testing.assert_array_equal(result.values[:, 0], [1.0, 2.0])


def test_result_is_frozen_and_slotted():
    graph = CausalDAG(("X",), (), ("X",))
    result = evaluate_structural_causal_model(
        graph, {"X": lambda parents, noise: noise}, {"X": [1.0]}
    )
    assert isinstance(result, StructuralCausalModelResult)
    assert not hasattr(result, "__dict__")
    with pytest.raises(FrozenInstanceError):
        result.node_order = ()


@pytest.mark.parametrize(
    "mechanisms,exogenous,exception",
    [
        ((), {"X": [1.0]}, TypeError),
        ({"X": lambda parents, noise: noise}, (), TypeError),
        ({}, {"X": [1.0]}, ValueError),
        ({"X": lambda parents, noise: noise}, {}, ValueError),
        ({"X": 1}, {"X": [1.0]}, TypeError),
        ({"X": lambda parents, noise: noise}, {"X": []}, ValueError),
        ({"X": lambda parents, noise: noise}, {"X": [[1.0]]}, ValueError),
        ({"X": lambda parents, noise: noise}, {"X": [True]}, ValueError),
        ({"X": lambda parents, noise: noise}, {"X": [np.inf]}, ValueError),
    ],
)
def test_structural_model_boundary_validation(mechanisms, exogenous, exception):
    graph = CausalDAG(("X",), (), ("X",))
    with pytest.raises(exception):
        evaluate_structural_causal_model(graph, mechanisms, exogenous)


@pytest.mark.parametrize(
    "mechanism",
    [
        lambda parents, noise: 1.0,
        lambda parents, noise: [[1.0]],
        lambda parents, noise: [1.0, 2.0],
        lambda parents, noise: [np.nan],
        lambda parents, noise: [1.0 + 1.0j],
        lambda parents, noise: [True],
    ],
)
def test_invalid_mechanism_results_fail_closed(mechanism):
    graph = CausalDAG(("X",), (), ("X",))
    with pytest.raises(ValueError):
        evaluate_structural_causal_model(graph, {"X": mechanism}, {"X": [0.0]})


def test_exogenous_lengths_must_match_before_any_mechanism_runs():
    graph = CausalDAG(("X", "Y"), (("X", "Y"),), ("X", "Y"))
    called = []
    mechanisms = {
        "X": lambda parents, noise: called.append("X") or noise,
        "Y": lambda parents, noise: called.append("Y") or noise,
    }
    with pytest.raises(ValueError, match="equal length"):
        evaluate_structural_causal_model(
            graph, mechanisms, {"X": [1.0], "Y": [1.0, 2.0]}
        )
    assert called == []
