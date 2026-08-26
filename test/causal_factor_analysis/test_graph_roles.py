"""Independent tests for accepted-DAG and one-hop graph roles."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from RiskLabAI.causal_factor_analysis.graph_identification import CausalDAG
from RiskLabAI.causal_factor_analysis.graph_roles import (
    FactorControlRoles,
    TreatmentOutcomeRole,
    TreatmentOutcomeRoleEvidence,
    classify_treatment_outcome_role,
    factor_control_roles,
)


def _dag(edges=(), *, nodes=("N", "X", "Y"), observed=None):
    if observed is None:
        observed = nodes
    return CausalDAG(nodes, edges, observed)


def test_factor_control_roles_partition_observed_and_report_latent_reachability():
    graph = CausalDAG(
        ("A", "D", "L_A", "L_D", "O", "T"),
        (("A", "T"), ("D", "L_D"), ("L_A", "A"), ("T", "D")),
        ("A", "D", "O", "T"),
    )

    roles = factor_control_roles(graph, "T")

    assert roles == FactorControlRoles(
        target_factor="T",
        ancestor_factors=("A",),
        descendant_factors=("D",),
        other_factors=("O",),
        unobserved_ancestors=("L_A",),
        unobserved_descendants=("L_D",),
    )
    assert not hasattr(roles, "__dict__")
    with pytest.raises(FrozenInstanceError):
        roles.target_factor = "O"


def test_factor_control_roles_is_invariant_to_declaration_order():
    edges = (("Z", "T"), ("T", "B"), ("A", "Z"))
    first = CausalDAG(("A", "B", "T", "X", "Z"), edges, ("A", "B", "T", "X", "Z"))
    second = CausalDAG(
        ("X", "T", "Z", "B", "A"),
        tuple(reversed(edges)),
        ("Z", "X", "T", "B", "A"),
    )
    assert factor_control_roles(first, "T") == factor_control_roles(second, "T")


@pytest.mark.parametrize(
    "edges,role,signature",
    [
        ((("N", "X"),), TreatmentOutcomeRole.CAUSE_OF_TREATMENT, (1, 0, 0, 0)),
        ((("X", "N"),), TreatmentOutcomeRole.CONSEQUENCE_OF_TREATMENT, (0, 1, 0, 0)),
        ((("N", "Y"),), TreatmentOutcomeRole.CAUSE_OF_OUTCOME, (0, 0, 1, 0)),
        ((("Y", "N"),), TreatmentOutcomeRole.CONSEQUENCE_OF_OUTCOME, (0, 0, 0, 1)),
        (
            (("N", "X"), ("N", "Y")),
            TreatmentOutcomeRole.CONFOUNDER,
            (1, 0, 1, 0),
        ),
        (
            (("X", "N"), ("Y", "N")),
            TreatmentOutcomeRole.COLLIDER,
            (0, 1, 0, 1),
        ),
        (
            (("X", "N"), ("N", "Y")),
            TreatmentOutcomeRole.MEDIATOR,
            (0, 1, 1, 0),
        ),
        ((), TreatmentOutcomeRole.INDEPENDENT, (0, 0, 0, 0)),
    ],
)
def test_all_eight_one_hop_signatures(edges, role, signature):
    evidence = classify_treatment_outcome_role(_dag(edges), "X", "Y", "N")

    assert isinstance(evidence, TreatmentOutcomeRoleEvidence)
    assert evidence.node == "N"
    assert evidence.treatment == "X"
    assert evidence.outcome == "Y"
    assert evidence.role is role
    assert (
        evidence.node_to_treatment,
        evidence.treatment_to_node,
        evidence.node_to_outcome,
        evidence.outcome_to_node,
    ) == tuple(bool(value) for value in signature)


def test_role_classifier_rejects_reverse_mediation():
    graph = _dag((("N", "X"), ("Y", "N")))
    with pytest.raises(ValueError, match="eight admitted roles"):
        classify_treatment_outcome_role(graph, "X", "Y", "N")


def test_role_classifier_uses_direct_edges_only():
    graph = CausalDAG(
        ("A", "N", "X", "Y"),
        (("N", "A"), ("A", "X"), ("N", "Y")),
        ("A", "N", "X", "Y"),
    )
    evidence = classify_treatment_outcome_role(graph, "X", "Y", "N")
    assert evidence.role is TreatmentOutcomeRole.CAUSE_OF_OUTCOME
    assert not evidence.node_to_treatment


@pytest.mark.parametrize(
    "operation,exception",
    [
        (lambda: factor_control_roles(object(), "X"), TypeError),
        (lambda: factor_control_roles(_dag(), 1), TypeError),
        (lambda: factor_control_roles(_dag(), "missing"), ValueError),
        (
            lambda: factor_control_roles(
                _dag(nodes=("L", "N", "X", "Y"), observed=("N", "X", "Y")),
                "L",
            ),
            ValueError,
        ),
        (lambda: classify_treatment_outcome_role(_dag(), "X", "X", "N"), ValueError),
        (lambda: classify_treatment_outcome_role(_dag(), "X", "Y", 1), TypeError),
        (
            lambda: classify_treatment_outcome_role(
                _dag(nodes=("L", "N", "X", "Y"), observed=("N", "X", "Y")),
                "X",
                "Y",
                "L",
            ),
            ValueError,
        ),
    ],
)
def test_graph_role_boundary_validation(operation, exception):
    with pytest.raises(exception):
        operation()


def test_treatment_outcome_role_values_are_frozen():
    assert [(role.name, role.value) for role in TreatmentOutcomeRole] == [
        ("CAUSE_OF_TREATMENT", "cause_of_treatment"),
        ("CONSEQUENCE_OF_TREATMENT", "consequence_of_treatment"),
        ("CAUSE_OF_OUTCOME", "cause_of_outcome"),
        ("CONSEQUENCE_OF_OUTCOME", "consequence_of_outcome"),
        ("CONFOUNDER", "confounder"),
        ("COLLIDER", "collider"),
        ("MEDIATOR", "mediator"),
        ("INDEPENDENT", "independent"),
    ]
