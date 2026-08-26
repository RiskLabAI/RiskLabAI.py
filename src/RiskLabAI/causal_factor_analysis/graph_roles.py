"""Accepted-DAG factor roles and one-hop treatment/outcome roles.

These routines classify implications of a supplied :class:`CausalDAG`.  They
do not discover a graph or assert that every ancestor is a sufficient or
minimal adjustment set.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .graph_identification import CausalDAG

__all__ = [
    "FactorControlRoles",
    "TreatmentOutcomeRole",
    "TreatmentOutcomeRoleEvidence",
    "classify_treatment_outcome_role",
    "factor_control_roles",
]


@dataclass(frozen=True, slots=True)
class FactorControlRoles:
    """Reachability roles for factors relative to one observed target."""

    target_factor: str
    ancestor_factors: tuple[str, ...]
    descendant_factors: tuple[str, ...]
    other_factors: tuple[str, ...]
    unobserved_ancestors: tuple[str, ...]
    unobserved_descendants: tuple[str, ...]


class TreatmentOutcomeRole(Enum):
    """The eight admitted one-hop treatment/outcome roles."""

    CAUSE_OF_TREATMENT = "cause_of_treatment"
    CONSEQUENCE_OF_TREATMENT = "consequence_of_treatment"
    CAUSE_OF_OUTCOME = "cause_of_outcome"
    CONSEQUENCE_OF_OUTCOME = "consequence_of_outcome"
    CONFOUNDER = "confounder"
    COLLIDER = "collider"
    MEDIATOR = "mediator"
    INDEPENDENT = "independent"


@dataclass(frozen=True, slots=True)
class TreatmentOutcomeRoleEvidence:
    """Direct-edge evidence for one admitted treatment/outcome role."""

    node: str
    treatment: str
    outcome: str
    role: TreatmentOutcomeRole
    node_to_treatment: bool
    treatment_to_node: bool
    node_to_outcome: bool
    outcome_to_node: bool


_ROLE_BY_SIGNATURE = {
    (True, False, False, False): TreatmentOutcomeRole.CAUSE_OF_TREATMENT,
    (False, True, False, False): TreatmentOutcomeRole.CONSEQUENCE_OF_TREATMENT,
    (False, False, True, False): TreatmentOutcomeRole.CAUSE_OF_OUTCOME,
    (False, False, False, True): TreatmentOutcomeRole.CONSEQUENCE_OF_OUTCOME,
    (True, False, True, False): TreatmentOutcomeRole.CONFOUNDER,
    (False, True, False, True): TreatmentOutcomeRole.COLLIDER,
    (False, True, True, False): TreatmentOutcomeRole.MEDIATOR,
    (False, False, False, False): TreatmentOutcomeRole.INDEPENDENT,
}


def _validate_dag(dag: CausalDAG) -> CausalDAG:
    if not isinstance(dag, CausalDAG):
        raise TypeError("dag must be a CausalDAG")
    return dag


def _require_observed_node(dag: CausalDAG, value: str, subject: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{subject} must be a string node identifier")
    if not value.strip():
        raise ValueError(f"{subject} must be nonblank")
    if value not in dag.nodes:
        raise ValueError(f"{subject} must be a known graph node")
    if value not in dag.observed_nodes:
        raise ValueError(f"{subject} must be an observed graph node")
    return value


def _adjacency(dag: CausalDAG, *, reverse: bool) -> dict[str, set[str]]:
    adjacency = {node: set() for node in dag.nodes}
    for source, target in dag.directed_edges:
        left, right = (target, source) if reverse else (source, target)
        adjacency[left].add(right)
    return adjacency


def _reachable(adjacency: dict[str, set[str]], start: str) -> set[str]:
    visited: set[str] = set()
    pending = list(adjacency[start])
    while pending:
        node = pending.pop()
        if node in visited:
            continue
        visited.add(node)
        pending.extend(adjacency[node] - visited)
    return visited


def factor_control_roles(dag: CausalDAG, target_factor: str) -> FactorControlRoles:
    """Partition factors by reachability in an accepted DAG.

    Observed non-target nodes are partitioned into ancestors, descendants, and
    all remaining factors.  Latent ancestors and descendants are reported
    separately as evidence that may limit adjustment or interpretation.
    """

    dag = _validate_dag(dag)
    target_factor = _require_observed_node(dag, target_factor, "target_factor")

    ancestors = _reachable(_adjacency(dag, reverse=True), target_factor)
    descendants = _reachable(_adjacency(dag, reverse=False), target_factor)
    observed_candidates = set(dag.observed_nodes) - {target_factor}
    unobserved = set(dag.nodes) - set(dag.observed_nodes)

    ancestor_factors = observed_candidates & ancestors
    descendant_factors = observed_candidates & descendants
    other_factors = observed_candidates - ancestor_factors - descendant_factors

    return FactorControlRoles(
        target_factor=target_factor,
        ancestor_factors=tuple(sorted(ancestor_factors)),
        descendant_factors=tuple(sorted(descendant_factors)),
        other_factors=tuple(sorted(other_factors)),
        unobserved_ancestors=tuple(sorted(unobserved & ancestors)),
        unobserved_descendants=tuple(sorted(unobserved & descendants)),
    )


def classify_treatment_outcome_role(
    dag: CausalDAG,
    treatment: str,
    outcome: str,
    node: str,
) -> TreatmentOutcomeRoleEvidence:
    """Classify one observed node using the frozen four-edge signature.

    Only the eight published one-hop signatures are admitted.  Reverse
    mediation and every other signature fail closed with ``ValueError``.
    """

    dag = _validate_dag(dag)
    treatment = _require_observed_node(dag, treatment, "treatment")
    outcome = _require_observed_node(dag, outcome, "outcome")
    node = _require_observed_node(dag, node, "node")
    if len({treatment, outcome, node}) != 3:
        raise ValueError("treatment, outcome, and node must be distinct")

    edges = set(dag.directed_edges)
    signature = (
        (node, treatment) in edges,
        (treatment, node) in edges,
        (node, outcome) in edges,
        (outcome, node) in edges,
    )
    try:
        role = _ROLE_BY_SIGNATURE[signature]
    except KeyError as exc:
        raise ValueError(
            "the direct treatment/outcome edge signature is not one of the "
            "eight admitted roles"
        ) from exc

    return TreatmentOutcomeRoleEvidence(
        node=node,
        treatment=treatment,
        outcome=outcome,
        role=role,
        node_to_treatment=signature[0],
        treatment_to_node=signature[1],
        node_to_outcome=signature[2],
        outcome_to_node=signature[3],
    )
