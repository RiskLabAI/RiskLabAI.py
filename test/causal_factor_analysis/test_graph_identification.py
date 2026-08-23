"""Source-locked tests for clean graphical identification evidence."""

from __future__ import annotations

import ast
import itertools
import os
import subprocess
import sys
from pathlib import Path

import pytest

from RiskLabAI.causal_factor_analysis.graph_identification import (
    BackdoorAdjustmentEvidence,
    CausalDAG,
    DSeparationEvidence,
    FrontdoorAdjustmentEvidence,
    InstrumentEvidence,
    NodeRoleEvidence,
    PathEvidence,
    causal_role_evidence,
    check_backdoor_adjustment_set,
    check_frontdoor_adjustment_set,
    check_instrument,
    d_separation,
    minimal_backdoor_adjustment_sets,
    minimal_frontdoor_adjustment_sets,
)

PUBLIC_NAMES = {
    "BackdoorAdjustmentEvidence",
    "CausalDAG",
    "DSeparationEvidence",
    "FrontdoorAdjustmentEvidence",
    "InstrumentEvidence",
    "NodeRoleEvidence",
    "PathEvidence",
    "causal_role_evidence",
    "check_backdoor_adjustment_set",
    "check_frontdoor_adjustment_set",
    "check_instrument",
    "d_separation",
    "minimal_backdoor_adjustment_sets",
    "minimal_frontdoor_adjustment_sets",
}


def _dag(edges, *, observed=None, extra=()):
    nodes = sorted(set(extra) | {node for edge in edges for node in edge})
    if observed is None:
        observed = nodes
    return CausalDAG(nodes, edges, observed)


@pytest.mark.parametrize(
    "conditioned,separated",
    [((), False), (("U",), True)],
)
def test_fork_d_separation(conditioned, separated):
    graph = _dag((("U", "X"), ("U", "Y")))
    evidence = d_separation(graph, ("X",), ("Y",), conditioned)
    assert evidence.separated is separated
    assert evidence.paths[0].nodes == ("X", "U", "Y")


@pytest.mark.parametrize(
    "conditioned,separated",
    [((), False), (("M",), True)],
)
def test_chain_d_separation(conditioned, separated):
    graph = _dag((("X", "M"), ("M", "Y")))
    assert d_separation(graph, ("X",), ("Y",), conditioned).separated is separated


def test_collider_and_conditioned_descendant_reverse_path_status():
    graph = _dag((("X", "C"), ("Y", "C"), ("C", "D")))
    closed = d_separation(graph, ("X",), ("Y",))
    opened_at_collider = d_separation(graph, ("X",), ("Y",), ("C",))
    opened_at_descendant = d_separation(graph, ("X",), ("Y",), ("D",))

    assert closed.separated
    assert closed.paths[0].blocking_colliders == ("C",)
    assert not opened_at_collider.separated
    assert opened_at_collider.paths[0].activated_colliders == ("C",)
    assert not opened_at_descendant.separated
    assert opened_at_descendant.paths[0].activated_colliders == ("C",)


def test_collider_is_path_relative_not_a_global_node_label():
    graph = _dag((("A", "X"), ("A", "C"), ("B", "C"), ("B", "Y")))
    empty = d_separation(graph, ("X",), ("Y",))
    selected = d_separation(graph, ("X",), ("Y",), ("C",))
    roles = causal_role_evidence(graph, "X", "Y", "C")

    assert empty.separated
    assert not selected.separated
    assert roles.collider_on_paths == (("X", "A", "C", "B", "Y"),)
    assert roles.backdoor_noncollider_on_paths == ()


def test_d_separation_is_symmetric_with_oriented_path_evidence():
    graph = _dag((("U", "X"), ("U", "Y"), ("X", "M"), ("M", "Y")))
    forward = d_separation(graph, ("X",), ("Y",), ("U", "M"))
    reverse = d_separation(graph, ("Y",), ("X",), ("U", "M"))
    assert forward.separated == reverse.separated
    assert {path.nodes for path in forward.paths} == {
        tuple(reversed(path.nodes)) for path in reverse.paths
    }


@pytest.mark.parametrize(
    "edges,expected",
    [
        (("X", "Y"), ((),)),
        (("Y", "X"), ()),
        ((), ((),)),
    ],
)
def test_empty_or_missing_backdoor_sets_are_distinguished(edges, expected):
    graph = _dag(
        tuple((edges[index], edges[index + 1]) for index in range(0, len(edges), 2)),
        extra=("X", "Y"),
    )
    assert minimal_backdoor_adjustment_sets(graph, "X", "Y") == expected


@pytest.mark.parametrize(
    "observed,expected",
    [
        (("U", "X", "Y", "Z"), (("U",), ("Z",))),
        (("X", "Y", "Z"), (("Z",),)),
        (("U", "X", "Y"), (("U",),)),
        (("X", "Y"), ()),
    ],
)
def test_backdoor_search_respects_observability(observed, expected):
    graph = _dag(
        (("U", "Z"), ("Z", "X"), ("U", "Y"), ("X", "Y")),
        observed=observed,
    )
    assert minimal_backdoor_adjustment_sets(graph, "X", "Y") == expected


def test_backdoor_returns_all_inclusion_minimal_sets_not_only_smallest():
    graph = _dag(
        (
            ("A", "X"),
            ("A", "B"),
            ("B", "Y"),
            ("A", "C"),
            ("C", "Y"),
            ("X", "Y"),
        )
    )
    assert minimal_backdoor_adjustment_sets(graph, "X", "Y") == (
        ("A",),
        ("B", "C"),
    )


def test_indegree_two_node_can_be_the_required_backdoor_blocker():
    graph = _dag((("A", "C"), ("B", "C"), ("C", "X"), ("C", "Y"), ("X", "Y")))
    assert minimal_backdoor_adjustment_sets(graph, "X", "Y") == (("C",),)


def test_conditioning_on_a_collider_can_invalidate_an_empty_backdoor_set():
    graph = _dag((("A", "X"), ("A", "C"), ("B", "C"), ("B", "Y"), ("X", "Y")))
    assert check_backdoor_adjustment_set(graph, "X", "Y", ()).admissible
    selected = check_backdoor_adjustment_set(graph, "X", "Y", ("C",))
    assert not selected.admissible
    assert selected.open_backdoor_paths


def test_descendant_control_is_reported_and_rejected():
    graph = _dag((("X", "Y"), ("X", "D")))
    evidence = check_backdoor_adjustment_set(graph, "X", "Y", ("D",))
    assert evidence.descendants_in_set == ("D",)
    assert not evidence.admissible


def test_classic_frontdoor_set_is_graphically_admissible_but_needs_positivity():
    graph = _dag(
        (("U", "X"), ("U", "Y"), ("X", "M"), ("M", "Y")),
        observed=("X", "M", "Y"),
    )
    evidence = check_frontdoor_adjustment_set(graph, "X", "Y", ("M",))
    assert evidence.graphically_admissible
    assert evidence.positivity_required
    assert minimal_frontdoor_adjustment_sets(graph, "X", "Y") == (("M",),)
    assert minimal_backdoor_adjustment_sets(graph, "X", "Y") == ()


def test_frontdoor_checker_accepts_redundant_superset_but_search_removes_it():
    graph = _dag(
        (("U", "X"), ("U", "Y"), ("X", "M"), ("M", "Y")),
        observed=("X", "M", "N", "Y"),
        extra=("N",),
    )
    assert check_frontdoor_adjustment_set(
        graph, "X", "Y", ("M", "N")
    ).graphically_admissible
    assert minimal_frontdoor_adjustment_sets(graph, "X", "Y") == (("M",),)


@pytest.mark.parametrize(
    "extra_edge,failed_field",
    [
        (("X", "Y"), "unintercepted_directed_paths"),
        (("V", "M"), "open_treatment_mediator_backdoor_paths"),
        (("W", "Y"), "open_mediator_outcome_backdoor_paths_given_treatment"),
    ],
)
def test_each_frontdoor_criterion_fails_independently(extra_edge, failed_field):
    edges = [("U", "X"), ("U", "Y"), ("X", "M"), ("M", "Y")]
    if extra_edge == ("V", "M"):
        edges.append(("V", "X"))
    if extra_edge == ("W", "Y"):
        edges.append(("W", "M"))
    edges.append(extra_edge)
    observed = tuple(
        node for node in {item for edge in edges for item in edge} if node != "U"
    )
    graph = _dag(tuple(edges), observed=observed)
    evidence = check_frontdoor_adjustment_set(graph, "X", "Y", ("M",))
    assert not evidence.graphically_admissible
    assert getattr(evidence, failed_field)


def test_parallel_and_serial_mediator_sets_have_correct_minimality():
    parallel = _dag(
        (
            ("U", "X"),
            ("U", "Y"),
            ("X", "M1"),
            ("M1", "Y"),
            ("X", "M2"),
            ("M2", "Y"),
        ),
        observed=("X", "M1", "M2", "Y"),
    )
    serial = _dag(
        (("U", "X"), ("U", "Y"), ("X", "M1"), ("M1", "M2"), ("M2", "Y")),
        observed=("X", "M1", "M2", "Y"),
    )
    assert minimal_frontdoor_adjustment_sets(parallel, "X", "Y") == (("M1", "M2"),)
    assert minimal_frontdoor_adjustment_sets(serial, "X", "Y") == (
        ("M1",),
        ("M2",),
    )


def test_latent_or_irrelevant_frontdoor_set_is_not_returned():
    hidden = _dag(
        (("U", "X"), ("U", "Y"), ("X", "M"), ("M", "Y")),
        observed=("X", "Y"),
    )
    unrelated = _dag((), observed=("X", "Y", "M"), extra=("X", "Y", "M"))
    assert not check_frontdoor_adjustment_set(
        hidden, "X", "Y", ("M",)
    ).graphically_admissible
    assert minimal_frontdoor_adjustment_sets(hidden, "X", "Y") == ()
    assert minimal_frontdoor_adjustment_sets(unrelated, "X", "Y") == ()


def test_pearl_and_cfi_instrument_profiles_are_not_collapsed():
    graph = _dag((("U", "Z"), ("U", "X"), ("X", "Y")))
    evidence = check_instrument(graph, "Z", "X", "Y")
    assert evidence.pearl_relevance
    assert evidence.pearl_exclusion_exogeneity
    assert evidence.pearl_graphical
    assert not evidence.cfi_direct_relevance
    assert not evidence.cfi_simple


def test_simple_direct_instrument_satisfies_both_profiles():
    graph = _dag((("Z", "X"), ("X", "Y")))
    evidence = check_instrument(graph, "Z", "X", "Y")
    assert evidence.pearl_graphical
    assert evidence.cfi_direct_relevance
    assert evidence.cfi_full_mediation
    assert evidence.cfi_exogeneity
    assert evidence.cfi_simple


def test_cfi_simple_profile_is_unconditional_when_pearl_controls_are_supplied():
    graph = _dag((("Z", "X"), ("X", "Y")), extra=("W",))
    evidence = check_instrument(graph, "Z", "X", "Y", ("W",))
    assert evidence.pearl_graphical
    assert evidence.cfi_simple


def test_conditional_instrument_blocks_observed_common_cause_for_pearl_only():
    graph = _dag((("W", "Z"), ("W", "Y"), ("Z", "X"), ("X", "Y")))
    unconditional = check_instrument(graph, "Z", "X", "Y")
    conditional = check_instrument(graph, "Z", "X", "Y", ("W",))
    assert not unconditional.pearl_graphical
    assert conditional.pearl_graphical
    assert not conditional.cfi_simple


@pytest.mark.parametrize(
    "edges,conditioned,failed",
    [
        ((("Z", "X"), ("X", "Y"), ("Z", "Y")), (), "exclusion"),
        ((("U", "Z"), ("U", "Y"), ("Z", "X"), ("X", "Y")), (), "exogeneity"),
        (
            (("Z", "C"), ("U", "C"), ("U", "Y"), ("Z", "X"), ("X", "Y")),
            ("C",),
            "collider",
        ),
    ],
)
def test_instrument_exclusion_and_exogeneity_failures(edges, conditioned, failed):
    graph = _dag(edges)
    evidence = check_instrument(graph, "Z", "X", "Y", conditioned)
    assert not evidence.pearl_graphical, failed


def test_treatment_descendant_conditioning_invalidates_pearl_profile():
    graph = _dag((("Z", "X"), ("X", "Y"), ("X", "D")))
    evidence = check_instrument(graph, "Z", "X", "Y", ("D",))
    assert not evidence.conditioning_unaffected_by_treatment
    assert not evidence.pearl_graphical


def test_post_treatment_proxy_is_not_a_pearl_instrument():
    graph = _dag(
        (("U", "X"), ("U", "Y"), ("X", "Y"), ("X", "Z")),
        observed=("X", "Y", "Z"),
    )
    evidence = check_instrument(graph, "Z", "X", "Y")
    assert evidence.pearl_relevance
    assert not evidence.pearl_exclusion_exogeneity
    assert not evidence.pearl_graphical


def test_node_role_evidence_is_nonexclusive_and_path_witnessed():
    graph = _dag(
        (
            ("X", "M"),
            ("M", "Y"),
            ("A", "M"),
            ("A", "X"),
            ("B", "M"),
            ("B", "Y"),
        )
    )
    evidence = causal_role_evidence(graph, "X", "Y", "M")
    assert evidence.mediator_on_directed_paths
    assert evidence.collider_on_paths
    assert evidence.noncollider_on_paths
    assert evidence.descendant_of_treatment
    assert evidence.ancestor_of_outcome


def test_common_cause_and_remote_backdoor_witnesses_are_distinct():
    graph = _dag((("U", "Z"), ("Z", "X"), ("U", "Y"), ("X", "Y")))
    upstream = causal_role_evidence(graph, "X", "Y", "U")
    downstream = causal_role_evidence(graph, "X", "Y", "Z")
    assert upstream.common_cause_paths
    assert upstream.backdoor_noncollider_on_paths
    assert downstream.backdoor_noncollider_on_paths
    assert downstream.common_cause_paths == ()


def test_common_cause_witness_requires_a_pathwise_directed_fork():
    graph = _dag((("U", "X"), ("U", "Y"), ("X", "D"), ("U", "A"), ("A", "D")))
    evidence = causal_role_evidence(graph, "X", "Y", "U")
    assert evidence.common_cause_paths == (("X", "U", "Y"),)
    assert ("X", "D", "A", "U", "Y") in evidence.noncollider_on_paths


def test_causal_dag_normalizes_without_mutating_callers_and_is_frozen():
    nodes = ["Y", "X", "U"]
    edges = [["U", "Y"], ["U", "X"]]
    observed = ["Y", "X"]
    graph = CausalDAG(nodes, edges, observed)
    nodes.append("LATER")
    edges[0][0] = "CHANGED"
    observed.clear()
    assert graph.nodes == ("U", "X", "Y")
    assert graph.directed_edges == (("U", "X"), ("U", "Y"))
    assert graph.observed_nodes == ("X", "Y")
    assert not hasattr(graph, "__dict__")
    with pytest.raises(AttributeError):
        graph.nodes = ()


def test_causal_dag_namedtuple_helpers_cannot_bypass_validation():
    graph = _dag((("X", "Y"),))
    with pytest.raises(ValueError):
        graph._replace(nodes=("ONLY",))
    with pytest.raises(ValueError):
        CausalDAG._make((("X", "Y"), (("X", "Y"), ("Y", "X")), ("X", "Y")))
    with pytest.raises(TypeError):
        CausalDAG._make((graph.nodes, graph.directed_edges))


@pytest.mark.parametrize(
    "args,error",
    [
        (("XYZ", (), ()), TypeError),
        ((("X", 1), (), ("X",)), TypeError),
        ((("X", ""), (), ("X",)), ValueError),
        ((("X", "X"), (), ("X",)), ValueError),
        ((("X", "Y"), (("X",),), ("X", "Y")), ValueError),
        ((("X", "Y"), ({"X", "Y"},), ("X", "Y")), TypeError),
        ((("X", "Y"), (("X", "Z"),), ("X", "Y")), ValueError),
        ((("X", "Y"), (("X", "X"),), ("X", "Y")), ValueError),
        ((("X", "Y"), (("X", "Y"), ("Y", "X")), ("X", "Y")), ValueError),
    ],
)
def test_malformed_dag_fails_closed(args, error):
    with pytest.raises(error):
        CausalDAG(*args)


def test_query_validation_fails_closed_without_scalar_string_iteration():
    graph = _dag((("X", "Y"),), extra=("Z",))
    with pytest.raises(TypeError):
        d_separation(graph, "X", ("Y",))
    with pytest.raises(ValueError):
        d_separation(graph, ("X",), ("Y",), ("UNKNOWN",))
    with pytest.raises(ValueError):
        d_separation(graph, ("X",), ("Y",), ("X",))
    with pytest.raises(ValueError):
        check_backdoor_adjustment_set(graph, "X", "X")
    with pytest.raises(ValueError):
        check_instrument(graph, "X", "X", "Y")


def test_known_latent_and_endpoint_sets_are_substantive_negatives():
    graph = _dag(
        (("U", "X"), ("U", "Y"), ("X", "M"), ("M", "Y")), observed=("X", "Y", "M")
    )
    latent = check_backdoor_adjustment_set(graph, "X", "Y", ("U",))
    endpoint = check_backdoor_adjustment_set(graph, "X", "Y", ("X",))
    frontdoor = check_frontdoor_adjustment_set(graph, "X", "Y", ("U",))
    assert not latent.all_observed and not latent.admissible
    assert endpoint.endpoints_in_set == ("X",) and not endpoint.admissible
    assert not frontdoor.all_observed and not frontdoor.graphically_admissible


def _oracle_children(nodes, edges):
    result = {node: set() for node in nodes}
    for source, target in edges:
        result[source].add(target)
    return result


def _oracle_descendants(nodes, edges, start):
    children = _oracle_children(nodes, edges)
    result = set()
    pending = list(children[start])
    while pending:
        node = pending.pop()
        if node not in result:
            result.add(node)
            pending.extend(children[node])
    return result


def _oracle_simple_paths(nodes, edges, start, finish):
    adjacency = {node: set() for node in nodes}
    for source, target in edges:
        adjacency[source].add(target)
        adjacency[target].add(source)
    result = []

    def visit(node, path):
        if node == finish:
            result.append(tuple(path))
            return
        for neighbor in sorted(adjacency[node]):
            if neighbor not in path:
                visit(neighbor, path + [neighbor])

    visit(start, [start])
    return result


def _oracle_path_open(nodes, edges, path, conditioned):
    edge_set = set(edges)
    conditioned = set(conditioned)
    for index in range(1, len(path) - 1):
        previous, node, following = path[index - 1 : index + 2]
        collider = (previous, node) in edge_set and (following, node) in edge_set
        if collider:
            family = {node} | _oracle_descendants(nodes, edges, node)
            if not family.intersection(conditioned):
                return False
        elif node in conditioned:
            return False
    return True


def _oracle_d_separated(nodes, edges, left, right, conditioned):
    return not any(
        _oracle_path_open(nodes, edges, path, conditioned)
        for path in _oracle_simple_paths(nodes, edges, left, right)
    )


def _oracle_directed_paths(nodes, edges, start, finish):
    edge_set = set(edges)
    return tuple(
        sorted(
            (
                path
                for path in _oracle_simple_paths(nodes, edges, start, finish)
                if all(
                    (path[index], path[index + 1]) in edge_set
                    for index in range(len(path) - 1)
                )
            ),
            key=lambda path: (len(path), path),
        )
    )


def _oracle_frontdoor_admissible(nodes, edges, treatment, outcome, mediators):
    mediators = tuple(sorted(mediators))
    directed = _oracle_directed_paths(nodes, edges, treatment, outcome)
    if not mediators or not directed:
        return False
    if any(not set(path[1:-1]).intersection(mediators) for path in directed):
        return False
    treatment_graph = tuple(edge for edge in edges if edge[0] != treatment)
    if any(
        not _oracle_d_separated(nodes, treatment_graph, treatment, node, ())
        for node in mediators
    ):
        return False
    mediator_graph = tuple(edge for edge in edges if edge[0] not in mediators)
    return all(
        _oracle_d_separated(nodes, mediator_graph, node, outcome, (treatment,))
        for node in mediators
    )


def _oracle_role(nodes, edges, treatment, outcome, node):
    edge_set = set(edges)
    paths = tuple(
        sorted(
            _oracle_simple_paths(nodes, edges, treatment, outcome),
            key=lambda path: (len(path), path),
        )
    )
    colliders = []
    noncolliders = []
    mediators = []
    backdoors = []
    common_causes = []
    for path in paths:
        index = path.index(node) if node in path[1:-1] else None
        if index is None:
            continue
        previous, following = path[index - 1], path[index + 1]
        is_collider = (previous, node) in edge_set and (following, node) in edge_set
        if is_collider:
            colliders.append(path)
        else:
            noncolliders.append(path)
        if all(
            (path[position], path[position + 1]) in edge_set
            for position in range(len(path) - 1)
        ):
            mediators.append(path)
        if (path[1], treatment) in edge_set and not is_collider:
            backdoors.append(path)
        directed_to_treatment = all(
            (path[position], path[position - 1]) in edge_set
            for position in range(index, 0, -1)
        )
        directed_to_outcome = all(
            (path[position], path[position + 1]) in edge_set
            for position in range(index, len(path) - 1)
        )
        if not is_collider and directed_to_treatment and directed_to_outcome:
            common_causes.append(path)
    return {
        "collider_on_paths": tuple(colliders),
        "noncollider_on_paths": tuple(noncolliders),
        "mediator_on_directed_paths": tuple(mediators),
        "backdoor_noncollider_on_paths": tuple(backdoors),
        "common_cause_paths": tuple(common_causes),
        "descendant_of_treatment": node in _oracle_descendants(nodes, edges, treatment),
        "ancestor_of_outcome": outcome in _oracle_descendants(nodes, edges, node),
    }


def _oracle_is_acyclic(nodes, edges):
    descendants = _oracle_children(nodes, edges)
    indegree = {node: 0 for node in nodes}
    for _, target in edges:
        indegree[target] += 1
    pending = [node for node in nodes if indegree[node] == 0]
    visited = 0
    while pending:
        node = pending.pop()
        visited += 1
        for child in descendants[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                pending.append(child)
    return visited == len(nodes)


def _all_four_node_dags():
    nodes = ("A", "B", "C", "D")
    pairs = tuple(itertools.combinations(nodes, 2))
    for states in itertools.product((0, 1, 2), repeat=len(pairs)):
        edges = []
        for state, (first, second) in zip(states, pairs):
            if state == 1:
                edges.append((first, second))
            elif state == 2:
                edges.append((second, first))
        edges = tuple(edges)
        if _oracle_is_acyclic(nodes, edges):
            yield nodes, edges


def test_all_543_four_node_dags_match_independent_d_separation_oracle():
    count = 0
    for nodes, edges in _all_four_node_dags():
        count += 1
        graph = CausalDAG(nodes, tuple(reversed(edges)), tuple(reversed(nodes)))
        for left, right in itertools.permutations(nodes, 2):
            remaining = tuple(node for node in nodes if node not in {left, right})
            for size in range(len(remaining) + 1):
                for conditioned in itertools.combinations(remaining, size):
                    expected = _oracle_d_separated(
                        nodes, edges, left, right, conditioned
                    )
                    actual = d_separation(
                        graph, (left,), (right,), conditioned
                    ).separated
                    assert actual is expected
    assert count == 543


def test_all_four_node_backdoor_sets_match_independent_powerset_oracle():
    for nodes, edges in _all_four_node_dags():
        graph = CausalDAG(nodes, edges, nodes)
        for treatment, outcome in itertools.permutations(nodes, 2):
            descendants = _oracle_descendants(nodes, edges, treatment)
            candidates = tuple(
                node
                for node in nodes
                if node not in {treatment, outcome} and node not in descendants
            )
            mutilated = tuple(edge for edge in edges if edge[0] != treatment)
            valid = []
            for size in range(len(candidates) + 1):
                for candidate in itertools.combinations(candidates, size):
                    if _oracle_d_separated(
                        nodes, mutilated, treatment, outcome, candidate
                    ):
                        valid.append(candidate)
            expected = tuple(
                candidate
                for candidate in sorted(valid, key=lambda item: (len(item), item))
                if not any(set(other) < set(candidate) for other in valid)
            )
            assert (
                minimal_backdoor_adjustment_sets(graph, treatment, outcome) == expected
            )


def test_all_four_node_frontdoor_sets_match_independent_oracle():
    queries = 0
    for nodes, edges in _all_four_node_dags():
        graph = CausalDAG(nodes, tuple(reversed(edges)), tuple(reversed(nodes)))
        for treatment, outcome in itertools.permutations(nodes, 2):
            queries += 1
            candidates = tuple(
                node for node in nodes if node not in {treatment, outcome}
            )
            valid = []
            for size in range(1, len(candidates) + 1):
                for mediators in itertools.combinations(candidates, size):
                    expected = _oracle_frontdoor_admissible(
                        nodes, edges, treatment, outcome, mediators
                    )
                    actual = check_frontdoor_adjustment_set(
                        graph, treatment, outcome, mediators
                    ).graphically_admissible
                    assert actual is expected
                    if expected:
                        valid.append(mediators)
            expected_minimal = tuple(
                candidate
                for candidate in sorted(valid, key=lambda item: (len(item), item))
                if not any(set(other) < set(candidate) for other in valid)
            )
            assert (
                minimal_frontdoor_adjustment_sets(graph, treatment, outcome)
                == expected_minimal
            )
    assert queries == 6_516


def test_all_four_node_instrument_profiles_match_independent_oracles():
    assessments = 0
    for nodes, edges in _all_four_node_dags():
        graph = CausalDAG(nodes, tuple(reversed(edges)), tuple(reversed(nodes)))
        edge_set = set(edges)
        for treatment, outcome in itertools.permutations(nodes, 2):
            for instrument in (
                node for node in nodes if node not in {treatment, outcome}
            ):
                control = next(
                    node
                    for node in nodes
                    if node not in {instrument, treatment, outcome}
                )
                for conditioned in ((), (control,)):
                    assessments += 1
                    unaffected = not set(conditioned).intersection(
                        _oracle_descendants(nodes, edges, treatment)
                    )
                    pearl_relevance = not _oracle_d_separated(
                        nodes, edges, instrument, treatment, conditioned
                    )
                    intervention_graph = tuple(
                        edge for edge in edges if edge[1] != treatment
                    )
                    pearl_exclusion = _oracle_d_separated(
                        nodes,
                        intervention_graph,
                        instrument,
                        outcome,
                        conditioned,
                    )
                    directed = _oracle_directed_paths(nodes, edges, instrument, outcome)
                    cfi_direct = (instrument, treatment) in edge_set
                    cfi_full_mediation = bool(directed) and all(
                        treatment in path[1:-1] for path in directed
                    )
                    instrument_graph = tuple(
                        edge for edge in edges if edge[0] != instrument
                    )
                    cfi_exogeneity = _oracle_d_separated(
                        nodes, instrument_graph, instrument, outcome, ()
                    )
                    evidence = check_instrument(
                        graph,
                        instrument,
                        treatment,
                        outcome,
                        conditioned,
                    )
                    assert evidence.conditioning_unaffected_by_treatment is unaffected
                    assert evidence.pearl_relevance is pearl_relevance
                    assert evidence.pearl_exclusion_exogeneity is pearl_exclusion
                    assert evidence.pearl_graphical is (
                        unaffected and pearl_relevance and pearl_exclusion
                    )
                    assert evidence.cfi_direct_relevance is cfi_direct
                    assert evidence.cfi_full_mediation is cfi_full_mediation
                    assert evidence.cfi_exogeneity is cfi_exogeneity
                    assert evidence.cfi_simple is (
                        cfi_direct and cfi_full_mediation and cfi_exogeneity
                    )
    assert assessments == 26_064


def test_all_four_node_role_evidence_matches_path_oracle():
    assessments = 0
    for nodes, edges in _all_four_node_dags():
        graph = CausalDAG(nodes, tuple(reversed(edges)), tuple(reversed(nodes)))
        for treatment, outcome in itertools.permutations(nodes, 2):
            for node in nodes:
                if node in {treatment, outcome}:
                    continue
                assessments += 1
                expected = _oracle_role(nodes, edges, treatment, outcome, node)
                actual = causal_role_evidence(graph, treatment, outcome, node)
                for field, value in expected.items():
                    assert getattr(actual, field) == value
    assert assessments == 13_032


def test_complete_path_evidence_fails_closed_at_explicit_budget():
    graph = _dag((("X", "A"), ("A", "Y"), ("X", "B"), ("B", "Y")))
    assert len(d_separation(graph, ("X",), ("Y",), max_paths=2).paths) == 2
    with pytest.raises(ValueError, match="max_paths"):
        d_separation(graph, ("X",), ("Y",), max_paths=1)
    with pytest.raises(ValueError, match="max_paths"):
        causal_role_evidence(graph, "X", "Y", "A", max_paths=1)


def test_path_state_budget_bounds_dense_dead_end_search():
    lobe = tuple(f"A{index:02d}" for index in range(6))
    edges = [("X", "Y")]
    edges.extend(("X", node) for node in lobe)
    edges.extend(
        (left, right) for index, left in enumerate(lobe) for right in lobe[index + 1 :]
    )
    graph = _dag(tuple(edges))
    with pytest.raises(ValueError, match="max_path_states"):
        d_separation(
            graph,
            ("X",),
            ("Y",),
            max_paths=1,
            max_path_states=50,
        )
    evidence = check_frontdoor_adjustment_set(
        graph,
        "X",
        "Y",
        (),
        max_paths=1,
        max_path_states=2,
    )
    assert not evidence.graphically_admissible


def test_minimal_search_budget_fails_closed_without_partial_results():
    graph = _dag((("Y", "X"),), extra=("A", "B"))
    with pytest.raises(ValueError, match="max_candidates"):
        minimal_backdoor_adjustment_sets(graph, "X", "Y", max_candidates=1)

    frontdoor_graph = _dag((("X", "Y"),), extra=("A", "B"))
    with pytest.raises(ValueError, match="max_candidates"):
        minimal_frontdoor_adjustment_sets(frontdoor_graph, "X", "Y", max_candidates=1)


def test_large_candidate_space_hits_budget_without_recursion_failure():
    irrelevant = tuple(f"N{index:04d}" for index in range(1_100))
    graph = _dag((("Y", "X"),), extra=irrelevant)
    with pytest.raises(ValueError, match="max_candidates"):
        minimal_backdoor_adjustment_sets(graph, "X", "Y", max_candidates=2_000)


def test_minimal_search_prunes_supersets_of_many_singleton_solutions():
    chain = tuple(f"A{index:02d}" for index in range(20))
    edges = [(chain[index], chain[index - 1]) for index in range(1, 20)]
    edges.extend(((chain[0], "X"), (chain[-1], "Y")))
    graph = _dag(tuple(edges))
    assert minimal_backdoor_adjustment_sets(
        graph, "X", "Y", max_candidates=100
    ) == tuple((node,) for node in chain)


def test_empty_backdoor_set_short_circuits_candidate_search_exactly():
    irrelevant = tuple(f"N{index:02d}" for index in range(20))
    graph = _dag((("X", "Y"),), extra=irrelevant)
    assert minimal_backdoor_adjustment_sets(graph, "X", "Y", max_candidates=1) == ((),)


@pytest.mark.parametrize("value,error", [(True, TypeError), (0, ValueError)])
def test_resource_budgets_require_positive_integers_or_none(value, error):
    graph = _dag((("X", "Y"),), extra=("Z",))
    with pytest.raises(error):
        d_separation(graph, ("X",), ("Y",), max_paths=value)
    with pytest.raises(error):
        d_separation(graph, ("X",), ("Y",), max_path_states=value)
    with pytest.raises(error):
        minimal_backdoor_adjustment_sets(graph, "X", "Y", max_candidates=value)


def test_protocol_stage_two_graph_tuples_construct_the_same_dag():
    from RiskLabAI.causal_factor_analysis.protocol import CausalDiscoveryStage

    stage = CausalDiscoveryStage(
        method_labels=("PC",),
        graph_id="resolved-dag-v1",
        graph_nodes=("Y", "X", "U"),
        directed_edges=(("U", "Y"), ("X", "Y"), ("U", "X")),
        graph_kind="DAG",
        ambiguous_edges=(),
        assumptions=("Graph accepted after external causal review.",),
    )
    graph = CausalDAG(
        stage.graph_nodes,
        stage.directed_edges,
        stage.graph_nodes,
    )
    assert graph.nodes == ("U", "X", "Y")
    assert minimal_backdoor_adjustment_sets(graph, "X", "Y") == (("U",),)


def test_public_records_and_exports_are_exact_and_frozen():
    import RiskLabAI.causal_factor_analysis.graph_identification as module

    assert set(module.__all__) == PUBLIC_NAMES
    graph = _dag((("X", "Y"),), extra=("Z",))
    records = (
        d_separation(graph, ("X",), ("Y",)),
        check_backdoor_adjustment_set(graph, "X", "Y"),
        check_frontdoor_adjustment_set(graph, "X", "Y", ("Z",)),
        check_instrument(_dag((("Z", "X"), ("X", "Y"))), "Z", "X", "Y"),
        causal_role_evidence(_dag((("X", "Z"), ("Z", "Y"))), "X", "Y", "Z"),
    )
    assert isinstance(records[0], DSeparationEvidence)
    assert isinstance(records[0].paths[0], PathEvidence)
    assert isinstance(records[1], BackdoorAdjustmentEvidence)
    assert isinstance(records[2], FrontdoorAdjustmentEvidence)
    assert isinstance(records[3], InstrumentEvidence)
    assert isinstance(records[4], NodeRoleEvidence)
    for record in records:
        assert not hasattr(record, "__dict__")
    with pytest.raises(AttributeError):
        records[0].separated = False


def test_module_is_python39_and_standard_library_only():
    module_path = (
        Path(__file__).parents[2]
        / "src"
        / "RiskLabAI"
        / "causal_factor_analysis"
        / "graph_identification.py"
    )
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source, feature_version=(3, 9))
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots <= {
        "__future__",
        "collections",
        "dataclasses",
        "itertools",
        "typing",
    }


def test_package_import_does_not_load_legacy_or_private_namespaces():
    project = Path(__file__).parents[2]
    public_names = repr(sorted(PUBLIC_NAMES))
    program = f"""
import sys
import RiskLabAI.causal_factor_analysis as package
import RiskLabAI.causal_factor_analysis.graph_identification as module
expected = set({public_names})
assert expected <= set(package.__all__)
assert expected == set(module.__all__)
blocked = (
    'RiskLabAI.causal',
    'RiskLabAI.backtest',
    'RiskLabAI.optimization',
    'RiskLabAI.risk_control',
)
assert not any(name == root or name.startswith(root + '.') for name in sys.modules for root in blocked)
"""
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(
        [sys.executable, "-B", "-c", program],
        cwd=project,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
