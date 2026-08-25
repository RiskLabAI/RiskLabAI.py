# ruff: noqa: UP045
"""Graphical identification evidence for causal-factor research.

The routines in this module operate on an explicit, immutable directed acyclic
graph.  They implement path activation, d-separation, the back-door and
front-door criteria, and two distinct graphical instrument profiles.  The
results describe implications of an accepted graph; they do not discover a
graph, verify its causal assumptions, establish positivity, or estimate an
effect.
"""

from __future__ import annotations

from collections import namedtuple
from collections.abc import Callable, Iterable, Sequence
from typing import NamedTuple, Optional

__all__ = [
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
]


NodeTuple = tuple[str, ...]
EdgeTuple = tuple[tuple[str, str], ...]
PathTuple = tuple[str, ...]
_DEFAULT_MAX_PATHS = 10_000
_DEFAULT_MAX_PATH_STATES = 100_000
_DEFAULT_MAX_CANDIDATES = 65_536


def _materialize_collection(value, subject: str) -> tuple:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise TypeError(f"{subject} must be a collection, not a scalar string.")
    try:
        return tuple(value)
    except TypeError as exc:
        raise TypeError(f"{subject} must be an iterable collection.") from exc


def _require_node(value, subject: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{subject} must be a string node identifier.")
    if not value.strip():
        raise ValueError(f"{subject} must be nonblank.")
    return value


def _normalize_nodes(value, subject: str, *, allow_empty: bool) -> NodeTuple:
    items = _materialize_collection(value, subject)
    normalized = tuple(_require_node(item, f"{subject} member") for item in items)
    if not allow_empty and not normalized:
        raise ValueError(f"{subject} must be nonempty.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{subject} must not contain duplicate nodes.")
    return tuple(sorted(normalized))


def _normalize_edges(value) -> EdgeTuple:
    items = _materialize_collection(value, "directed_edges")
    normalized = []
    for raw_edge in items:
        if isinstance(raw_edge, (str, bytes)) or not isinstance(raw_edge, Sequence):
            raise TypeError("Each directed edge must be an ordered two-node sequence.")
        edge = _materialize_collection(raw_edge, "each directed edge")
        if len(edge) != 2:
            raise ValueError("Each directed edge must contain exactly two nodes.")
        source = _require_node(edge[0], "directed-edge source")
        target = _require_node(edge[1], "directed-edge target")
        normalized.append((source, target))
    result = tuple(normalized)
    if len(set(result)) != len(result):
        raise ValueError("directed_edges must not contain duplicate edges.")
    return tuple(sorted(result))


def _children(nodes: NodeTuple, edges: EdgeTuple):
    result = {node: set() for node in nodes}
    for source, target in edges:
        result[source].add(target)
    return result


def _parents(nodes: NodeTuple, edges: EdgeTuple):
    result = {node: set() for node in nodes}
    for source, target in edges:
        result[target].add(source)
    return result


def _is_acyclic(nodes: NodeTuple, edges: EdgeTuple) -> bool:
    children = _children(nodes, edges)
    indegree = {node: 0 for node in nodes}
    for _, target in edges:
        indegree[target] += 1
    ready = sorted(node for node, degree in indegree.items() if degree == 0)
    visited = 0
    while ready:
        node = ready.pop()
        visited += 1
        for child in sorted(children[node], reverse=True):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    return visited == len(nodes)


_CausalDAGBase = namedtuple("CausalDAG", ("nodes", "directed_edges", "observed_nodes"))


class CausalDAG(_CausalDAGBase):
    """An immutable DAG with explicit observed and latent nodes.

    Nodes absent from ``observed_nodes`` remain in every graphical calculation
    but are never eligible for an adjustment or mediator set.
    """

    __slots__ = ()

    def __new__(cls, nodes, directed_edges, observed_nodes):
        nodes = _normalize_nodes(nodes, "nodes", allow_empty=False)
        edges = _normalize_edges(directed_edges)
        observed = _normalize_nodes(observed_nodes, "observed_nodes", allow_empty=True)
        node_set = set(nodes)
        unknown_observed = set(observed) - node_set
        if unknown_observed:
            raise ValueError("observed_nodes must be a subset of nodes.")
        for source, target in edges:
            if source not in node_set or target not in node_set:
                raise ValueError("Every directed edge endpoint must be a graph node.")
            if source == target:
                raise ValueError("Directed self-loops are not admissible in a DAG.")
        if not _is_acyclic(nodes, edges):
            raise ValueError("directed_edges must define an acyclic graph.")
        return super().__new__(cls, nodes, edges, observed)

    @classmethod
    def _make(cls, iterable):
        values = tuple(iterable)
        if len(values) != 3:
            raise TypeError("CausalDAG._make requires exactly three values.")
        return cls(*values)

    def _replace(self, **changes):
        unknown = set(changes) - set(self._fields)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise TypeError(f"Unexpected CausalDAG field names: {names}.")
        return type(self)(
            changes.get("nodes", self.nodes),
            changes.get("directed_edges", self.directed_edges),
            changes.get("observed_nodes", self.observed_nodes),
        )


class PathEvidence(NamedTuple):
    """Path-relative collider, blocking, and direction evidence."""

    nodes: PathTuple
    colliders: NodeTuple
    noncolliders: NodeTuple
    conditioned_noncolliders: NodeTuple
    activated_colliders: NodeTuple
    blocking_colliders: NodeTuple
    is_open: bool
    is_directed_from_left: bool
    is_backdoor_from_left: bool


class DSeparationEvidence(NamedTuple):
    """Complete simple-path evidence for one d-separation query."""

    left: NodeTuple
    right: NodeTuple
    conditioned: NodeTuple
    separated: bool
    paths: tuple[PathEvidence, ...]


class BackdoorAdjustmentEvidence(NamedTuple):
    """Evidence for the ordinary total-effect back-door criterion."""

    treatment: str
    outcome: str
    adjustment_set: NodeTuple
    all_observed: bool
    endpoints_in_set: NodeTuple
    descendants_in_set: NodeTuple
    open_backdoor_paths: tuple[PathEvidence, ...]
    admissible: bool


class FrontdoorAdjustmentEvidence(NamedTuple):
    """Graphical front-door evidence; positivity remains an external premise."""

    treatment: str
    outcome: str
    mediator_set: NodeTuple
    all_observed: bool
    endpoints_in_set: NodeTuple
    directed_path_exists: bool
    unintercepted_directed_paths: tuple[PathEvidence, ...]
    open_treatment_mediator_backdoor_paths: tuple[PathEvidence, ...]
    open_mediator_outcome_backdoor_paths_given_treatment: tuple[PathEvidence, ...]
    graphically_admissible: bool
    positivity_required: bool


class InstrumentEvidence(NamedTuple):
    """Separate Pearl-graphical and simple CFI instrument premises."""

    instrument: str
    treatment: str
    outcome: str
    conditioned: NodeTuple
    all_observed: bool
    conditioning_disjoint: bool
    conditioning_unaffected_by_treatment: bool
    pearl_relevance: bool
    pearl_exclusion_exogeneity: bool
    pearl_graphical: bool
    cfi_direct_relevance: bool
    cfi_full_mediation: bool
    cfi_exogeneity: bool
    cfi_simple: bool


class NodeRoleEvidence(NamedTuple):
    """Non-exclusive, path-relative evidence for one internal graph node."""

    node: str
    treatment: str
    outcome: str
    collider_on_paths: tuple[PathTuple, ...]
    noncollider_on_paths: tuple[PathTuple, ...]
    mediator_on_directed_paths: tuple[PathTuple, ...]
    backdoor_noncollider_on_paths: tuple[PathTuple, ...]
    common_cause_paths: tuple[PathTuple, ...]
    descendant_of_treatment: bool
    ancestor_of_outcome: bool


def _validate_dag(dag) -> CausalDAG:
    if not isinstance(dag, CausalDAG):
        raise TypeError("dag must be a CausalDAG.")
    return dag


def _known_node_set(dag: CausalDAG, value, subject: str, *, allow_empty: bool):
    nodes = _normalize_nodes(value, subject, allow_empty=allow_empty)
    if not set(nodes).issubset(dag.nodes):
        raise ValueError(f"{subject} must contain only known graph nodes.")
    return nodes


def _validate_endpoints(dag: CausalDAG, treatment, outcome) -> tuple[str, str]:
    treatment = _require_node(treatment, "treatment")
    outcome = _require_node(outcome, "outcome")
    if treatment == outcome:
        raise ValueError("treatment and outcome must be distinct nodes.")
    if treatment not in dag.nodes or outcome not in dag.nodes:
        raise ValueError("treatment and outcome must be known graph nodes.")
    observed = set(dag.observed_nodes)
    if treatment not in observed or outcome not in observed:
        raise ValueError("treatment and outcome must both be observed nodes.")
    return treatment, outcome


def _descendants(dag: CausalDAG, node: str, edges: Optional[EdgeTuple] = None):
    active_edges = dag.directed_edges if edges is None else edges
    children = _children(dag.nodes, active_edges)
    visited = set()
    pending = list(children[node])
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        pending.extend(children[current])
    return visited


def _ancestors(dag: CausalDAG, node: str):
    parents = _parents(dag.nodes, dag.directed_edges)
    visited = set()
    pending = list(parents[node])
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        pending.extend(parents[current])
    return visited


def _normalize_limit(value, subject: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{subject} must be a positive integer or None.")
    if value < 1:
        raise ValueError(f"{subject} must be positive.")
    return value


def _has_directed_path(
    dag: CausalDAG,
    left: str,
    right: str,
    edges: Optional[EdgeTuple] = None,
    blocked: Iterable[str] = (),
) -> bool:
    active_edges = dag.directed_edges if edges is None else edges
    blocked_nodes = set(blocked)
    if left in blocked_nodes or right in blocked_nodes:
        return False
    children = _children(dag.nodes, active_edges)
    pending = [left]
    visited = set()
    while pending:
        node = pending.pop()
        if node == right:
            return True
        if node in visited:
            continue
        visited.add(node)
        pending.extend(
            child
            for child in children[node]
            if child not in visited and child not in blocked_nodes
        )
    return False


def _is_d_separated(
    dag: CausalDAG,
    left: NodeTuple,
    right: NodeTuple,
    conditioned: NodeTuple,
    edges: Optional[EdgeTuple] = None,
) -> bool:
    """Test d-separation through ancestral-subgraph moralization."""

    active_edges = dag.directed_edges if edges is None else edges
    parents = _parents(dag.nodes, active_edges)
    relevant = set(left).union(right, conditioned)
    pending = list(relevant)
    while pending:
        node = pending.pop()
        for parent in parents[node]:
            if parent not in relevant:
                relevant.add(parent)
                pending.append(parent)

    moral = {node: set() for node in relevant}
    for source, target in active_edges:
        if source in relevant and target in relevant:
            moral[source].add(target)
            moral[target].add(source)
    for child in relevant:
        child_parents = sorted(
            parent for parent in parents[child] if parent in relevant
        )
        for index, first in enumerate(child_parents):
            for second in child_parents[index + 1 :]:
                moral[first].add(second)
                moral[second].add(first)

    blocked = set(conditioned)
    targets = set(right)
    reachable = [node for node in left if node not in blocked]
    visited = set()
    while reachable:
        node = reachable.pop()
        if node in targets:
            return False
        if node in visited:
            continue
        visited.add(node)
        reachable.extend(
            neighbor
            for neighbor in moral[node]
            if neighbor not in blocked and neighbor not in visited
        )
    return True


def _without_outgoing(edges: EdgeTuple, nodes) -> EdgeTuple:
    removed = set(nodes)
    return tuple(edge for edge in edges if edge[0] not in removed)


def _without_incoming(edges: EdgeTuple, nodes) -> EdgeTuple:
    removed = set(nodes)
    return tuple(edge for edge in edges if edge[1] not in removed)


def _simple_paths(
    dag: CausalDAG,
    left: NodeTuple,
    right: NodeTuple,
    edges: Optional[EdgeTuple] = None,
    max_paths: Optional[int] = _DEFAULT_MAX_PATHS,
    max_path_states: Optional[int] = _DEFAULT_MAX_PATH_STATES,
) -> tuple[PathTuple, ...]:
    active_edges = dag.directed_edges if edges is None else edges
    adjacency = {node: set() for node in dag.nodes}
    for source, target in active_edges:
        adjacency[source].add(target)
        adjacency[target].add(source)
    paths = []
    states = 0
    for start in left:
        for finish in right:
            stack = [(start, (start,), frozenset((start,)))]
            while stack:
                states += 1
                if max_path_states is not None and states > max_path_states:
                    raise ValueError(
                        "Complete path search exceeds max_path_states; "
                        "increase the explicit budget or simplify the query."
                    )
                node, path, visited = stack.pop()
                if node == finish:
                    paths.append(path)
                    if max_paths is not None and len(paths) > max_paths:
                        raise ValueError(
                            "Complete path evidence exceeds max_paths; "
                            "increase the explicit budget or simplify the query."
                        )
                    continue
                for neighbor in sorted(adjacency[node], reverse=True):
                    if neighbor not in visited:
                        stack.append(
                            (neighbor, path + (neighbor,), visited | {neighbor})
                        )
    return tuple(sorted(paths, key=lambda path: (len(path), path)))


def _path_evidence(
    dag: CausalDAG,
    path: PathTuple,
    conditioned: NodeTuple,
    edges: Optional[EdgeTuple] = None,
) -> PathEvidence:
    active_edges = dag.directed_edges if edges is None else edges
    edge_set = set(active_edges)
    conditioned_set = set(conditioned)
    colliders = []
    noncolliders = []
    for index in range(1, len(path) - 1):
        previous, node, following = path[index - 1 : index + 2]
        if (previous, node) in edge_set and (following, node) in edge_set:
            colliders.append(node)
        else:
            noncolliders.append(node)
    activated = []
    blocking = []
    for collider in colliders:
        collider_family = {collider} | _descendants(dag, collider, active_edges)
        if collider_family.intersection(conditioned_set):
            activated.append(collider)
        else:
            blocking.append(collider)
    conditioned_noncolliders = tuple(
        node for node in noncolliders if node in conditioned_set
    )
    is_directed = all(
        (path[index], path[index + 1]) in edge_set for index in range(len(path) - 1)
    )
    is_backdoor = len(path) > 1 and (path[1], path[0]) in edge_set
    return PathEvidence(
        nodes=path,
        colliders=tuple(colliders),
        noncolliders=tuple(noncolliders),
        conditioned_noncolliders=conditioned_noncolliders,
        activated_colliders=tuple(activated),
        blocking_colliders=tuple(blocking),
        is_open=not conditioned_noncolliders and not blocking,
        is_directed_from_left=is_directed,
        is_backdoor_from_left=is_backdoor,
    )


def _d_separation_from_normalized(
    dag: CausalDAG,
    left: NodeTuple,
    right: NodeTuple,
    conditioned: NodeTuple,
    edges: Optional[EdgeTuple] = None,
    max_paths: Optional[int] = _DEFAULT_MAX_PATHS,
    max_path_states: Optional[int] = _DEFAULT_MAX_PATH_STATES,
) -> DSeparationEvidence:
    paths = tuple(
        _path_evidence(dag, path, conditioned, edges)
        for path in _simple_paths(
            dag,
            left,
            right,
            edges,
            max_paths,
            max_path_states,
        )
    )
    return DSeparationEvidence(
        left=left,
        right=right,
        conditioned=conditioned,
        separated=all(not path.is_open for path in paths),
        paths=paths,
    )


def d_separation(
    dag: CausalDAG,
    left: Iterable[str],
    right: Iterable[str],
    conditioned: Iterable[str] = (),
    *,
    max_paths: Optional[int] = _DEFAULT_MAX_PATHS,
    max_path_states: Optional[int] = _DEFAULT_MAX_PATH_STATES,
) -> DSeparationEvidence:
    """Return complete simple-path evidence for a d-separation query.

    The three node sets must be pairwise disjoint.  ``separated`` describes a
    graphical implication for distributions compatible with ``dag``; its
    negation is not a sample-level dependence test.  Complete path evidence is
    output-sensitive.  ``max_paths`` bounds returned witnesses and
    ``max_path_states`` bounds search expansions.  ``None`` explicitly opts
    out of the respective bound.  Evidence is never truncated.
    """

    dag = _validate_dag(dag)
    left_nodes = _known_node_set(dag, left, "left", allow_empty=False)
    right_nodes = _known_node_set(dag, right, "right", allow_empty=False)
    conditioned_nodes = _known_node_set(
        dag, conditioned, "conditioned", allow_empty=True
    )
    max_paths = _normalize_limit(max_paths, "max_paths")
    max_path_states = _normalize_limit(max_path_states, "max_path_states")
    if (
        set(left_nodes).intersection(right_nodes)
        or set(left_nodes).intersection(conditioned_nodes)
        or set(right_nodes).intersection(conditioned_nodes)
    ):
        raise ValueError("left, right, and conditioned must be pairwise disjoint.")
    return _d_separation_from_normalized(
        dag,
        left_nodes,
        right_nodes,
        conditioned_nodes,
        max_paths=max_paths,
        max_path_states=max_path_states,
    )


def check_backdoor_adjustment_set(
    dag: CausalDAG,
    treatment: str,
    outcome: str,
    adjustment_set: Iterable[str] = (),
    *,
    max_paths: Optional[int] = _DEFAULT_MAX_PATHS,
    max_path_states: Optional[int] = _DEFAULT_MAX_PATH_STATES,
) -> BackdoorAdjustmentEvidence:
    """Check one observed set against the ordinary back-door criterion.

    ``max_paths`` bounds returned witnesses and ``max_path_states`` bounds
    search expansions.  Evidence is never truncated.
    """

    dag = _validate_dag(dag)
    treatment, outcome = _validate_endpoints(dag, treatment, outcome)
    adjustment = _known_node_set(
        dag, adjustment_set, "adjustment_set", allow_empty=True
    )
    max_paths = _normalize_limit(max_paths, "max_paths")
    max_path_states = _normalize_limit(max_path_states, "max_path_states")
    observed = set(dag.observed_nodes)
    endpoints = tuple(sorted(set(adjustment).intersection((treatment, outcome))))
    descendants = tuple(
        sorted(set(adjustment).intersection(_descendants(dag, treatment)))
    )
    backdoor_edges = _without_outgoing(dag.directed_edges, (treatment,))
    separation = _d_separation_from_normalized(
        dag,
        (treatment,),
        (outcome,),
        adjustment,
        backdoor_edges,
        max_paths,
        max_path_states,
    )
    open_paths = tuple(path for path in separation.paths if path.is_open)
    all_observed = set(adjustment).issubset(observed)
    admissible = all_observed and not endpoints and not descendants and not open_paths
    return BackdoorAdjustmentEvidence(
        treatment=treatment,
        outcome=outcome,
        adjustment_set=adjustment,
        all_observed=all_observed,
        endpoints_in_set=endpoints,
        descendants_in_set=descendants,
        open_backdoor_paths=open_paths,
        admissible=admissible,
    )


def _inclusion_minimal_sets(valid_sets):
    ordered = sorted(valid_sets, key=lambda item: (len(item), item))
    result = []
    for candidate in ordered:
        candidate_set = set(candidate)
        if not any(set(existing).issubset(candidate_set) for existing in result):
            result.append(candidate)
    return tuple(result)


def _search_inclusion_minimal_sets(
    candidates: NodeTuple,
    is_valid: Callable[[NodeTuple], bool],
    max_candidates: Optional[int],
) -> tuple[NodeTuple, ...]:
    valid_sets = []
    selected = []
    frames = [[0, False]]
    work = 0
    while frames:
        next_index, evaluated = frames[-1]
        candidate = tuple(selected)
        if not evaluated:
            work += 1 + len(candidate)
            if max_candidates is not None and work > max_candidates:
                raise ValueError(
                    "Exact minimal-set search exceeds max_candidates; "
                    "increase the explicit budget or simplify the graph."
                )
            candidate_set = set(candidate)
            if any(set(existing).issubset(candidate_set) for existing in valid_sets):
                frames.pop()
                if selected:
                    selected.pop()
                continue
            if is_valid(candidate):
                if not candidate:
                    return ((),)
                valid_sets.append(candidate)
                frames.pop()
                selected.pop()
                continue
            frames[-1][1] = True
            continue
        if next_index >= len(candidates):
            frames.pop()
            if selected:
                selected.pop()
            continue
        frames[-1][0] += 1
        selected.append(candidates[next_index])
        frames.append([next_index + 1, False])
    return _inclusion_minimal_sets(valid_sets)


def minimal_backdoor_adjustment_sets(
    dag: CausalDAG,
    treatment: str,
    outcome: str,
    *,
    max_candidates: Optional[int] = _DEFAULT_MAX_CANDIDATES,
) -> tuple[NodeTuple, ...]:
    """Return all inclusion-minimal observed back-door adjustment sets.

    ``((),)`` means that the empty set is admissible.  ``()`` means that no
    admissible set exists among the observed, non-descendant graph nodes.
    Search is exact and stops with ``ValueError`` before exceeding the explicit
    ``max_candidates`` budget; no partial result is returned.
    """

    dag = _validate_dag(dag)
    treatment, outcome = _validate_endpoints(dag, treatment, outcome)
    max_candidates = _normalize_limit(max_candidates, "max_candidates")
    descendants = _descendants(dag, treatment)
    candidates = tuple(
        node
        for node in dag.observed_nodes
        if node not in {treatment, outcome} and node not in descendants
    )
    backdoor_edges = _without_outgoing(dag.directed_edges, (treatment,))

    def is_valid(candidate: NodeTuple) -> bool:
        return _is_d_separated(dag, (treatment,), (outcome,), candidate, backdoor_edges)

    return _search_inclusion_minimal_sets(candidates, is_valid, max_candidates)


def _directed_path_tuples(
    dag: CausalDAG,
    left: str,
    right: str,
    max_paths: Optional[int] = _DEFAULT_MAX_PATHS,
    max_path_states: Optional[int] = _DEFAULT_MAX_PATH_STATES,
) -> tuple[PathTuple, ...]:
    children = _children(dag.nodes, dag.directed_edges)
    parents = _parents(dag.nodes, dag.directed_edges)
    can_reach_right = {right}
    pending = [right]
    while pending:
        node = pending.pop()
        for parent in parents[node]:
            if parent not in can_reach_right:
                can_reach_right.add(parent)
                pending.append(parent)
    paths = []
    stack = [(left, (left,))]
    states = 0
    while stack:
        states += 1
        if max_path_states is not None and states > max_path_states:
            raise ValueError(
                "Complete directed-path search exceeds max_path_states; "
                "increase the explicit budget or simplify the query."
            )
        node, path = stack.pop()
        if node == right:
            paths.append(path)
            if max_paths is not None and len(paths) > max_paths:
                raise ValueError(
                    "Complete directed-path evidence exceeds max_paths; "
                    "increase the explicit budget or simplify the query."
                )
            continue
        for child in sorted(children[node], reverse=True):
            if child in can_reach_right:
                stack.append((child, path + (child,)))
    return tuple(sorted(paths, key=lambda path: (len(path), path)))


def _directed_paths(
    dag: CausalDAG,
    left: str,
    right: str,
    max_paths: Optional[int] = _DEFAULT_MAX_PATHS,
    max_path_states: Optional[int] = _DEFAULT_MAX_PATH_STATES,
) -> tuple[PathEvidence, ...]:
    return tuple(
        _path_evidence(dag, path, ())
        for path in _directed_path_tuples(dag, left, right, max_paths, max_path_states)
    )


def check_frontdoor_adjustment_set(
    dag: CausalDAG,
    treatment: str,
    outcome: str,
    mediator_set: Iterable[str],
    *,
    max_paths: Optional[int] = _DEFAULT_MAX_PATHS,
    max_path_states: Optional[int] = _DEFAULT_MAX_PATH_STATES,
) -> FrontdoorAdjustmentEvidence:
    """Check Pearl's front-door conditions under PROTOCOL-01 eligibility.

    The policy layer requires a nonempty observed mediator set and at least one
    directed treatment-to-outcome path, avoiding vacuous eligibility when no
    causal path exists.  A positive result is deliberately called
    ``graphically_admissible``.  Positivity is required by the identification
    theorem but cannot be checked from a DAG.  ``max_paths`` bounds returned
    witnesses and ``max_path_states`` bounds search expansions.  Evidence is
    never truncated.
    """

    dag = _validate_dag(dag)
    treatment, outcome = _validate_endpoints(dag, treatment, outcome)
    mediators = _known_node_set(dag, mediator_set, "mediator_set", allow_empty=True)
    max_paths = _normalize_limit(max_paths, "max_paths")
    max_path_states = _normalize_limit(max_path_states, "max_path_states")
    observed = set(dag.observed_nodes)
    endpoints = tuple(sorted(set(mediators).intersection((treatment, outcome))))
    directed_paths = _directed_paths(
        dag, treatment, outcome, max_paths, max_path_states
    )
    unintercepted = tuple(
        path
        for path in directed_paths
        if not set(path.nodes[1:-1]).intersection(mediators)
    )
    treatment_edges = _without_outgoing(dag.directed_edges, (treatment,))
    treatment_mediator = (
        _d_separation_from_normalized(
            dag,
            (treatment,),
            mediators,
            (),
            treatment_edges,
            max_paths,
            max_path_states,
        )
        if mediators and not endpoints
        else None
    )
    mediator_edges = _without_outgoing(dag.directed_edges, mediators)
    mediator_outcome = (
        _d_separation_from_normalized(
            dag,
            mediators,
            (outcome,),
            (treatment,),
            mediator_edges,
            max_paths,
            max_path_states,
        )
        if mediators and not endpoints
        else None
    )
    open_treatment_mediator = (
        tuple(path for path in treatment_mediator.paths if path.is_open)
        if treatment_mediator is not None
        else ()
    )
    open_mediator_outcome = (
        tuple(path for path in mediator_outcome.paths if path.is_open)
        if mediator_outcome is not None
        else ()
    )
    all_observed = set(mediators).issubset(observed)
    directed_path_exists = bool(directed_paths)
    admissible = (
        bool(mediators)
        and all_observed
        and not endpoints
        and directed_path_exists
        and not unintercepted
        and not open_treatment_mediator
        and not open_mediator_outcome
    )
    return FrontdoorAdjustmentEvidence(
        treatment=treatment,
        outcome=outcome,
        mediator_set=mediators,
        all_observed=all_observed,
        endpoints_in_set=endpoints,
        directed_path_exists=directed_path_exists,
        unintercepted_directed_paths=unintercepted,
        open_treatment_mediator_backdoor_paths=open_treatment_mediator,
        open_mediator_outcome_backdoor_paths_given_treatment=open_mediator_outcome,
        graphically_admissible=admissible,
        positivity_required=True,
    )


def minimal_frontdoor_adjustment_sets(
    dag: CausalDAG,
    treatment: str,
    outcome: str,
    *,
    max_candidates: Optional[int] = _DEFAULT_MAX_CANDIDATES,
) -> tuple[NodeTuple, ...]:
    """Return all inclusion-minimal observed PROTOCOL-eligible front-door sets.

    Search is exact and stops with ``ValueError`` before exceeding the explicit
    ``max_candidates`` budget; no partial result is returned.
    """

    dag = _validate_dag(dag)
    treatment, outcome = _validate_endpoints(dag, treatment, outcome)
    max_candidates = _normalize_limit(max_candidates, "max_candidates")
    if not _has_directed_path(dag, treatment, outcome):
        return ()
    candidates = tuple(
        node for node in dag.observed_nodes if node not in {treatment, outcome}
    )
    treatment_edges = _without_outgoing(dag.directed_edges, (treatment,))

    def is_valid(candidate: NodeTuple) -> bool:
        if not candidate:
            return False
        if _has_directed_path(dag, treatment, outcome, blocked=candidate):
            return False
        if not _is_d_separated(dag, (treatment,), candidate, (), treatment_edges):
            return False
        mediator_edges = _without_outgoing(dag.directed_edges, candidate)
        return _is_d_separated(dag, candidate, (outcome,), (treatment,), mediator_edges)

    return _search_inclusion_minimal_sets(candidates, is_valid, max_candidates)


def check_instrument(
    dag: CausalDAG,
    instrument: str,
    treatment: str,
    outcome: str,
    conditioned: Iterable[str] = (),
) -> InstrumentEvidence:
    """Check Pearl's conditional and CFI's simple instrument profiles.

    ``pearl_graphical`` uses d-connected relevance and a conditioning set.
    ``conditioned`` applies only to that Pearl profile.  Every ``cfi_*`` field
    evaluates the stricter, unconditional simple CFI profile on the same DAG
    and can remain true when ``conditioned`` is nonempty.  Neither Boolean is
    an effect-estimation or nonparametric-identification claim.
    """

    dag = _validate_dag(dag)
    treatment, outcome = _validate_endpoints(dag, treatment, outcome)
    instrument = _require_node(instrument, "instrument")
    if instrument not in dag.nodes:
        raise ValueError("instrument must be a known graph node.")
    if instrument in {treatment, outcome}:
        raise ValueError("instrument, treatment, and outcome must be distinct.")
    controls = _known_node_set(dag, conditioned, "conditioned", allow_empty=True)
    forbidden = {instrument, treatment, outcome}
    conditioning_disjoint = not set(controls).intersection(forbidden)
    observed = set(dag.observed_nodes)
    all_observed = {instrument, treatment, outcome}.union(controls).issubset(observed)
    unaffected = not set(controls).intersection(_descendants(dag, treatment))
    if conditioning_disjoint:
        treatment_edges = _without_incoming(dag.directed_edges, (treatment,))
        pearl_relevance = not _is_d_separated(
            dag, (instrument,), (treatment,), controls
        )
        pearl_exclusion = _is_d_separated(
            dag, (instrument,), (outcome,), controls, treatment_edges
        )
    else:
        pearl_relevance = False
        pearl_exclusion = False
    pearl_graphical = (
        all_observed
        and conditioning_disjoint
        and unaffected
        and pearl_relevance
        and pearl_exclusion
    )

    edge_set = set(dag.directed_edges)
    cfi_direct = (instrument, treatment) in edge_set
    instrument_outcome_path = _has_directed_path(dag, instrument, outcome)
    bypasses_treatment = _has_directed_path(
        dag, instrument, outcome, blocked=(treatment,)
    )
    cfi_full_mediation = instrument_outcome_path and not bypasses_treatment
    instrument_edges = _without_outgoing(dag.directed_edges, (instrument,))
    cfi_exogeneity = _is_d_separated(
        dag, (instrument,), (outcome,), (), instrument_edges
    )
    cfi_simple = (
        instrument in observed and cfi_direct and cfi_full_mediation and cfi_exogeneity
    )
    return InstrumentEvidence(
        instrument=instrument,
        treatment=treatment,
        outcome=outcome,
        conditioned=controls,
        all_observed=all_observed,
        conditioning_disjoint=conditioning_disjoint,
        conditioning_unaffected_by_treatment=unaffected,
        pearl_relevance=pearl_relevance,
        pearl_exclusion_exogeneity=pearl_exclusion,
        pearl_graphical=pearl_graphical,
        cfi_direct_relevance=cfi_direct,
        cfi_full_mediation=cfi_full_mediation,
        cfi_exogeneity=cfi_exogeneity,
        cfi_simple=cfi_simple,
    )


def causal_role_evidence(
    dag: CausalDAG,
    treatment: str,
    outcome: str,
    node: str,
    *,
    max_paths: Optional[int] = _DEFAULT_MAX_PATHS,
    max_path_states: Optional[int] = _DEFAULT_MAX_PATH_STATES,
) -> NodeRoleEvidence:
    """Return non-exclusive path witnesses for one internal graph node.

    ``max_paths`` bounds returned witnesses and ``max_path_states`` bounds
    search expansions.  Evidence is never truncated.
    """

    dag = _validate_dag(dag)
    treatment, outcome = _validate_endpoints(dag, treatment, outcome)
    node = _require_node(node, "node")
    if node not in dag.nodes:
        raise ValueError("node must be a known graph node.")
    if node in {treatment, outcome}:
        raise ValueError("node must be distinct from treatment and outcome.")
    max_paths = _normalize_limit(max_paths, "max_paths")
    max_path_states = _normalize_limit(max_path_states, "max_path_states")
    paths = tuple(
        _path_evidence(dag, path, ())
        for path in _simple_paths(
            dag,
            (treatment,),
            (outcome,),
            max_paths=max_paths,
            max_path_states=max_path_states,
        )
    )
    collider_paths = tuple(path.nodes for path in paths if node in path.colliders)
    noncollider_paths = tuple(path.nodes for path in paths if node in path.noncolliders)
    mediator_paths = tuple(
        path.nodes
        for path in paths
        if path.is_directed_from_left and node in path.nodes[1:-1]
    )
    backdoor_noncollider_paths = tuple(
        path.nodes
        for path in paths
        if path.is_backdoor_from_left and node in path.noncolliders
    )
    edge_set = set(dag.directed_edges)
    common_cause_paths = []
    for path in paths:
        if node not in path.noncolliders:
            continue
        index = path.nodes.index(node)
        directed_to_treatment = all(
            (path.nodes[position], path.nodes[position - 1]) in edge_set
            for position in range(index, 0, -1)
        )
        directed_to_outcome = all(
            (path.nodes[position], path.nodes[position + 1]) in edge_set
            for position in range(index, len(path.nodes) - 1)
        )
        if directed_to_treatment and directed_to_outcome:
            common_cause_paths.append(path.nodes)
    return NodeRoleEvidence(
        node=node,
        treatment=treatment,
        outcome=outcome,
        collider_on_paths=collider_paths,
        noncollider_on_paths=noncollider_paths,
        mediator_on_directed_paths=mediator_paths,
        backdoor_noncollider_on_paths=backdoor_noncollider_paths,
        common_cause_paths=tuple(common_cause_paths),
        descendant_of_treatment=node in _descendants(dag, treatment),
        ancestor_of_outcome=node in _ancestors(dag, outcome),
    )
