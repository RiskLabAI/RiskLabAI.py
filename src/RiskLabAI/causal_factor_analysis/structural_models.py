"""Deterministic evaluation of supplied structural causal mechanisms."""

from __future__ import annotations

import heapq
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .graph_identification import CausalDAG

__all__ = ["StructuralCausalModelResult", "evaluate_structural_causal_model"]


@dataclass(frozen=True, slots=True, eq=False)
class StructuralCausalModelResult:
    """Canonical node order and sample-by-node structural values."""

    node_order: tuple[str, ...]
    values: NDArray[np.float64]


def _validate_dag(dag: CausalDAG) -> CausalDAG:
    if not isinstance(dag, CausalDAG):
        raise TypeError("dag must be a CausalDAG")
    return dag


def _readonly_snapshot(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a C-contiguous array backed by immutable independent bytes."""

    contiguous = np.ascontiguousarray(values, dtype=np.float64)
    snapshot = np.frombuffer(contiguous.tobytes(order="C"), dtype=np.float64)
    return snapshot.reshape(contiguous.shape)


def _as_finite_vector(
    values: ArrayLike,
    subject: str,
    *,
    expected_length: int | None = None,
) -> NDArray[np.float64]:
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{subject} must be a real numeric vector") from exc
    if raw.ndim != 1:
        raise ValueError(f"{subject} must be a one-dimensional vector")
    if raw.dtype.kind == "b":
        raise ValueError(f"{subject} must not be boolean-valued")
    if np.iscomplexobj(raw):
        raise ValueError(f"{subject} must be real-valued")
    if raw.dtype.kind in {"S", "U", "V"}:
        raise ValueError(f"{subject} must be a real numeric vector")
    try:
        vector = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{subject} must be a real numeric vector") from exc
    if expected_length is not None and vector.shape[0] != expected_length:
        raise ValueError(f"{subject} must have length {expected_length}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{subject} must contain only finite values")
    return _readonly_snapshot(vector)


def _mapping_keys(mapping: Mapping, subject: str, nodes: set[str]) -> None:
    try:
        keys = set(mapping.keys())
    except (AttributeError, TypeError) as exc:
        raise TypeError(f"{subject} must expose hashable mapping keys") from exc
    if keys != nodes:
        raise ValueError(f"{subject} must contain exactly one entry per graph node")


def _parent_map(dag: CausalDAG) -> dict[str, tuple[str, ...]]:
    parents: dict[str, list[str]] = {node: [] for node in dag.nodes}
    for source, target in dag.directed_edges:
        parents[target].append(source)
    return {node: tuple(sorted(values)) for node, values in parents.items()}


def _lexical_topological_order(
    dag: CausalDAG, parents: dict[str, tuple[str, ...]]
) -> tuple[str, ...]:
    children: dict[str, list[str]] = {node: [] for node in dag.nodes}
    indegree = {node: len(parents[node]) for node in dag.nodes}
    for source, target in dag.directed_edges:
        children[source].append(target)
    for values in children.values():
        values.sort()

    ready = [node for node in dag.nodes if indegree[node] == 0]
    heapq.heapify(ready)
    ordered: list[str] = []
    while ready:
        node = heapq.heappop(ready)
        ordered.append(node)
        for child in children[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)
    if len(ordered) != len(dag.nodes):
        raise ValueError("dag must be acyclic")
    return tuple(ordered)


def evaluate_structural_causal_model(
    dag: CausalDAG,
    mechanisms: Mapping[
        str,
        Callable[[NDArray[np.float64], NDArray[np.float64]], ArrayLike],
    ],
    exogenous_inputs: Mapping[str, ArrayLike],
) -> StructuralCausalModelResult:
    """Evaluate one deterministic structural mechanism per graph node.

    Nodes are evaluated in a lexical tie-broken topological order.  A node's
    mechanism receives an ``(n_samples, n_parents)`` parent matrix whose
    columns follow lexical parent order and an ``n_samples`` exogenous vector.
    Both inputs are protected independent snapshots.  The mechanism must
    return one finite vector of length ``n_samples``.
    """

    dag = _validate_dag(dag)
    if not isinstance(mechanisms, Mapping):
        raise TypeError("mechanisms must be a mapping")
    if not isinstance(exogenous_inputs, Mapping):
        raise TypeError("exogenous_inputs must be a mapping")

    graph_nodes = set(dag.nodes)
    _mapping_keys(mechanisms, "mechanisms", graph_nodes)
    _mapping_keys(exogenous_inputs, "exogenous_inputs", graph_nodes)

    mechanism_snapshot = {node: mechanisms[node] for node in dag.nodes}
    for node, mechanism in mechanism_snapshot.items():
        if not callable(mechanism):
            raise TypeError(f"mechanism for node {node!r} must be callable")

    exogenous_snapshot: dict[str, NDArray[np.float64]] = {}
    n_samples: int | None = None
    for node in dag.nodes:
        vector = _as_finite_vector(
            exogenous_inputs[node], f"exogenous input for node {node!r}"
        )
        if n_samples is None:
            n_samples = vector.shape[0]
            if n_samples == 0:
                raise ValueError(
                    "exogenous input vectors must contain at least one sample"
                )
        elif vector.shape[0] != n_samples:
            raise ValueError("all exogenous input vectors must have equal length")
        exogenous_snapshot[node] = vector

    if n_samples is None:
        raise ValueError("dag must contain at least one node")

    parents = _parent_map(dag)
    node_order = _lexical_topological_order(dag, parents)
    evaluated: dict[str, NDArray[np.float64]] = {}
    for node in node_order:
        parent_nodes = parents[node]
        if parent_nodes:
            parent_matrix = np.column_stack(
                tuple(evaluated[parent] for parent in parent_nodes)
            )
        else:
            parent_matrix = np.empty((n_samples, 0), dtype=np.float64)
        parent_snapshot = _readonly_snapshot(parent_matrix)
        raw_result = mechanism_snapshot[node](
            parent_snapshot,
            exogenous_snapshot[node],
        )
        evaluated[node] = _as_finite_vector(
            raw_result,
            f"mechanism result for node {node!r}",
            expected_length=n_samples,
        )

    values = np.column_stack(tuple(evaluated[node] for node in node_order))
    return StructuralCausalModelResult(
        node_order=node_order,
        values=_readonly_snapshot(values),
    )
