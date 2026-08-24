# ruff: noqa: UP045
"""Immutable evidence contract for the causal-factor research protocol.

The module represents the seven stages in Lopez de Prado and Zoonekynd
(2026), PDF pages 18-22 (journal pages 27-31) and Exhibit 10.  It validates a
completed research record; it does not execute discovery, estimation,
portfolio construction, backtesting, or multiple-testing algorithms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Optional

__all__ = [
    "BacktestStage",
    "CausalAdjustmentSetStage",
    "CausalDiscoveryStage",
    "CausalExplanatoryAndPredictivePowerStage",
    "CausalFactorProtocolReport",
    "CausalPortfolioConstructionStage",
    "EventHorizon",
    "FoldEvidence",
    "MultipleTestingAdjustmentsStage",
    "ValidationEvidence",
    "VariableSelectionStage",
    "validate_causal_factor_protocol",
]


_STAGE_NAMES = (
    "Variable Selection",
    "Causal Discovery",
    "Causal Adjustment Set",
    "Causal Explanatory and Predictive Power",
    "Causal Portfolio Construction",
    "Backtest",
    "Multiple Testing Adjustments",
)

_VARIABLE_SELECTION_METHODS = {
    "MUTUAL_INFORMATION",
    "SHAPLEY_VALUES",
    "MEAN_DECREASE_IMPURITY",
    "PERMUTATION_FEATURE_IMPORTANCE",
}
_RESEARCH_PURPOSES = {"CAUSAL_ATTRIBUTION", "RISK_PREMIA_HARVESTING"}
_DISCOVERY_METHODS = {
    "PC",
    "LINGAM",
    "ECONOMIC_REASONING",
    "EX_ANTE_VIEWS",
    "OBSERVED_OUTCOMES",
    "PEER_REVIEWED_ASSUMPTIONS",
    "DOMAIN_EXPERTISE",
}
_ADJUSTMENT_METHODS = {
    "BACKDOOR_ADJUSTMENT",
    "FRONT_DOOR_ADJUSTMENT",
    "INSTRUMENTAL_VARIABLES",
}
_VALIDATION_METHODS = {
    "PURGED_CROSS_VALIDATION",
    "WALK_FORWARD",
    "RESAMPLING",
    "COMBINATORIAL_PURGED_CROSS_VALIDATION",
}
_PURGED_VALIDATION_METHODS = {
    "PURGED_CROSS_VALIDATION",
    "COMBINATORIAL_PURGED_CROSS_VALIDATION",
}
_METRICS_BY_TASK = {
    "PROBABILITY": {"LOG_LOSS", "BRIER_SCORE"},
    "RANKING": {
        "ROC_CURVE",
        "PRECISION_RECALL_CURVE",
        "MEAN_RECIPROCAL_RANK",
        "CLASSIFICATION_ACCURACY",
    },
    "RETURN_SIZE": {
        "MEAN_SQUARED_ERROR",
        "R_SQUARED",
        "SPEARMAN_CORRELATION",
    },
}
_PORTFOLIO_METHODS = {
    "POSITION_SIZING",
    "EXPOSURE_CONTROL",
    "ECONOMIC_RATIONALE",
    "FRAGILITY_STRESS_TEST",
    "TRANSACTION_COST_OPTIMIZATION",
    "TRANSFER_COEFFICIENT",
}
_BACKTEST_METHODS = {
    "WALK_FORWARD",
    "RESAMPLING",
    "COMBINATORIAL_PURGED_CROSS_VALIDATION",
    "MONTE_CARLO",
}
_MULTIPLE_TESTING_METHODS = {
    "HOLM",
    "HOCHBERG",
    "BENJAMINI_HOCHBERG",
    "DEFLATED_SHARPE_RATIO",
}
_P_VALUE_METHODS = {"HOLM", "HOCHBERG", "BENJAMINI_HOCHBERG"}
_REFIT_COMPONENTS = {
    "VARIABLE_SELECTION",
    "CAUSAL_ESTIMATION",
    "PORTFOLIO_CONSTRUCTION",
}


def _freeze_tuple(value):
    if isinstance(value, list):
        return tuple(value)
    return value


def _freeze_nested_tuple(value):
    outer = _freeze_tuple(value)
    if isinstance(outer, tuple):
        return tuple(_freeze_tuple(item) for item in outer)
    return outer


def _freeze_partitions(value):
    outer = _freeze_tuple(value)
    if not isinstance(outer, tuple):
        return outer
    result = []
    for item in outer:
        pair = _freeze_tuple(item)
        if isinstance(pair, tuple) and len(pair) == 2:
            pair = (pair[0], _freeze_tuple(pair[1]))
        result.append(pair)
    return tuple(result)


def _store_normalized_field(instance, field_name, value):
    """Store a normalized value while a frozen record is being constructed."""

    instance.__dict__[field_name] = value


@dataclass(frozen=True)
class EventHorizon:
    """Observation identifier and its half-open return interval."""

    observation_id: str
    start: Real
    end: Real


@dataclass(frozen=True)
class FoldEvidence:
    """Membership and training-only pipeline components for one fold."""

    fold_id: str
    path_id: str
    train_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    fit_ids: tuple[str, ...]
    prediction_ids: tuple[str, ...]
    refit_id: str
    refit_components: tuple[str, ...]

    def __post_init__(self):
        for name in (
            "train_ids",
            "test_ids",
            "fit_ids",
            "prediction_ids",
            "refit_components",
        ):
            _store_normalized_field(self, name, _freeze_tuple(getattr(self, name)))


@dataclass(frozen=True)
class ValidationEvidence:
    """Temporal validation evidence shared by source stages that require it."""

    method_labels: tuple[str, ...]
    event_horizons: tuple[EventHorizon, ...]
    folds: tuple[FoldEvidence, ...]
    embargo: Real = 0.0

    def __post_init__(self):
        _store_normalized_field(
            self, "method_labels", _freeze_tuple(self.method_labels)
        )
        _store_normalized_field(
            self, "event_horizons", _freeze_tuple(self.event_horizons)
        )
        _store_normalized_field(self, "folds", _freeze_tuple(self.folds))


@dataclass(frozen=True)
class VariableSelectionStage:
    """Stage 1 evidence for candidate-variable selection and leakage controls."""

    purpose: str
    selected_variables: tuple[str, ...]
    method_labels: tuple[str, ...]
    overlapping_returns: bool
    strong_time_dependence: bool
    validation: Optional[ValidationEvidence]

    def __post_init__(self):
        _store_normalized_field(
            self, "selected_variables", _freeze_tuple(self.selected_variables)
        )
        _store_normalized_field(
            self, "method_labels", _freeze_tuple(self.method_labels)
        )


@dataclass(frozen=True)
class CausalDiscoveryStage:
    """Stage 2 evidence for an explicit, resolved causal graph."""

    method_labels: tuple[str, ...]
    graph_id: str
    graph_nodes: tuple[str, ...]
    directed_edges: tuple[tuple[str, str], ...]
    graph_kind: str
    ambiguous_edges: tuple[tuple[str, str], ...]
    assumptions: tuple[str, ...]

    def __post_init__(self):
        _store_normalized_field(
            self, "method_labels", _freeze_tuple(self.method_labels)
        )
        _store_normalized_field(self, "graph_nodes", _freeze_tuple(self.graph_nodes))
        _store_normalized_field(
            self, "directed_edges", _freeze_nested_tuple(self.directed_edges)
        )
        _store_normalized_field(
            self, "ambiguous_edges", _freeze_nested_tuple(self.ambiguous_edges)
        )
        _store_normalized_field(self, "assumptions", _freeze_tuple(self.assumptions))


@dataclass(frozen=True)
class CausalAdjustmentSetStage:
    """Stage 3 evidence for identification and admissible ordinary controls."""

    treatment: str
    outcome: str
    method_label: str
    identified: bool
    admissible_adjustment_sets: tuple[tuple[str, ...], ...]
    selected_adjustment_set: tuple[str, ...]
    confounders: tuple[str, ...]
    descendants: tuple[str, ...]
    mediators: tuple[str, ...]
    colliders: tuple[str, ...]
    instruments: tuple[str, ...]
    control_justifications: tuple[tuple[str, str], ...]
    open_backdoor_paths: tuple[str, ...]
    frontdoor_criteria_satisfied: Optional[bool] = None
    instrument_relevance_satisfied: Optional[bool] = None
    instrument_exclusion_satisfied: Optional[bool] = None
    instrument_exogeneity_satisfied: Optional[bool] = None

    def __post_init__(self):
        _store_normalized_field(
            self,
            "admissible_adjustment_sets",
            _freeze_nested_tuple(self.admissible_adjustment_sets),
        )
        for name in (
            "selected_adjustment_set",
            "confounders",
            "descendants",
            "mediators",
            "colliders",
            "instruments",
            "open_backdoor_paths",
        ):
            _store_normalized_field(self, name, _freeze_tuple(getattr(self, name)))
        _store_normalized_field(
            self,
            "control_justifications",
            _freeze_nested_tuple(self.control_justifications),
        )


@dataclass(frozen=True)
class CausalExplanatoryAndPredictivePowerStage:
    """Stage 4 generalization evidence, separated by research purpose."""

    task_types: tuple[str, ...]
    estimator_label: str
    explanatory_metric_labels: tuple[str, ...]
    predictive_metric_labels: tuple[str, ...]
    explanatory_evidence_id: Optional[str]
    predictive_evidence_id: Optional[str]
    naive_benchmark_id: str
    validation: ValidationEvidence
    multiclass_encoding: Optional[str] = None
    averaging_method: Optional[str] = None

    def __post_init__(self):
        _store_normalized_field(self, "task_types", _freeze_tuple(self.task_types))
        _store_normalized_field(
            self,
            "explanatory_metric_labels",
            _freeze_tuple(self.explanatory_metric_labels),
        )
        _store_normalized_field(
            self,
            "predictive_metric_labels",
            _freeze_tuple(self.predictive_metric_labels),
        )


@dataclass(frozen=True)
class CausalPortfolioConstructionStage:
    """Stage 5 evidence for a causally motivated implemented portfolio."""

    method_labels: tuple[str, ...]
    causal_exposures: tuple[str, ...]
    controlled_unintended_exposures: tuple[str, ...]
    cost_model_id: str
    constraint_set_id: str
    economic_rationale: str
    fragility_scenarios: tuple[str, ...]
    transfer_coefficient: Real

    def __post_init__(self):
        for name in (
            "method_labels",
            "causal_exposures",
            "controlled_unintended_exposures",
            "fragility_scenarios",
        ):
            _store_normalized_field(self, name, _freeze_tuple(getattr(self, name)))


@dataclass(frozen=True)
class BacktestStage:
    """Stage 6 evidence for one or more source-listed backtest types."""

    method_labels: tuple[str, ...]
    trial_family_id: str
    declared_trial_ids: tuple[str, ...]
    validation: Optional[ValidationEvidence] = None
    monte_carlo_dgp: Optional[str] = None

    def __post_init__(self):
        _store_normalized_field(
            self, "method_labels", _freeze_tuple(self.method_labels)
        )
        _store_normalized_field(
            self, "declared_trial_ids", _freeze_tuple(self.declared_trial_ids)
        )


@dataclass(frozen=True)
class MultipleTestingAdjustmentsStage:
    """Stage 7 evidence for the complete declared family of trials."""

    method_labels: tuple[str, ...]
    trial_family_id: str
    declared_trial_ids: tuple[str, ...]
    family_partitions: tuple[tuple[str, tuple[str, ...]], ...]
    alpha: Real
    backtests_independent: bool
    p_value_estimator: Optional[str] = None
    time_dependence_model: Optional[str] = None
    p_value_inputs_assume_independence: Optional[bool] = None
    sharpe_variance: Optional[Real] = None
    effective_trials: Optional[Real] = None
    sample_length: Optional[Integral] = None
    skewness: Optional[Real] = None
    kurtosis: Optional[Real] = None
    selection_bias_evidence_id: Optional[str] = None

    def __post_init__(self):
        _store_normalized_field(
            self, "method_labels", _freeze_tuple(self.method_labels)
        )
        _store_normalized_field(
            self, "declared_trial_ids", _freeze_tuple(self.declared_trial_ids)
        )
        _store_normalized_field(
            self, "family_partitions", _freeze_partitions(self.family_partitions)
        )


@dataclass(frozen=True)
class CausalFactorProtocolReport:
    """Complete immutable record of the seven published protocol stages."""

    trial_family_id: str
    declared_trial_ids: tuple[str, ...]
    variable_selection: VariableSelectionStage
    causal_discovery: CausalDiscoveryStage
    causal_adjustment_set: CausalAdjustmentSetStage
    causal_explanatory_and_predictive_power: CausalExplanatoryAndPredictivePowerStage
    causal_portfolio_construction: CausalPortfolioConstructionStage
    backtest: BacktestStage
    multiple_testing_adjustments: MultipleTestingAdjustmentsStage
    source_oracle: str = field(default="PROTOCOL-01", init=False)
    source_locator: str = field(
        default="PDF pages 18-22 (journal pages 27-31), Exhibit 10",
        init=False,
    )

    def __post_init__(self):
        _store_normalized_field(
            self, "declared_trial_ids", _freeze_tuple(self.declared_trial_ids)
        )

    @property
    def stages(self):
        """Return the seven named stage records in the source order."""

        return (
            (_STAGE_NAMES[0], self.variable_selection),
            (_STAGE_NAMES[1], self.causal_discovery),
            (_STAGE_NAMES[2], self.causal_adjustment_set),
            (_STAGE_NAMES[3], self.causal_explanatory_and_predictive_power),
            (_STAGE_NAMES[4], self.causal_portfolio_construction),
            (_STAGE_NAMES[5], self.backtest),
            (_STAGE_NAMES[6], self.multiple_testing_adjustments),
        )

    @property
    def stage_names(self):
        """Return the exact seven source stage names."""

        return _STAGE_NAMES


def _add_error(errors: list[str], message: str) -> None:
    if message not in errors:
        errors.append(message)


def _nonblank(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _safe_label_set(value) -> set[str]:
    if not isinstance(value, tuple) or not all(isinstance(item, str) for item in value):
        return set()
    return set(value)


def _finite_real(value) -> bool:
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _validate_string_tuple(
    value,
    stage: str,
    subject: str,
    errors: list[str],
    *,
    allow_empty: bool = False,
) -> bool:
    if not isinstance(value, tuple):
        _add_error(errors, f"{stage}: {subject} must be a tuple.")
        return False
    if not allow_empty and not value:
        _add_error(errors, f"{stage}: {subject} must not be empty.")
        return False
    if not all(_nonblank(item) for item in value):
        _add_error(errors, f"{stage}: {subject} contains an invalid identifier.")
        return False
    if len(set(value)) != len(value):
        _add_error(errors, f"{stage}: {subject} contains duplicate identifiers.")
        return False
    return True


def _validate_labels(
    value,
    allowed: set[str],
    stage: str,
    subject: str,
    errors: list[str],
    *,
    allow_empty: bool = False,
) -> bool:
    valid = _validate_string_tuple(
        value, stage, subject, errors, allow_empty=allow_empty
    )
    if valid and not set(value).issubset(allowed):
        _add_error(errors, f"{stage}: {subject} contains a non-source label.")
        return False
    return valid


def _validate_optional_identifier(
    value, stage: str, subject: str, errors: list[str]
) -> bool:
    if value is None:
        return True
    if not _nonblank(value):
        _add_error(errors, f"{stage}: {subject} is invalid.")
        return False
    return True


def _validate_event_horizons(
    evidence: ValidationEvidence, stage: str, errors: list[str]
) -> dict[str, tuple[float, float]]:
    horizons = evidence.event_horizons
    if not isinstance(horizons, tuple) or not horizons:
        _add_error(errors, f"{stage}: event horizons must be a nonempty tuple.")
        return {}

    result: dict[str, tuple[float, float]] = {}
    for horizon in horizons:
        if not isinstance(horizon, EventHorizon):
            _add_error(errors, f"{stage}: an event-horizon record has an invalid type.")
            continue
        if not _nonblank(horizon.observation_id):
            _add_error(errors, f"{stage}: an event horizon has an invalid identifier.")
            continue
        if horizon.observation_id in result:
            _add_error(errors, f"{stage}: event-horizon identifiers must be unique.")
            continue
        if not _finite_real(horizon.start) or not _finite_real(horizon.end):
            _add_error(errors, f"{stage}: event-horizon bounds must be finite numbers.")
            continue
        start = float(horizon.start)
        end = float(horizon.end)
        if not start < end:
            _add_error(
                errors, f"{stage}: every event horizon must have positive length."
            )
            continue
        result[horizon.observation_id] = (start, end)
    return result


def _intervals_overlap(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _has_any_overlap(horizons: dict[str, tuple[float, float]]) -> bool:
    intervals = list(horizons.values())
    for left_index, left in enumerate(intervals):
        for right in intervals[left_index + 1 :]:
            if _intervals_overlap(left, right):
                return True
    return False


def _validate_fold_evidence(
    evidence: ValidationEvidence,
    stage: str,
    errors: list[str],
    *,
    required_refit_components: set[str],
) -> dict[str, tuple[float, float]]:
    _validate_labels(
        evidence.method_labels,
        _VALIDATION_METHODS,
        stage,
        "validation method labels",
        errors,
    )
    horizon_map = _validate_event_horizons(evidence, stage, errors)
    if not _finite_real(evidence.embargo) or float(evidence.embargo) < 0.0:
        _add_error(errors, f"{stage}: embargo must be a finite nonnegative number.")
        embargo = 0.0
    else:
        embargo = float(evidence.embargo)

    folds = evidence.folds
    if not isinstance(folds, tuple) or not folds:
        _add_error(errors, f"{stage}: validation folds must be a nonempty tuple.")
        return horizon_map

    fold_ids: set[str] = set()
    refit_training_sets: dict[str, frozenset] = {}
    path_predictions: set[tuple[str, str]] = set()
    path_test_ids: dict[str, set[str]] = {}
    path_partitions: dict[str, list[tuple[str, ...]]] = {}
    nonresampled_test_ids: set[str] = set()
    known_ids = set(horizon_map)
    validation_methods = _safe_label_set(evidence.method_labels)

    for fold in folds:
        if not isinstance(fold, FoldEvidence):
            _add_error(errors, f"{stage}: a fold record has an invalid type.")
            continue
        if not _nonblank(fold.fold_id) or not _nonblank(fold.path_id):
            _add_error(errors, f"{stage}: fold and path identifiers must be nonblank.")
        elif fold.fold_id in fold_ids:
            _add_error(errors, f"{stage}: fold identifiers must be unique.")
        else:
            fold_ids.add(fold.fold_id)
        if not _nonblank(fold.refit_id):
            _add_error(errors, f"{stage}: every fold must record a refit identifier.")
        components_valid = _validate_labels(
            fold.refit_components,
            _REFIT_COMPONENTS,
            stage,
            "refit components",
            errors,
        )
        if components_valid and not required_refit_components.issubset(
            set(fold.refit_components)
        ):
            _add_error(
                errors,
                f"{stage}: every fold must refit all required pipeline components.",
            )

        train_valid = _validate_string_tuple(
            fold.train_ids, stage, "training identifiers", errors
        )
        test_valid = _validate_string_tuple(
            fold.test_ids, stage, "test identifiers", errors
        )
        fit_valid = _validate_string_tuple(
            fold.fit_ids, stage, "fit identifiers", errors
        )
        prediction_valid = _validate_string_tuple(
            fold.prediction_ids, stage, "prediction identifiers", errors
        )
        if not (train_valid and test_valid and fit_valid and prediction_valid):
            continue

        train_set = set(fold.train_ids)
        test_set = set(fold.test_ids)
        fit_set = set(fold.fit_ids)
        prediction_set = set(fold.prediction_ids)
        if train_set.intersection(test_set):
            _add_error(
                errors, f"{stage}: training and test membership must be disjoint."
            )
        if fit_set != train_set or fit_set.intersection(test_set):
            _add_error(
                errors,
                f"{stage}: every fold must refit on its complete training membership only.",
            )
        if prediction_set != test_set:
            _add_error(errors, f"{stage}: predictions must match the test membership.")
        if not (train_set | test_set | fit_set | prediction_set).issubset(known_ids):
            _add_error(
                errors, f"{stage}: every fold identifier must have an event horizon."
            )
        if validation_methods.intersection({"PURGED_CROSS_VALIDATION", "WALK_FORWARD"}):
            if nonresampled_test_ids.intersection(test_set):
                _add_error(
                    errors,
                    f"{stage}: ordinary temporal validation cannot duplicate test predictions.",
                )
            nonresampled_test_ids.update(test_set.intersection(known_ids))

        if _nonblank(fold.refit_id):
            training_signature = frozenset(fold.train_ids)
            previous = refit_training_sets.get(fold.refit_id)
            if previous is not None and previous != training_signature:
                _add_error(
                    errors,
                    f"{stage}: a refit identifier cannot represent different training sets.",
                )
            refit_training_sets[fold.refit_id] = training_signature

        if _nonblank(fold.path_id):
            path_test_ids.setdefault(fold.path_id, set())
            path_partitions.setdefault(fold.path_id, [])
            path_partitions[fold.path_id].append(tuple(sorted(test_set)))
            for observation_id in fold.test_ids:
                contribution = (fold.path_id, observation_id)
                if contribution in path_predictions:
                    _add_error(
                        errors,
                        f"{stage}: a path cannot contain duplicate test contributions.",
                    )
                path_predictions.add(contribution)
                path_test_ids[fold.path_id].add(observation_id)

        if not (train_set | test_set).issubset(known_ids):
            continue
        for train_id in fold.train_ids:
            train_horizon = horizon_map[train_id]
            for test_id in fold.test_ids:
                test_horizon = horizon_map[test_id]
                if _intervals_overlap(train_horizon, test_horizon):
                    _add_error(
                        errors,
                        f"{stage}: train and test event horizons must be purged.",
                    )
                if (
                    embargo > 0.0
                    and train_horizon[0] >= test_horizon[1]
                    and train_horizon[0] < test_horizon[1] + embargo
                ):
                    _add_error(
                        errors,
                        f"{stage}: post-test embargo observations must be excluded.",
                    )
        if "WALK_FORWARD" in validation_methods and train_set and test_set:
            latest_train_end = max(horizon_map[item][1] for item in train_set)
            earliest_test_start = min(horizon_map[item][0] for item in test_set)
            if latest_train_end > earliest_test_start:
                _add_error(
                    errors,
                    f"{stage}: walk-forward training must precede the test interval.",
                )

    if "COMBINATORIAL_PURGED_CROSS_VALIDATION" in validation_methods:
        if len(path_test_ids) < 2:
            _add_error(errors, f"{stage}: CPCV requires multiple complete paths.")
        if known_ids and any(ids != known_ids for ids in path_test_ids.values()):
            _add_error(
                errors,
                f"{stage}: every CPCV path must predict each declared observation once.",
            )
        partition_signatures = [
            tuple(sorted(partition)) for partition in path_partitions.values()
        ]
        if len(set(partition_signatures)) != len(partition_signatures):
            _add_error(
                errors, f"{stage}: CPCV paths must have distinct test partitions."
            )
    if (
        "PURGED_CROSS_VALIDATION" in validation_methods
        and known_ids
        and nonresampled_test_ids != known_ids
    ):
        _add_error(
            errors,
            f"{stage}: purged cross-validation must predict every observation once.",
        )
    return horizon_map


def _valid_edge_tuple(value, stage: str, subject: str, errors: list[str]) -> bool:
    if not isinstance(value, tuple):
        _add_error(errors, f"{stage}: {subject} must be a tuple.")
        return False
    valid = True
    seen: set[tuple[str, str]] = set()
    for edge in value:
        if (
            not isinstance(edge, tuple)
            or len(edge) != 2
            or not _nonblank(edge[0])
            or not _nonblank(edge[1])
        ):
            _add_error(errors, f"{stage}: {subject} contains an invalid edge.")
            valid = False
            continue
        if edge in seen:
            _add_error(errors, f"{stage}: {subject} contains duplicate edges.")
            valid = False
        seen.add(edge)
    return valid


def _adjacency(nodes: set[str], edges: tuple[tuple[str, str], ...]):
    result = {node: set() for node in nodes}
    for source, target in edges:
        if source in result and target in result:
            result[source].add(target)
    return result


def _is_acyclic(nodes: set[str], edges: tuple[tuple[str, str], ...]) -> bool:
    indegree = {node: 0 for node in nodes}
    adjacency = _adjacency(nodes, edges)
    for targets in adjacency.values():
        for target in targets:
            indegree[target] += 1
    ready = [node for node, degree in indegree.items() if degree == 0]
    visited = 0
    while ready:
        node = ready.pop()
        visited += 1
        for target in adjacency[node]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    return visited == len(nodes)


def _reachable(adjacency, start: str) -> set[str]:
    visited: set[str] = set()
    pending = list(adjacency.get(start, ()))
    while pending:
        node = pending.pop()
        if node in visited:
            continue
        visited.add(node)
        pending.extend(adjacency.get(node, ()))
    return visited


def _reachable_avoiding(adjacency, start: str, blocked: set[str]) -> set[str]:
    if start in blocked:
        return set()
    visited: set[str] = set()
    pending = [node for node in adjacency.get(start, ()) if node not in blocked]
    while pending:
        node = pending.pop()
        if node in visited or node in blocked:
            continue
        visited.add(node)
        pending.extend(
            target for target in adjacency.get(node, ()) if target not in blocked
        )
    return visited


def _d_separated(
    edges: tuple[tuple[str, str], ...],
    left: str,
    right: str,
    conditioned: set[str],
) -> bool:
    nodes = {left, right} | conditioned
    for source, target in edges:
        nodes.add(source)
        nodes.add(target)
    parents = {node: set() for node in nodes}
    for source, target in edges:
        parents[target].add(source)

    ancestors = {left, right} | conditioned
    pending = list(ancestors)
    while pending:
        node = pending.pop()
        for parent in parents.get(node, ()):
            if parent not in ancestors:
                ancestors.add(parent)
                pending.append(parent)

    moral = {node: set() for node in ancestors}
    for source, target in edges:
        if source in ancestors and target in ancestors:
            moral[source].add(target)
            moral[target].add(source)
    for child in ancestors:
        child_parents = [parent for parent in parents[child] if parent in ancestors]
        for index, first in enumerate(child_parents):
            for second in child_parents[index + 1 :]:
                moral[first].add(second)
                moral[second].add(first)

    if left in conditioned or right in conditioned:
        return True
    visited = set(conditioned)
    pending = [left]
    while pending:
        node = pending.pop()
        if node == right:
            return False
        if node in visited:
            continue
        visited.add(node)
        pending.extend(neighbor for neighbor in moral[node] if neighbor not in visited)
    return True


def _backdoor_d_separated(
    edges: tuple[tuple[str, str], ...],
    exposure: str,
    outcome: str,
    conditioned: set[str],
) -> bool:
    backdoor_edges = tuple(edge for edge in edges if edge[0] != exposure)
    return _d_separated(backdoor_edges, exposure, outcome, conditioned)


def _validate_discovery(
    stage: CausalDiscoveryStage,
    selected_variables: set[str],
    outcome: str,
    errors: list[str],
):
    label = "Stage 2"
    _validate_labels(
        stage.method_labels,
        _DISCOVERY_METHODS,
        label,
        "causal-discovery method labels",
        errors,
    )
    if not _nonblank(stage.graph_id):
        _add_error(errors, f"{label}: graph identifier must be nonblank.")
    nodes_valid = _validate_string_tuple(
        stage.graph_nodes, label, "graph nodes", errors
    )
    nodes = set(stage.graph_nodes) if nodes_valid else set()
    if nodes_valid and nodes != selected_variables | {outcome}:
        _add_error(
            errors,
            f"{label}: graph nodes must match the selected variables and outcome.",
        )
    edges_valid = _valid_edge_tuple(
        stage.directed_edges, label, "directed edges", errors
    )
    if edges_valid:
        for source, target in stage.directed_edges:
            if source not in nodes or target not in nodes:
                _add_error(
                    errors, f"{label}: every directed edge must use graph nodes."
                )
            if source == target:
                _add_error(errors, f"{label}: directed self-loops are not admissible.")
        if nodes and not _is_acyclic(nodes, stage.directed_edges):
            _add_error(errors, f"{label}: the resolved graph must be acyclic.")
    if stage.graph_kind != "DAG":
        _add_error(errors, f"{label}: the accepted graph must be a resolved DAG.")
    if not isinstance(stage.ambiguous_edges, tuple):
        _add_error(errors, f"{label}: ambiguous edges must be a tuple.")
    elif stage.ambiguous_edges:
        _valid_edge_tuple(stage.ambiguous_edges, label, "ambiguous edges", errors)
        _add_error(
            errors, f"{label}: identification-relevant ambiguity must be resolved."
        )
    _validate_string_tuple(stage.assumptions, label, "graph assumptions", errors)
    safe_edges = stage.directed_edges if edges_valid else ()
    return nodes, _adjacency(nodes, safe_edges), safe_edges


def _validate_adjustment_sets(
    value, stage: str, errors: list[str]
) -> tuple[set[frozenset], bool]:
    if not isinstance(value, tuple) or not value:
        _add_error(errors, f"{stage}: admissible adjustment sets must be nonempty.")
        return set(), False
    result: set[frozenset] = set()
    valid = True
    for adjustment_set in value:
        if not _validate_string_tuple(
            adjustment_set,
            stage,
            "an admissible adjustment set",
            errors,
            allow_empty=True,
        ):
            valid = False
            continue
        frozen = frozenset(adjustment_set)
        if frozen in result:
            _add_error(errors, f"{stage}: admissible adjustment sets must be distinct.")
            valid = False
        result.add(frozen)
    return result, valid


def _validate_justifications(value, stage: str, errors: list[str]):
    if not isinstance(value, tuple):
        _add_error(errors, f"{stage}: control justifications must be a tuple.")
        return {}
    result = {}
    for item in value:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not _nonblank(item[0])
            or not _nonblank(item[1])
        ):
            _add_error(errors, f"{stage}: a control justification is invalid.")
            continue
        if item[0] in result:
            _add_error(errors, f"{stage}: every control must have one justification.")
        result[item[0]] = item[1]
    return result


def _validate_adjustment(
    stage: CausalAdjustmentSetStage,
    graph_nodes: set[str],
    adjacency,
    directed_edges: tuple[tuple[str, str], ...],
    errors: list[str],
) -> None:
    label = "Stage 3"
    treatment = stage.treatment if _nonblank(stage.treatment) else ""
    outcome = stage.outcome if _nonblank(stage.outcome) else ""
    if not _nonblank(stage.treatment) or not _nonblank(stage.outcome):
        _add_error(errors, f"{label}: treatment and outcome must be nonblank.")
    elif treatment == outcome:
        _add_error(errors, f"{label}: treatment and outcome must be distinct.")
    if treatment not in graph_nodes or outcome not in graph_nodes:
        _add_error(errors, f"{label}: treatment and outcome must be graph nodes.")
    if (
        not _nonblank(stage.method_label)
        or stage.method_label not in _ADJUSTMENT_METHODS
    ):
        _add_error(errors, f"{label}: adjustment method is not source-listed.")
    if not isinstance(stage.identified, bool) or not stage.identified:
        _add_error(errors, f"{label}: the causal effect must be identified.")

    selected_valid = _validate_string_tuple(
        stage.selected_adjustment_set,
        label,
        "selected adjustment set",
        errors,
        allow_empty=True,
    )
    admissible_sets, _ = _validate_adjustment_sets(
        stage.admissible_adjustment_sets, label, errors
    )
    selected = set(stage.selected_adjustment_set) if selected_valid else set()
    if selected_valid and frozenset(selected) not in admissible_sets:
        _add_error(
            errors,
            f"{label}: the selected adjustment set must be declared admissible.",
        )

    role_names = {}
    for field_name, subject in (
        ("confounders", "confounders"),
        ("descendants", "descendants"),
        ("mediators", "mediators"),
        ("colliders", "colliders"),
        ("instruments", "instruments"),
    ):
        value = getattr(stage, field_name)
        valid = _validate_string_tuple(value, label, subject, errors, allow_empty=True)
        role_names[field_name] = set(value) if valid else set()
        if valid and not set(value).issubset(graph_nodes):
            _add_error(
                errors, f"{label}: every declared causal role must be a graph node."
            )
        if valid and {treatment, outcome}.intersection(value):
            _add_error(
                errors, f"{label}: treatment and outcome cannot be role variables."
            )

    computed_descendants = _reachable(adjacency, treatment)
    computed_descendants.discard(outcome)
    if role_names["descendants"] != computed_descendants:
        _add_error(
            errors,
            f"{label}: declared descendants must exactly match the graph.",
        )
    for mediator in role_names["mediators"]:
        if mediator not in computed_descendants or outcome not in _reachable(
            adjacency, mediator
        ):
            _add_error(
                errors, f"{label}: declared mediators must lie on a causal path."
            )
    incoming = {node: 0 for node in graph_nodes}
    for targets in adjacency.values():
        for target in targets:
            incoming[target] += 1
    if any(incoming.get(collider, 0) < 2 for collider in role_names["colliders"]):
        _add_error(errors, f"{label}: declared colliders must have converging arrows.")
    if outcome not in _reachable(adjacency, treatment):
        _add_error(
            errors, f"{label}: the graph must retain a treatment-to-outcome path."
        )

    graph_confounders = {
        node
        for node in graph_nodes - {treatment, outcome}
        if treatment in _reachable(adjacency, node)
        and outcome in _reachable_avoiding(adjacency, node, {treatment})
    }
    if role_names["confounders"] != graph_confounders:
        _add_error(
            errors,
            f"{label}: declared confounders must match graph-implied common causes.",
        )
    graph_colliders = {
        node
        for node, degree in incoming.items()
        if degree >= 2 and node not in {treatment, outcome}
    }
    if role_names["colliders"] != graph_colliders:
        _add_error(
            errors,
            f"{label}: declared colliders must match graph-implied converging nodes.",
        )
    graph_mediators = {
        node
        for node in graph_nodes - {treatment, outcome}
        if node in _reachable(adjacency, treatment)
        and outcome in _reachable(adjacency, node)
    }
    if role_names["mediators"] != graph_mediators:
        _add_error(
            errors,
            f"{label}: declared mediators must match graph-implied causal-path nodes.",
        )
    if role_names["confounders"].intersection(role_names["instruments"]):
        _add_error(
            errors,
            f"{label}: a variable cannot be both a confounder and an instrument.",
        )
    instrument_edges = tuple(edge for edge in directed_edges if edge[0] != treatment)
    for instrument in role_names["instruments"]:
        if treatment not in _reachable_avoiding(adjacency, instrument, selected):
            _add_error(
                errors,
                f"{label}: every declared instrument must be relevant in the graph.",
            )
        if outcome in _reachable_avoiding(adjacency, instrument, {treatment}):
            _add_error(
                errors,
                f"{label}: every declared instrument must satisfy graph-level exclusion.",
            )
        if not _d_separated(instrument_edges, instrument, outcome, selected):
            _add_error(
                errors,
                f"{label}: every declared instrument must be graph-separated from outcome shocks.",
            )

    forbidden_controls = (
        computed_descendants
        | role_names["descendants"]
        | role_names["mediators"]
        | role_names["colliders"]
        | role_names["instruments"]
        | {treatment, outcome}
    )
    if selected.intersection(forbidden_controls):
        _add_error(
            errors,
            f"{label}: ordinary controls cannot include descendants, mediators, colliders, or instruments.",
        )
    for admissible in admissible_sets:
        candidate = set(admissible)
        if not candidate.issubset(graph_nodes):
            _add_error(
                errors,
                f"{label}: every declared admissible set must use graph nodes.",
            )
        if candidate.intersection(forbidden_controls):
            _add_error(
                errors,
                f"{label}: no declared admissible set may contain a descendant, mediator, collider, or instrument.",
            )
        if stage.method_label == "BACKDOOR_ADJUSTMENT" and not _backdoor_d_separated(
            directed_edges,
            treatment,
            outcome,
            candidate,
        ):
            _add_error(
                errors,
                f"{label}: every declared admissible set must block graph-implied backdoor paths.",
            )
    if not selected.issubset(graph_nodes):
        _add_error(errors, f"{label}: every selected control must be a graph node.")
    justifications = _validate_justifications(
        stage.control_justifications, label, errors
    )
    if set(justifications) != selected:
        _add_error(errors, f"{label}: every selected control needs one justification.")
    if not _validate_string_tuple(
        stage.open_backdoor_paths,
        label,
        "open backdoor paths",
        errors,
        allow_empty=True,
    ):
        pass
    elif stage.open_backdoor_paths:
        _add_error(errors, f"{label}: no backdoor path may remain open.")

    if (
        stage.method_label != "FRONT_DOOR_ADJUSTMENT"
        and stage.frontdoor_criteria_satisfied is not None
    ):
        _add_error(
            errors,
            f"{label}: front-door evidence requires the front-door method.",
        )
    instrument_premises = (
        stage.instrument_relevance_satisfied,
        stage.instrument_exclusion_satisfied,
        stage.instrument_exogeneity_satisfied,
    )
    if stage.method_label != "INSTRUMENTAL_VARIABLES" and any(
        value is not None for value in instrument_premises
    ):
        _add_error(
            errors,
            f"{label}: instrumental-variable premises require the IV method.",
        )

    if stage.method_label == "BACKDOOR_ADJUSTMENT":
        if not _backdoor_d_separated(
            directed_edges,
            treatment,
            outcome,
            selected,
        ):
            _add_error(
                errors,
                f"{label}: selected controls must block every graph-implied backdoor path.",
            )
    elif stage.method_label == "FRONT_DOOR_ADJUSTMENT":
        if (
            not role_names["mediators"]
            or stage.frontdoor_criteria_satisfied is not True
        ):
            _add_error(
                errors, f"{label}: front-door criteria must be explicitly satisfied."
            )
        elif outcome in _reachable_avoiding(
            adjacency, treatment, role_names["mediators"]
        ):
            _add_error(
                errors,
                f"{label}: front-door mediators must intercept every directed causal path.",
            )
        for mediator in role_names["mediators"]:
            if not _backdoor_d_separated(
                directed_edges, treatment, mediator, selected
            ) or not _backdoor_d_separated(
                directed_edges, mediator, outcome, {treatment} | selected
            ):
                _add_error(
                    errors,
                    f"{label}: graph-implied front-door backdoor criteria must hold.",
                )
    elif stage.method_label == "INSTRUMENTAL_VARIABLES":
        if not role_names["instruments"] or any(
            value is not True for value in instrument_premises
        ):
            _add_error(
                errors, f"{label}: instrumental-variable premises must all hold."
            )


def _same_horizons(
    left: dict[str, tuple[float, float]], right: dict[str, tuple[float, float]]
) -> bool:
    return left == right


def validate_causal_factor_protocol(
    report: CausalFactorProtocolReport,
) -> CausalFactorProtocolReport:
    """Validate and return a complete seven-stage protocol evidence record.

    The validation is structural and fail-closed.  A successful result means
    the record conforms to the published stage contract; it does not certify
    that the caller's empirical evidence or causal assumptions are true.

    Raises
    ------
    ValueError
        If any stage is missing, has the wrong record type, uses a method label
        outside the source contract, or fails a cross-stage safety invariant.
    """

    if not isinstance(report, CausalFactorProtocolReport):
        raise ValueError("Protocol: report type is invalid.")

    errors: list[str] = []
    if not _nonblank(report.trial_family_id):
        _add_error(errors, "Protocol: trial-family identifier must be nonblank.")
    trials_valid = _validate_string_tuple(
        report.declared_trial_ids,
        "Protocol",
        "declared trial identifiers",
        errors,
    )

    stage_types = (
        ("Stage 1", report.variable_selection, VariableSelectionStage),
        ("Stage 2", report.causal_discovery, CausalDiscoveryStage),
        ("Stage 3", report.causal_adjustment_set, CausalAdjustmentSetStage),
        (
            "Stage 4",
            report.causal_explanatory_and_predictive_power,
            CausalExplanatoryAndPredictivePowerStage,
        ),
        (
            "Stage 5",
            report.causal_portfolio_construction,
            CausalPortfolioConstructionStage,
        ),
        ("Stage 6", report.backtest, BacktestStage),
        (
            "Stage 7",
            report.multiple_testing_adjustments,
            MultipleTestingAdjustmentsStage,
        ),
    )
    for label, value, expected in stage_types:
        if not isinstance(value, expected):
            _add_error(errors, f"{label}: stage record type is invalid.")
    if any(not isinstance(value, expected) for _, value, expected in stage_types):
        raise ValueError("Causal factor protocol is invalid. " + " ".join(errors))

    selection = report.variable_selection
    label = "Stage 1"
    if not _nonblank(selection.purpose) or selection.purpose not in _RESEARCH_PURPOSES:
        _add_error(errors, f"{label}: research purpose is not source-listed.")
    selected_valid = _validate_string_tuple(
        selection.selected_variables, label, "selected variables", errors
    )
    selected_variables = set(selection.selected_variables) if selected_valid else set()
    selection_methods_valid = _validate_labels(
        selection.method_labels,
        _VARIABLE_SELECTION_METHODS,
        label,
        "variable-selection method labels",
        errors,
    )
    if not isinstance(selection.overlapping_returns, bool):
        _add_error(errors, f"{label}: overlapping-returns flag must be boolean.")
    if not isinstance(selection.strong_time_dependence, bool):
        _add_error(errors, f"{label}: time-dependence flag must be boolean.")
    selection_methods = (
        set(selection.method_labels) if selection_methods_valid else set()
    )
    if selection.validation is None:
        selection_horizons = {}
        if (
            selection.overlapping_returns is True
            or selection.strong_time_dependence is True
            or "PERMUTATION_FEATURE_IMPORTANCE" in selection_methods
        ):
            _add_error(
                errors,
                f"{label}: the declared selection design requires temporal validation.",
            )
    elif not isinstance(selection.validation, ValidationEvidence):
        _add_error(
            errors, f"{label}: temporal validation evidence has an invalid type."
        )
        selection_horizons = {}
    else:
        selection_horizons = _validate_fold_evidence(
            selection.validation,
            label,
            errors,
            required_refit_components={"VARIABLE_SELECTION"},
        )
        actual_overlap = _has_any_overlap(selection_horizons)
        if isinstance(selection.overlapping_returns, bool) and (
            selection.overlapping_returns != actual_overlap
        ):
            _add_error(
                errors, f"{label}: overlap declaration must match event horizons."
            )
        validation_methods = _safe_label_set(selection.validation.method_labels)
        if actual_overlap and not validation_methods.intersection(
            _PURGED_VALIDATION_METHODS
        ):
            _add_error(
                errors, f"{label}: overlapping returns require purged validation."
            )
        if selection.strong_time_dependence and (
            not _finite_real(selection.validation.embargo)
            or float(selection.validation.embargo) <= 0.0
        ):
            _add_error(errors, f"{label}: strong time dependence requires an embargo.")

    adjustment = report.causal_adjustment_set
    safe_outcome = adjustment.outcome if _nonblank(adjustment.outcome) else ""
    graph_nodes, graph_adjacency, graph_edges = _validate_discovery(
        report.causal_discovery,
        selected_variables,
        safe_outcome,
        errors,
    )
    _validate_adjustment(
        adjustment,
        graph_nodes,
        graph_adjacency,
        graph_edges,
        errors,
    )

    stage4 = report.causal_explanatory_and_predictive_power
    label = "Stage 4"
    task_types_valid = _validate_labels(
        stage4.task_types,
        set(_METRICS_BY_TASK),
        label,
        "task types",
        errors,
    )
    allowed_metrics = set()
    if task_types_valid:
        for task_type in stage4.task_types:
            allowed_metrics.update(_METRICS_BY_TASK[task_type])
    if not _nonblank(stage4.estimator_label):
        _add_error(errors, f"{label}: estimator label must be nonblank.")
    explanatory_valid = _validate_labels(
        stage4.explanatory_metric_labels,
        allowed_metrics,
        label,
        "explanatory metric labels",
        errors,
        allow_empty=True,
    )
    predictive_valid = _validate_labels(
        stage4.predictive_metric_labels,
        allowed_metrics,
        label,
        "predictive metric labels",
        errors,
        allow_empty=True,
    )
    has_explanatory = explanatory_valid and bool(stage4.explanatory_metric_labels)
    has_predictive = predictive_valid and bool(stage4.predictive_metric_labels)
    if not has_explanatory and not has_predictive:
        _add_error(errors, f"{label}: at least one performance aspect is required.")
    reported_metrics = set()
    if explanatory_valid:
        reported_metrics.update(stage4.explanatory_metric_labels)
    if predictive_valid:
        reported_metrics.update(stage4.predictive_metric_labels)
    if task_types_valid:
        for task_type in stage4.task_types:
            if not reported_metrics.intersection(_METRICS_BY_TASK[task_type]):
                _add_error(
                    errors,
                    f"{label}: every declared task needs a compatible metric.",
                )
    if has_explanatory and not _nonblank(stage4.explanatory_evidence_id):
        _add_error(errors, f"{label}: explanatory evidence identifier is required.")
    if not has_explanatory and stage4.explanatory_evidence_id is not None:
        _add_error(
            errors, f"{label}: explanatory evidence must match reported metrics."
        )
    if has_predictive and not _nonblank(stage4.predictive_evidence_id):
        _add_error(errors, f"{label}: predictive evidence identifier is required.")
    if not has_predictive and stage4.predictive_evidence_id is not None:
        _add_error(errors, f"{label}: predictive evidence must match reported metrics.")
    _validate_optional_identifier(
        stage4.explanatory_evidence_id,
        label,
        "explanatory evidence identifier",
        errors,
    )
    _validate_optional_identifier(
        stage4.predictive_evidence_id,
        label,
        "predictive evidence identifier",
        errors,
    )
    if (
        has_explanatory
        and has_predictive
        and stage4.explanatory_evidence_id == stage4.predictive_evidence_id
    ):
        _add_error(
            errors, f"{label}: explanatory and predictive evidence must be separate."
        )
    if selection.purpose == "CAUSAL_ATTRIBUTION" and not has_explanatory:
        _add_error(
            errors, f"{label}: causal attribution requires explanatory evidence."
        )
    if selection.purpose == "RISK_PREMIA_HARVESTING" and not has_predictive:
        _add_error(
            errors, f"{label}: risk-premia harvesting requires predictive evidence."
        )
    if not _nonblank(stage4.naive_benchmark_id):
        _add_error(errors, f"{label}: a naive benchmark identifier is required.")
    multiclass_fields = (stage4.multiclass_encoding, stage4.averaging_method)
    if any(value is not None for value in multiclass_fields):
        if not isinstance(stage4.multiclass_encoding, str) or (
            stage4.multiclass_encoding != "ONE_VS_REST"
        ):
            _add_error(errors, f"{label}: multiclass encoding is not source-listed.")
        if not isinstance(stage4.averaging_method, str) or (
            stage4.averaging_method not in {"MICRO", "MACRO", "WEIGHTED", "SAMPLES"}
        ):
            _add_error(errors, f"{label}: multiclass averaging is not source-listed.")
        if not _safe_label_set(stage4.task_types).intersection(
            {"PROBABILITY", "RANKING"}
        ):
            _add_error(
                errors, f"{label}: multiclass settings require a classification task."
            )
    if not isinstance(stage4.validation, ValidationEvidence):
        _add_error(errors, f"{label}: purged validation evidence is required.")
    else:
        stage4_horizons = _validate_fold_evidence(
            stage4.validation,
            label,
            errors,
            required_refit_components={"VARIABLE_SELECTION", "CAUSAL_ESTIMATION"},
        )
        if not _safe_label_set(stage4.validation.method_labels).intersection(
            _PURGED_VALIDATION_METHODS
        ):
            _add_error(errors, f"{label}: purged cross-validation is required.")
        if selection_horizons and not _same_horizons(
            selection_horizons, stage4_horizons
        ):
            _add_error(errors, f"{label}: event horizons must match Stage 1.")
        if selection.strong_time_dependence and (
            not _finite_real(stage4.validation.embargo)
            or float(stage4.validation.embargo) <= 0.0
        ):
            _add_error(errors, f"{label}: strong time dependence requires an embargo.")

    portfolio = report.causal_portfolio_construction
    label = "Stage 5"
    portfolio_labels_valid = _validate_labels(
        portfolio.method_labels,
        _PORTFOLIO_METHODS,
        label,
        "portfolio-construction method labels",
        errors,
    )
    if portfolio_labels_valid and set(portfolio.method_labels) != _PORTFOLIO_METHODS:
        _add_error(
            errors, f"{label}: all source portfolio considerations are required."
        )
    causal_valid = _validate_string_tuple(
        portfolio.causal_exposures, label, "causal exposures", errors
    )
    neutral_valid = _validate_string_tuple(
        portfolio.controlled_unintended_exposures,
        label,
        "controlled unintended exposures",
        errors,
        allow_empty=True,
    )
    causal_exposures = set(portfolio.causal_exposures) if causal_valid else set()
    neutral_exposures = (
        set(portfolio.controlled_unintended_exposures) if neutral_valid else set()
    )
    if causal_valid and not causal_exposures.issubset(graph_nodes):
        _add_error(errors, f"{label}: causal exposures must be graph variables.")
    if causal_valid and any(
        safe_outcome not in _reachable(graph_adjacency, exposure)
        for exposure in causal_exposures
    ):
        _add_error(
            errors,
            f"{label}: every causal exposure must have a directed path to the outcome.",
        )
    if neutral_valid and not neutral_exposures.issubset(graph_nodes):
        _add_error(
            errors,
            f"{label}: controlled unintended exposures must be graph variables.",
        )
    safe_treatment = adjustment.treatment if _nonblank(adjustment.treatment) else ""
    if safe_treatment not in causal_exposures:
        _add_error(
            errors, f"{label}: the target causal factor must drive position sizing."
        )
    if causal_exposures.intersection(neutral_exposures):
        _add_error(
            errors, f"{label}: causal and neutralized exposures must be disjoint."
        )
    declared_colliders = _safe_label_set(adjustment.colliders)
    if not declared_colliders.issubset(neutral_exposures):
        _add_error(errors, f"{label}: declared colliders must be neutralized.")
    for value, subject in (
        (portfolio.cost_model_id, "cost model identifier"),
        (portfolio.constraint_set_id, "constraint-set identifier"),
        (portfolio.economic_rationale, "economic rationale"),
    ):
        if not _nonblank(value):
            _add_error(errors, f"{label}: {subject} must be nonblank.")
    _validate_string_tuple(
        portfolio.fragility_scenarios,
        label,
        "causal-fragility scenarios",
        errors,
    )
    if not _finite_real(portfolio.transfer_coefficient) or not (
        -1.0 <= float(portfolio.transfer_coefficient) <= 1.0
    ):
        _add_error(
            errors,
            f"{label}: transfer coefficient must be finite and between minus one and one.",
        )

    backtest = report.backtest
    label = "Stage 6"
    backtest_labels_valid = _validate_labels(
        backtest.method_labels,
        _BACKTEST_METHODS,
        label,
        "backtest method labels",
        errors,
    )
    if backtest.trial_family_id != report.trial_family_id:
        _add_error(errors, f"{label}: trial-family identifier must match the report.")
    if backtest.declared_trial_ids != report.declared_trial_ids:
        _add_error(errors, f"{label}: declared trials must match the report exactly.")
    non_monte_carlo = (
        set(backtest.method_labels) - {"MONTE_CARLO"}
        if backtest_labels_valid
        else set()
    )
    if non_monte_carlo:
        if not isinstance(backtest.validation, ValidationEvidence):
            _add_error(
                errors, f"{label}: temporal backtests require validation evidence."
            )
        else:
            _validate_fold_evidence(
                backtest.validation,
                label,
                errors,
                required_refit_components=_REFIT_COMPONENTS,
            )
            if selection.strong_time_dependence and (
                not _finite_real(backtest.validation.embargo)
                or float(backtest.validation.embargo) <= 0.0
            ):
                _add_error(
                    errors,
                    f"{label}: strong time dependence requires a backtest embargo.",
                )
            if not non_monte_carlo.issubset(
                _safe_label_set(backtest.validation.method_labels)
            ):
                _add_error(
                    errors,
                    f"{label}: backtest and validation method labels must agree.",
                )
    elif backtest.validation is not None:
        _add_error(
            errors,
            f"{label}: Monte Carlo-only evidence cannot carry temporal folds.",
        )
    if backtest_labels_valid and "MONTE_CARLO" in backtest.method_labels:
        if not _nonblank(backtest.monte_carlo_dgp):
            _add_error(errors, f"{label}: Monte Carlo requires an explicit DGP.")
    elif backtest.monte_carlo_dgp is not None:
        _add_error(
            errors, f"{label}: a Monte Carlo DGP requires the Monte Carlo method."
        )

    multiple_testing = report.multiple_testing_adjustments
    label = "Stage 7"
    methods_valid = _validate_labels(
        multiple_testing.method_labels,
        _MULTIPLE_TESTING_METHODS,
        label,
        "multiple-testing method labels",
        errors,
    )
    if multiple_testing.trial_family_id != report.trial_family_id:
        _add_error(errors, f"{label}: trial-family identifier must match the report.")
    if multiple_testing.declared_trial_ids != report.declared_trial_ids:
        _add_error(errors, f"{label}: declared trials must match the report exactly.")

    partitions = multiple_testing.family_partitions
    covered_trials: set[str] = set()
    partition_ids: set[str] = set()
    if not isinstance(partitions, tuple) or not partitions:
        _add_error(errors, f"{label}: family partitions must be a nonempty tuple.")
    else:
        for partition in partitions:
            if (
                not isinstance(partition, tuple)
                or len(partition) != 2
                or not _nonblank(partition[0])
            ):
                _add_error(errors, f"{label}: a family partition is invalid.")
                continue
            if partition[0] in partition_ids:
                _add_error(
                    errors, f"{label}: family partition identifiers must be unique."
                )
            partition_ids.add(partition[0])
            if not _validate_string_tuple(
                partition[1], label, "partition trial identifiers", errors
            ):
                continue
            for trial_id in partition[1]:
                if trial_id in covered_trials:
                    _add_error(errors, f"{label}: family partitions must be disjoint.")
                covered_trials.add(trial_id)
        if trials_valid and covered_trials != set(report.declared_trial_ids):
            _add_error(
                errors,
                f"{label}: family partitions must cover declared trials exactly.",
            )

    if not _finite_real(multiple_testing.alpha) or not (
        0.0 < float(multiple_testing.alpha) < 1.0
    ):
        _add_error(
            errors, f"{label}: alpha must be a finite number between zero and one."
        )
    if not isinstance(multiple_testing.backtests_independent, bool):
        _add_error(errors, f"{label}: backtest-dependence flag must be boolean.")
    elif multiple_testing.backtests_independent:
        _add_error(errors, f"{label}: backtest dependence must be represented.")

    methods = set(multiple_testing.method_labels) if methods_valid else set()
    if methods.intersection(_P_VALUE_METHODS):
        if not _nonblank(multiple_testing.p_value_estimator):
            _add_error(errors, f"{label}: p-value estimator must be declared.")
        if not _nonblank(multiple_testing.time_dependence_model):
            _add_error(errors, f"{label}: time dependence treatment must be declared.")
        if multiple_testing.p_value_inputs_assume_independence is not False:
            _add_error(
                errors,
                f"{label}: p-value inputs must explicitly reject independence assumptions.",
            )
    elif any(
        value is not None
        for value in (
            multiple_testing.p_value_estimator,
            multiple_testing.time_dependence_model,
            multiple_testing.p_value_inputs_assume_independence,
        )
    ):
        _add_error(
            errors,
            f"{label}: p-value inputs require a p-value correction.",
        )
    if "DEFLATED_SHARPE_RATIO" in methods:
        trial_count = (
            len(report.declared_trial_ids)
            if isinstance(report.declared_trial_ids, tuple)
            else 0
        )
        if (
            not _finite_real(multiple_testing.sharpe_variance)
            or float(multiple_testing.sharpe_variance) <= 0.0
        ):
            _add_error(errors, f"{label}: positive Sharpe-ratio variance is required.")
        if not _finite_real(multiple_testing.effective_trials) or not (
            1.0 <= float(multiple_testing.effective_trials) < trial_count
        ):
            _add_error(errors, f"{label}: effective trials must be below total trials.")
        if (
            isinstance(multiple_testing.sample_length, bool)
            or not isinstance(multiple_testing.sample_length, Integral)
            or int(multiple_testing.sample_length) <= 1
        ):
            _add_error(errors, f"{label}: sample length must be an integer above one.")
        if not _finite_real(multiple_testing.skewness):
            _add_error(errors, f"{label}: skewness must be finite.")
        if (
            not _finite_real(multiple_testing.kurtosis)
            or float(multiple_testing.kurtosis) < 1.0
        ):
            _add_error(
                errors, f"{label}: Pearson kurtosis must be finite and at least one."
            )
        elif _finite_real(multiple_testing.skewness) and abs(
            float(multiple_testing.skewness)
        ) > math.sqrt(float(multiple_testing.kurtosis) - 1.0):
            _add_error(
                errors,
                f"{label}: skewness and Pearson kurtosis must satisfy the moment inequality.",
            )
        if not _nonblank(multiple_testing.selection_bias_evidence_id):
            _add_error(errors, f"{label}: selection-bias evidence is required.")
    elif any(
        value is not None
        for value in (
            multiple_testing.sharpe_variance,
            multiple_testing.effective_trials,
            multiple_testing.sample_length,
            multiple_testing.skewness,
            multiple_testing.kurtosis,
            multiple_testing.selection_bias_evidence_id,
        )
    ):
        _add_error(
            errors,
            f"{label}: Sharpe-correction inputs require the deflated Sharpe ratio.",
        )

    if errors:
        raise ValueError("Causal factor protocol is invalid. " + " ".join(errors))
    return report
