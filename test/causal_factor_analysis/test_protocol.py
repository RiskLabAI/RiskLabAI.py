"""Source-locked tests for the seven-stage causal-factor protocol."""

from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from RiskLabAI.causal_factor_analysis.protocol import (
    BacktestStage,
    CausalAdjustmentSetStage,
    CausalDiscoveryStage,
    CausalExplanatoryAndPredictivePowerStage,
    CausalFactorProtocolReport,
    CausalPortfolioConstructionStage,
    EventHorizon,
    FoldEvidence,
    MultipleTestingAdjustmentsStage,
    ValidationEvidence,
    VariableSelectionStage,
    validate_causal_factor_protocol,
)

PUBLIC_NAMES = {
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
}


def _horizons():
    return (
        EventHorizon("o1", 0.0, 2.0),
        EventHorizon("o2", 1.0, 3.0),
        EventHorizon("o3", 4.0, 5.0),
        EventHorizon("o4", 6.0, 7.0),
        EventHorizon("o5", 8.0, 9.0),
        EventHorizon("o6", 10.0, 11.0),
    )


def _fold(fold_id, path_id, train_ids, test_ids, refit_id):
    return FoldEvidence(
        fold_id=fold_id,
        path_id=path_id,
        train_ids=train_ids,
        test_ids=test_ids,
        fit_ids=train_ids,
        prediction_ids=test_ids,
        refit_id=refit_id,
        refit_components=(
            "VARIABLE_SELECTION",
            "CAUSAL_ESTIMATION",
            "PORTFOLIO_CONSTRUCTION",
        ),
    )


def _validation(method_labels=("PURGED_CROSS_VALIDATION",)):
    return ValidationEvidence(
        method_labels=method_labels,
        event_horizons=_horizons(),
        folds=(
            _fold("f1", "p1", ("o3", "o4", "o5", "o6"), ("o1", "o2"), "r1"),
            _fold("f2", "p1", ("o1", "o2", "o5", "o6"), ("o3", "o4"), "r2"),
            _fold("f3", "p1", ("o1", "o2", "o3", "o4"), ("o5", "o6"), "r3"),
        ),
        embargo=1.0,
    )


def _cpcv_validation():
    return ValidationEvidence(
        method_labels=("COMBINATORIAL_PURGED_CROSS_VALIDATION",),
        event_horizons=_horizons(),
        folds=(
            _fold("p1-f1", "p1", ("o3", "o4", "o5", "o6"), ("o1", "o2"), "p1-r1"),
            _fold("p1-f2", "p1", ("o1", "o2", "o5", "o6"), ("o3", "o4"), "p1-r2"),
            _fold("p1-f3", "p1", ("o1", "o2", "o3", "o4"), ("o5", "o6"), "p1-r3"),
            _fold("p2-f1", "p2", ("o4", "o5", "o6"), ("o1", "o3"), "p2-r1"),
            _fold("p2-f2", "p2", ("o3", "o4", "o6"), ("o2", "o5"), "p2-r2"),
            _fold("p2-f3", "p2", ("o1", "o2", "o3"), ("o4", "o6"), "p2-r3"),
        ),
        embargo=1.0,
    )


def _valid_report():
    trial_ids = ("strategy-a", "strategy-b", "strategy-c")
    family_id = "preregistered-family"
    selection_validation = _validation()
    evaluation_validation = _validation()
    backtest_validation = _cpcv_validation()

    return CausalFactorProtocolReport(
        trial_family_id=family_id,
        declared_trial_ids=trial_ids,
        variable_selection=VariableSelectionStage(
            purpose="RISK_PREMIA_HARVESTING",
            selected_variables=("X", "Z", "I", "M", "C", "D"),
            method_labels=(
                "MUTUAL_INFORMATION",
                "SHAPLEY_VALUES",
                "MEAN_DECREASE_IMPURITY",
                "PERMUTATION_FEATURE_IMPORTANCE",
            ),
            overlapping_returns=True,
            strong_time_dependence=True,
            validation=selection_validation,
        ),
        causal_discovery=CausalDiscoveryStage(
            method_labels=(
                "PC",
                "ECONOMIC_REASONING",
                "DOMAIN_EXPERTISE",
                "OBSERVED_OUTCOMES",
            ),
            graph_id="resolved-dag",
            graph_nodes=("X", "Y", "Z", "I", "M", "C", "D"),
            directed_edges=(
                ("Z", "X"),
                ("Z", "Y"),
                ("I", "X"),
                ("X", "M"),
                ("M", "Y"),
                ("X", "Y"),
                ("X", "C"),
                ("Y", "C"),
                ("X", "D"),
            ),
            graph_kind="DAG",
            ambiguous_edges=(),
            assumptions=(
                "The recorded variables are causally sufficient for the target effect.",
                "Edge directions use temporal and economic restrictions.",
            ),
        ),
        causal_adjustment_set=CausalAdjustmentSetStage(
            treatment="X",
            outcome="Y",
            method_label="BACKDOOR_ADJUSTMENT",
            identified=True,
            admissible_adjustment_sets=(("Z",),),
            selected_adjustment_set=("Z",),
            confounders=("Z",),
            descendants=("M", "C", "D"),
            mediators=("M",),
            colliders=("C",),
            instruments=("I",),
            control_justifications=(("Z", "Common cause of factor and return."),),
            open_backdoor_paths=(),
        ),
        causal_explanatory_and_predictive_power=(
            CausalExplanatoryAndPredictivePowerStage(
                task_types=("RETURN_SIZE",),
                estimator_label="fold-refit causal regression",
                explanatory_metric_labels=("R_SQUARED",),
                predictive_metric_labels=(
                    "MEAN_SQUARED_ERROR",
                    "SPEARMAN_CORRELATION",
                ),
                explanatory_evidence_id="explanatory-evidence",
                predictive_evidence_id="predictive-evidence",
                naive_benchmark_id="training-mean-benchmark",
                validation=evaluation_validation,
            )
        ),
        causal_portfolio_construction=CausalPortfolioConstructionStage(
            method_labels=(
                "POSITION_SIZING",
                "EXPOSURE_CONTROL",
                "ECONOMIC_RATIONALE",
                "FRAGILITY_STRESS_TEST",
                "TRANSACTION_COST_OPTIMIZATION",
                "TRANSFER_COEFFICIENT",
            ),
            causal_exposures=("X",),
            controlled_unintended_exposures=("C", "D"),
            cost_model_id="cost-model-v1",
            constraint_set_id="constraints-v1",
            economic_rationale="Positions target the identified X-to-Y effect.",
            fragility_scenarios=("weaken X-to-Y", "perturb Z-to-X"),
            transfer_coefficient=0.82,
        ),
        backtest=BacktestStage(
            method_labels=("COMBINATORIAL_PURGED_CROSS_VALIDATION",),
            trial_family_id=family_id,
            declared_trial_ids=trial_ids,
            validation=backtest_validation,
        ),
        multiple_testing_adjustments=MultipleTestingAdjustmentsStage(
            method_labels=("HOLM", "DEFLATED_SHARPE_RATIO"),
            trial_family_id=family_id,
            declared_trial_ids=trial_ids,
            family_partitions=(
                ("primary", ("strategy-a", "strategy-b")),
                ("robustness", ("strategy-c",)),
            ),
            alpha=0.05,
            backtests_independent=False,
            p_value_estimator="dependence-adjusted-estimator",
            time_dependence_model="purged temporal folds",
            p_value_inputs_assume_independence=False,
            sharpe_variance=0.04,
            effective_trials=2.0,
            sample_length=252,
            skewness=0.1,
            kurtosis=3.2,
            selection_bias_evidence_id="selection-bias-evidence",
        ),
    )


def _validate_with(report, **changes):
    return validate_causal_factor_protocol(replace(report, **changes))


def test_protocol_01_success_has_exact_stage_order_and_provenance():
    report = validate_causal_factor_protocol(_valid_report())

    assert report.stage_names == (
        "Variable Selection",
        "Causal Discovery",
        "Causal Adjustment Set",
        "Causal Explanatory and Predictive Power",
        "Causal Portfolio Construction",
        "Backtest",
        "Multiple Testing Adjustments",
    )
    assert len(report.stages) == 7
    assert report.source_oracle == "PROTOCOL-01"
    assert report.source_locator == (
        "PDF pages 18-22 (journal pages 27-31), Exhibit 10"
    )


def test_valid_records_are_frozen_and_sequence_inputs_are_tuple_backed():
    stage = VariableSelectionStage(
        purpose="CAUSAL_ATTRIBUTION",
        selected_variables=["X"],
        method_labels=["MUTUAL_INFORMATION"],
        overlapping_returns=False,
        strong_time_dependence=False,
        validation=_validation(),
    )
    assert stage.selected_variables == ("X",)
    assert stage.method_labels == ("MUTUAL_INFORMATION",)
    with pytest.raises(FrozenInstanceError):
        stage.purpose = "RISK_PREMIA_HARVESTING"


def test_wrong_stage_record_type_is_rejected():
    report = _valid_report()
    with pytest.raises(ValueError, match="Stage 2"):
        _validate_with(report, causal_discovery=report.variable_selection)


def test_boundary_touching_horizons_are_not_overlapping():
    report = _valid_report()
    horizons = tuple(
        EventHorizon(f"b{index}", float(index), float(index + 1)) for index in range(6)
    )
    validation = replace(
        report.variable_selection.validation,
        event_horizons=horizons,
        folds=(
            _fold("b1", "bp", ("b2", "b3", "b4", "b5"), ("b0", "b1"), "br1"),
            _fold("b2", "bp", ("b0", "b1", "b4", "b5"), ("b2", "b3"), "br2"),
            _fold("b3", "bp", ("b0", "b1", "b2", "b3"), ("b4", "b5"), "br3"),
        ),
        embargo=0.0,
    )
    selection = replace(
        report.variable_selection,
        overlapping_returns=False,
        strong_time_dependence=False,
        validation=validation,
    )
    evaluation = replace(
        report.causal_explanatory_and_predictive_power,
        validation=validation,
    )
    validate_causal_factor_protocol(
        replace(
            report,
            variable_selection=selection,
            causal_explanatory_and_predictive_power=evaluation,
        )
    )


def test_overlap_declaration_and_purging_are_required():
    report = _valid_report()
    selection = replace(report.variable_selection, overlapping_returns=False)
    with pytest.raises(ValueError, match="Stage 1"):
        _validate_with(report, variable_selection=selection)


def test_cpcv_is_a_source_supported_purged_validation_method():
    report = _valid_report()
    cpcv = _cpcv_validation()
    selection = replace(report.variable_selection, validation=cpcv)
    evaluation = replace(
        report.causal_explanatory_and_predictive_power,
        validation=cpcv,
    )
    validate_causal_factor_protocol(
        replace(
            report,
            variable_selection=selection,
            causal_explanatory_and_predictive_power=evaluation,
        )
    )

    validation = replace(
        report.variable_selection.validation,
        method_labels=("WALK_FORWARD",),
    )
    selection = replace(report.variable_selection, validation=validation)
    with pytest.raises(ValueError, match="Stage 1"):
        _validate_with(report, variable_selection=selection)


def test_strong_time_dependence_requires_positive_embargo():
    report = _valid_report()
    validation = replace(report.variable_selection.validation, embargo=0.0)
    selection = replace(report.variable_selection, validation=validation)
    with pytest.raises(ValueError, match="Stage 1"):
        _validate_with(report, variable_selection=selection)


def test_in_sample_stage1_methods_do_not_require_temporal_folds_without_guards():
    report = _valid_report()
    selection = replace(
        report.variable_selection,
        method_labels=("MUTUAL_INFORMATION", "SHAPLEY_VALUES"),
        overlapping_returns=False,
        strong_time_dependence=False,
        validation=None,
    )
    validate_causal_factor_protocol(replace(report, variable_selection=selection))

    with pytest.raises(ValueError, match="Stage 1"):
        validate_causal_factor_protocol(
            replace(
                report,
                variable_selection=replace(
                    selection,
                    method_labels=("PERMUTATION_FEATURE_IMPORTANCE",),
                ),
            )
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("train_ids", ("o3", "o4", "missing")),
        ("test_ids", ("o1", "o3")),
        ("fit_ids", ("o1",)),
        ("fit_ids", ("o3",)),
        ("prediction_ids", ("o1",)),
        ("refit_id", ""),
        ("refit_components", ("CAUSAL_ESTIMATION",)),
    ],
)
def test_fold_membership_and_training_only_refit_evidence(field, value):
    report = _valid_report()
    validation = report.causal_explanatory_and_predictive_power.validation
    bad_fold = replace(validation.folds[0], **{field: value})
    bad_validation = replace(validation, folds=(bad_fold,) + validation.folds[1:])
    stage = replace(
        report.causal_explanatory_and_predictive_power,
        validation=bad_validation,
    )
    with pytest.raises(ValueError, match="Stage 4"):
        _validate_with(report, causal_explanatory_and_predictive_power=stage)


def test_refit_id_cannot_represent_different_training_sets():
    report = _valid_report()
    validation = report.causal_explanatory_and_predictive_power.validation
    second = replace(validation.folds[1], refit_id=validation.folds[0].refit_id)
    stage = replace(
        report.causal_explanatory_and_predictive_power,
        validation=replace(
            validation, folds=(validation.folds[0], second, validation.folds[2])
        ),
    )
    with pytest.raises(ValueError, match="Stage 4"):
        _validate_with(report, causal_explanatory_and_predictive_power=stage)


def test_temporal_overlap_and_embargo_leakage_are_rejected():
    report = _valid_report()
    validation = report.causal_explanatory_and_predictive_power.validation
    overlapping_fold = _fold("bad", "p2", ("o2", "o3"), ("o1",), "bad-refit")
    stage = replace(
        report.causal_explanatory_and_predictive_power,
        validation=replace(validation, folds=(overlapping_fold,)),
    )
    with pytest.raises(ValueError, match="Stage 4"):
        _validate_with(report, causal_explanatory_and_predictive_power=stage)

    horizons = tuple(
        EventHorizon(item.observation_id, item.start, item.end)
        for item in validation.event_horizons
    ) + (
        EventHorizon("embargoed", 3.5, 3.8),
    )
    embargo_fold = _fold("bad2", "p3", ("embargoed", "o3"), ("o1", "o2"), "bad2-refit")
    bad_validation = replace(validation, event_horizons=horizons, folds=(embargo_fold,))
    stage = replace(
        report.causal_explanatory_and_predictive_power,
        validation=bad_validation,
    )
    with pytest.raises(ValueError, match="Stage 4"):
        _validate_with(report, causal_explanatory_and_predictive_power=stage)


@pytest.mark.parametrize(
    "changes",
    [
        {"graph_kind": "CPDAG"},
        {"ambiguous_edges": (("X", "Y"),)},
        {"directed_edges": (("X", "X"),)},
        {"directed_edges": (("X", "M"), ("M", "X"))},
        {"assumptions": ()},
    ],
)
def test_discovery_requires_explicit_resolved_acyclic_dag(changes):
    report = _valid_report()
    discovery = replace(report.causal_discovery, **changes)
    with pytest.raises(ValueError, match="Stage 2"):
        _validate_with(report, causal_discovery=discovery)


def test_discovery_graph_contains_every_selected_variable():
    report = _valid_report()
    discovery = replace(
        report.causal_discovery,
        graph_nodes=tuple(
            node for node in report.causal_discovery.graph_nodes if node != "D"
        ),
        directed_edges=tuple(
            edge for edge in report.causal_discovery.directed_edges if "D" not in edge
        ),
    )
    with pytest.raises(ValueError, match="Stage 2"):
        _validate_with(report, causal_discovery=discovery)


@pytest.mark.parametrize("forbidden", ["C", "M", "D"])
def test_ordinary_controls_exclude_colliders_mediators_and_descendants(forbidden):
    report = _valid_report()
    adjustment = replace(
        report.causal_adjustment_set,
        admissible_adjustment_sets=((forbidden,),),
        selected_adjustment_set=(forbidden,),
        control_justifications=((forbidden, "Not admissible as an ordinary control."),),
    )
    with pytest.raises(ValueError, match="Stage 3"):
        _validate_with(report, causal_adjustment_set=adjustment)


@pytest.mark.parametrize("inadmissible", ["C", "M", "D", "I"])
def test_every_declared_admissible_adjustment_set_is_graph_valid(inadmissible):
    report = _valid_report()
    adjustment = replace(
        report.causal_adjustment_set,
        admissible_adjustment_sets=(("Z",), (inadmissible,)),
    )
    with pytest.raises(ValueError, match="Stage 3"):
        _validate_with(report, causal_adjustment_set=adjustment)


def test_declared_descendants_must_exactly_match_the_graph():
    report = _valid_report()
    adjustment = replace(
        report.causal_adjustment_set,
        descendants=("M", "C"),
    )

    with pytest.raises(ValueError, match="exactly match the graph"):
        _validate_with(report, causal_adjustment_set=adjustment)


@pytest.mark.parametrize(
    "changes",
    [
        {"identified": False},
        {"open_backdoor_paths": ("X<-Z->Y",)},
        {"control_justifications": ()},
        {"selected_adjustment_set": ("I",)},
    ],
)
def test_backdoor_adjustment_fails_closed(changes):
    report = _valid_report()
    adjustment = replace(report.causal_adjustment_set, **changes)
    with pytest.raises(ValueError, match="Stage 3"):
        _validate_with(report, causal_adjustment_set=adjustment)


def test_frontdoor_mediator_is_a_mechanism_not_an_ordinary_control():
    report = _valid_report()
    adjustment = replace(
        report.causal_adjustment_set,
        method_label="FRONT_DOOR_ADJUSTMENT",
        admissible_adjustment_sets=((),),
        selected_adjustment_set=(),
        control_justifications=(),
        frontdoor_criteria_satisfied=True,
    )
    with pytest.raises(ValueError, match="Stage 3"):
        validate_causal_factor_protocol(
            replace(report, causal_adjustment_set=adjustment)
        )

    discovery = replace(
        report.causal_discovery,
        directed_edges=tuple(
            edge
            for edge in report.causal_discovery.directed_edges
            if edge != ("X", "Y")
        ),
    )
    validate_causal_factor_protocol(
        replace(
            report,
            causal_discovery=discovery,
            causal_adjustment_set=adjustment,
        )
    )
    with pytest.raises(ValueError, match="Stage 3"):
        validate_causal_factor_protocol(
            replace(
                report,
                causal_adjustment_set=replace(
                    adjustment,
                    admissible_adjustment_sets=(("M",),),
                    selected_adjustment_set=("M",),
                    control_justifications=(("M", "ordinary control"),),
                ),
            )
        )


def test_instrumental_variables_requires_all_identification_premises():
    report = _valid_report()
    adjustment = replace(
        report.causal_adjustment_set,
        method_label="INSTRUMENTAL_VARIABLES",
        admissible_adjustment_sets=((),),
        selected_adjustment_set=(),
        control_justifications=(),
        instrument_relevance_satisfied=True,
        instrument_exclusion_satisfied=True,
        instrument_exogeneity_satisfied=True,
    )
    validate_causal_factor_protocol(replace(report, causal_adjustment_set=adjustment))
    for field in (
        "instrument_relevance_satisfied",
        "instrument_exclusion_satisfied",
        "instrument_exogeneity_satisfied",
    ):
        with pytest.raises(ValueError, match="Stage 3"):
            validate_causal_factor_protocol(
                replace(
                    report,
                    causal_adjustment_set=replace(adjustment, **{field: False}),
                )
            )

    discovery = replace(
        report.causal_discovery,
        directed_edges=tuple(
            edge
            for edge in report.causal_discovery.directed_edges
            if edge != ("I", "X")
        ),
    )
    with pytest.raises(ValueError, match="Stage 3"):
        validate_causal_factor_protocol(
            replace(
                report,
                causal_discovery=discovery,
                causal_adjustment_set=adjustment,
            )
        )


def test_instrumental_variable_can_be_conditionally_exogenous():
    report = _valid_report()
    discovery = replace(
        report.causal_discovery,
        directed_edges=(
            ("Z", "I"),
            ("I", "X"),
            ("Z", "Y"),
            ("X", "M"),
            ("M", "Y"),
            ("X", "Y"),
            ("X", "C"),
            ("Y", "C"),
            ("X", "D"),
        ),
    )
    adjustment = replace(
        report.causal_adjustment_set,
        method_label="INSTRUMENTAL_VARIABLES",
        instrument_relevance_satisfied=True,
        instrument_exclusion_satisfied=True,
        instrument_exogeneity_satisfied=True,
    )
    validate_causal_factor_protocol(
        replace(
            report,
            causal_discovery=discovery,
            causal_adjustment_set=adjustment,
        )
    )


def test_instrument_cannot_also_be_an_ordinary_control():
    report = _valid_report()
    adjustment = replace(
        report.causal_adjustment_set,
        method_label="INSTRUMENTAL_VARIABLES",
        admissible_adjustment_sets=(("I",),),
        selected_adjustment_set=("I",),
        control_justifications=(("I", "Not an ordinary control."),),
        instrument_relevance_satisfied=True,
        instrument_exclusion_satisfied=True,
        instrument_exogeneity_satisfied=True,
    )
    with pytest.raises(ValueError, match="Stage 3"):
        _validate_with(report, causal_adjustment_set=adjustment)


def test_selected_controls_cannot_block_instrument_relevance():
    report = _valid_report()
    discovery = replace(
        report.causal_discovery,
        directed_edges=(
            ("I", "Z"),
            ("Z", "X"),
            ("D", "X"),
            ("D", "Y"),
            ("X", "M"),
            ("M", "Y"),
            ("X", "Y"),
            ("X", "C"),
            ("Y", "C"),
        ),
    )
    adjustment = replace(
        report.causal_adjustment_set,
        method_label="INSTRUMENTAL_VARIABLES",
        admissible_adjustment_sets=(("D", "Z"),),
        selected_adjustment_set=("D", "Z"),
        confounders=("D",),
        descendants=("M", "C"),
        control_justifications=(
            ("D", "Blocks the graph-implied backdoor path."),
            ("Z", "Declared conditional control."),
        ),
        instrument_relevance_satisfied=True,
        instrument_exclusion_satisfied=True,
        instrument_exogeneity_satisfied=True,
    )
    with pytest.raises(ValueError, match="Stage 3"):
        validate_causal_factor_protocol(
            replace(
                report,
                causal_discovery=discovery,
                causal_adjustment_set=adjustment,
            )
        )


def test_declared_instruments_are_graph_valid_for_every_adjustment_method():
    report = _valid_report()
    adjustment = replace(report.causal_adjustment_set, instruments=("D",))
    with pytest.raises(ValueError, match="Stage 3"):
        _validate_with(report, causal_adjustment_set=adjustment)


def test_graph_implied_confounders_cannot_be_omitted_or_used_as_instruments():
    report = _valid_report()
    omitted = replace(
        report.causal_adjustment_set,
        admissible_adjustment_sets=((),),
        selected_adjustment_set=(),
        confounders=(),
        control_justifications=(),
    )
    with pytest.raises(ValueError, match="Stage 3"):
        validate_causal_factor_protocol(replace(report, causal_adjustment_set=omitted))

    invalid_instrument = replace(
        report.causal_adjustment_set,
        method_label="INSTRUMENTAL_VARIABLES",
        admissible_adjustment_sets=((),),
        selected_adjustment_set=(),
        instruments=("Z",),
        control_justifications=(),
        instrument_relevance_satisfied=True,
        instrument_exclusion_satisfied=True,
        instrument_exogeneity_satisfied=True,
    )
    with pytest.raises(ValueError, match="Stage 3"):
        validate_causal_factor_protocol(
            replace(report, causal_adjustment_set=invalid_instrument)
        )


@pytest.mark.parametrize(
    "field",
    [
        "frontdoor_criteria_satisfied",
        "instrument_relevance_satisfied",
        "instrument_exclusion_satisfied",
        "instrument_exogeneity_satisfied",
    ],
)
def test_adjustment_method_specific_evidence_is_inapplicable_by_default(field):
    report = _valid_report()
    adjustment = replace(report.causal_adjustment_set, **{field: []})
    with pytest.raises(ValueError, match="Stage 3"):
        _validate_with(report, causal_adjustment_set=adjustment)


def test_graph_valid_backdoor_blocker_need_not_be_the_remote_common_cause():
    report = _valid_report()
    discovery = replace(
        report.causal_discovery,
        directed_edges=(
            ("Z", "I"),
            ("I", "X"),
            ("Z", "Y"),
            ("X", "M"),
            ("M", "Y"),
            ("X", "Y"),
            ("X", "C"),
            ("Y", "C"),
            ("X", "D"),
        ),
    )
    adjustment = replace(
        report.causal_adjustment_set,
        admissible_adjustment_sets=(("I",),),
        selected_adjustment_set=("I",),
        instruments=(),
        control_justifications=(("I", "Blocks the graph-implied backdoor path."),),
    )
    validate_causal_factor_protocol(
        replace(
            report,
            causal_discovery=discovery,
            causal_adjustment_set=adjustment,
        )
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"predictive_metric_labels": ("LOG_LOSS",)},
        {"naive_benchmark_id": ""},
        {"predictive_evidence_id": "explanatory-evidence"},
        {"validation": replace(_validation(), method_labels=("WALK_FORWARD",))},
    ],
)
def test_stage4_metrics_baseline_separation_and_purging(changes):
    report = _valid_report()
    stage = replace(report.causal_explanatory_and_predictive_power, **changes)
    with pytest.raises(ValueError, match="Stage 4"):
        _validate_with(report, causal_explanatory_and_predictive_power=stage)


def test_stage4_refits_variable_selection_inside_each_training_fold():
    report = _valid_report()
    validation = report.causal_explanatory_and_predictive_power.validation
    first_fold = replace(
        validation.folds[0],
        refit_components=("CAUSAL_ESTIMATION",),
    )
    stage = replace(
        report.causal_explanatory_and_predictive_power,
        validation=replace(
            validation,
            folds=(first_fold,) + validation.folds[1:],
        ),
    )
    with pytest.raises(ValueError, match="Stage 4"):
        _validate_with(report, causal_explanatory_and_predictive_power=stage)


def test_purged_cross_validation_covers_every_out_of_sample_observation():
    report = _valid_report()
    validation = report.causal_explanatory_and_predictive_power.validation
    stage = replace(
        report.causal_explanatory_and_predictive_power,
        validation=replace(validation, folds=validation.folds[:2]),
    )
    with pytest.raises(ValueError, match="Stage 4"):
        _validate_with(report, causal_explanatory_and_predictive_power=stage)


def test_one_source_supported_stage4_aspect_is_sufficient():
    report = _valid_report()
    stage = replace(
        report.causal_explanatory_and_predictive_power,
        explanatory_metric_labels=(),
        explanatory_evidence_id=None,
    )
    validate_causal_factor_protocol(
        replace(report, causal_explanatory_and_predictive_power=stage)
    )


def test_complementary_stage4_tasks_and_multiclass_options_are_supported():
    report = _valid_report()
    stage = replace(
        report.causal_explanatory_and_predictive_power,
        task_types=("PROBABILITY", "RANKING", "RETURN_SIZE"),
        explanatory_metric_labels=("BRIER_SCORE", "R_SQUARED"),
        predictive_metric_labels=(
            "LOG_LOSS",
            "ROC_CURVE",
            "CLASSIFICATION_ACCURACY",
            "MEAN_SQUARED_ERROR",
        ),
        multiclass_encoding="ONE_VS_REST",
        averaging_method="MACRO",
    )
    validate_causal_factor_protocol(
        replace(report, causal_explanatory_and_predictive_power=stage)
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"cost_model_id": ""},
        {"constraint_set_id": ""},
        {"economic_rationale": ""},
        {"fragility_scenarios": ()},
        {"transfer_coefficient": math.inf},
        {"transfer_coefficient": -1.01},
        {"transfer_coefficient": 1.01},
        {"causal_exposures": ()},
        {
            "causal_exposures": ("X", "D"),
            "controlled_unintended_exposures": ("C",),
        },
        {"controlled_unintended_exposures": ("C", "D", "NOT_IN_GRAPH")},
    ],
)
def test_portfolio_construction_requires_complete_evidence(changes):
    report = _valid_report()
    stage = replace(report.causal_portfolio_construction, **changes)
    with pytest.raises(ValueError, match="Stage 5"):
        _validate_with(report, causal_portfolio_construction=stage)


def test_monte_carlo_requires_an_explicit_dgp():
    report = _valid_report()
    stage = replace(
        report.backtest,
        method_labels=("MONTE_CARLO",),
        validation=None,
        monte_carlo_dgp="structural market simulator v1",
    )
    validate_causal_factor_protocol(replace(report, backtest=stage))
    with pytest.raises(ValueError, match="Stage 6"):
        validate_causal_factor_protocol(
            replace(report, backtest=replace(stage, monte_carlo_dgp=""))
        )
    with pytest.raises(ValueError, match="Stage 6"):
        validate_causal_factor_protocol(
            replace(report, backtest=replace(stage, validation="invalid"))
        )
    with pytest.raises(ValueError, match="Stage 6"):
        _validate_with(
            report,
            backtest=replace(report.backtest, monte_carlo_dgp=[]),
        )


def test_backtest_labels_and_cpcv_path_contributions_fail_closed():
    report = _valid_report()
    with pytest.raises(ValueError, match="Stage 6"):
        _validate_with(
            report,
            backtest=replace(report.backtest, method_labels=("PBO",)),
        )

    validation = report.backtest.validation
    duplicate = replace(
        validation.folds[1],
        path_id=validation.folds[0].path_id,
        test_ids=("o1",),
        prediction_ids=("o1",),
    )
    stage = replace(
        report.backtest,
        validation=replace(validation, folds=(validation.folds[0], duplicate)),
    )
    with pytest.raises(ValueError, match="Stage 6"):
        _validate_with(report, backtest=stage)

    incomplete = replace(
        report.backtest,
        validation=replace(
            report.backtest.validation,
            folds=(report.backtest.validation.folds[0],),
        ),
    )
    with pytest.raises(ValueError, match="Stage 6"):
        _validate_with(report, backtest=incomplete)


def test_cpcv_paths_must_encode_distinct_test_partitions():
    report = _valid_report()
    first_path = report.backtest.validation.folds[:3]
    copied_path = tuple(
        replace(
            fold,
            fold_id=f"copy-{fold.fold_id}",
            path_id="copied-path",
            refit_id=f"copy-{fold.refit_id}",
        )
        for fold in first_path
    )
    backtest = replace(
        report.backtest,
        validation=replace(
            report.backtest.validation,
            folds=first_path + copied_path,
        ),
    )
    with pytest.raises(ValueError, match="Stage 6"):
        _validate_with(report, backtest=backtest)


def test_walk_forward_cannot_train_on_future_observations():
    report = _valid_report()
    stage = replace(
        report.backtest,
        method_labels=("WALK_FORWARD",),
        validation=_validation(("WALK_FORWARD",)),
    )
    with pytest.raises(ValueError, match="Stage 6"):
        _validate_with(report, backtest=stage)


def test_walk_forward_cannot_duplicate_out_of_sample_predictions():
    report = _valid_report()
    first = _fold("wf1", "wf-path-1", ("o1", "o2"), ("o3", "o4"), "wf-r1")
    second = replace(
        first,
        fold_id="wf2",
        path_id="wf-path-2",
        refit_id="wf-r2",
    )
    validation = ValidationEvidence(
        method_labels=("WALK_FORWARD",),
        event_horizons=_horizons(),
        folds=(first, second),
        embargo=1.0,
    )
    backtest = replace(
        report.backtest,
        method_labels=("WALK_FORWARD",),
        validation=validation,
    )
    with pytest.raises(ValueError, match="Stage 6"):
        _validate_with(report, backtest=backtest)


def test_strong_time_dependence_requires_a_backtest_embargo():
    report = _valid_report()
    backtest = replace(
        report.backtest,
        validation=replace(report.backtest.validation, embargo=0.0),
    )
    with pytest.raises(ValueError, match="Stage 6"):
        _validate_with(report, backtest=backtest)


@pytest.mark.parametrize(
    "stage_name,changes",
    [
        ("backtest", {"trial_family_id": "other-family"}),
        ("backtest", {"declared_trial_ids": ("strategy-a", "fabricated")}),
        ("multiple_testing_adjustments", {"trial_family_id": "other-family"}),
        ("multiple_testing_adjustments", {"declared_trial_ids": ("strategy-a",)}),
    ],
)
def test_trial_family_identity_is_exact_across_stages(stage_name, changes):
    report = _valid_report()
    stage = replace(getattr(report, stage_name), **changes)
    with pytest.raises(ValueError, match="Stage [67]"):
        _validate_with(report, **{stage_name: stage})


@pytest.mark.parametrize(
    "changes",
    [
        {"method_labels": ("BHY",)},
        {"alpha": True},
        {"alpha": 1.0},
        {"backtests_independent": True},
        {"family_partitions": (("primary", ("strategy-a",)),)},
        {"p_value_estimator": ""},
        {"time_dependence_model": ""},
        {"p_value_inputs_assume_independence": True},
        {"p_value_inputs_assume_independence": None},
        {"sharpe_variance": 0.0},
        {"effective_trials": 0.1},
        {"effective_trials": 3.0},
        {"sample_length": 1},
        {"skewness": math.nan},
        {"skewness": 3.0, "kurtosis": 2.0},
        {"kurtosis": -100.0},
        {"selection_bias_evidence_id": ""},
    ],
)
def test_multiple_testing_inputs_and_methods_fail_closed(changes):
    report = _valid_report()
    stage = replace(report.multiple_testing_adjustments, **changes)
    with pytest.raises(ValueError, match="Stage 7"):
        _validate_with(report, multiple_testing_adjustments=stage)


def test_non_iid_p_values_and_dsr_only_evidence_are_supported():
    report = _valid_report()
    dependence_adjusted = replace(
        report.multiple_testing_adjustments,
        p_value_estimator="non-IID block bootstrap",
        time_dependence_model="serial-dependence block bootstrap",
        p_value_inputs_assume_independence=False,
    )
    _validate_with(report, multiple_testing_adjustments=dependence_adjusted)

    dsr_only = replace(
        report.multiple_testing_adjustments,
        method_labels=("DEFLATED_SHARPE_RATIO",),
        p_value_estimator=None,
        time_dependence_model=None,
        p_value_inputs_assume_independence=None,
    )
    _validate_with(report, multiple_testing_adjustments=dsr_only)

    p_value_only = replace(
        report.multiple_testing_adjustments,
        method_labels=("HOLM",),
        sharpe_variance=None,
        effective_trials=None,
        sample_length=None,
        skewness=None,
        kurtosis=None,
        selection_bias_evidence_id=None,
    )
    _validate_with(report, multiple_testing_adjustments=p_value_only)


@pytest.mark.parametrize(
    "stage_name,stage",
    [
        (
            "variable_selection",
            replace(_valid_report().variable_selection, purpose=[]),
        ),
        (
            "causal_discovery",
            replace(_valid_report().causal_discovery, method_labels=(["PC"],)),
        ),
        (
            "causal_adjustment_set",
            replace(_valid_report().causal_adjustment_set, method_label=[]),
        ),
        (
            "causal_adjustment_set",
            replace(_valid_report().causal_adjustment_set, treatment=[]),
        ),
        (
            "causal_adjustment_set",
            replace(_valid_report().causal_adjustment_set, outcome=[]),
        ),
        (
            "causal_adjustment_set",
            replace(_valid_report().causal_adjustment_set, colliders=None),
        ),
        (
            "causal_explanatory_and_predictive_power",
            replace(
                _valid_report().causal_explanatory_and_predictive_power,
                task_types=(["RETURN_SIZE"],),
            ),
        ),
        (
            "causal_explanatory_and_predictive_power",
            replace(
                _valid_report().causal_explanatory_and_predictive_power,
                averaging_method=[],
            ),
        ),
        (
            "variable_selection",
            replace(
                _valid_report().variable_selection,
                validation=replace(
                    _valid_report().variable_selection.validation,
                    method_labels=(["PURGED_CROSS_VALIDATION"],),
                ),
            ),
        ),
    ],
)
def test_malformed_labels_fail_with_value_error(stage_name, stage):
    with pytest.raises(ValueError):
        _validate_with(_valid_report(), **{stage_name: stage})


def test_module_and_package_exports_are_exact_and_isolated():
    repository_root = Path(__file__).resolve().parents[2]
    program = f"""
import sys
import RiskLabAI.causal_factor_analysis as package
import RiskLabAI.causal_factor_analysis.protocol as module

expected = {sorted(PUBLIC_NAMES)!r}
if sorted(module.__all__) != expected:
    raise SystemExit('protocol public surface mismatch')
if not set(expected).issubset(package.__all__):
    raise SystemExit('package public surface omits protocol names')

blocked = {{
    'RiskLabAI.causal',
    'RiskLabAI.optimization',
    'RiskLabAI.backtest',
    'RiskLabAI.risk_control',
}}
if blocked.intersection(sys.modules):
    raise SystemExit('private-dependent namespace loaded')
"""
    subprocess.run(
        [sys.executable, "-c", program],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
