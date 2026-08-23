# Causal factor analysis

The public causal API implements the source-consistent analytical scope of *Causal Factor Investing* and exposes 57 concepts. The Python namespace is `RiskLabAI.causal_factor_analysis`; the matching Julia namespace is `RiskLabAI.CausalFactorAnalysis`.

## Public surface

| Group | Records and functions |
|---|---|
| Optimizer (1) | `minimum_variance_factor_weights` |
| Factor-mirage analytics (10) | `StrategyPerformance`, `ColliderCoefficients`, `ColliderDiagnostics`, `confounder_undercontrolled_coefficient`, `confounder_factor_return`, `confounder_forecast_return`, `collider_overcontrolled_coefficients`, `collider_factor_return`, `collider_forecast_return`, `collider_model_diagnostics` |
| Protocol (12) | `EventHorizon`, `FoldEvidence`, `ValidationEvidence`, `VariableSelectionStage`, `CausalDiscoveryStage`, `CausalAdjustmentSetStage`, `CausalExplanatoryAndPredictivePowerStage`, `CausalPortfolioConstructionStage`, `BacktestStage`, `MultipleTestingAdjustmentsStage`, `CausalFactorProtocolReport`, `validate_causal_factor_protocol` |
| Graph identification (14) | `CausalDAG`, `PathEvidence`, `DSeparationEvidence`, `BackdoorAdjustmentEvidence`, `FrontdoorAdjustmentEvidence`, `InstrumentEvidence`, `NodeRoleEvidence`, `d_separation`, `check_backdoor_adjustment_set`, `minimal_backdoor_adjustment_sets`, `check_frontdoor_adjustment_set`, `minimal_frontdoor_adjustment_sets`, `check_instrument`, `causal_role_evidence` |
| Treatment effects (10) | `DifferenceInDifferencesEstimate`, `TreatmentEffectDecomposition`, `average_treatment_effect`, `backdoor_adjusted_average_treatment_effect`, `backdoor_adjusted_expectation`, `difference_in_differences`, `frontdoor_adjusted_probability`, `linear_instrumental_variable_effect`, `randomized_mean_difference`, `treatment_effect_decomposition` |
| Specification diagnostics (10) | `PopulationRegressionDiagnostics`, `RegressionDiagnostics`, `SpecificationExperimentResult`, `SpecificationPopulationResult`, `collider_population_diagnostics`, `collider_specification_experiment`, `confounded_mediator_population_diagnostics`, `confounded_mediator_specification_experiment`, `fork_population_diagnostics`, `fork_specification_experiment` |

## Behavioral contract

The optimizer requires a symmetric positive-definite covariance matrix and a full-column-rank exposure matrix. It rejects singular or numerically unresolved constraints rather than adding hidden regularization.

Graph routines operate on a caller-supplied acyclic graph with an explicit observed-node set. They provide path evidence for d-separation, back-door, front-door, and instrument checks. These are graphical criteria, not causal-discovery procedures, positivity proofs, empirical instrument-strength tests, or claims that the supplied graph is true. Inclusion-minimal adjustment sets are not necessarily variance-optimal or minimum-cost sets.

Treatment-effect functions evaluate explicit identification formulas from caller-supplied interventional, conditional, or grouped quantities. They do not infer that the identification assumptions hold in observed data.

Protocol records are immutable evidence containers. The validator checks the declared seven-stage structure and fail-closed evidence rules; it does not execute empirical research or certify the truth of a declaration.

Specification diagnostics expose exact population identities and deterministic experiments for fork, collider, and confounded-mediator structures. They are mathematical diagnostics rather than automated model selection.

## Cross-language behavior

Python returns immutable tuple-backed evidence records and NumPy arrays; Julia returns immutable structs and Julia vectors or matrices. Python raises `ValueError` for invalid analytical inputs; Julia raises `ArgumentError`. Canonical node and set ordering, graphical criteria, result fields, resource limits, numerical tolerances, and estimands otherwise match.

Both implementations are tested against independent analytical results and graph oracles. Agreement between the languages is additional evidence, never the sole correctness test.

## Exclusions

Three later-source results remain excluded because their sources conflict: `CFA-CONF-01`, `CFA-COLL-01`, and `MIRAGE-COLL-SHIFT-01`. The clean namespace does not delegate to the excluded legacy causal implementation.

See `../examples/causal_factor_analysis_quickstart.py` for a deterministic example and `compatibility.md` for the tested runtime policy.

The exact ordered names, signatures, immutable record fields, implementation
modules, and source hashes are frozen in `../PUBLIC_API.json`. The seven causal
test files and their hashes are part of the exact package-scoped inventory in
`../TEST_INVENTORY.json`.

Cross-language parity is defined over the exact 57-name set and behavior, not
over incidental export-list order. Python preserves its approved `__all__`
order; the Julia inventory uses canonical lexical order. Both language-native
digests and the canonical name-set digest are recorded in the blocked
pre-package contract.
