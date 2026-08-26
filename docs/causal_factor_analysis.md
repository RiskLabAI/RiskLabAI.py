# Causal factor analysis

The public causal API implements the source-consistent analytical scope of *Causal Factor Investing* and the sufficiently specified causal-factor methods in the reviewed public papers. It exposes 87 concepts: the released 57-name contract plus 30 additive names. The Python namespace is `RiskLabAI.causal_factor_analysis`; the matching Julia namespace is `RiskLabAI.CausalFactorAnalysis`.

## Public surface

| Group | Records and functions |
|---|---|
| Optimizer (1) | `minimum_variance_factor_weights` |
| Factor-mirage analytics (10) | `StrategyPerformance`, `ColliderCoefficients`, `ColliderDiagnostics`, `confounder_undercontrolled_coefficient`, `confounder_factor_return`, `confounder_forecast_return`, `collider_overcontrolled_coefficients`, `collider_factor_return`, `collider_forecast_return`, `collider_model_diagnostics` |
| Protocol (12) | `EventHorizon`, `FoldEvidence`, `ValidationEvidence`, `VariableSelectionStage`, `CausalDiscoveryStage`, `CausalAdjustmentSetStage`, `CausalExplanatoryAndPredictivePowerStage`, `CausalPortfolioConstructionStage`, `BacktestStage`, `MultipleTestingAdjustmentsStage`, `CausalFactorProtocolReport`, `validate_causal_factor_protocol` |
| Graph identification (14) | `CausalDAG`, `PathEvidence`, `DSeparationEvidence`, `BackdoorAdjustmentEvidence`, `FrontdoorAdjustmentEvidence`, `InstrumentEvidence`, `NodeRoleEvidence`, `d_separation`, `check_backdoor_adjustment_set`, `minimal_backdoor_adjustment_sets`, `check_frontdoor_adjustment_set`, `minimal_frontdoor_adjustment_sets`, `check_instrument`, `causal_role_evidence` |
| Treatment effects (10) | `DifferenceInDifferencesEstimate`, `TreatmentEffectDecomposition`, `average_treatment_effect`, `backdoor_adjusted_average_treatment_effect`, `backdoor_adjusted_expectation`, `difference_in_differences`, `frontdoor_adjusted_probability`, `linear_instrumental_variable_effect`, `randomized_mean_difference`, `treatment_effect_decomposition` |
| Specification diagnostics (10) | `PopulationRegressionDiagnostics`, `RegressionDiagnostics`, `SpecificationExperimentResult`, `SpecificationPopulationResult`, `collider_population_diagnostics`, `collider_specification_experiment`, `confounded_mediator_population_diagnostics`, `confounded_mediator_specification_experiment`, `fork_population_diagnostics`, `fork_specification_experiment` |
| General-variance mirage analytics (2) | `generalized_confounder_undercontrolled_coefficient`, `generalized_collider_overcontrolled_coefficients` |
| Allocation misspecification (2) | `AllocationMisspecificationDiagnostics`, `allocation_misspecification_diagnostics` |
| Factor roles and structural models (7) | `FactorControlRoles`, `TreatmentOutcomeRole`, `TreatmentOutcomeRoleEvidence`, `StructuralCausalModelResult`, `factor_control_roles`, `classify_treatment_outcome_role`, `evaluate_structural_causal_model` |
| Search-adjusted false discovery (19) | `MaxSelectionFamilyErrors`, `FDRComparisonEvidence`, `GaussianTrialMixture`, `GaussianSearchAdjustedFDR`, `FDRNonIdentificationWitness`, `SelectionLevelProbabilityEvidence`, `single_trial_false_discovery_rate`, `family_level_false_discovery_rate`, `max_selection_family_errors`, `compare_single_and_family_fdr`, `maximum_mixture_cdf`, `conditional_upper_tail_probability`, `gaussian_trial_mixture_cdf`, `gaussian_max_selection_cdf`, `gaussian_max_selection_log_density`, `gaussian_max_selection_log_likelihood`, `gaussian_search_adjusted_false_discovery_rate`, `fdr_nonidentification_witness`, `max_selection_null_probability` |

## Behavioral contract

The optimizer requires a symmetric positive-definite covariance matrix and a full-column-rank exposure matrix. It rejects singular or numerically unresolved constraints rather than adding hidden regularization.

Graph routines operate on a caller-supplied acyclic graph with an explicit observed-node set. They provide path evidence for d-separation, back-door, front-door, and instrument checks. These are graphical criteria, not causal-discovery procedures, positivity proofs, empirical instrument-strength tests, or claims that the supplied graph is true. Inclusion-minimal adjustment sets are not necessarily variance-optimal or minimum-cost sets.

Treatment-effect functions evaluate explicit identification formulas from caller-supplied interventional, conditional, or grouped quantities. They do not infer that the identification assumptions hold in observed data.

Protocol records are immutable evidence containers. The validator checks the declared seven-stage structure and fail-closed evidence rules; it does not execute empirical research or certify the truth of a declaration.

Specification diagnostics expose exact population identities and deterministic experiments for fork, collider, and confounded-mediator structures. They are mathematical diagnostics rather than automated model selection.

The general-variance mirage functions retain the source's explicit disturbance variances. Allocation diagnostics solve the two released minimum-variance systems, then evaluate misspecified weights against the reference exposure system. They do not select an exposure model or repair a misspecified graph.

Factor-role functions classify a supplied accepted DAG. `factor_control_roles` uses reachability, while `classify_treatment_outcome_role` uses the eight published one-hop signatures and rejects every unlisted signature. The structural-model evaluator applies caller-supplied mechanisms in a deterministic lexical topological order; it is not a random graph or mechanism generator.

False-discovery functions keep two estimands separate. Family-level FDR treats the searched family as null only when all trials are null. `max_selection_null_probability` instead conditions on the selected winner being null. The Gaussian likelihood is exposed, but the paper's empirical fitting pipeline is not: its data, optimization, initialization, and model-selection rules are not sufficiently frozen.

## Cross-language behavior

Python returns frozen evidence records and protected NumPy snapshots; Julia returns immutable structs containing independent Julia vector or matrix snapshots. Python raises `ValueError` for invalid analytical values and `TypeError` for invalid callback boundaries; Julia raises `ArgumentError`. Python represents a treatment/outcome role with `Enum`, while Julia uses a validated symbol-backed value. Python uses SciPy adaptive quadrature and Julia uses QuadGK for the selected-winner integral, so reported integration-error estimates are backend-specific. Canonical ordering, formulas, evidence fields, validation intent, tolerances, and estimands otherwise match.

Both implementations are tested against independent analytical results and graph oracles. Agreement between the languages is additional evidence, never the sole correctness test.

## Exclusions

The review leaves the following method units source-blocked rather than inventing missing mathematics:

| Method unit | Blocking issue |
|---|---|
| `ADIA-METRIC-01` | The displayed eight-class accuracy formula conflicts with the prose and stated random baseline. |
| `ADIA-PIPE-01` | Competition pipelines depend on unavailable trained assets, data, and unstated hyperparameters. |
| `ADIA-GEN-01` | Random graph and mechanism factories omit necessary distributions, scales, and policies. |
| `CDNOTS-01` | The end-to-end discovery schedule and orientation rules are incomplete and internally inconsistent. |
| `CDNOTS-02` | The conditional-independence tests are delegated to absent defining sources. |
| `CDNOTS-03` | The printed diagnostic and Ornstein-Uhlenbeck constructions conflict with their own equations. |
| `NECESSARY-COLL-01` | The same collider experiment prints three incompatible coefficient systems. |
| `NECESSARY-MC-01` | The paper-specific simulations and empirical studies lack a complete reproducibility contract. |
| `PRIMER-PC-01` | The proprietary data and consensus-discovery procedure are unavailable. |
| `MIRAGE-COLL-SHIFT-01` | The main exhibit and appendix condition on different quantities without reconciliation. |
| `MIRAGE-EMP-01` | The empirical study lacks data and complete discovery settings. |
| `FDR-12` | The empirical fit omits a frozen dataset, optimizer, initialization, tolerances, and trial-count policy. |
| `ADIA-VOL-14` | The model-assisted workflow is not a deterministic general identification algorithm. |

The clean namespace does not delegate to excluded legacy causal implementations.

See `../examples/causal_factor_analysis_quickstart.py` for a deterministic example and `compatibility.md` for the tested runtime policy.

The exact ordered names, signatures, immutable record fields, implementation
modules, and source hashes are recorded in `../PUBLIC_API.json`. The causal
test files, independent mathematical oracles, and shared numerical parity
fixture are recorded in `../TEST_INVENTORY.json`.

Cross-language parity is defined over the exact 87-name set and behavior, not
over incidental export-list order. Python preserves its approved `__all__`
order; the Julia inventory uses canonical lexical order. Both language-native
digests and the canonical name-set digest are recorded in the blocked
pre-package contract.
