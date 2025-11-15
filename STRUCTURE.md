# Project File Structure

```
📁 RiskLabAI.py/
├── 📁 docs/
│   └── 📄 delete
├── 📁 RiskLabAI/
│   ├── 📁 backtest/
│   │   ├── 📁 validation/
│   │   │   ├── 📄 __init__.py
│   │   │   ├── 📄 adaptive_combinatorial_purged.py
│   │   │   ├── 📄 bagged_combinatorial_purged.py
│   │   │   ├── 📄 combinatorial_purged.py
│   │   │   ├── 📄 cross_validator_controller.py
│   │   │   ├── 📄 cross_validator_factory.py
│   │   │   ├── 📄 cross_validator_interface.py
│   │   │   ├── 📄 kfold.py
│   │   │   ├── 📄 purged_kfold.py
│   │   │   └── 📄 walk_forward.py
│   │   ├── 📄 __init__.py
│   │   ├── 📄 backtest_overfitting_simulation.py
│   │   ├── 📄 backtest_statistics.py
│   │   ├── 📄 backtest_synthetic_data.py
│   │   ├── 📄 bet_sizing.py
│   │   ├── 📄 probabilistic_sharpe_ratio.py
│   │   ├── 📄 probability_of_backtest_overfitting.py
│   │   ├── 📄 strategy_risk.py
│   │   └── 📄 test_set_overfitting.py
│   ├── 📁 cluster/
│   │   ├── 📄 __init__.py
│   │   └── 📄 clustering.py
│   ├── 📁 controller/
│   │   ├── 📄 __init__.py
│   │   ├── 📄 bars_initializer.py
│   │   └── 📄 data_structure_controller.py
│   ├── 📁 data/
│   │   ├── 📁 denoise/
│   │   │   ├── 📄 __init__.py
│   │   │   └── 📄 denoising.py
│   │   ├── 📁 differentiation/
│   │   │   ├── 📄 __init__.py
│   │   │   └── 📄 differentiation.py
│   │   ├── 📁 distance/
│   │   │   ├── 📄 __init__.py
│   │   │   └── 📄 distance_metric.py
│   │   ├── 📁 labeling/
│   │   │   ├── 📄 __init__.py
│   │   │   ├── 📄 financial_labels.py
│   │   │   └── 📄 labeling.py
│   │   ├── 📁 structures/
│   │   │   ├── 📄 __init__.py
│   │   │   ├── 📄 abstract_bars.py
│   │   │   ├── 📄 abstract_imbalance_bars.py
│   │   │   ├── 📄 abstract_information_driven_bars.py
│   │   │   ├── 📄 abstract_run_bars.py
│   │   │   ├── 📄 imbalance_bars.py
│   │   │   ├── 📄 run_bars.py
│   │   │   ├── 📄 standard_bars.py
│   │   │   └── 📄 time_bars.py
│   │   ├── 📁 synthetic_data/
│   │   │   ├── 📄 __init__.py
│   │   │   ├── 📄 drift_burst_hypothesis.py
│   │   │   ├── 📄 simulation.py
│   │   │   └── 📄 synthetic_controlled_environment.py
│   │   ├── 📁 weights/
│   │   │   ├── 📄 __init__.py
│   │   │   └── 📄 sample_weights.py
│   │   └── 📄 __init__.py
│   ├── 📁 ensemble/
│   │   ├── 📄 __init__.py
│   │   ├── 📄 bagging_classifier_accuracy.py
│   │   └── 📄 empirical_bagging_accuracy.py
│   ├── 📁 features/
│   │   ├── 📁 entropy_features/
│   │   │   ├── 📄 __init__.py
│   │   │   ├── 📄 entropy.py
│   │   │   ├── 📄 kontoyiannis.py
│   │   │   ├── 📄 lempel_ziv.py
│   │   │   ├── 📄 plug_in.py
│   │   │   ├── 📄 pmf.py
│   │   │   └── 📄 shannon.py
│   │   ├── 📁 feature_importance/
│   │   │   ├── 📄 __init__.py
│   │   │   ├── 📄 clustered_feature_importance_mda.py
│   │   │   ├── 📄 clustered_feature_importance_mdi.py
│   │   │   ├── 📄 feature_importance_controller.py
│   │   │   ├── 📄 feature_importance_factory.py
│   │   │   ├── 📄 feature_importance_mda.py
│   │   │   ├── 📄 feature_importance_mdi.py
│   │   │   ├── 📄 feature_importance_sfi.py
│   │   │   ├── 📄 feature_importance_strategy.py
│   │   │   ├── 📄 generate_synthetic_data.py
│   │   │   ├── 📄 orthogonal_features.py
│   │   │   └── 📄 weighted_tau.py
│   │   ├── 📁 microstructural_features/
│   │   │   ├── 📄 __init__.py
│   │   │   ├── 📄 bekker_parkinson_volatility_estimator.py
│   │   │   └── 📄 corwin_schultz.py
│   │   ├── 📁 structural_breaks/
│   │   │   ├── 📄 __init__.py
│   │   │   └── 📄 structural_breaks.py
│   │   └── 📄 __init__.py
│   ├── 📁 hpc/
│   │   ├── 📄 __init__.py
│   │   └── 📄 hpc.py
│   ├── 📁 optimization/
│   │   ├── 📄 __init__.py
│   │   ├── 📄 hedging.py
│   │   ├── 📄 hrp.py
│   │   ├── 📄 hyper_parameter_tuning.py
│   │   └── 📄 nco.py
│   ├── 📁 pde/
│   │   ├── 📄 __init__.py
│   │   ├── 📄 equation.py
│   │   ├── 📄 model.py
│   │   └── 📄 solver.py
│   ├── 📁 utils/
│   │   ├── 📄 __init__.py
│   │   ├── 📄 constants.py
│   │   ├── 📄 ewma.py
│   │   ├── 📄 momentum_mean_reverting_strategy_sides.py
│   │   ├── 📄 progress.py
│   │   ├── 📄 publication_plots.py
│   │   ├── 📄 smoothing_average.py
│   │   ├── 📄 update_figure_layout.py
│   │   └── 📄 utilities_lopez.py
│   └── 📄 __init__.py
├── 📁 test/
│   ├── 📁 backtest/
│   │   ├── 📁 validation/
│   │   │   ├── 📄 test_adaptive_combinatorial_purged.py
│   │   │   ├── 📄 test_bagged_combinatorial_purged.py
│   │   │   ├── 📄 test_combinatorial_purged.py
│   │   │   ├── 📄 test_cross_validator_controller.py
│   │   │   ├── 📄 test_cross_validator_factory.py
│   │   │   ├── 📄 test_kfold.py
│   │   │   ├── 📄 test_purged_kfold.py
│   │   │   └── 📄 test_walk_forward.py
│   │   ├── 📄 test_backtest_statistics.py
│   │   ├── 📄 test_backtest_synthetic_data.py
│   │   ├── 📄 test_bet_sizing.py
│   │   ├── 📄 test_probabilistic_sharpe_ratio.py
│   │   ├── 📄 test_probability_of_backtest_overfitting.py
│   │   ├── 📄 test_strategy_risk.py
│   │   ├── 📄 test_test_set_overfitting.py
│   │   └── 📄 teste_backtest_overfitting_simulation.py
│   ├── 📁 cluster/
│   │   └── 📄 test_clustering.py
│   ├── 📁 controller/
│   │   ├── 📄 test_bars_initializer.py
│   │   └── 📄 test_data_structure_controller.py
│   ├── 📁 data/
│   │   ├── 📁 denoise/
│   │   │   └── 📄 test_denoising.py
│   │   ├── 📁 differentiation/
│   │   │   └── 📄 test_differentiation.py
│   │   ├── 📁 distance/
│   │   │   └── 📄 test_distance_metric.py
│   │   ├── 📁 labeling/
│   │   │   ├── 📄 test_financial_labels.py
│   │   │   └── 📄 test_labeling.py
│   │   ├── 📁 structures/
│   │   │   ├── 📄 test_imbalance_bars.py
│   │   │   ├── 📄 test_run_bars.py
│   │   │   ├── 📄 test_standard_bars.py
│   │   │   └── 📄 test_time_bars.py
│   │   ├── 📁 synthetic_data/
│   │   │   ├── 📄 test_drift_burst_hypothesis.py
│   │   │   └── 📄 test_synthetic_controlled_environment.py
│   │   └── 📁 weights/
│   │       └── 📄 test_sample_weights.py
│   ├── 📁 ensemble/
│   │   └── 📄 test_bagging_classifier_accuracy.py
│   ├── 📁 features/
│   │   ├── 📁 entropy_features/
│   │   │   └── 📄 test_entropy.py
│   │   ├── 📁 feature_importance/
│   │   │   ├── 📄 test_feature_importance.py
│   │   │   ├── 📄 test_generate_synthetic_data.py
│   │   │   ├── 📄 test_orthogonal_features.py
│   │   │   └── 📄 test_weighted_tau.py
│   │   ├── 📁 microstructural_features/
│   │   │   └── 📄 test_microstructure.py
│   │   └── 📁 structural_breaks/
│   │       └── 📄 test_structural_breaks.py
│   ├── 📁 hpc/
│   │   └── 📄 test_hpc.py
│   ├── 📁 optimization/
│   │   ├── 📄 test_hedging.py
│   │   ├── 📄 test_hrp.py
│   │   ├── 📄 test_hyper_parameter_tuning.py
│   │   └── 📄 test_nco.py
│   ├── 📁 pde/
│   │   └── 📄 test_pde_solver.py
│   └── 📁 utils/
│       ├── 📄 test_ewma.py
│       ├── 📄 test_momentum_mean_reverting_strategy_sides.py
│       └── 📄 test_progress.py
├── 📄 .gitignore
├── 📄 .pypirc
├── 📄 desktop.ini
├── 📄 DOCUMENTATION.md
├── 📄 documenter.py
├── 📄 INSTALLATION.md
├── 📄 LICENSE
├── 📄 pyproject.toml
├── 📄 README.md
├── 📄 STRUCTURE.md
├── 📄 style_guide.md
└── 📄 tree.py
```
