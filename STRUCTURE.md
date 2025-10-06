# Project File Structure

```
📁 RiskLabAI.py/
├── 📁 data/
│   └── 📄 all.py
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
│   │   ├── 📄 backtest_statistics.py
│   │   ├── 📄 backtest_synthetic_data.py
│   │   ├── 📄 backtset_overfitting_in_the_machine_learning_era_simulation.py
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
│   │   │   ├── 📄 data_structures_lopez.py
│   │   │   ├── 📄 filtering_lopez.py
│   │   │   ├── 📄 hedging.py
│   │   │   ├── 📄 imbalance_bars.py
│   │   │   ├── 📄 infomation_driven_bars.py
│   │   │   ├── 📄 run_bars.py
│   │   │   ├── 📄 standard_bars.py
│   │   │   ├── 📄 standard_bars_lopez.py
│   │   │   ├── 📄 time_bars.py
│   │   │   └── 📄 utilities_lopez.py
│   │   ├── 📁 synthetic_data/
│   │   │   ├── 📄 __init__.py
│   │   │   ├── 📄 drift_burst_hypothesis.py
│   │   │   └── 📄 synthetic_controlled_environment.py
│   │   ├── 📁 weights/
│   │   │   ├── 📄 __init__.py
│   │   │   └── 📄 sample_weights.py
│   │   └── 📄 __init__.py
│   ├── 📁 ensemble/
│   │   ├── 📄 __init__.py
│   │   └── 📄 bagging_classifier_accuracy.py
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
│   │   │   ├── 📄 clustering.py
│   │   │   ├── 📄 feature_importance_controller.py
│   │   │   ├── 📄 feature_importance_factory.py
│   │   │   ├── 📄 feature_importance_mda.py
│   │   │   ├── 📄 feature_importance_mdi.py
│   │   │   ├── 📄 feature_importance_sfi.py
│   │   │   ├── 📄 feature_importance_strategy.py
│   │   │   ├── 📄 FeatureImportance.ipynb
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
│   │   ├── 📄 __init__.py
│   │   └── 📄 test.ipynb
│   ├── 📁 hpc/
│   │   ├── 📄 __init__.py
│   │   └── 📄 hpc.py
│   ├── 📁 optimization/
│   │   ├── 📄 __init__.py
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
│   │   ├── 📄 smoothing_average.py
│   │   └── 📄 update_figure_layout.py
│   └── 📄 __init__.py
├── 📁 test/
│   └── 📄 delete
├── 📄 .gitignore
├── 📄 .pypirc
├── 📄 LICENSE
├── 📄 pyproject.toml
├── 📄 README.md
├── 📄 setup.cfg
├── 📄 STRUCTURE.md
├── 📄 style_guide.md
└── 📄 tree.py
```
