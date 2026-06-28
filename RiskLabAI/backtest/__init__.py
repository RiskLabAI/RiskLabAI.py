"""
RiskLabAI Backtesting Module

This package provides tools for:
- Backtest simulation
- Bet sizing and strategy risk analysis
- Overfitting detection (PBO, PSR)
- Advanced cross-validation (in the .validation submodule)
"""

from . import validation
from .advanced_bet_sizing import (
    PlattCalibrator,
    distributionally_robust_kelly_fraction,
    expected_calibration_error,
    kelly_bet_fraction,
)
from .backtest_overfitting_simulation import (
    backtest_overfitting_simulation_financial_metrics_rank_correlation,
    backtest_overfitting_simulation_model_complexity,
    format_cpu_info,
    get_cpu_info,
    measure_all_cv_computational_requirements,
    measure_cpcv_parallelization,
    measure_cpcv_scalability,
    noised_backtest_overfitting_simulation,
    # This file contains many functions, exporting the main ones
    overall_backtest_overfitting_simulation,
    overall_novel_methods_backtest_overfitting_simulation,
    temporal_backtest_overfitting_simulation,
    time_temporal_backtest_overfitting_simulation,
    varying_embargo_backtest_overfitting_simulation,
)
from .backtest_statistics import (
    bet_timing,
    calculate_hhi,
    calculate_hhi_concentration,
    calculate_holding_period,
    compute_drawdowns_time_under_water,
)
from .backtest_statistics import (
    sharpe_ratio as pbo_sharpe_ratio,
)
from .backtest_synthetic_data import synthetic_back_testing
from .bet_sizing import (
    Signal,
    TPos,
    average_bet_sizes,
    avg_active_signals,
    avgActiveSignals,
    bet_size_sigmoid,
    betSize,
    compute_sigmoid_width,
    discrete_signal,
    discreteSignal,
    generate_signal,
    getW,
    inverse_price,
    inversePrice,
    limit_price,
    limitPrice,
    mp_avg_active_signals,
    mpAvgActiveSignals,
    probability_bet_size,
    strategy_bet_sizing,
    target_position,
)
from .multiple_testing import (
    benjamini_hochberg_yekutieli_adjusted_p_values,
    haircut_sharpe_ratios,
    holm_adjusted_p_values,
    sharpe_ratio_p_values,
)
from .ou_trading_rules import (
    fit_ornstein_uhlenbeck,
    optimal_ou_trading_rule,
    ou_rule_metrics,
)
from .probabilistic_sharpe_ratio import (
    benchmark_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from .probability_of_backtest_overfitting import (
    performance_evaluation,
    probability_of_backtest_overfitting,
)
from .robust_statistics import (
    conditional_expected_drawdown,
    sharpe_difference_test,
)
from .sharpe_inference import (
    lplz_sharpe_inference,
    newey_west_automatic_lag,
    newey_west_long_run_variance,
    sharpe_ratio_influence_function,
)
from .strategy_risk import (
    bin_frequency,
    binomial_sharpe_ratio,
    calculate_strategy_risk,
    failure_probability,
    implied_precision,
    mix_gaussians,
    sharpe_ratio_trials,
    target_sharpe_ratio_symbolic,
)
from .test_set_overfitting import (
    estimated_sharpe_ratio_z_statistics,
    expected_max_sharpe_ratio,
    generate_max_sharpe_ratios,
    mean_std_error,
    strategy_type1_error_probability,
    strategy_type2_error_probability,
    theta_for_type2_error,
)

# Define what `from RiskLabAI.backtest import *` will import
__all__ = [
    "validation",
    # from advanced_bet_sizing (Appraisal 18 admits)
    "PlattCalibrator",
    "distributionally_robust_kelly_fraction",
    "expected_calibration_error",
    "kelly_bet_fraction",
    # from robust_statistics + ou_trading_rules (Appraisal 22, 23 admits)
    "conditional_expected_drawdown",
    "sharpe_difference_test",
    "optimal_ou_trading_rule",
    "ou_rule_metrics",
    "fit_ornstein_uhlenbeck",
    # from backtest_statistics
    "bet_timing",
    "calculate_holding_period",
    "calculate_hhi_concentration",
    "calculate_hhi",
    "compute_drawdowns_time_under_water",
    # from backtest_synthetic_data
    "synthetic_back_testing",
    # from bet_sizing (canonical 2.0.0 names)
    "probability_bet_size",
    "average_bet_sizes",
    "strategy_bet_sizing",
    "avg_active_signals",
    "mp_avg_active_signals",
    "discrete_signal",
    "generate_signal",
    "bet_size_sigmoid",
    "target_position",
    "inverse_price",
    "limit_price",
    "compute_sigmoid_width",
    # from bet_sizing (deprecated aliases, removed in 2.1.0)
    "avgActiveSignals",
    "mpAvgActiveSignals",
    "discreteSignal",
    "Signal",
    "betSize",
    "TPos",
    "inversePrice",
    "limitPrice",
    "getW",
    # from strategy_risk
    "sharpe_ratio_trials",
    "target_sharpe_ratio_symbolic",
    "implied_precision",
    "bin_frequency",
    "binomial_sharpe_ratio",
    "mix_gaussians",
    "failure_probability",
    "calculate_strategy_risk",
    # from test_set_overfitting
    "expected_max_sharpe_ratio",
    "generate_max_sharpe_ratios",
    "mean_std_error",
    "estimated_sharpe_ratio_z_statistics",
    "strategy_type1_error_probability",
    "theta_for_type2_error",
    "strategy_type2_error_probability",
    # from probability_of_backtest_overfitting
    "pbo_sharpe_ratio",
    "performance_evaluation",
    "probability_of_backtest_overfitting",
    # from probabilistic_sharpe_ratio
    "probabilistic_sharpe_ratio",
    "benchmark_sharpe_ratio",
    # from multiple_testing
    "sharpe_ratio_p_values",
    "holm_adjusted_p_values",
    "benjamini_hochberg_yekutieli_adjusted_p_values",
    "haircut_sharpe_ratios",
    # from sharpe_inference
    "sharpe_ratio_influence_function",
    "newey_west_long_run_variance",
    "newey_west_automatic_lag",
    "lplz_sharpe_inference",
    # from backtest_overfitting_..._simulation
    "overall_backtest_overfitting_simulation",
    "temporal_backtest_overfitting_simulation",
    "time_temporal_backtest_overfitting_simulation",
    "varying_embargo_backtest_overfitting_simulation",
    "backtest_overfitting_simulation_financial_metrics_rank_correlation",
    "backtest_overfitting_simulation_model_complexity",
    "noised_backtest_overfitting_simulation",
    "overall_novel_methods_backtest_overfitting_simulation",
    "measure_all_cv_computational_requirements",
    "measure_cpcv_parallelization",
    "measure_cpcv_scalability",
    "get_cpu_info",
    "format_cpu_info",
]
