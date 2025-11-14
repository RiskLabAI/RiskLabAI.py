"""
RiskLabAI Backtesting Module

This package provides tools for:
- Backtest simulation
- Bet sizing and strategy risk analysis
- Overfitting detection (PBO, PSR)
- Advanced cross-validation (in the .validation submodule)
"""

from . import validation

from .backtest_statistics import (
    bet_timing,
    calculate_holding_period,
    calculate_hhi_concentration,
    calculate_hhi,
    compute_drawdowns_time_under_water,
    sharpe_ratio as pbo_sharpe_ratio,
)
from .backtest_synthetic_data import synthetic_back_testing
from .bet_sizing import (
    probability_bet_size,
    average_bet_sizes,
    strategy_bet_sizing,
    avgActiveSignals,
    mpAvgActiveSignals,
    discreteSignal,
    Signal,
    betSize,
    TPos,
    inversePrice,
    limitPrice,
    getW,
)
from .strategy_risk import (
    sharpe_ratio_trials,
    target_sharpe_ratio_symbolic,
    implied_precision,
    bin_frequency,
    binomial_sharpe_ratio,
    mix_gaussians,
    failure_probability,
    calculate_strategy_risk,
)
from .test_set_overfitting import (
    expected_max_sharpe_ratio,
    generate_max_sharpe_ratios,
    mean_std_error,
    estimated_sharpe_ratio_z_statistics,
    strategy_type1_error_probability,
    theta_for_type2_error,
    strategy_type2_error_probability,
)
from .probability_of_backtest_overfitting import (
    performance_evaluation,
    probability_of_backtest_overfitting,
)
from .probabilistic_sharpe_ratio import (
    probabilistic_sharpe_ratio,
    benchmark_sharpe_ratio,
)
from .backtest_overfitting_simulation import (
    # This file contains many functions, exporting the main ones
    overall_backtest_overfitting_simulation,
    temporal_backtest_overfitting_simulation,
    time_temporal_backtest_overfitting_simulation,
    varying_embargo_backtest_overfitting_simulation,
    backtest_overfitting_simulation_financial_metrics_rank_correlation,
    backtest_overfitting_simulation_model_complexity,
    noised_backtest_overfitting_simulation,
    overall_novel_methods_backtest_overfitting_simulation,
    measure_all_cv_computational_requirements,
    measure_cpcv_parallelization,
    measure_cpcv_scalability,
    get_cpu_info,
    format_cpu_info,
)

# Define what `from RiskLabAI.backtest import *` will import
__all__ = [
    "validation",
    # from backtest_statistics
    "bet_timing",
    "calculate_holding_period",
    "calculate_hhi_concentration",
    "calculate_hhi",
    "compute_drawdowns_time_under_water",
    # from backtest_synthetic_data
    "synthetic_back_testing",
    # from bet_sizing
    "probability_bet_size",
    "average_bet_sizes",
    "strategy_bet_sizing",
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