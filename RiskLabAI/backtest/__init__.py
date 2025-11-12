"""
RiskLabAI Backtesting Module

Implements a comprehensive suite of modern backtesting tools,
including performance metrics, Probabilistic Sharpe Ratio (PSR),
Probability of Backtest Overfitting (PBO), bet sizing, and
overfitting detection via CSCV.

Reference:
    De Prado, M. (2018) Advances in financial machine learning.
    John Wiley & Sons, Chapters 10-14.
"""

from .backtest_statistics import (
    sharpe_ratio,
    information_ratio,
    drawdown_and_time_under_water,
    compute_drawdown_series,
    annualized_sharpe_ratio,
    annualized_sortino_ratio,
    annualized_downside_deviation,
    calmar_ratio,
    compute_log_returns,
    compute_poly_fit,
)

from .bet_sizing import (
    get_signal,
    avg_active_signals,
    bet_size_probability,
    bet_size_dynamic,
    bet_size_power,
    get_target_pos,
)

from .probabilistic_sharpe_ratio import (
    probabilistic_sharpe_ratio,
    num_independent_trials,
    compute_psr_curve,
    evaluate_strategy_psr,
)

from .probability_of_backtest_overfitting import (
    get_backtest_sharpe_ratios,
    get_optimal_sharpe_ratio,
    get_pbo,
    pbo_overfitting_plot,
)

from .strategy_risk import (
    compute_tail_risk,
)

from .test_set_overfitting import (
    get_cs_folds,
    cs_cross_validation,
    log_loss,
    combinatorial_symmetric_cross_validation,
)

from .backtest_synthetic_data import (
    generate_market_data,
    get_signal_from_t_test,
)

from .backtest_overfitting_simulation import (
    run_backtest_overfitting_simulation,
)


__all__ = [
    # backtest_statistics.py
    "sharpe_ratio",
    "information_ratio",
    "drawdown_and_time_under_water",
    "compute_drawdown_series",
    "annualized_sharpe_ratio",
    "annualized_sortino_ratio",
    "annualized_downside_deviation",
    "calmar_ratio",
    "compute_log_returns",
    "compute_poly_fit",
    
    # bet_sizing.py
    "get_signal",
    "avg_active_signals",
    "bet_size_probability",
    "bet_size_dynamic",
    "bet_size_power",
    "get_target_pos",
    
    # probabilistic_sharpe_ratio.py
    "probabilistic_sharpe_ratio",
    "num_independent_trials",
    "compute_psr_curve",
    "evaluate_strategy_psr",
    
    # probability_of_backtest_overfitting.py
    "get_backtest_sharpe_ratios",
    "get_optimal_sharpe_ratio",
    "get_pbo",
    "pbo_overfitting_plot",
    
    # strategy_risk.py
    "compute_tail_risk",
    
    # test_set_overfitting.py
    "get_cs_folds",
    "cs_cross_validation",
    "log_loss",
    "combinatorial_symmetric_cross_validation",
    
    # backtest_synthetic_data.py
    "generate_market_data",
    "get_signal_from_t_test",
    
    # backtest_overfitting_simulation.py
    "run_backtest_overfitting_simulation",
]