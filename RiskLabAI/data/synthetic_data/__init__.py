"""
RiskLabAI Synthetic Data Module

Provides tools for generating synthetic financial data, including:
- Drift-Burst Hypothesis (DBH) model.
- Heston-Merton model with Markov-switching regimes.
"""

from .drift_burst_hypothesis import drift_volatility_burst
from .synthetic_controlled_environment import (
    compute_log_returns,
    heston_merton_log_returns,
    align_params_length,
    generate_prices_from_regimes,
    parallel_generate_prices,
)

__all__ = [
    "drift_volatility_burst",
    "compute_log_returns",
    "heston_merton_log_returns",
    "align_params_length",
    "generate_prices_from_regimes",
    "parallel_generate_prices",
]