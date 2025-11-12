"""
RiskLabAI Synthetic Data Module

Provides tools for generating synthetic financial data, from
simple block-diagonal covariance matrices to complex,
regime-switching Heston-Merton price paths.
"""

from .drift_burst_hypothesis import drift_volatility_burst
from .synthetic_controlled_environment import (
    compute_log_returns,
    heston_merton_log_returns,
    align_params_length,
    generate_prices_from_regimes,
    parallel_generate_prices,
)
from .simulation import (
    random_cov,
    form_block_matrix,
    form_true_matrix,
    simulates_cov_mu,
)

__all__ = [
    "drift_volatility_burst",
    "compute_log_returns",
    "heston_merton_log_returns",
    "align_params_length",
    "generate_prices_from_regimes",
    "parallel_generate_prices",
    "random_cov",
    "form_block_matrix",
    "form_true_matrix",
    "simulates_cov_mu",
]