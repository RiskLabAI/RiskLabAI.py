"""
RiskLabAI Optimization Module

Implements advanced portfolio optimization techniques, including
Nested Clustered Optimization (NCO) and Hierarchical Risk Parity (HRP).
"""

from .hrp import (
    inverse_variance_weights,
    cluster_variance,
    quasi_diagonal,
    recursive_bisection,
    distance_corr,
    hrp,
)
from .nco import (
    get_optimal_portfolio_weights,
    get_optimal_portfolio_weights_nco,
)
from .hyper_parameter_tuning import (
    MyPipeline,
    clf_hyper_fit,
)

__all__ = [
    # HRP
    "inverse_variance_weights",
    "cluster_variance",
    "quasi_diagonal",
    "recursive_bisection",
    "distance_corr",
    "hrp",
    
    # NCO
    "get_optimal_portfolio_weights",
    "get_optimal_portfolio_weights_nco",
    
    # Tuning
    "MyPipeline",
    "clf_hyper_fit",
]