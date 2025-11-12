"""
RiskLabAI Portfolio Optimization Module

Implements advanced portfolio optimization techniques, including:
- Hierarchical Risk Parity (HRP)
- Nested Clustered Optimisation (NCO)
- PCA-Based Hedging
- Custom Hyper-Parameter Tuning
"""

from .hrp import (
    get_cluster_var,
    get_quasi_diag,
    get_rec_bipart,
    hrp_alloc,
)

from .nco import (
    cluster_kmeans_base,
    optimal_portfolio,
    nco_alloc,
)

from .hedging import (
    pca_hedge_weights,
)

from .hyper_parameter_tuning import (
    HyperParameterTuning,
)

__all__ = [
    # hrp.py
    "get_cluster_var",
    "get_quasi_diag",
    "get_rec_bipart",
    "hrp_alloc",
    
    # nco.py
    "cluster_kmeans_base",
    "optimal_portfolio",
    "nco_alloc",
    
    # hedging.py
    "pca_hedge_weights",
    
    # hyper_parameter_tuning.py
    "HyperParameterTuning",
]