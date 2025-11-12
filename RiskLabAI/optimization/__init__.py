"""
RiskLabAI Portfolio Optimization Module

Implements advanced portfolio optimization techniques, including:
- Hierarchical Risk Parity (HRP)
- Nested Clustered Optimisation (NCO)
- PCA-Based Hedging
- Custom Hyper-Parameter Tuning
"""

from .hrp import (
    cluster_variance,
    quasi_diagonal,
    recursive_bisection,
    hrp,
)

from .nco import (
    # cluster_kmeans_base is imported into nco.py, not defined there.
    # It should be imported from RiskLabAI.cluster.clustering directly
    # in any file that needs it, not from here.
    get_optimal_portfolio_weights,
    get_optimal_portfolio_weights_nco,
)

from .hedging import (
    pca_weights,
)

from .hyper_parameter_tuning import (
    MyPipeline,
    clf_hyper_fit,
)

__all__ = [
    # hrp.py
    "cluster_variance",
    "quasi_diagonal",
    "recursive_bisection",
    "hrp",
    
    # nco.py
    "get_optimal_portfolio_weights",
    "get_optimal_portfolio_weights_nco",
    
    # hedging.py
    "pca_weights",
    
    # hyper_parameter_tuning.py
    "MyPipeline",
    "clf_hyper_fit",
]