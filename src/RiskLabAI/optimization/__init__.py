"""
RiskLabAI Portfolio Optimization Module

Implements advanced portfolio optimization techniques, including:
- Hierarchical Risk Parity (HRP)
- Nested Clustered Optimisation (NCO)
- PCA-Based Hedging
- Custom Hyper-Parameter Tuning
"""

from .clearing import eisenberg_noe_clearing
from .constrained import constrained_minimum_variance, robust_mean_variance
from .decision_focused import spo_plus_loss, spo_plus_torch
from .execution import almgren_chriss_execution
from .hedging import (
    pca_weights,
)
from .hrp import (
    cluster_variance,
    hrp,
    quasi_diagonal,
    recursive_bisection,
)
from .hyper_parameter_tuning import (
    MyPipeline,  # deprecated alias (scheduled for removal in 4.0.0)
    SampleWeightedPipeline,
    clf_hyper_fit,
)
from .majorization import (
    birkhoff_von_neumann_decomposition,
    is_doubly_stochastic,
    is_majorized,
    majorization_matrix,
)
from .nco import (
    # cluster_kmeans_base is imported into nco.py, not defined there.
    # It should be imported from RiskLabAI.cluster.clustering directly
    # in any file that needs it, not from here.
    get_optimal_portfolio_weights,
    get_optimal_portfolio_weights_nco,
)

__all__ = [
    "constrained_minimum_variance",
    "robust_mean_variance",
    "spo_plus_loss",
    "spo_plus_torch",
    "eisenberg_noe_clearing",
    "almgren_chriss_execution",
    "birkhoff_von_neumann_decomposition",
    "is_doubly_stochastic",
    "is_majorized",
    "majorization_matrix",
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
    "SampleWeightedPipeline",
    "MyPipeline",  # deprecated alias (scheduled for removal in 4.0.0)
    "clf_hyper_fit",
]
