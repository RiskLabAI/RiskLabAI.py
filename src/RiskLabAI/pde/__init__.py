"""
RiskLabAI PDE Solver Module

Implements a Deep BSDE (Backward Stochastic Differential Equation)
solver for various financial PDEs.

This sub-package requires PyTorch, which is an optional dependency of
RiskLabAI. The base install stays torch-free: ``import RiskLabAI`` never pulls
in this sub-package (it is loaded lazily). Importing ``RiskLabAI.pde`` without
torch installed raises a clear, actionable error instead of a bare
``ModuleNotFoundError``.
"""

try:
    import torch as _torch  # noqa: F401  (presence check only)
except ImportError as _exc:  # pragma: no cover - exercised only without torch
    raise ImportError(
        "RiskLabAI.pde requires PyTorch, which is an optional dependency. "
        "Install it with:  pip install 'RiskLabAI[pde]'  (or: pip install torch)."
    ) from _exc

from .equation import (
    HJBLQ,
    BlackScholesBarenblatt,
    Equation,
    PricingDefaultRisk,
    PricingDiffRate,
)
from .model import (
    ISAB,
    MAB,
    PMA,
    SAB,
    DeepBSDE,
    DeepTimeSetTransformer,
    FBSNNNetwork,
    Net1,
    TimeDependentNetwork,
    TimeDependentNetworkMonteCarlo,
    TimeNet,
    TimeNetForSet,
)
from .solver import (
    FBSDESolver,
    FBSNNolver,  # deprecated alias (scheduled for removal in 4.0.0)
    FBSNNSolver,
    initialize_weights,
)
from .neural_sde import (
    normalized_call_grid_constraints,
    normalized_call_grid_witness,
    affine_price_state_interval,
    IntervalNeuralSDE,
    PolytopeNeuralSDE,
    fit_interval_neural_sde,
    simulate_interval_neural_sde,
    fit_polytope_neural_sde,
    simulate_polytope_neural_sde,
    hjm_drift_residual,
)

__all__ = [
    "normalized_call_grid_constraints",
    "normalized_call_grid_witness",
    "affine_price_state_interval",
    "IntervalNeuralSDE",
    "PolytopeNeuralSDE",
    "fit_interval_neural_sde",
    "simulate_interval_neural_sde",
    "fit_polytope_neural_sde",
    "simulate_polytope_neural_sde",
    "hjm_drift_residual",
    # Equations
    "Equation",
    "PricingDefaultRisk",
    "HJBLQ",
    "BlackScholesBarenblatt",
    "PricingDiffRate",
    # Models
    "TimeNet",
    "Net1",
    "MAB",
    "SAB",
    "ISAB",
    "PMA",
    "TimeNetForSet",
    "DeepTimeSetTransformer",
    "FBSNNNetwork",
    "DeepBSDE",
    "TimeDependentNetwork",
    "TimeDependentNetworkMonteCarlo",
    # Solvers
    "initialize_weights",
    "FBSDESolver",
    "FBSNNSolver",
    "FBSNNolver",  # deprecated alias (scheduled for removal in 4.0.0)
]
