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
    FBSNNolver,  # deprecated alias (removed in 2.1.0)
    FBSNNSolver,
    initialize_weights,
)

__all__ = [
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
    "FBSNNolver",  # deprecated alias (removed in 2.1.0)
]
