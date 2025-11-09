"""
RiskLabAI PDE Solver Module

Implements a Deep BSDE (Backward Stochastic Differential Equation)
solver for various financial PDEs.
"""

from .equation import (
    Equation,
    PricingDefaultRisk,
    HJBLQ,
    BlackScholesBarenblatt,
    PricingDiffRate,
)
from .model import (
    TimeNet,
    Net1,
    MAB,
    SAB,
    ISAB,
    PMA,
    TimeNetForSet,
    DeepTimeSetTransformer,
    FBSNNNetwork,
    DeepBSDE,
    TimeDependentNetwork,
    TimeDependentNetworkMonteCarlo,
)
from .solver import (
    initialize_weights,
    FBSDESolver,
    FBSNNolver, # Note: Typo in original filename?
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
    "MAB", "SAB", "ISAB", "PMA",
    "TimeNetForSet",
    "DeepTimeSetTransformer",
    "FBSNNNetwork",
    "DeepBSDE",
    "TimeDependentNetwork",
    "TimeDependentNetworkMonteCarlo",
    
    # Solvers
    "initialize_weights",
    "FBSDESolver",
    "FBSNNolver",
]