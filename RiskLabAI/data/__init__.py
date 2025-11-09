"""
RiskLabAI Data Module

This package contains all modules related to financial data processing,
including:
- Bar data structures (standard, time, imbalance, run)
- Labeling (triple-barrier, trend-scanning)
- Fractional differentiation
- Covariance matrix denoising
- Distance metrics
- Sample weighting
- Synthetic data generation
"""

# Import sub-packages
from . import denoise
from . import differentiation
from . import distance
from . import labeling
from . import structures
from . import synthetic_data
from . import weights

__all__ = [
    "denoise",
    "differentiation",
    "distance",
    "labeling",
    "structures",
    "synthetic_data",
    "weights",
]