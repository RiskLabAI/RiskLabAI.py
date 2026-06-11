"""
RiskLabAI: Financial AI with Python

This library is a Python-based implementation of advanced
methods for quantitative finance and financial machine learning,
based on the work of Marcos López de Prado.

Sub-modules
-----------
backtest
    Tools for robust backtesting, including PBO, PSR, and
    advanced cross-validation (PurgedKFold, CPCV).
cluster
    Algorithms for portfolio clustering (ONC).
controller
    High-level controllers for bar generation and data processing.
data
    Core modules for data processing, including bar generation,
    labeling, denoising, and synthetic data.
ensemble
    Functions related to ensemble methods.
features
    Modules for feature generation and importance analysis
    (MDI, MDA, SFI, structural breaks, entropy, microstructure).
hpc
    Utilities for high-performance computing and parallel processing.
optimization
    Portfolio optimization algorithms, including HRP and NCO.
pde
    Implementation of a Deep BSDE solver for PDEs.
utils
    Common helper functions, constants, and plotting utilities.
"""

# Import all sub-packages to make them available
from . import backtest
from . import cluster
from . import controller
from . import data
from . import ensemble
from . import features
from . import hpc
from . import optimization
# from . import pde # Temporarily disabled to prevent torch crash
from . import utils

# Single source of truth for the version is pyproject.toml.
try:
    from importlib.metadata import version as _version

    __version__ = _version("RiskLabAI")
except Exception:  # pragma: no cover - package not installed (e.g. source tree)
    __version__ = "0.0.0+unknown"

__all__ = [
    "backtest",
    "cluster",
    "controller",
    "data",
    "ensemble",
    "features",
    "hpc",
    "optimization",
    "pde",
    "utils",
    "__version__",
]