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
core
    The extension layer: a component registry, base interfaces per model
    family, and a built-in catalogue. Start here to add a new model. See
    ``EXTENDING.md``.
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

Logging
-------
RiskLabAI uses the standard ``logging`` module under the ``"RiskLabAI"`` logger
and is silent by default (a ``NullHandler`` is attached). To see progress and
diagnostics, configure logging in your application, e.g.::

    import logging
    logging.basicConfig(level=logging.INFO)
"""

import logging as _logging
from importlib import import_module

# Attach a NullHandler so the library never emits "No handlers could be found"
# and stays silent unless the application opts in by configuring logging.
_logging.getLogger(__name__).addHandler(_logging.NullHandler())

_SUBMODULES = frozenset(
    {
        "backtest",
        "cluster",
        "controller",
        "core",
        "data",
        "ensemble",
        "features",
        "hpc",
        "optimization",
        "pde",
        "utils",
    }
)

# Single source of truth for the version is pyproject.toml.
try:
    from importlib.metadata import version as _version

    __version__ = _version("RiskLabAI")
except Exception:  # pragma: no cover - package not installed (e.g. source tree)
    __version__ = "0.0.0+unknown"

__all__ = sorted(_SUBMODULES) + ["__version__"]


def __getattr__(name):
    """
    Lazily import sub-packages on first attribute access (PEP 562).

    Keeps `import RiskLabAI` lightweight: optional heavy dependencies
    (torch for `pde`, plotting libraries, etc.) are only imported when
    the sub-package that needs them is actually used.
    """
    if name in _SUBMODULES:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
