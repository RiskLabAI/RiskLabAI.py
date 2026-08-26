"""RiskLabAI financial machine-learning and causal-factor toolkit."""

from importlib import import_module as _import_module

from . import causal_factor_analysis

__version__ = "3.1.0"

_LEGACY_SUBMODULES = frozenset(
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

__all__ = [
    "__version__",
    "backtest",
    "causal_factor_analysis",
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
]


def __getattr__(name):
    """Load only a declared legacy subpackage when it is first requested."""
    if name in _LEGACY_SUBMODULES:
        module = _import_module(".{}".format(name), __name__)
        globals()[name] = module
        return module
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))


def __dir__():
    """Return the exact declared package surface."""
    return list(__all__)
