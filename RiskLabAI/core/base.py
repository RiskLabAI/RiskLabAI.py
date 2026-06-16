"""
Base interfaces (contracts) for RiskLabAI model families.

This module gives every model family a single, documented contract that new
implementations can target. Two kinds of contract live here:

1. **Canonical interfaces that already exist** elsewhere in the library are
   re-exported from here so there is one obvious place to look:
   :class:`AbstractBars`, :class:`CrossValidator`, and
   :class:`FeatureImportanceStrategy`. These are re-exported *lazily* (PEP 562)
   so that ``import RiskLabAI.core`` does not eagerly pull in pandas/numba and
   the rest of the data/backtest/features sub-packages.

2. **New, optional abstract base classes** for families that are currently
   exposed only as free functions (labeling, bet sizing, portfolio
   optimization). They are purely additive: existing functions keep working
   unchanged. A *new* model can subclass one of these to advertise a uniform
   interface (and to be registered and discovered through
   :mod:`RiskLabAI.core.registry`).

Plus a structural :class:`Estimator` protocol describing the scikit-learn-style
``fit``/``predict`` objects the cross-validation and feature-importance code
already expects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Protocol, runtime_checkable

import pandas as pd

__all__ = [
    # Structural protocol
    "Estimator",
    # New optional contracts for free-function families
    "BaseLabeler",
    "BaseBetSizer",
    "BasePortfolioOptimizer",
    # Re-exported canonical interfaces (lazy)
    "AbstractBars",
    "BarBuilder",
    "CrossValidator",
    "FeatureImportanceStrategy",
]


@runtime_checkable
class Estimator(Protocol):
    """
    Structural type for a scikit-learn-style estimator.

    Anything with ``fit`` and ``predict`` satisfies this protocol; no
    inheritance is required (it is ``runtime_checkable``, so
    ``isinstance(model, Estimator)`` works for duck-typed models). This is the
    object the cross-validation and feature-importance machinery consumes.
    """

    def fit(self, X: Any, y: Any = None, **kwargs: Any) -> Any: ...

    def predict(self, X: Any) -> Any: ...


class BaseLabeler(ABC):
    """
    Optional contract for a labeling method (e.g. triple-barrier, trend-scanning).

    The library's existing labelers are free functions and remain so. New
    class-based labelers may subclass this to expose a uniform ``label`` entry
    point and be registered in the ``labelers`` registry. See
    ``EXTENDING.md`` for a worked example.
    """

    @abstractmethod
    def label(
        self,
        prices: pd.Series,
        events: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Produce labels for the given price series.

        Parameters
        ----------
        prices : pd.Series
            Price (or log-price) series indexed by timestamp.
        events : pd.DataFrame, optional
            Event definitions (e.g. event start times, vertical barriers,
            target sizes) where applicable.
        **kwargs
            Method-specific options.

        Returns
        -------
        pd.DataFrame
            Labels, conventionally indexed by event time.
        """
        raise NotImplementedError


class BaseBetSizer(ABC):
    """
    Optional contract for a bet-sizing method.

    Maps model outputs (e.g. predicted probabilities) to a position size in
    ``[-1, 1]``. New class-based bet sizers may subclass this and register in
    the ``bet_sizers`` registry.
    """

    @abstractmethod
    def bet_size(self, probabilities: pd.Series, **kwargs: Any) -> pd.Series:
        """
        Convert prediction probabilities into signed bet sizes.

        Parameters
        ----------
        probabilities : pd.Series
            Predicted probabilities (or scores) indexed by timestamp.
        **kwargs
            Method-specific options.

        Returns
        -------
        pd.Series
            Signed bet sizes, conventionally in ``[-1, 1]``.
        """
        raise NotImplementedError


class BasePortfolioOptimizer(ABC):
    """
    Optional contract for a portfolio-construction method (e.g. HRP, NCO).

    New class-based optimizers may subclass this and register in the
    ``portfolio_optimizers`` registry.
    """

    @abstractmethod
    def weights(self, returns: pd.DataFrame, **kwargs: Any) -> pd.Series:
        """
        Compute portfolio weights.

        Parameters
        ----------
        returns : pd.DataFrame
            Asset returns (rows = observations, columns = assets). Some
            optimizers may accept a covariance matrix instead; document the
            expectation in the concrete subclass.
        **kwargs
            Method-specific options.

        Returns
        -------
        pd.Series
            Portfolio weights indexed by asset.
        """
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Lazy re-exports of the canonical interfaces that already live in the library.
# Importing them here eagerly would pull in the heavy data/backtest/features
# sub-package __init__ chains, defeating the lazy-import design. PEP 562
# __getattr__ defers that cost until the name is actually accessed.
# --------------------------------------------------------------------------- #
_LAZY_REEXPORTS = {
    "AbstractBars": "RiskLabAI.data.structures.abstract_bars:AbstractBars",
    "BarBuilder": "RiskLabAI.data.structures.abstract_bars:AbstractBars",
    "CrossValidator": (
        "RiskLabAI.backtest.validation.cross_validator_interface:CrossValidator"
    ),
    "FeatureImportanceStrategy": (
        "RiskLabAI.features.feature_importance.feature_importance_strategy"
        ":FeatureImportanceStrategy"
    ),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_REEXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module_path, _, attribute = target.partition(":")
    obj = getattr(importlib.import_module(module_path), attribute)
    globals()[name] = obj  # cache for next access
    return obj


def __dir__() -> list:
    return sorted(__all__)
