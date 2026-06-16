"""
RiskLabAI.core — the extension layer.

This package holds the small set of abstractions that make RiskLabAI *modular,
extendable and maintainable*: a reusable component :class:`Registry`, the base
interfaces every model family targets, and a built-in catalogue of the
components the library already ships.

It is the recommended entry point for adding a **new** model (for example, an
extension or alternative to a method from López de Prado's books): implement the
relevant base interface, register it in the matching registry, and it becomes
discoverable and constructible by name — without editing any central file. See
``EXTENDING.md`` at the repository root for a step-by-step guide.

Importing this package is cheap: every component is registered lazily, so heavy
dependencies are only imported when a component is actually created.

Quick tour
----------
>>> from RiskLabAI.core import list_components, CROSS_VALIDATORS
>>> sorted(list_components())                                    # doctest: +SKIP
['bars', 'bet_sizers', 'cross_validators', ...]
>>> "purgedkfold" in CROSS_VALIDATORS
True
>>> cv = CROSS_VALIDATORS.create("purgedkfold", n_splits=5)      # doctest: +SKIP
"""

from __future__ import annotations

from .base import (
    BaseBetSizer,
    BaseLabeler,
    BasePortfolioOptimizer,
    Estimator,
)
from ._builtins import (
    BARS,
    BET_SIZERS,
    CROSS_VALIDATORS,
    FEATURE_IMPORTANCE,
    LABELERS,
    PORTFOLIO_OPTIMIZERS,
    REGISTRIES,
    get_registry,
    list_components,
)
from .registry import Registry

__all__ = [
    # Registry machinery
    "Registry",
    "REGISTRIES",
    "get_registry",
    "list_components",
    # Per-family registries
    "BARS",
    "CROSS_VALIDATORS",
    "FEATURE_IMPORTANCE",
    "LABELERS",
    "BET_SIZERS",
    "PORTFOLIO_OPTIMIZERS",
    # Base contracts
    "Estimator",
    "BaseLabeler",
    "BaseBetSizer",
    "BasePortfolioOptimizer",
]
