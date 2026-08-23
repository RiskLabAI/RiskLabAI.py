"""
The built-in component catalogue.

Defines one :class:`~RiskLabAI.core.registry.Registry` per model family and
populates it with the components the library already ships. Registration is
**lazy** (by import path), so importing this module — and therefore
``RiskLabAI.core`` — does not import pandas, numba, scikit-learn or torch. The
heavy import only happens when a component is actually created.

Families currently exposed only as free functions (labeling, bet sizing,
portfolio optimization) get an empty registry here: it is the extension point
where new class-based models register themselves (see ``EXTENDING.md``).

The registries are deliberately a *superset* of the existing hand-written
factories (``CrossValidatorFactory``, ``FeatureImportanceFactory``,
``BarsInitializerController``), which remain unchanged for backward
compatibility. ``test/core/test_builtin_parity.py`` asserts the registries stay
in sync with those factories so the two cannot silently drift apart.
"""

from __future__ import annotations

from .registry import Registry

__all__ = [
    "BARS",
    "CROSS_VALIDATORS",
    "FEATURE_IMPORTANCE",
    "LABELERS",
    "BET_SIZERS",
    "PORTFOLIO_OPTIMIZERS",
    "REGISTRIES",
    "get_registry",
    "list_components",
]

# --------------------------------------------------------------------------- #
# Bar structures. The extension point for a new bar type is a new AbstractBars
# subclass; register it here (or via @BARS.register in your own code).
# --------------------------------------------------------------------------- #
BARS = Registry("bars")
BARS.register_lazy(
    "standard_bars",
    "RiskLabAI.data.structures.standard_bars:StandardBars",
    metadata={"family": "bars", "afml_chapter": 2},
)
BARS.register_lazy(
    "time_bars",
    "RiskLabAI.data.structures.time_bars:TimeBars",
    metadata={"family": "bars", "afml_chapter": 2},
)
BARS.register_lazy(
    "expected_imbalance_bars",
    "RiskLabAI.data.structures.imbalance_bars:ExpectedImbalanceBars",
    metadata={"family": "bars", "afml_chapter": 2},
)
BARS.register_lazy(
    "fixed_imbalance_bars",
    "RiskLabAI.data.structures.imbalance_bars:FixedImbalanceBars",
    metadata={"family": "bars", "afml_chapter": 2},
)
BARS.register_lazy(
    "expected_run_bars",
    "RiskLabAI.data.structures.run_bars:ExpectedRunBars",
    metadata={"family": "bars", "afml_chapter": 2},
)
BARS.register_lazy(
    "fixed_run_bars",
    "RiskLabAI.data.structures.run_bars:FixedRunBars",
    metadata={"family": "bars", "afml_chapter": 2},
)

# --------------------------------------------------------------------------- #
# Cross-validators. Keys mirror CrossValidatorFactory.VALIDATORS exactly.
# --------------------------------------------------------------------------- #
CROSS_VALIDATORS = Registry("cross_validators")
CROSS_VALIDATORS.register_lazy(
    "kfold",
    "RiskLabAI.backtest.validation.kfold:KFold",
    metadata={"family": "cross_validator"},
)
CROSS_VALIDATORS.register_lazy(
    "walkforward",
    "RiskLabAI.backtest.validation.walk_forward:WalkForward",
    metadata={"family": "cross_validator"},
)
CROSS_VALIDATORS.register_lazy(
    "purgedkfold",
    "RiskLabAI.backtest.validation.purged_kfold:PurgedKFold",
    metadata={"family": "cross_validator"},
)
CROSS_VALIDATORS.register_lazy(
    "combinatorialpurged",
    "RiskLabAI.backtest.validation.combinatorial_purged:CombinatorialPurged",
    metadata={"family": "cross_validator"},
)
CROSS_VALIDATORS.register_lazy(
    "baggedcombinatorialpurged",
    "RiskLabAI.backtest.validation.bagged_combinatorial_purged:BaggedCombinatorialPurged",
    metadata={"family": "cross_validator"},
)
CROSS_VALIDATORS.register_lazy(
    "adaptivecombinatorialpurged",
    "RiskLabAI.backtest.validation.adaptive_combinatorial_purged:AdaptiveCombinatorialPurged",
    metadata={"family": "cross_validator"},
)

# --------------------------------------------------------------------------- #
# Feature-importance strategies. Keys mirror FeatureImportanceFactory exactly.
# --------------------------------------------------------------------------- #
FEATURE_IMPORTANCE = Registry("feature_importance")
FEATURE_IMPORTANCE.register_lazy(
    "MDI",
    "RiskLabAI.features.feature_importance.feature_importance_mdi:FeatureImportanceMDI",
    metadata={"family": "feature_importance"},
)
FEATURE_IMPORTANCE.register_lazy(
    "ClusteredMDI",
    "RiskLabAI.features.feature_importance.clustered_feature_importance_mdi:ClusteredFeatureImportanceMDI",
    metadata={"family": "feature_importance"},
)
FEATURE_IMPORTANCE.register_lazy(
    "MDA",
    "RiskLabAI.features.feature_importance.feature_importance_mda:FeatureImportanceMDA",
    metadata={"family": "feature_importance"},
)
FEATURE_IMPORTANCE.register_lazy(
    "ClusteredMDA",
    "RiskLabAI.features.feature_importance.clustered_feature_importance_mda:ClusteredFeatureImportanceMDA",
    metadata={"family": "feature_importance"},
)
FEATURE_IMPORTANCE.register_lazy(
    "SFI",
    "RiskLabAI.features.feature_importance.feature_importance_sfi:FeatureImportanceSFI",
    metadata={"family": "feature_importance"},
)

# --------------------------------------------------------------------------- #
# Free-function families: empty registries, ready for new class-based models.
# These correspond to the BaseLabeler / BaseBetSizer / BasePortfolioOptimizer
# contracts in RiskLabAI.core.base.
# --------------------------------------------------------------------------- #
LABELERS = Registry("labelers")
BET_SIZERS = Registry("bet_sizers")
PORTFOLIO_OPTIMIZERS = Registry("portfolio_optimizers")

# --------------------------------------------------------------------------- #
# Family name -> registry, for discovery and a unified catalogue.
# --------------------------------------------------------------------------- #
REGISTRIES: dict[str, Registry] = {
    "bars": BARS,
    "cross_validators": CROSS_VALIDATORS,
    "feature_importance": FEATURE_IMPORTANCE,
    "labelers": LABELERS,
    "bet_sizers": BET_SIZERS,
    "portfolio_optimizers": PORTFOLIO_OPTIMIZERS,
}


def get_registry(family: str) -> Registry:
    """
    Return the registry for a model family.

    Parameters
    ----------
    family : str
        One of the keys of :data:`REGISTRIES` (e.g. ``"bars"``,
        ``"cross_validators"``).

    Raises
    ------
    KeyError
        If the family name is unknown (the message lists valid families).
    """
    try:
        return REGISTRIES[family]
    except KeyError:
        valid = ", ".join(repr(k) for k in sorted(REGISTRIES))
        raise KeyError(
            f"{family!r} is not a known model family. Valid families: {valid}."
        ) from None


def list_components() -> dict[str, list[str]]:
    """
    Return a catalogue mapping each family name to its available component keys.

    Useful for discovery and for documentation that should not drift::

        >>> from RiskLabAI.core import list_components
        >>> catalogue = list_components()
        >>> catalogue["cross_validators"]              # doctest: +SKIP
        ['adaptivecombinatorialpurged', 'baggedcombinatorialpurged', ...]
    """
    return {family: reg.available() for family, reg in REGISTRIES.items()}
