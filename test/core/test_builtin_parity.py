"""
Parity tests: the built-in registries must stay in sync with the existing
hand-written factories. If someone adds a validator/strategy to a factory but
forgets the registry (or vice versa), these tests fail.
"""

import pytest

from RiskLabAI.core import (
    BARS,
    BET_SIZERS,
    CROSS_VALIDATORS,
    FEATURE_IMPORTANCE,
    LABELERS,
    PORTFOLIO_OPTIMIZERS,
    get_registry,
    list_components,
)
from RiskLabAI.core.base import CrossValidator


# --------------------------------------------------------------------------- #
# Cross-validators
# --------------------------------------------------------------------------- #
def test_cross_validator_registry_matches_factory():
    from RiskLabAI.backtest.validation.cross_validator_factory import (
        CrossValidatorFactory,
    )

    factory_keys = set(CrossValidatorFactory.VALIDATORS)
    registry_keys = {k.lower() for k in CROSS_VALIDATORS.available()}
    assert registry_keys == factory_keys

    # Same class object behind each key.
    for key, cls in CrossValidatorFactory.VALIDATORS.items():
        assert CROSS_VALIDATORS.get(key) is cls


def test_cross_validator_create_end_to_end():
    # KFold needs only n_splits; PurgedKFold etc. require a `times` series.
    cv = CROSS_VALIDATORS.create("kfold", n_splits=3, filter_unknown_kwargs=True)
    assert isinstance(cv, CrossValidator)


# --------------------------------------------------------------------------- #
# Feature importance
# --------------------------------------------------------------------------- #
def test_feature_importance_registry_matches_implementations():
    from RiskLabAI.features.feature_importance.clustered_feature_importance_mda import (
        ClusteredFeatureImportanceMDA,
    )
    from RiskLabAI.features.feature_importance.clustered_feature_importance_mdi import (
        ClusteredFeatureImportanceMDI,
    )
    from RiskLabAI.features.feature_importance.feature_importance_mda import (
        FeatureImportanceMDA,
    )
    from RiskLabAI.features.feature_importance.feature_importance_mdi import (
        FeatureImportanceMDI,
    )
    from RiskLabAI.features.feature_importance.feature_importance_sfi import (
        FeatureImportanceSFI,
    )

    expected = {
        "MDI": FeatureImportanceMDI,
        "ClusteredMDI": ClusteredFeatureImportanceMDI,
        "MDA": FeatureImportanceMDA,
        "ClusteredMDA": ClusteredFeatureImportanceMDA,
        "SFI": FeatureImportanceSFI,
    }
    assert set(FEATURE_IMPORTANCE.available()) == set(expected)
    for key, cls in expected.items():
        assert FEATURE_IMPORTANCE.get(key) is cls


# --------------------------------------------------------------------------- #
# Bars
# --------------------------------------------------------------------------- #
def test_bars_registry_matches_bar_classes():
    from RiskLabAI.data.structures.imbalance_bars import (
        ExpectedImbalanceBars,
        FixedImbalanceBars,
    )
    from RiskLabAI.data.structures.run_bars import (
        ExpectedRunBars,
        FixedRunBars,
    )
    from RiskLabAI.data.structures.standard_bars import StandardBars
    from RiskLabAI.data.structures.time_bars import TimeBars

    expected = {
        "standard_bars": StandardBars,
        "time_bars": TimeBars,
        "expected_imbalance_bars": ExpectedImbalanceBars,
        "fixed_imbalance_bars": FixedImbalanceBars,
        "expected_run_bars": ExpectedRunBars,
        "fixed_run_bars": FixedRunBars,
    }
    assert set(BARS.available()) == set(expected)
    for key, cls in expected.items():
        assert BARS.get(key) is cls


# --------------------------------------------------------------------------- #
# Catalogue / helpers / extension points
# --------------------------------------------------------------------------- #
def test_list_components_covers_all_families():
    catalogue = list_components()
    assert set(catalogue) == {
        "bars",
        "cross_validators",
        "feature_importance",
        "labelers",
        "bet_sizers",
        "portfolio_optimizers",
    }


def test_free_function_families_start_empty():
    # These are extension points for new class-based models.
    assert LABELERS.available() == []
    assert BET_SIZERS.available() == []
    assert PORTFOLIO_OPTIMIZERS.available() == []


def test_get_registry_known_and_unknown():
    assert get_registry("bars") is BARS
    with pytest.raises(KeyError):
        get_registry("nonexistent_family")


def test_new_model_can_register_into_a_family():
    import pandas as pd

    from RiskLabAI.core.base import BaseLabeler

    # Simulate a user/extension paper adding a new labeler.
    @LABELERS.register("dummy_const")
    class DummyConst(BaseLabeler):
        def label(self, prices, events=None, **kwargs):
            return pd.DataFrame({"label": 0}, index=prices.index)

    try:
        assert "dummy_const" in LABELERS
        obj = LABELERS.create("dummy_const")
        assert isinstance(obj, BaseLabeler)
    finally:
        LABELERS.unregister("dummy_const")  # keep global registry clean
