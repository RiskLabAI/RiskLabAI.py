"""Tests for the base interfaces and lazy re-exports in RiskLabAI.core.base."""

import pandas as pd
import pytest

from RiskLabAI.core import base
from RiskLabAI.core.base import (
    BaseBetSizer,
    BaseLabeler,
    BasePortfolioOptimizer,
    Estimator,
)


# --------------------------------------------------------------------------- #
# Estimator structural protocol
# --------------------------------------------------------------------------- #
def test_estimator_protocol_is_structural():
    class Model:
        def fit(self, X, y=None, **kwargs):
            return self

        def predict(self, X):
            return X

    class NotAModel:
        def fit(self, X, y=None):
            return self

    assert isinstance(Model(), Estimator)
    assert not isinstance(NotAModel(), Estimator)  # missing predict


# --------------------------------------------------------------------------- #
# New optional contracts cannot be instantiated until implemented
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", [BaseLabeler, BaseBetSizer, BasePortfolioOptimizer])
def test_base_contracts_are_abstract(cls):
    with pytest.raises(TypeError):
        cls()


def test_concrete_labeler_satisfies_contract():
    class ConstantLabeler(BaseLabeler):
        def label(self, prices, events=None, **kwargs):
            return pd.DataFrame({"label": [1] * len(prices)}, index=prices.index)

    labeler = ConstantLabeler()
    out = labeler.label(pd.Series([10.0, 11.0, 12.0]))
    assert isinstance(out, pd.DataFrame)
    assert list(out["label"]) == [1, 1, 1]
    assert isinstance(labeler, BaseLabeler)


# --------------------------------------------------------------------------- #
# Lazy re-exports of the canonical interfaces
# --------------------------------------------------------------------------- #
def test_lazy_reexports_resolve_to_real_interfaces():
    from RiskLabAI.data.structures.abstract_bars import AbstractBars
    from RiskLabAI.backtest.validation.cross_validator_interface import (
        CrossValidator,
    )
    from RiskLabAI.features.feature_importance.feature_importance_strategy import (
        FeatureImportanceStrategy,
    )

    assert base.AbstractBars is AbstractBars
    assert base.BarBuilder is AbstractBars
    assert base.CrossValidator is CrossValidator
    assert base.FeatureImportanceStrategy is FeatureImportanceStrategy


def test_base_getattr_rejects_unknown_name():
    with pytest.raises(AttributeError):
        _ = base.DoesNotExist


def test_base_dir_lists_public_names():
    names = dir(base)
    assert "BaseLabeler" in names
    assert "CrossValidator" in names
