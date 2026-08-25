import builtins

import numpy as np
import pandas as pd
import pytest

from RiskLabAI.backtest.backtest_overfitting_simulation import (
    _average_directional_index,
    _average_true_range,
    _commodity_channel_index,
    _ichimoku_lines,
    _relative_strength_index,
    _stochastic_oscillator,
    _wilder_change_average,
    financial_features_backtest_overfitting_simulation,
)


def test_wilder_indicators_match_hand_calculations():
    prices = pd.Series([1.0, 2.0, 3.0, 2.0, 4.0])
    np.testing.assert_allclose(
        _relative_strength_index(prices, 2).to_numpy()[2:],
        [100.0, 50.0, 100.0 * 1.25 / 1.5],
        rtol=0.0,
        atol=1e-12,
    )

    true_range_prices = pd.Series([1.0, 3.0, 2.0, 6.0])
    np.testing.assert_allclose(
        _average_true_range(true_range_prices, 2).to_numpy()[2:],
        [1.5, 2.75],
        rtol=0.0,
        atol=1e-12,
    )

    directional_prices = pd.Series([1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        _average_directional_index(directional_prices, 2).to_numpy()[3:],
        [50.0, 50.0, 37.5, 50.0],
        rtol=0.0,
        atol=1e-12,
    )


def test_rolling_indicators_match_hand_calculations():
    prices = pd.Series([1.0, 2.0, 3.0, 2.0, 4.0])
    np.testing.assert_allclose(
        _commodity_channel_index(prices.iloc[:4], 3).to_numpy()[2:],
        [100.0, -50.0],
        rtol=0.0,
        atol=1e-12,
    )

    oscillator, signal = _stochastic_oscillator(prices, 3, 3)
    np.testing.assert_allclose(
        oscillator.to_numpy()[2:], [100.0, 0.0, 100.0], rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        signal.to_numpy()[4:], [200.0 / 3.0], rtol=0.0, atol=1e-12
    )

    conversion, base, span_a, span_b = _ichimoku_lines(
        pd.Series(np.arange(1.0, 7.0)), 2, 3, 4
    )
    np.testing.assert_allclose(conversion.to_numpy()[1:], [1.5, 2.5, 3.5, 4.5, 5.5])
    np.testing.assert_allclose(base.to_numpy()[2:], [2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(span_a.to_numpy()[2:], [2.25, 3.25, 4.25, 5.25])
    np.testing.assert_allclose(span_b.to_numpy()[3:], [2.5, 3.5, 4.5])


def test_constant_prices_produce_neutral_mature_indicators():
    prices = pd.Series(np.ones(60))
    assert _relative_strength_index(prices).iloc[-1] == 50.0
    assert _average_true_range(prices).iloc[-1] == 0.0
    assert _average_directional_index(prices).iloc[-1] == 0.0
    assert _commodity_channel_index(prices).iloc[-1] == 0.0
    oscillator, signal = _stochastic_oscillator(prices)
    assert oscillator.iloc[-1] == 50.0
    assert signal.iloc[-1] == 50.0


@pytest.mark.parametrize("window", [0, -1, True, 1.5])
def test_wilder_average_rejects_invalid_windows(window):
    with pytest.raises(ValueError, match="positive integer"):
        _wilder_change_average(np.array([0.0, 1.0, 2.0]), window)


def test_financial_feature_generation_is_self_contained(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "ta" or name.startswith("ta."):
            raise AssertionError("unexpected external technical-analysis import")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    generator = np.random.default_rng(7)
    prices = pd.Series(
        100.0 * np.exp(np.cumsum(generator.normal(0.0, 0.01, 300))),
        index=pd.date_range("2024-01-01", periods=300),
    )
    features = financial_features_backtest_overfitting_simulation(prices)

    assert not features.empty
    assert features.index.isin(prices.index).all()
    assert np.isfinite(features.dropna().to_numpy()).all()
    assert {
        "ADX",
        "RSI",
        "CCI",
        "Stochastic",
        "ROC",
        "ATR",
        "Kumo Breakout",
        "TK Position",
        "Price Kumo Position",
        "Cloud Thickness",
        "Momentum Confirmation",
    }.issubset(features.columns)
