"""Validate the discrete execution benchmark against quadratic optimization."""

import numpy as np
import pytest

from RiskLabAI.optimization import almgren_chriss_execution


@pytest.mark.parametrize("risk,volatility", [(0, 2), (1, 0), (0, 0), (1e-30, 2)])
def test_constant_rate_limit(risk, volatility):
    result = almgren_chriss_execution(
        100,
        2,
        8,
        volatility=volatility,
        temporary_impact=0.4,
        permanent_impact=0.1,
        risk_aversion=risk,
    )
    np.testing.assert_allclose(result["inventory"], np.linspace(100, 0, 9))
    np.testing.assert_allclose(result["trades"], np.full(8, 12.5))
    assert result["expected_cost"] == pytest.approx(0.05 * 100**2 + 0.3875 * 100**2 / 2)
    assert result["cost_variance"] == pytest.approx(
        volatility**2 * 0.25 * sum((100 - k * 12.5) ** 2 for k in range(1, 9))
    )


@pytest.mark.parametrize("intervals", [1, 2, 5, 20])
@pytest.mark.parametrize("risk", [0, 1e-4, 0.3, 1000])
def test_schedule_matches_independent_quadratic_system(intervals, risk):
    quantity, horizon, sigma, eta, gamma = 80, 3, 1.7, 0.8, 0.1
    result = almgren_chriss_execution(
        quantity,
        horizon,
        intervals,
        volatility=sigma,
        temporary_impact=eta,
        permanent_impact=gamma,
        risk_aversion=risk,
    )
    tau = horizon / intervals
    difference = np.eye(intervals, intervals + 1) - np.eye(
        intervals, intervals + 1, k=1
    )
    boundary = np.zeros(intervals + 1)
    boundary[0] = quantity
    interior = difference[:, 1:-1]
    coefficient = eta / tau - gamma / 2
    hessian = coefficient * interior.T @ interior + risk * sigma**2 * tau * np.eye(
        intervals - 1
    )
    linear = coefficient * interior.T @ (difference @ boundary)
    expected = boundary.copy()
    if intervals > 1:
        expected[1:-1] = np.linalg.solve(hessian, -linear)
    np.testing.assert_allclose(result["inventory"], expected, atol=1e-10)
    inventory = result["inventory"]
    trades = result["trades"]
    assert inventory[0] == quantity and inventory[-1] == 0
    assert np.all(trades >= 0)
    assert trades.sum() == pytest.approx(quantity)
    direct_cost = gamma * np.dot(inventory[1:], trades) + eta / tau * np.dot(
        trades, trades
    )
    assert result["expected_cost"] == pytest.approx(direct_cost)
    assert result["cost_variance"] == pytest.approx(
        sigma**2 * tau * np.dot(inventory[1:], inventory[1:])
    )
    np.testing.assert_allclose(result["times"], np.linspace(0, horizon, intervals + 1))


def test_increased_risk_aversion_trades_earlier():
    paths = [
        almgren_chriss_execution(
            50, 1, 10, volatility=2, temporary_impact=0.1, risk_aversion=r
        )
        for r in [0, 0.1, 10]
    ]
    for slower, faster in zip(paths, paths[1:]):
        assert np.all(faster["inventory"] <= slower["inventory"] + 1e-12)
        assert faster["cost_variance"] < slower["cost_variance"]
        assert faster["expected_cost"] > slower["expected_cost"]


def test_extreme_risk_aversion_is_finite():
    result = almgren_chriss_execution(
        50, 1, 10, volatility=2, temporary_impact=0.1, risk_aversion=1e308
    )
    np.testing.assert_allclose(result["inventory"], [50] + [0] * 10, atol=1e-300)
    assert np.isfinite(result["expected_cost"])


def test_zero_inventory_and_single_interval():
    zero = almgren_chriss_execution(0, 1, 4, volatility=2, temporary_impact=1)
    assert zero["expected_cost"] == zero["cost_variance"] == 0
    np.testing.assert_array_equal(zero["inventory"], np.zeros(5))
    single = almgren_chriss_execution(
        20,
        2,
        1,
        volatility=2,
        temporary_impact=1,
        permanent_impact=0.1,
        risk_aversion=5,
    )
    assert single["expected_cost"] == pytest.approx(200)
    assert single["cost_variance"] == 0


@pytest.mark.parametrize(
    "name",
    [
        "inventory",
        "horizon",
        "volatility",
        "temporary_impact",
        "permanent_impact",
        "risk_aversion",
    ],
)
@pytest.mark.parametrize("bad", [-1, np.nan, np.inf, True, "1", [1], 1j])
def test_invalid_scalar_parameters(name, bad):
    arguments = dict(
        inventory=10,
        horizon=1,
        intervals=5,
        volatility=1,
        temporary_impact=1,
        permanent_impact=0,
        risk_aversion=1,
    )
    arguments[name] = bad
    with pytest.raises(ValueError):
        almgren_chriss_execution(**arguments)


@pytest.mark.parametrize("intervals", [0, -1, 2.5, True, "2"])
def test_invalid_interval_count(intervals):
    with pytest.raises(ValueError):
        almgren_chriss_execution(10, 1, intervals, volatility=1, temporary_impact=1)


@pytest.mark.parametrize(
    "horizon,eta,gamma", [(0, 1, 0), (1, 0, 0), (2, 0.1, 1), (1, 0.1, 1)]
)
def test_invalid_discrete_convexity(horizon, eta, gamma):
    with pytest.raises(ValueError):
        almgren_chriss_execution(
            10, horizon, 5, volatility=1, temporary_impact=eta, permanent_impact=gamma
        )


def test_cost_overflow_rejected():
    with pytest.raises(ValueError, match="floating-point"):
        almgren_chriss_execution(1e300, 1, 5, volatility=1, temporary_impact=1)
