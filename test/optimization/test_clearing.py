"""Check proportional clearing against hand calculations and independent iteration."""

from types import SimpleNamespace

import numpy as np
import pytest

from RiskLabAI.optimization import eisenberg_noe_clearing
from RiskLabAI.optimization import clearing


@pytest.mark.parametrize(
    "assets,liabilities,relative,expected",
    [
        ([2, 8], [5, 5], [[0, 0], [0, 0]], [2, 5]),
        ([3, 0, 0], [5, 4, 2], [[0, 1, 0], [0, 0, 1], [0, 0, 0]], [3, 3, 2]),
        ([0, 0], [2, 3], [[0, 1], [1, 0]], [2, 2]),
        ([1, 0], [10, 10], [[0, 0.5], [0.5, 0]], [4 / 3, 2 / 3]),
        ([0, 0], [2, 3], [[0, 1], [1, 0]], [2, 2]),
        ([100, 100], [2, 3], [[0, 0.5], [0.5, 0]], [2, 3]),
        ([1, 2], [0, 0], [[0, 0], [0, 0]], [0, 0]),
        ([1, 0], [2, 0], [[0, 1], [0, 0]], [1, 0]),
        ([0], [7], [[1]], [7]),
    ],
)
def test_hand_calculated_networks(assets, liabilities, relative, expected):
    np.testing.assert_allclose(
        eisenberg_noe_clearing(assets, liabilities, relative), expected
    )


def test_random_networks_match_contractive_iteration():
    rng = np.random.default_rng(971)
    for size in range(1, 9):
        for _ in range(10):
            relative = rng.uniform(size=(size, size))
            relative *= 0.8 / relative.sum(axis=1, keepdims=True)
            liabilities = rng.uniform(0, 5, size)
            assets = rng.uniform(0, 1, size)
            expected = liabilities.copy()
            for _ in range(250):
                expected = np.minimum(liabilities, assets + relative.T @ expected)
            actual = eisenberg_noe_clearing(assets, liabilities, relative)
            np.testing.assert_allclose(actual, expected, atol=1e-9)


@pytest.mark.parametrize("scale", [1e-12, 1, 1e12])
def test_monetary_units_and_input_preservation(scale):
    assets = np.array([1, 0.0]) * scale
    liabilities = np.array([10, 10.0]) * scale
    relative = np.array([[0, 0.5], [0.5, 0]])
    before = [x.copy() for x in (assets, liabilities, relative)]
    actual = eisenberg_noe_clearing(assets, liabilities, relative)
    np.testing.assert_allclose(actual / scale, [4 / 3, 2 / 3])
    for array, original in zip((assets, liabilities, relative), before):
        np.testing.assert_array_equal(array, original)


@pytest.mark.parametrize(
    "assets,liabilities,relative",
    [
        ([], [], []),
        ([[1]], [1], [[0]]),
        ([1], [1, 2], [[0]]),
        ([1], [1], [0]),
        ([-1], [1], [[0]]),
        ([1], [-1], [[0]]),
        ([1], [1], [[-0.1]]),
        ([1], [1], [[1.01]]),
        ([np.nan], [1], [[0]]),
        ([1], [np.inf], [[0]]),
        ([1], [1], [[np.nan]]),
        ([1j], [1], [[0]]),
        (["1"], [1], [[0]]),
    ],
)
def test_reject_invalid_network(assets, liabilities, relative):
    with pytest.raises(ValueError):
        eisenberg_noe_clearing(assets, liabilities, relative)


@pytest.mark.parametrize("atol", [0, -1, 1, np.nan, np.inf, True, "0.1", [0.1]])
def test_reject_invalid_tolerance(atol):
    with pytest.raises(ValueError):
        eisenberg_noe_clearing([1], [2], [[0]], atol=atol)


@pytest.mark.parametrize(
    "success,payments",
    [(False, None), (True, [np.nan]), (True, [-1]), (True, [2]), (True, [0])],
)
def test_reject_failed_or_unchecked_solver(monkeypatch, success, payments):
    monkeypatch.setattr(
        clearing,
        "linprog",
        lambda *args, **kwargs: SimpleNamespace(
            success=success, x=payments, message="test"
        ),
    )
    with pytest.raises(RuntimeError):
        eisenberg_noe_clearing([1], [2], [[0]])
