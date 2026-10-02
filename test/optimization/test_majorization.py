"""Mathematical oracles for majorization and permutation-mixture reconstruction."""

from itertools import permutations
from types import SimpleNamespace

import numpy as np
import pytest

from RiskLabAI.optimization import (
    birkhoff_von_neumann_decomposition,
    is_doubly_stochastic,
    is_majorized,
    majorization_matrix,
)
from RiskLabAI.optimization import majorization as module


@pytest.mark.parametrize(
    "target,source,expected",
    [
        ([0.5, 0.5], [1.0, 0.0], True),
        ([1.0, 0.0], [0.5, 0.5], False),
        ([2, 2, 2], [4, 1, 1], True),
        ([3, 3, 0], [4, 1, 1], False),
        ([4, 1, 1], [3, 3, 0], False),
        ([-1, 0, 1], [-2, 0, 2], True),
        ([2], [2], True),
        ([1], [2], False),
        ([0, 0], [1, 0], False),
    ],
)
def test_majorization_partial_sum_cases(target, source, expected):
    assert is_majorized(target, source) is expected
    for ordering in permutations(range(len(target))):
        assert is_majorized(np.asarray(target)[list(ordering)], source) is expected


@pytest.mark.parametrize(
    "target,source",
    [
        ([], []),
        ([1], [1, 2]),
        ([[1]], [[1]]),
        ([np.nan], [1]),
        ([1], [np.inf]),
        ([1j], [1]),
        (["1"], [1]),
    ],
)
def test_vector_inputs_rejected(target, source):
    with pytest.raises(ValueError):
        is_majorized(target, source)
    with pytest.raises(ValueError):
        majorization_matrix(target, source)


@pytest.mark.parametrize("atol", [0, -1, 1, np.inf, np.nan, True, "1e-10"])
def test_invalid_tolerance_rejected(atol):
    for function, args in [
        (is_majorized, ([1], [1])),
        (majorization_matrix, ([1], [1])),
        (is_doubly_stochastic, ([[1]],)),
        (birkhoff_von_neumann_decomposition, ([[1]],)),
    ]:
        with pytest.raises(ValueError, match="atol"):
            function(*args, atol=atol)


@pytest.mark.parametrize(
    "matrix",
    [
        [],
        [1],
        [[1, 0]],
        [[np.nan]],
        [[np.inf]],
        [[1j]],
        [["1"]],
        [[0.4, 0.6], [0.4, 0.6]],
        [[1.1, -0.1], [-0.1, 1.1]],
    ],
)
def test_invalid_stochastic_matrix_rejected(matrix):
    assert not is_doubly_stochastic(matrix)
    with pytest.raises(ValueError):
        birkhoff_von_neumann_decomposition(matrix)


@pytest.mark.parametrize(
    "target,source",
    [
        ([0.5, 0.5], [1, 0]),
        ([2, 2, 2], [4, 1, 1]),
        ([-1, 0, 1], [-2, 0, 2]),
        ([2, -2, 0], [-2, 0, 2]),
        ([3], [3]),
        ([1, 1, 1], [1, 1, 1]),
    ],
)
def test_witness_direction_and_constraints(target, source):
    target = np.asarray(target, dtype=float)
    source = np.asarray(source, dtype=float)
    old_target, old_source = target.copy(), source.copy()
    matrix = majorization_matrix(target, source)
    assert np.min(matrix) >= -1e-10
    np.testing.assert_allclose(matrix.sum(axis=0), 1, atol=1e-10, rtol=0)
    np.testing.assert_allclose(matrix.sum(axis=1), 1, atol=1e-10, rtol=0)
    np.testing.assert_allclose(matrix @ source, target, atol=1e-10, rtol=0)
    np.testing.assert_array_equal(target, old_target)
    np.testing.assert_array_equal(source, old_source)


def test_non_majorized_pair_has_no_witness():
    with pytest.raises(ValueError, match="not majorized"):
        majorization_matrix([3, 3, 0], [4, 1, 1])


@pytest.mark.parametrize("size", [1, 2, 3, 5, 10])
def test_mixture_reconstruction_and_witness(size):
    generator = np.random.default_rng(304 + size)
    terms = [np.eye(size)[generator.permutation(size)] for _ in range(20)]
    coefficients = generator.dirichlet(np.ones(len(terms)))
    matrix = sum(weight * term for weight, term in zip(coefficients, terms))
    before = matrix.copy()
    weights, orders = birkhoff_von_neumann_decomposition(matrix)
    assert len(weights) == len(orders)
    assert np.all(weights > 0)
    assert np.isclose(weights.sum(), 1, atol=1e-10, rtol=0)
    for order in orders:
        np.testing.assert_array_equal(np.sort(order), np.arange(size))
    restored = sum(
        weight * np.eye(size)[order] for weight, order in zip(weights, orders)
    )
    np.testing.assert_allclose(restored, matrix, atol=1e-10, rtol=0)
    np.testing.assert_array_equal(matrix, before)
    source = generator.normal(size=size)
    target = matrix @ source
    assert is_majorized(target, source)
    witness = majorization_matrix(target, source)
    np.testing.assert_allclose(witness @ source, target, atol=1e-10, rtol=0)


def test_single_permutation_and_tiny_mixture_component():
    matrix = np.eye(3)[[2, 0, 1]]
    weights, orders = birkhoff_von_neumann_decomposition(matrix)
    np.testing.assert_array_equal(weights, [1.0])
    np.testing.assert_array_equal(orders, [[2, 0, 1]])
    tiny = 5e-12
    matrix = (1 - tiny) * np.eye(3) + tiny * np.eye(3)[[1, 2, 0]]
    weights, orders = birkhoff_von_neumann_decomposition(matrix, atol=1e-13)
    restored = sum(w * np.eye(3)[p] for w, p in zip(weights, orders))
    np.testing.assert_allclose(restored, matrix, atol=1e-13, rtol=0)
    assert np.isclose(weights.sum(), 1, atol=1e-13, rtol=0)


def test_tolerance_does_not_normalize_inputs():
    matrix = np.eye(2) * (1 + 5e-11)
    before = matrix.copy()
    assert is_doubly_stochastic(matrix)
    assert not is_doubly_stochastic(matrix, atol=1e-12)
    weights, orders = birkhoff_von_neumann_decomposition(matrix)
    restored = sum(w * np.eye(2)[p] for w, p in zip(weights, orders))
    np.testing.assert_array_equal(restored, matrix)
    np.testing.assert_array_equal(matrix, before)


def test_solver_failure_and_unchecked_result_are_not_returned(monkeypatch):
    monkeypatch.setattr(
        module,
        "linprog",
        lambda *args, **kwargs: SimpleNamespace(
            success=False, x=None, message="test failure"
        ),
    )
    with pytest.raises(RuntimeError, match="test failure"):
        majorization_matrix([0.5, 0.5], [1, 0])
    monkeypatch.setattr(
        module,
        "linprog",
        lambda *args, **kwargs: SimpleNamespace(
            success=True, x=np.eye(2).ravel(), message="unchecked"
        ),
    )
    with pytest.raises(RuntimeError, match="residual"):
        majorization_matrix([0.5, 0.5], [1, 0])


def test_matching_failure_is_reported(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("test failure")

    monkeypatch.setattr(module, "linear_sum_assignment", fail)
    with pytest.raises(RuntimeError, match="matching"):
        birkhoff_von_neumann_decomposition(np.eye(2))
