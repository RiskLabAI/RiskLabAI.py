"""
Tests for data/weights/sample_weights.py
"""

import numpy as np
import pandas as pd
import pytest

from RiskLabAI.data.weights.sample_weights import (
    calculate_average_uniqueness,
    calculate_time_decay,
    expand_label_for_meta_labeling,
    sample_weight_absolute_return_meta_labeling,
    sequential_bootstrap,
)


@pytest.fixture
def sample_events():
    """Fixture for sample events and price index."""
    close_index = pd.to_datetime(pd.date_range("2020-01-01", periods=10))

    # Event 1: [0, 4]
    # Event 2: [2, 6]
    # Event 3: [8, 9]
    timestamp = pd.Series(
        pd.to_datetime(["2020-01-05", "2020-01-07", "2020-01-10"]),
        index=pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-09"]),
    )
    molecule = timestamp.index
    return close_index, timestamp, molecule


def test_expand_label_for_meta_labeling(sample_events):
    """Test the concurrency calculation."""
    close_index, timestamp, molecule = sample_events

    concurrency = expand_label_for_meta_labeling(close_index, timestamp, molecule)

    # Concurrency:
    # 01-01: 1
    # 01-02: 1
    # 01-03: 2
    # 01-04: 2
    # 01-05: 2
    # 01-06: 1
    # 01-07: 1
    # 01-08: 0
    # 01-09: 1
    # 01-10: 1
    expected_values = [1, 1, 2, 2, 2, 1, 1, 0, 1, 1]
    expected_index = pd.to_datetime(pd.date_range("2020-01-01", periods=10))

    pd.testing.assert_series_equal(
        concurrency, pd.Series(expected_values, index=expected_index), check_dtype=False
    )


def test_calculate_average_uniqueness():
    """Test average uniqueness calculation."""
    # T=4, N=3
    # Event 0: [0, 2]
    # Event 1: [1, 3]
    # Event 2: [0, 1]
    idx_matrix = pd.DataFrame(
        [
            [1, 0, 1],  # t=0, c=2
            [1, 1, 1],  # t=1, c=3
            [1, 1, 0],  # t=2, c=2
            [0, 1, 0],  # t=3, c=1
        ]
    )
    # Uniqueness =
    #   [1/2, 0, 1/2]
    #   [1/3, 1/3, 1/3]
    #   [1/2, 1/2, 0]
    #   [0,   1,   0]

    # Avg Uniqueness (by column):
    # E0: (1/2 + 1/3 + 1/2) / 3 = (0.5 + 0.333 + 0.5) / 3 = 1.333 / 3 = 0.444
    # E1: (1/3 + 1/2 + 1) / 3 = (0.333 + 0.5 + 1) / 3 = 1.833 / 3 = 0.611
    # E2: (1/2 + 1/3) / 2 = (0.5 + 0.333) / 2 = 0.833 / 2 = 0.416

    avg_u = calculate_average_uniqueness(idx_matrix)

    assert np.isclose(avg_u[0], (0.5 + 1 / 3 + 0.5) / 3)
    assert np.isclose(avg_u[1], (1 / 3 + 0.5 + 1) / 3)
    assert np.isclose(avg_u[2], (0.5 + 1 / 3) / 2)


def test_sample_weight_absolute_return(sample_events):
    """Test sample weighting by absolute return."""
    close_index, timestamp, molecule = sample_events
    prices = pd.Series([10, 11, 12, 13, 12, 11, 10, 11, 12, 13], index=close_index)

    weights = sample_weight_absolute_return_meta_labeling(timestamp, prices, molecule)

    assert weights.shape == (3,)
    assert np.isclose(weights.sum(), 3.0)  # Normalized to N
    assert weights.loc["2020-01-01"] > 0
    assert weights.loc["2020-01-03"] > 0
    assert weights.loc["2020-01-09"] > 0


def test_calculate_time_decay():
    """Test time decay weighting."""
    weights = pd.Series(1.0, index=pd.date_range("2020-01-01", periods=10))

    # Test 1: No decay
    decayed_1 = calculate_time_decay(weights, clf_last_weight=1.0)
    assert np.allclose(decayed_1, 1.0)

    # Test 2: Linear decay to 0
    decayed_0 = calculate_time_decay(weights, clf_last_weight=0.0)
    # cumsum = [1, 2, ..., 10]
    # slope = (1-0) / 10 = 0.1
    # const = 1 - 0.1 * 10 = 0
    # new_weights = 0 + 0.1 * [1, 2, ..., 10] = [0.1, 0.2, ..., 1.0]
    expected_0 = np.arange(1, 11) * 0.1
    assert np.allclose(decayed_0, expected_0)

    # Test 3: Linear decay to 0.5
    decayed_05 = calculate_time_decay(weights, clf_last_weight=0.5)
    # slope = (1-0.5) / 10 = 0.05
    # const = 1 - 0.05 * 10 = 0.5
    # new_weights = 0.5 + 0.05 * [1, ..., 10] = [0.55, 0.6, ..., 1.0]
    expected_05 = 0.5 + 0.05 * np.arange(1, 11)
    assert np.allclose(decayed_05, expected_05)


@pytest.mark.parametrize(
    "start_positions,end_positions,molecule_positions",
    [
        ([0, 1], [2, 3], [1]),
        ([0, 1], [2, 3], [0]),
        ([0, 1, 3], [6, 4, 5], [1]),
        ([0, 1, 4], [1, 3, 5], [1]),
        ([0, 1, 3], [4, 2, None], [1]),
        ([0, 1, 3], [4, None, 5], [1]),
        ([0, 1, 3, 5], [2, 4, 6, 6], [0, 2]),
    ],
)
def test_concurrency_subset_counts_all_overlapping_events(
    start_positions, end_positions, molecule_positions
):
    """Subset concurrency equals the full event indicator sum on its interval."""
    close_index = pd.date_range("2024-01-01", periods=7)
    event_starts = close_index[start_positions]
    event_ends = pd.Series(
        [close_index[end] if end is not None else pd.NaT for end in end_positions],
        index=event_starts,
    )
    molecule = event_starts[molecule_positions]
    resolved_ends = event_ends.fillna(close_index[-1])
    requested_index = close_index[
        (close_index >= molecule[0])
        & (close_index <= resolved_ends.loc[molecule].max())
    ]
    expected = pd.Series(
        [
            sum(start <= date <= end for start, end in resolved_ends.items())
            for date in requested_index
        ],
        index=requested_index,
        dtype=int,
    )

    actual = expand_label_for_meta_labeling(close_index, event_ends, molecule)

    pd.testing.assert_series_equal(actual, expected)


class _RecordingGenerator(np.random.Generator):
    """Record conditional probabilities while supplying predetermined draws."""

    def __init__(self, draws):
        super().__init__(np.random.PCG64(0))
        self.draws = iter(draws)
        self.probabilities = []

    def choice(self, a, p):
        self.probabilities.append(np.array(p, copy=True))
        result = next(self.draws)
        assert 0 <= result < a
        return result


def test_sequential_bootstrap_exact_conditional_probabilities():
    """The next-draw probabilities match exact mean-uniqueness fractions."""
    matrix = np.array([[1, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]])
    generator = _RecordingGenerator([1, 2, 0, 2])

    actual = sequential_bootstrap(matrix, sample_length=4, random_state=generator)

    np.testing.assert_array_equal(actual, [1, 2, 0, 2])
    np.testing.assert_allclose(
        generator.probabilities,
        [
            [1 / 3, 1 / 3, 1 / 3],
            [5 / 14, 3 / 14, 6 / 14],
            [5 / 11, 3 / 11, 3 / 11],
            [16 / 49, 15 / 49, 18 / 49],
        ],
        rtol=1e-14,
        atol=1e-14,
    )


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_sequential_bootstrap_matches_rebuilt_indicator_reference(seed):
    """Repeated draws agree with rebuilding the full selected-event matrix."""
    matrix = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=float)
    generator = np.random.default_rng(seed)
    expected = []
    for _ in range(20):
        scores = []
        for candidate in range(matrix.shape[1]):
            reduced = matrix[:, expected + [candidate]]
            active = reduced[:, -1] > 0
            scores.append(np.mean(1.0 / reduced[active].sum(axis=1)))
        probabilities = np.asarray(scores) / sum(scores)
        expected.append(generator.choice(matrix.shape[1], p=probabilities))

    actual = sequential_bootstrap(matrix, sample_length=20, random_state=seed)

    np.testing.assert_array_equal(actual, expected)


def test_sequential_bootstrap_dataframe_and_generator_contract():
    """Labels do not replace positional output, and generator state is consumed."""
    matrix = pd.DataFrame([[1, 0], [1, 1], [0, 1]], columns=["event B", "event A"])
    original = matrix.copy(deep=True)
    generator = np.random.default_rng(12)
    reference_generator = np.random.default_rng(12)
    for _ in range(2):
        actual = sequential_bootstrap(matrix, random_state=generator)
        expected = sequential_bootstrap(
            matrix.to_numpy(), random_state=reference_generator
        )
        np.testing.assert_array_equal(actual, expected)
        assert actual.shape == (2,)
        assert np.issubdtype(actual.dtype, np.integer)
    pd.testing.assert_frame_equal(matrix, original)
    assert generator.bit_generator.state == reference_generator.bit_generator.state


def test_sequential_bootstrap_leaves_legacy_global_rng_unchanged():
    """Both seeded and unseeded calls avoid the legacy global RNG."""
    before = np.random.get_state()
    sequential_bootstrap(np.ones((2, 2)), random_state=3)
    sequential_bootstrap(np.ones((2, 2)))
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


@pytest.mark.parametrize("shape", [(0, 0), (3, 0), (2, 2)])
def test_sequential_bootstrap_zero_length_does_not_consume_rng(shape):
    """Zero draws return integer positions without advancing a supplied RNG."""
    generator = np.random.default_rng(5)
    before = generator.bit_generator.state
    actual = sequential_bootstrap(
        np.ones(shape), sample_length=0, random_state=generator
    )
    assert actual.shape == (0,)
    assert np.issubdtype(actual.dtype, np.integer)
    assert generator.bit_generator.state == before


def test_sequential_bootstrap_single_and_identical_events():
    """A single event repeats; identical events remain equally probable."""
    np.testing.assert_array_equal(
        sequential_bootstrap(np.ones((2, 1)), 5, 0), np.zeros(5, dtype=int)
    )
    generator = _RecordingGenerator([0, 0, 1, 0])
    sequential_bootstrap(np.ones((3, 2)), 4, generator)
    np.testing.assert_allclose(generator.probabilities, np.full((4, 2), 0.5))


@pytest.mark.parametrize(
    "matrix",
    [
        [1, 0],
        np.ones((1, 1, 1)),
        [[1, np.nan]],
        [[1, np.inf]],
        [[1, -1]],
        [[1, 0.5]],
        [[1, 2]],
        [[1, 0]],
        np.empty((0, 1)),
        [["1"]],
        [[1 + 1j]],
    ],
)
def test_sequential_bootstrap_rejects_invalid_indicator_matrix(matrix):
    """Only finite binary matrices with supported events define the algorithm."""
    with pytest.raises(ValueError):
        sequential_bootstrap(matrix, random_state=0)


@pytest.mark.parametrize("length", [-1, 1.5, True, np.bool_(False), "2"])
def test_sequential_bootstrap_rejects_invalid_lengths(length):
    """Draw counts must be nonnegative integers, not boolean flags."""
    with pytest.raises(ValueError, match="sample_length"):
        sequential_bootstrap(np.ones((1, 1)), sample_length=length, random_state=0)


@pytest.mark.parametrize("state", [-1, 1.5, True, "2", np.random.RandomState(0)])
def test_sequential_bootstrap_rejects_unsupported_random_state(state):
    """Unsupported RNG objects are rejected rather than silently reseeded."""
    with pytest.raises(ValueError, match="random_state"):
        sequential_bootstrap(np.ones((1, 1)), random_state=state)


def test_sequential_bootstrap_empty_population_and_numpy_integer_arguments():
    """Empty populations cannot produce positive draws; NumPy integers work."""
    assert sequential_bootstrap(np.empty((0, 0))).size == 0
    with pytest.raises(ValueError, match="no events"):
        sequential_bootstrap(np.empty((2, 0)), sample_length=1)
    np.testing.assert_array_equal(
        sequential_bootstrap(np.ones((1, 1)), np.int64(2), np.int64(4)), [0, 0]
    )


def test_sequential_bootstrap_public_export():
    """The sampling function is available through the sample-weights package."""
    from RiskLabAI.data import weights

    assert weights.sequential_bootstrap is sequential_bootstrap
    assert "sequential_bootstrap" in weights.__all__


def test_concurrency_chunks_match_full_event_counts():
    """Processing separate event groups leaves their shared counts unchanged."""
    close_index = pd.date_range("2024-01-01", periods=9)
    event_ends = pd.Series(close_index[[5, 4, 7, 8]], index=close_index[[0, 1, 3, 6]])
    full = expand_label_for_meta_labeling(close_index, event_ends, event_ends.index)

    for molecule in np.array_split(event_ends.index, 2):
        actual = expand_label_for_meta_labeling(close_index, event_ends, molecule)
        expected = full.loc[molecule[0] : event_ends.loc[molecule].max()]
        pd.testing.assert_series_equal(actual, expected)


def test_subset_return_weights_use_full_event_concurrency():
    """Return attribution uses all overlaps and retains subset normalization."""
    close_index = pd.date_range("2024-01-01", periods=7)
    event_ends = pd.Series(close_index[[4, 3, 6]], index=close_index[[0, 1, 3]])
    molecule = event_ends.index[1:]
    prices = pd.Series(np.exp(np.arange(7, dtype=float)), index=close_index)
    expected_raw = pd.Series([4.0 / 3.0, 17.0 / 6.0], index=molecule)
    expected = expected_raw * len(molecule) / expected_raw.sum()

    actual = sample_weight_absolute_return_meta_labeling(event_ends, prices, molecule)

    pd.testing.assert_series_equal(actual, expected)
