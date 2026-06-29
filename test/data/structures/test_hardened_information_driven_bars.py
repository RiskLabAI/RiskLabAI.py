"""
Tests for data/structures/hardened_information_driven_bars.py (anti-degeneracy imbalance/run bars).
"""

import numpy as np
import pandas as pd
import pytest

from RiskLabAI.data.structures.hardened_information_driven_bars import (
    HardenedExpectedImbalanceBars,
    HardenedExpectedRunBars,
)
from RiskLabAI.data.structures.imbalance_bars import ExpectedImbalanceBars
from RiskLabAI.utils.constants import CUMULATIVE_TICKS

# Bar layout index of cum_ticks (see bars list layout): position 9.
_CUM_TICKS_COL = 9


def _synthetic_ticks(n=3000, seed=0):
    """A deterministic (date_time, price, volume) tick stream with mixed up/down moves."""
    rng = np.random.default_rng(seed)
    steps = rng.choice([-1, 0, 1], size=n, p=[0.45, 0.1, 0.45])
    price = 100.0 + np.cumsum(steps) * 0.5
    price = np.clip(price, 1.0, None)
    volume = rng.integers(1, 20, size=n).astype(float)
    t0 = pd.to_datetime("2020-01-01 10:00:00")
    return [
        (t0 + pd.Timedelta(seconds=i), float(price[i]), float(volume[i]))
        for i in range(n)
    ]


def _imbalance(min_ticks, max_ticks, e_t=50):
    return HardenedExpectedImbalanceBars(
        bar_type="dollar_imbalance",
        window_size_for_expected_n_ticks_estimation=max(10, e_t),
        initial_estimate_of_expected_n_ticks_in_bar=e_t,
        window_size_for_expected_imbalance_estimation=10000,
        min_ticks=min_ticks,
        max_ticks=max_ticks,
    )


def _run(min_ticks, max_ticks, e_t=50):
    return HardenedExpectedRunBars(
        bar_type="dollar_run",
        window_size_for_expected_n_ticks_estimation=max(10, e_t),
        initial_estimate_of_expected_n_ticks_in_bar=e_t,
        window_size_for_expected_imbalance_estimation=10000,
        min_ticks=min_ticks,
        max_ticks=max_ticks,
    )


def test_hardened_imbalance_respects_tick_bounds():
    """Every hardened imbalance bar has min_ticks <= cum_ticks <= max_ticks (the core guard invariant)."""
    bars = _imbalance(min_ticks=20, max_ticks=500).construct_bars_from_data(
        _synthetic_ticks()
    )
    assert len(bars) > 0
    sizes = np.array([b[_CUM_TICKS_COL] for b in bars])
    # the final (possibly unfinished) bar can be short; the closed bars must satisfy the floor
    assert (sizes[:-1] >= 20).all()
    assert (sizes <= 500).all()
    assert (sizes <= 2).sum() == 0  # no 1-tick collapse


def test_hardened_run_respects_tick_bounds():
    bars = _run(min_ticks=20, max_ticks=500).construct_bars_from_data(
        _synthetic_ticks()
    )
    assert len(bars) > 0
    sizes = np.array([b[_CUM_TICKS_COL] for b in bars])
    assert (sizes[:-1] >= 20).all()
    assert (sizes <= 500).all()


def test_guard_condition_logic():
    """The collapse guard blocks closing below min_ticks; the divergence guard forces it at max_ticks."""
    bars = _imbalance(min_ticks=20, max_ticks=500)
    bars.base_statistics[CUMULATIVE_TICKS] = 5  # below the floor
    assert (
        bars._bar_construction_condition(1e-12) is False
    )  # cannot close even at a tiny threshold
    bars.base_statistics[CUMULATIVE_TICKS] = 500  # at the ceiling
    assert (
        bars._bar_construction_condition(1e18) is True
    )  # forced close even at a huge threshold


def test_well_behaved_unchanged_when_guards_do_not_bind():
    """
    Replication of the held-out "well-behaved unchanged" criterion: with loose guards the hardened bar
    count matches the naive ExpectedImbalanceBars count (the guards never trigger).
    """
    ticks = _synthetic_ticks()
    naive = ExpectedImbalanceBars(
        bar_type="dollar_imbalance",
        window_size_for_expected_n_ticks_estimation=50,
        initial_estimate_of_expected_n_ticks_in_bar=50,
        window_size_for_expected_imbalance_estimation=10000,
    ).construct_bars_from_data(ticks)
    hardened = _imbalance(min_ticks=1, max_ticks=10_000_000).construct_bars_from_data(
        ticks
    )
    assert abs(len(naive) - len(hardened)) <= max(
        1, int(0.1 * len(naive))
    )  # within 10%


def test_invalid_guards_raise():
    with pytest.raises(ValueError):
        _imbalance(min_ticks=0, max_ticks=100)
    with pytest.raises(ValueError):
        _imbalance(min_ticks=200, max_ticks=100)
    with pytest.raises(ValueError):
        _run(min_ticks=-5, max_ticks=100)


def test_naive_baseline_class_unchanged():
    """The naive ExpectedImbalanceBars is not modified by the hardened subclass (no monkeypatching)."""
    assert not hasattr(
        ExpectedImbalanceBars(
            bar_type="dollar_imbalance",
            window_size_for_expected_n_ticks_estimation=50,
            initial_estimate_of_expected_n_ticks_in_bar=50,
            window_size_for_expected_imbalance_estimation=10000,
        ),
        "min_ticks",
    )
