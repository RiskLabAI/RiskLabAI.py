"""
Tests for backtest/validation/path_bagged_cpcv.py (path-level Bagged CPCV PBO).
"""

import numpy as np

from RiskLabAI.backtest.probability_of_backtest_overfitting import (
    probability_of_backtest_overfitting,
)
from RiskLabAI.backtest.validation.path_bagged_cpcv import (
    bagged_probability_of_backtest_overfitting,
    moving_block_bootstrap_indices,
)


def _noisy_panel(seed, t_len=240, n_strategies=12):
    """A noisy strategy panel with a faint mean spread (the appraisal-style controlled setup)."""
    rng = np.random.default_rng(seed)
    means = np.linspace(-0.04, 0.04, n_strategies)
    return rng.standard_normal((t_len, n_strategies)) + means


def test_moving_block_indices_shape_and_range():
    rng = np.random.default_rng(0)
    idx = moving_block_bootstrap_indices(240, 12, rng)
    assert idx.shape == (240,)
    assert idx.min() >= 0 and idx.max() < 240


def test_bagged_pbo_in_unit_interval_and_shapes():
    perf = _noisy_panel(1)
    pbo, per_resample = bagged_probability_of_backtest_overfitting(
        perf, n_partitions=8, n_bag=20, seed=3
    )
    assert 0.0 <= pbo <= 1.0
    assert per_resample.shape == (20,)
    assert np.isclose(pbo, per_resample.mean())


def test_bagging_reduces_variance_vs_plain_pbo():
    """
    The core mechanism: across independent panels the bagged PBO has lower variance than the single
    plain CSCV PBO (replication of the appraisal's variance-reduction finding).
    """
    plain, bagged = [], []
    for seed in range(40):
        perf = _noisy_panel(seed)
        plain.append(probability_of_backtest_overfitting(perf, n_partitions=8)[0])
        bagged.append(
            bagged_probability_of_backtest_overfitting(
                perf, n_partitions=8, n_bag=25, seed=seed
            )[0]
        )
    assert np.var(bagged) < np.var(plain)


def _true_pbo(true_sharpes, t_len, n_partitions, n_sims):
    """Ground-truth symmetric-CSCV PBO scored by the TRUE Sharpe (non-circular), over fresh panels."""
    from itertools import combinations

    splits = [set(c) for c in combinations(range(n_partitions), n_partitions // 2)]
    median = np.median(true_sharpes)
    rng = np.random.default_rng(999)
    overfit = []
    for _ in range(n_sims):
        panel = rng.standard_normal((t_len, true_sharpes.size)) + true_sharpes
        parts = np.array_split(panel, n_partitions)
        for train_ids in splits:
            train = np.concatenate([parts[i] for i in train_ids], axis=0)
            best = int(np.argmax(train.mean(0) / train.std(0)))
            overfit.append(1.0 if true_sharpes[best] < median else 0.0)
    return float(np.mean(overfit))


def test_bagged_pbo_closer_to_truth_than_plain():
    """
    Replication of the appraisal's headline: the bagged PBO has lower estimation error against the
    known true PBO than the single plain CSCV PBO (variance reduction -> lower error).
    """
    true_sharpes = np.linspace(
        -0.04, 0.04, 12
    )  # tight spread => genuine overfitting risk
    true_pbo = _true_pbo(true_sharpes, t_len=240, n_partitions=8, n_sims=300)
    plain_err, bagged_err = [], []
    for seed in range(30):
        rng = np.random.default_rng(seed)
        perf = rng.standard_normal((240, 12)) + true_sharpes
        plain_err.append(
            abs(probability_of_backtest_overfitting(perf, n_partitions=8)[0] - true_pbo)
        )
        bagged_err.append(
            abs(
                bagged_probability_of_backtest_overfitting(
                    perf, n_partitions=8, n_bag=25, seed=seed
                )[0]
                - true_pbo
            )
        )
    assert np.mean(bagged_err) < np.mean(plain_err)


def test_deterministic_with_seed():
    perf = _noisy_panel(7)
    a = bagged_probability_of_backtest_overfitting(
        perf, n_partitions=8, n_bag=15, seed=11
    )[0]
    b = bagged_probability_of_backtest_overfitting(
        perf, n_partitions=8, n_bag=15, seed=11
    )[0]
    assert a == b
