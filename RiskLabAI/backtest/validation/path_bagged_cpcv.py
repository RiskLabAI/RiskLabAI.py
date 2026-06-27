"""
Path-level Bagged Combinatorial Purged Cross-Validation PBO (Arian-Norouzi-Seco 2024).

de Prado's CPCV / CSCV estimates the Probability of Backtest Overfitting (PBO) from a finite set of
combinatorial backtest paths, so the estimate carries estimation variance. Bagging the PATHS - taking
a moving-block bootstrap of the performance series, recomputing the CSCV PBO on each resample, and
averaging - reduces that variance, giving a more accurate PBO estimate whenever the path set is small
or noisy, and converging to plain CPCV in the data-rich limit.

This is the **path-level** mechanism from the Appraisal 09 pre-registration. It is distinct from, and
does NOT touch, the existing :class:`BaggedCombinatorialPurged`, which bags a sklearn *estimator* on an
ML backtest (a different mechanism). The plain-CPCV baseline here is the repo's CSCV PBO
(:func:`~RiskLabAI.backtest.probability_of_backtest_overfitting.probability_of_backtest_overfitting`),
which this function calls on each bootstrap resample.

Admitted in Appraisal 09 (CONTRIBUTIONS_LEDGER 2026-06-27; in-house method, held to the identical bar).
Regime tag, verbatim from the verdict:

    For a more accurate, lower-variance overfitting (PBO) estimate whenever the CPCV path set is small
    or noisy, converging to plain CPCV in the data-rich limit; neutral on model selection. Point-in-time
    and mechanism-backed.

Evidence and caveats: appraisals/09_verdict.md.

References
----------
Arian, H., Norouzi M., L. and Seco, L. (2024) Bagged and Adaptive Combinatorial Purged
    Cross-Validation. (Clean-room from the path-level mechanism; the verdict harness validated plain
    CPCV against probability_of_backtest_overfitting exactly.)
Bailey, D. H., Borwein, J., Lopez de Prado, M. and Zhu, Q. J. (2017) The probability of backtest
    overfitting. Journal of Computational Finance, 20(4), 39-69.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ..probability_of_backtest_overfitting import probability_of_backtest_overfitting


def moving_block_bootstrap_indices(
    n_observations: int, block_size: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Moving-block bootstrap row indices for a length-n series (blocks preserve serial structure).

    Random block start points are tiled, each contributing ``block_size`` consecutive (wrap-around)
    rows, truncated to ``n_observations``.

    Parameters
    ----------
    n_observations : int
        The series length n.
    block_size : int
        The block length.
    rng : np.random.Generator
        The random generator (seed it for reproducibility).

    Returns
    -------
    np.ndarray
        An index array of length ``n_observations``.
    """
    block_size = max(int(block_size), 1)
    n_blocks = int(np.ceil(n_observations / block_size))
    starts = rng.integers(0, n_observations, size=n_blocks)
    offsets = np.arange(block_size)
    return (starts[:, None] + offsets[None, :]).ravel()[
        :n_observations
    ] % n_observations


def bagged_probability_of_backtest_overfitting(
    performances: np.ndarray,
    n_partitions: int = 16,
    n_bag: int = 30,
    block_size: Optional[int] = None,
    risk_free_return: float = 0.0,
    metric: Optional[Callable[[np.ndarray, float], float]] = None,
    seed: int = 0,
) -> tuple[float, np.ndarray]:
    r"""
    Bagged (path-level) Probability of Backtest Overfitting.

    Takes ``n_bag`` moving-block bootstrap resamples of the performance matrix, computes the CSCV PBO
    (:func:`probability_of_backtest_overfitting`) on each, and returns the average - a lower-variance
    estimate than a single CPCV PBO. Prefer it whenever the CPCV path set is small or noisy; it
    converges to plain CPCV in the data-rich limit and is neutral on model selection. See the module
    docstring for the full regime tag and appraisals/09_verdict.md.

    Parameters
    ----------
    performances : np.ndarray
        A T-observations x N-strategies matrix of per-period returns.
    n_partitions : int, default 16
        The number of CSCV partitions S (must be even); passed through to the plain PBO.
    n_bag : int, default 30
        The number of moving-block bootstrap resamples to aggregate.
    block_size : int, optional
        The bootstrap block length. Defaults to ``max(T // 20, 5)``.
    risk_free_return : float, default 0.0
        Passed through to the metric.
    metric : Callable, optional
        The performance metric (defaults to the PBO module's Sharpe).
    seed : int, default 0
        The bootstrap seed (for reproducibility).

    Returns
    -------
    (float, np.ndarray)
        The bagged PBO (the mean over resamples) and the array of per-resample PBOs.

    Examples
    --------
    >>> import numpy as np
    >>> from RiskLabAI.backtest.validation import bagged_probability_of_backtest_overfitting
    >>> rng = np.random.default_rng(0)
    >>> perf = rng.standard_normal((480, 10)) * 0.05 + np.linspace(0, 0.01, 10)
    >>> pbo, per_resample = bagged_probability_of_backtest_overfitting(perf, n_partitions=8, n_bag=20)
    >>> 0.0 <= pbo <= 1.0
    True
    """
    performances = np.asarray(performances, dtype=float)
    t_len = performances.shape[0]
    if block_size is None:
        block_size = max(t_len // 20, 5)
    rng = np.random.default_rng(seed)
    pbos = np.empty(n_bag)
    for b in range(n_bag):
        rows = moving_block_bootstrap_indices(t_len, block_size, rng)
        pbo, _ = probability_of_backtest_overfitting(
            performances[rows], n_partitions, risk_free_return, metric
        )
        pbos[b] = pbo
    return float(pbos.mean()), pbos
