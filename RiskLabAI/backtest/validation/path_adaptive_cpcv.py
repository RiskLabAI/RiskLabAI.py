"""
Path-level Adaptive Combinatorial Purged Cross-Validation PBO (Arian-Norouzi-Seco 2024).

de Prado's CPCV / CSCV weights every combinatorial backtest path equally when it estimates the
Probability of Backtest Overfitting (PBO) and when it selects a configuration. When the train/test
regime shifts, that equal weighting biases the selection toward the wrong regime. Adaptive CPCV
weights each path by how much its TEST block resembles the FORWARD regime, where the regime is read
from an observable, point-in-time volatility signal (the cross-configuration dispersion). Down-weighting
paths that test on a non-representative regime lowers selection error when the regime is detectable and
selection is signal-bearing, and the weights collapse to uniform in a stationary single regime, so the
estimate converges to plain CPCV with no over-adaptation.

This is the **path-level** mechanism from the Appraisal 09 pre-registration. It is distinct from, and
does NOT touch, the existing :class:`AdaptiveCombinatorialPurged`, which adapts the split *boundaries*
of an ML backtest (a different mechanism). The plain-CPCV baseline here is the repo's CSCV PBO: the
per-path overfit indicator is taken from
:func:`~RiskLabAI.backtest.probability_of_backtest_overfitting.performance_evaluation`, so with uniform
weights this function reproduces
:func:`~RiskLabAI.backtest.probability_of_backtest_overfitting.probability_of_backtest_overfitting`
exactly.

Admitted in Appraisal 09b (CONTRIBUTIONS_LEDGER 2026-06-27; in-house method under a conflict of
interest, held to the identical bar, admitted only after a re-designed signal-bearing held-out).
Regime tag, verbatim from the 09b verdict:

    prefer Adaptive CPCV over plain CPCV for model selection when the train/test regime may shift, the
    shift is identifiable at decision time from observable volatility, and there is adequate data for
    selection (it does not help where selection is noise-dominated by too little data); it converges to
    plain CPCV in a stationary regime with no over-adaptation.

The advantage is, as of admission, Monte-Carlo evidence where the regime is observable through the
adaptive feature; a real-data regime-shift confirmation on a genuine crisis/volatility period is a
tracked obligation (``REAL_DATA_FOLLOWUPS.md``). Evidence and caveats: appraisals/09_verdict.md (the
09b re-appraisal).

References
----------
Arian, H., Norouzi M., L. and Seco, L. (2024) Bagged and Adaptive Combinatorial Purged
    Cross-Validation. (Clean-room from the path-level mechanism; the verdict harness validated plain
    CPCV against probability_of_backtest_overfitting exactly.)
Bailey, D. H., Borwein, J., Lopez de Prado, M. and Zhu, Q. J. (2017) The probability of backtest
    overfitting. Journal of Computational Finance, 20(4), 39-69.
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable

import numpy as np

from ..backtest_statistics import sharpe_ratio
from ..probability_of_backtest_overfitting import performance_evaluation


def estimate_volatility_regimes(
    performances: np.ndarray, n_regimes: int = 2, window: int | None = None
) -> np.ndarray:
    """
    Observable regime label per period from the cross-configuration volatility.

    The signal is the per-period dispersion across configurations (a market-stress proxy), smoothed by a
    **trailing** moving average so the volatility feature at period ``t`` uses only returns up to ``t``.
    Periods are then labelled high- vs low-volatility relative to the median of the observed signal (an
    evaluation-time normalisation over the in-sample window). The estimate keys on volatility alone: it
    is invariant to the sign of the returns and never uses the forward performance the method selects, so
    it carries no look-ahead into the quantity being predicted.

    Parameters
    ----------
    performances : np.ndarray
        A T-observations x N-configurations matrix of per-period returns.
    n_regimes : int, default 2
        The number of regimes (>= 2); 2 gives a low/high-volatility split at the observed median.
    window : int, optional
        The trailing smoothing window. Defaults to ``max(T // 20, 3)``.

    Returns
    -------
    np.ndarray
        An integer regime label in ``[0, n_regimes)`` for each of the T periods.
    """
    performances = np.asarray(performances, dtype=float)
    t_len = performances.shape[0]
    dispersion = performances.std(axis=1)
    if window is None:
        window = max(t_len // 20, 3)
    window = max(int(window), 1)
    # trailing (causal) moving average: cumulative-sum difference over a back-window
    cumulative = np.concatenate([[0.0], np.cumsum(dispersion)])
    counts = np.minimum(np.arange(1, t_len + 1), window)
    starts = np.maximum(np.arange(1, t_len + 1) - window, 0)
    smooth = (cumulative[1:] - cumulative[starts]) / counts
    if n_regimes <= 2:
        return (smooth > np.median(smooth)).astype(int)
    edges = np.quantile(smooth, np.linspace(0.0, 1.0, n_regimes + 1)[1:-1])
    return np.digitize(smooth, edges)


def _cscv_path_stats(
    performances: np.ndarray,
    n_partitions: int,
    metric: Callable[[np.ndarray, float], float],
    risk_free_return: float,
):
    """
    Per CSCV path: ``(is_overfit, test_config_metrics, test_partition_ids)``.

    The overfit indicator comes from the repo's
    :func:`~RiskLabAI.backtest.probability_of_backtest_overfitting.performance_evaluation`, over the same
    symmetric train/test combinations the plain PBO uses, so the unweighted mean reproduces the repo PBO
    exactly. The per-configuration test metrics are recomputed with the same metric for the (weighted)
    selection and weighting.
    """
    _, n_strategies = performances.shape
    partitions = np.array_split(performances, n_partitions)
    all_ids = list(range(n_partitions))
    stats = []
    for train_ids in combinations(all_ids, n_partitions // 2):
        test_ids = [i for i in all_ids if i not in train_ids]
        train = np.concatenate([partitions[i] for i in train_ids], axis=0)
        test = np.concatenate([partitions[i] for i in test_ids], axis=0)
        is_overfit, _logit = performance_evaluation(
            train, test, n_strategies, metric, risk_free_return
        )
        test_metrics = np.array(
            [metric(test[:, j], risk_free_return) for j in range(n_strategies)]
        )
        stats.append((1.0 if is_overfit else 0.0, test_metrics, test_ids))
    return stats


def adaptive_probability_of_backtest_overfitting(
    performances: np.ndarray,
    n_partitions: int = 16,
    target_fraction: float = 0.25,
    risk_free_return: float = 0.0,
    metric: Callable[[np.ndarray, float], float] | None = None,
    n_regimes: int = 2,
) -> tuple[float, int]:
    r"""
    Adaptive (path-level) Probability of Backtest Overfitting and regime-aware configuration selection.

    Each CSCV path is weighted by how much its test block resembles the forward regime (the regime of
    the most recent ``target_fraction`` of the sample, read from the observable volatility signal of
    :func:`estimate_volatility_regimes`). The regime-weighted PBO and the configuration with the highest
    regime-weighted out-of-sample metric are returned. In a stationary regime the weights are uniform,
    so the PBO converges to plain CPCV and the selection matches it; see the module docstring for the
    full regime tag and appraisals/09_verdict.md.

    Parameters
    ----------
    performances : np.ndarray
        A T-observations x N-configurations matrix of per-period returns.
    n_partitions : int, default 16
        The number of CSCV partitions S (must be even).
    target_fraction : float, default 0.25
        The trailing fraction of the sample whose regime defines the forward target (in ``(0, 1]``).
    risk_free_return : float, default 0.0
        Passed through to the metric.
    metric : Callable, optional
        The performance metric (defaults to the PBO module's Sharpe).
    n_regimes : int, default 2
        The number of volatility regimes to estimate.

    Returns
    -------
    (float, int)
        The regime-weighted PBO and the index of the selected configuration.

    Examples
    --------
    >>> import numpy as np
    >>> from RiskLabAI.backtest.validation import (
    ...     adaptive_probability_of_backtest_overfitting,
    ... )
    >>> rng = np.random.default_rng(0)
    >>> # a regime shift: config 0 leads early, config 9 leads in the volatile forward block
    >>> perf = rng.standard_normal((480, 10)) * 0.05
    >>> perf[:360, 0] += 0.02
    >>> perf[360:, 9] += 0.02 + rng.standard_normal((120, 1))[:, 0] * 0.05
    >>> pbo, selected = adaptive_probability_of_backtest_overfitting(perf, n_partitions=8)
    >>> 0.0 <= pbo <= 1.0 and 0 <= selected < 10
    True
    """
    performances = np.asarray(performances, dtype=float)
    if n_partitions % 2 != 0:
        raise ValueError("Number of partitions must be even.")
    if performances.ndim != 2:
        raise ValueError("performances must be a 2-D (T x N) matrix.")
    t_len, _ = performances.shape
    if t_len < n_partitions:
        raise ValueError(
            f"Too few observations ({t_len}) for {n_partitions} partitions."
        )
    if not 0.0 < target_fraction <= 1.0:
        raise ValueError("target_fraction must be in (0, 1].")
    if metric is None:
        metric = sharpe_ratio

    regimes = estimate_volatility_regimes(performances, n_regimes=n_regimes)
    forward = regimes[int((1.0 - target_fraction) * t_len) :]
    # majority regime label over the forward block (the target the selection is steered toward)
    forward_label = int(np.bincount(forward).argmax()) if forward.size else 0

    regime_partitions = np.array_split(regimes, n_partitions)
    stats = _cscv_path_stats(performances, n_partitions, metric, risk_free_return)

    overfit = np.array([s[0] for s in stats])
    test_metrics = np.array([s[1] for s in stats])  # (n_paths, n_configs)
    weights = np.empty(len(stats))
    for k, (_overfit, _metrics, test_ids) in enumerate(stats):
        test_regime = np.concatenate([regime_partitions[i] for i in test_ids])
        # similarity = share of the test block in the forward regime; epsilon avoids all-zero weights
        weights[k] = float(np.mean(test_regime == forward_label)) + 1e-3

    pbo = float(np.average(overfit, weights=weights))
    selected = int(np.argmax(np.average(test_metrics, axis=0, weights=weights)))
    return pbo, selected
