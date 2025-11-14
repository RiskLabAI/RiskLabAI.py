"""
Implements the Probability of Backtest Overfitting (PBO) calculation.

## TODO:
- [ ] Add a `get_pbo` wrapper function (as seen in the
      original notebook) that simplifies the call to
      `probability_of_backtest_overfitting` and returns
      only the PBO value.
- [ ] Add a `pbo_overfitting_plot` helper function to
      visualize the logit distribution (as seen in the
      original notebook).
"""

from typing import Tuple, Callable, List, Optional
from itertools import combinations
import numpy as np
from numba import jit
from joblib import Parallel, delayed

from .backtest_statistics import sharpe_ratio

def performance_evaluation(
    train_partition: np.ndarray,
    test_partition: np.ndarray,
    n_strategies: int,
    metric: Callable[[np.ndarray, float], float],
    risk_free_return: float,
) -> Tuple[bool, float]:
    """
    Evaluate strategy performance on train/test splits.

    This function finds the best strategy on the training partition and
    then ranks its performance on the test partition.

    Parameters
    ----------
    train_partition : np.ndarray
        Training data partition (T_train x N_strategies).
    test_partition : np.ndarray
        Testing data partition (T_test x N_strategies).
    n_strategies : int
        Number of strategies (N).
    metric : Callable
        Metric function (e.g., `sharpe_ratio`).
    risk_free_return : float
        Risk-free return for the metric.

    Returns
    -------
    Tuple[bool, float]
        - is_overfit: True if the best in-sample strategy is not
                      in the top half of out-of-sample strategies.
        - logit_value: The logit-transformed relative rank.
    """
    # 1. Find best strategy on training data
    evaluate_train = [
        metric(train_partition[:, i], risk_free_return)
        for i in range(n_strategies)
    ]
    best_strategy_idx = np.argmax(evaluate_train)

    # 2. Evaluate all strategies on test data
    evaluate_test = [
        metric(test_partition[:, i], risk_free_return)
        for i in range(n_strategies)
    ]

    # 3. Find rank of the best_strategy in the test set
    # Sort test evaluations and find where best_strategy_idx lands
    ranks = np.argsort(np.argsort(evaluate_test))
    rank_of_best_is_strategy = ranks[best_strategy_idx] + 1  # 1-based rank

    # 4. Calculate relative rank (omega_bar) and logit
    w_bar = rank_of_best_is_strategy / (n_strategies + 1)
    logit_value = np.log(w_bar / (1 - w_bar))

    # Overfit if logit is <= 0 (i.e., rank is in bottom half)
    return logit_value <= 0.0, logit_value


def probability_of_backtest_overfitting(
    performances: np.ndarray,
    n_partitions: int = 16,
    risk_free_return: float = 0.0,
    metric: Optional[Callable[[np.ndarray, float], float]] = None,
    n_jobs: int = 1,
) -> Tuple[float, np.ndarray]:
    r"""
    Compute the Probability of Backtest Overfitting (PBO).

    PBO is the frequency with which the best in-sample (IS) strategy
    underperforms relative to the median out-of-sample (OOS) performance.

    It uses combinatorial splits of the data into train/test partitions.
    Total combinations = C(S, S/2), where S = n_partitions.

    .. math::
        \left(\begin{array}{c}
        S \\
        S / 2
        \end{array}\right) = \prod_{i=0}^{S / 2^{-1}} \frac{S-i}{S / 2-i}

    Parameters
    ----------
    performances : np.ndarray
        Matrix of T observations x N strategies.
    n_partitions : int, default=16
        Number of partitions (S). Must be even.
    risk_free_return : float, default=0.0
        Risk-free return for calculating the metric (e.g., Sharpe ratio).
    metric : Callable, optional
        Metric function (e.g., `sharpe_ratio`). Defaults to the
        Numba-jitted `sharpe_ratio` in this module.
    n_jobs : int, default=1
        Number of parallel jobs for `joblib`.

    Returns
    -------
    Tuple[float, np.ndarray]
        - pbo: The Probability of Backtest Overfitting (0 to 1).
        - logit_values: Array of logit values from all combinations.
    """
    if n_partitions % 2 != 0:
        raise ValueError("Number of partitions must be even.")

    if metric is None:
        metric = sharpe_ratio

    _, n_strategies = performances.shape
    partitions = np.array_split(performances, n_partitions)
    partition_indices = list(range(n_partitions))
    
    # Get all combinations of training partition indices
    partition_combinations_indices = list(
        combinations(partition_indices, n_partitions // 2)
    )

    results = Parallel(n_jobs=n_jobs)(
        delayed(performance_evaluation)(
            np.concatenate([partitions[i] for i in train_indices], axis=0),
            np.concatenate(
                [
                    partitions[i]
                    for i in partition_indices
                    if i not in train_indices
                ],
                axis=0,
            ),
            n_strategies,
            metric,
            risk_free_return,
        )
        for train_indices in partition_combinations_indices
    )

    results_arr = np.array(results)

    pbo = results_arr[:, 0].mean()
    logit_values = results_arr[:, 1]

    return pbo, logit_values