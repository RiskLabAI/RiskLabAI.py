import numpy as np
from numba import jit
from joblib import Parallel, delayed
from typing import Tuple, Callable
from itertools import combinations
from joblib import Parallel, delayed


@jit(nopython=True)
def sharpe_ratio(
    returns: np.ndarray, 
    risk_free_rate: float = 0.0
) -> float:
    """
    Calculate the Sharpe Ratio for a given set of returns.

    :param returns: An array of returns for a portfolio.
    :param risk_free_rate: The risk-free rate.
    :return: The calculated Sharpe Ratio.

    .. math::

        \text{Sharpe Ratio} = \frac{\text{Mean Portfolio Return} - \text{Risk-Free Rate}}
                                  {\text{Standard Deviation of Portfolio Returns}}
    """
    excess_returns = returns - risk_free_rate
    std = np.std(excess_returns)
    if std != 0:
        
        return np.mean(excess_returns) / std
    
    else:
        return 0

def performance_evaluation(
    train_partition: np.ndarray,
    test_partition: np.ndarray,
    n_strategies: int,
    metric: Callable,
    risk_free_return: float
) -> Tuple[bool, float]:
    """
    Evaluate the performance of various strategies on given train and test partitions and 
    compute the logit value to determine if the best in-sample strategy is overfitting.

    :param train_partition: Training data partition used for evaluating in-sample performance.
    :type train_partition: np.ndarray
    :param test_partition: Testing data partition used for evaluating out-of-sample performance.
    :type test_partition: np.ndarray
    :param n_strategies: Number of strategies to evaluate.
    :type n_strategies: int
    :param metric: Metric function for evaluating strategy performance. 
                   The function should accept a data array and risk_free_return as arguments.
    :type metric: Callable
    :param risk_free_return: Risk-free return used in the metric function, often used for Sharpe ratio.
    :type risk_free_return: float

    :return: Tuple where the first value indicates if the best in-sample strategy is overfitting 
             (True if overfitting, False otherwise) and the second value is the logit value computed.
    :rtype: Tuple[bool, float]
    """

    evaluate_train = list(map(lambda i: metric(train_partition[:, i], risk_free_return), range(n_strategies)))
    best_strategy = np.argmax(evaluate_train)
    
    evaluate_test = list(map(lambda i: metric(test_partition[:, i], risk_free_return), range(n_strategies)))
    
    rank_of_best_is_strategy = np.argsort(evaluate_test).tolist().index(best_strategy) + 1
    
    w_bar = rank_of_best_is_strategy / (n_strategies + 1)
    logit_value = np.log(w_bar / (1 - w_bar))
    
    return logit_value <= 0.0, logit_value

def probability_of_backtest_overfitting(
    performances: np.ndarray, 
    n_partitions: int = 16, 
    risk_free_return: float = 0.0,
    metric: Callable = None, 
    n_jobs: int = 1
) -> Tuple[float, np.ndarray]:
    """
    Computes the Probability Of Backtest Overfitting.

    For instance, if \(S=16\), we will form 12,780 combinations.

    .. math::
        \left(\begin{array}{c}
        S \\
        S / 2
        \end{array}\right) = \prod_{i=0}^{S / 2^{-1}} \frac{S-i}{S / 2-i}

    :param performances: Matrix of T×N for T observations on N strategies.
    :type performances: np.ndarray
    :param n_partitions: Number of partitions (must be even).
    :type n_partitions: int
    :param metric: Metric function for evaluating strategy.
    :type metric: Callable
    :param risk_free_return: Risk-free return for calculating Sharpe ratio.
    :type risk_free_return: float
    :param n_jobs: Number of parallel jobs.
    :type n_jobs: int

    :return: Tuple containing Probability Of Backtest Overfitting and an array of logit values.
    :rtype: Tuple[float, List[float]]
    """

    if n_partitions % 2 == 1:
        raise ValueError("Number of partitions must be even.")
    
    if metric is None:
        metric = sharpe_ratio
    
    _, n_strategies = performances.shape
    partitions = np.array_split(performances, n_partitions)
    partition_indices = range(n_partitions)
    partition_combinations_indices = list(combinations(partition_indices, n_partitions // 2))

    results = Parallel(n_jobs=n_jobs)(
        delayed(performance_evaluation)(
            np.concatenate([partitions[i] for i in train_indices], axis=0),
            np.concatenate([partitions[i] for i in partition_indices 
                            if i not in train_indices], axis=0),
            n_strategies, 
            metric, 
            risk_free_return
        ) 
        for train_indices in partition_combinations_indices
    )

    results = np.array(results)  

    pbo = results[:, 0].mean(axis=0)
    logit_values = results[:, 1]
    
    return pbo, logit_values
