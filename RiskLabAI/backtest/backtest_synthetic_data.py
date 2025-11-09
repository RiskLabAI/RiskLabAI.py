"""
Performs backtesting on synthetic price data generated using the
Ornstein-Uhlenbeck (OU) process.
"""

from itertools import product
from random import gauss
from typing import List, Tuple

import numpy as np

def synthetic_back_testing(
    forecast: float,
    half_life: float,
    sigma: float,
    n_iteration: int = 100000,
    maximum_holding_period: int = 100,
    profit_taking_range: np.ndarray = np.linspace(0.5, 10, 20),
    stop_loss_range: np.ndarray = np.linspace(0.5, 10, 20),
    seed: int = 0,
) -> List[Tuple[float, float, float, float, float]]:
    r"""
    Perform backtesting on synthetic price data from an OU process.

    This function simulates multiple price paths based on the
    Ornstein-Uhlenbeck (OU) process and tests a grid of
    profit-taking (PT) and stop-loss (SL) levels.

    The OU process is given by:
    .. math::
        P_t = (1 - \rho) F + \rho P_{t-1} + \sigma Z_t

    Where:
    - \(P_t\) is the price at time t
    - \(F\) is the forecast price (long-term mean)
    - \(\rho\) is the autoregression coefficient
    - \(\sigma\) is the standard deviation of noise
    - \(Z_t \sim N(0, 1)\) is standard normal noise

    Parameters
    ----------
    forecast : float
        The forecasted price (long-term mean F).
    half_life : float
        The half-life of the mean-reversion process.
    sigma : float
        The standard deviation of the noise (\(\sigma\)).
    n_iteration : int, default=100000
        Number of price paths to simulate for each PT/SL combination.
    maximum_holding_period : int, default=100
        Maximum number of steps to hold the position.
    profit_taking_range : np.ndarray, default=np.linspace(0.5, 10, 20)
        An array of profit-taking levels to test.
    stop_loss_range : np.ndarray, default=np.linspace(0.5, 10, 20)
        An array of stop-loss levels to test.
    seed : int, default=0
        The initial price for all simulations.

    Returns
    -------
    List[Tuple[float, float, float, float, float]]
        A list of tuples. Each tuple contains:
        (profit_taking, stop_loss, mean_return, std_return, sharpe_ratio)
    """
    # compute ρ coefficient from half-life
    rho = 2 ** (-1.0 / half_life)
    back_test_results = []

    for profit_taking, stop_loss in product(profit_taking_range, stop_loss_range):
        stop_returns = []
        for _ in range(n_iteration):
            price, holding_period = float(seed), 0  # initial price
            while True:
                # Update price using O-U process
                price = (1 - rho) * forecast + rho * price + sigma * gauss(0, 1)
                gain = price - seed  # compute gain from initial seed price
                holding_period += 1

                # Check stop conditions
                if (
                    gain > profit_taking
                    or gain < -stop_loss
                    or holding_period > maximum_holding_period
                ):
                    stop_returns.append(gain)
                    break
        
        mean_return = np.mean(stop_returns)
        std_return = np.std(stop_returns)
        sharpe_ratio = mean_return / std_return if std_return > 0 else 0.0
        
        back_test_results.append(
            (profit_taking, stop_loss, mean_return, std_return, sharpe_ratio)
        )

    return back_test_results