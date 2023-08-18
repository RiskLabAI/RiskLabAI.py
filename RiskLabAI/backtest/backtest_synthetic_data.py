import numpy as np
from random import gauss
from itertools import product


def synthetic_back_testing(
    forecast: float,
    half_life: float,
    sigma: float,
    n_iteration: int = 100000,
    maximum_holding_period: int = 100,
    profit_taking_range: np.ndarray = np.linspace(0.5, 10, 20),
    stop_loss_range: np.ndarray = np.linspace(0.5, 10, 20),
    seed: int = 0
) -> list[tuple[float, float, float, float, float]]:
    """
    Perform backtesting on synthetic price data generated using the Ornstein-Uhlenbeck process.

    The Ornstein-Uhlenbeck process is given by:
    .. math:: P_t = (1 - \\rho) * F + \\rho * P_{t-1} + \\sigma * Z_t

    where:
    - \(P_t\) is the price at time t
    - \(F\) is the forecast price
    - \(\\rho\) is the autoregression coefficient
    - \(\\sigma\) is the standard deviation of noise
    - \(Z_t\) is a random noise with mean 0 and standard deviation 1

    Args:
        forecast (float): The forecasted price.
        half_life (float): The half-life time needed to reach half.
        sigma (float): The standard deviation of the noise.
        n_iteration (int): Number of iterations. Defaults to 100000.
        maximum_holding_period (int): Maximum holding period. Defaults to 100.
        profit_taking_range (np.ndarray): Profit taking range. Defaults to np.linspace(0.5, 10, 20).
        stop_loss_range (np.ndarray): Stop loss range. Defaults to np.linspace(0.5, 10, 20).
        seed (int): Initial seed value. Defaults to 0.

    Returns:
        list[tuple[float, float, float, float, float]]: List of tuples containing profit taking, stop loss, mean,
        standard deviation, and Sharpe ratio.
    """
    rho = 2 ** (-1. / half_life)  # compute ρ coefficient from half-life
    back_test = []

    for profit_taking, stop_loss in product(profit_taking_range, stop_loss_range):
        stop_prices = []
        for iteration in range(n_iteration):
            price, holding_period = seed, 0  # initial price
            while True:
                price = (1 - rho) * forecast + rho * price + sigma * gauss(0, 1)  # update price using O_U process
                gain = price - seed  # compute gain
                holding_period += 1
                if gain > profit_taking or gain < -stop_loss or holding_period > maximum_holding_period:  # check stop condition
                    stop_prices.append(gain)
                    break
        mean, std = np.mean(stop_prices), np.std(stop_prices)  # compute mean and std of samples
        back_test.append((profit_taking, stop_loss, mean, std, mean / std))  # add mean, std, and Sharpe ratio to backTest data

    return back_test
