import numpy as np
import pandas as pd
from numba import jit
import quantecon.markov as qe
from joblib import Parallel, delayed
from typing import Dict, List, Tuple, Union

@jit(nopython=True)
def compute_log_returns(
    N: int,
    mu_vector: np.ndarray,
    kappa_vector: np.ndarray,
    theta_vector: np.ndarray,
    xi_vector: np.ndarray,
    dwS: np.ndarray,
    dwV: np.ndarray,
    Y: np.ndarray,
    n: np.ndarray,
    dt: float,
    sqrt_dt: float,
    lambda_vector: np.ndarray,
    m_vector: np.ndarray,
    v_vector: np.ndarray,
    regime_change: np.ndarray,
) -> np.ndarray:
    r"""
    Computes the log returns based on the Heston-Merton model.

    :param N: Number of steps
    :param mu_vector: Drift vector of length N
    :param kappa_vector: Mean-reversion speed vector of length N
    :param theta_vector: Long-term mean vector of length N
    :param xi_vector: Volatility of volatility vector of length N
    :param dwS: Wiener process for stock
    :param dwV: Wiener process for volatility
    :param Y: Jump component
    :param n: Poisson random variable vector
    :param dt: Time step
    :param sqrt_dt: Square root of the time step
    :param lambda_vector: Intensity of the jump vector
    :param m_vector: Mean of jump size vector
    :param v_vector: Variance of jump size vector
    :param regime_change: Regime change booleans
    :return: Log returns based on the Heston-Merton model

    The Heston Merton model formulae for log returns are:
    .. math::
       v_{i+1} = v_i + \kappa_i (\theta_i - \max(v_i, 0)) dt + \xi_i \sqrt{\max(v_i, 0)} dwV_i \sqrt{dt}
       log\_returns_i = (\mu_i - 0.5 v_i - \lambda_i (m_i + \frac{v^2_i}{2})) dt + \sqrt{v_i} dwS_i \sqrt{dt} + dJ_i
    """

    v = np.empty(N + 1)
    dJ = np.multiply(n, Y)
    v[0] = theta_vector[0]
    log_returns = np.empty(N)
    for i in range(N):
        if regime_change[i]:
            v[i] = theta_vector[i]

        v[i + 1] = v[i] + kappa_vector[i] * (theta_vector[i] - max(v[i], 0)) * dt + \
                   xi_vector[i] * np.sqrt(max(v[i], 0)) * dwV[i] * sqrt_dt
        
        log_returns[i] = (mu_vector[i] - 0.5 * v[i] - lambda_vector[i] * 
                          (m_vector[i] + (v_vector[i] ** 2) / 2)) * dt + \
                         np.sqrt(v[i]) * dwS[i] * sqrt_dt + dJ[i]

    return log_returns

def heston_merton_log_returns(
    T: float,
    N: int,
    mu_vector: np.ndarray,
    kappa_vector: np.ndarray,
    theta_vector: np.ndarray,
    xi_vector: np.ndarray,
    rho_vector: np.ndarray,
    lambda_vector: np.ndarray,
    m_vector: np.ndarray,
    v_vector: np.ndarray,
    regime_change: np.ndarray,
    random_state=None
) -> np.ndarray:
    """
    Computes the log returns based on the Heston-Merton model using Gaussian random numbers.

    :param T: Total time
    :param N: Number of steps
    :param mu_vector: Drift vector of length N
    :param kappa_vector: Mean-reversion speed vector of length N
    :param theta_vector: Long-term mean vector of length N
    :param xi_vector: Volatility of volatility vector of length N
    :param rho_vector: Correlation coefficient vector of length N
    :param lambda_vector: Intensity of the jump vector
    :param m_vector: Mean of jump size vector
    :param v_vector: Variance of jump size vector
    :param random_state: Random state for reproducibility
    :param regime_change: Regime change booleans
    :return: Log returns based on the Heston-Merton model

    """

    # Ensure the length of parameter vectors is N
    assert len(mu_vector) == len(kappa_vector) == len(theta_vector) == len(xi_vector) == len(rho_vector) == N

    rng = np.random.default_rng(random_state)
    dt = T / N
    sqrt_dt = np.sqrt(dt)

    # Generate Gaussian random numbers
    z = np.zeros((N, 3))
    n = np.zeros(N)
    for i in range(N):
        z[i] = rng.multivariate_normal(
            [0, 0, m_vector[i]],
            [
                [1.0,           rho_vector[i],    0.0],
                [rho_vector[i], 1.0,              0.0],
                [0.0,           0.0, v_vector[i] ** 2]
            ]
        )
        n[i] = rng.poisson(lambda_vector[i] * dt)

    dwS = z[:, 0]
    dwV = z[:, 1]
    Y = z[:, 2]

    return compute_log_returns(N, mu_vector, kappa_vector, theta_vector, xi_vector, dwS, dwV, Y, n, dt, sqrt_dt,
                               lambda_vector, m_vector, v_vector, regime_change)

def align_params_length(
    regime_params: Dict[str, Union[float, List[float]]]
) -> Tuple[Dict[str, List[float]], int]:
    """
    Align the parameters' length within the provided regime parameters.

    :param regime_params: Dictionary of regime parameters. Values can be floats or lists.
    :return: A tuple containing the regime parameters with aligned lengths and the max length.
    """
    max_len = max([len(value) if isinstance(value, list) else 1 for value in regime_params.values()])

    for key, value in regime_params.items():
        if isinstance(value, list):
            if len(value) < max_len:
                regime_params[key].extend([value[-1]] * (max_len - len(value)))
        else:
            regime_params[key] = [value] * max_len

    return regime_params, max_len

def generate_prices_from_regimes(
    regimes: Dict[str, Dict[str, Union[float, List[float]]]],
    transition_matrix: np.ndarray,
    total_time: float,
    n_steps: int,
    random_state: int = None
) -> Tuple[pd.Series, np.ndarray]:
    """
    Generate prices based on provided regimes and a Markov Chain.

    :param regimes: Dictionary containing regime names and their respective parameters.
    :param transition_matrix: Markov Chain transition matrix.
    :param total_time: Total time for the simulation.
    :param n_steps: Number of discrete steps in the simulation.
    :param random_state: Seed for random number generation.
    :return: A tuple containing the generated prices as a pandas Series and the simulated regimes.
    """
    markov_chain = qe.MarkovChain(transition_matrix, state_values=list(regimes.keys()))
    simulated_regimes = markov_chain.simulate(ts_length=n_steps, random_state=np.random.default_rng(random_state))

    parameter_lists = {
        'mu': [],
        'kappa': [],
        'theta': [],
        'xi': [],
        'rho': [],
        'lam': [],
        'm': [],
        'v': []
    }

    extension = 0
    for i, regime in enumerate(np.copy(simulated_regimes)):
        params, max_len = align_params_length(regimes[regime].copy())
        for key in parameter_lists:
            parameter_lists[key].extend(params[key])

        if max_len > 1:
            simulated_regimes = np.insert(simulated_regimes, i + extension, [regime] * (max_len - 1))
            extension += max_len - 1

    # Truncate parameter vectors if they exceed n_steps
    for key in parameter_lists:
        parameter_lists[key] = np.array(parameter_lists[key][:n_steps])
    simulated_regimes = simulated_regimes[:n_steps]

    regime_change = simulated_regimes[1:] != simulated_regimes[:-1]
    regime_change = np.pad(regime_change, (1, 0), mode='constant', constant_values=False)

    log_returns = heston_merton_log_returns(
        total_time,
        n_steps,
        parameter_lists['mu'],
        parameter_lists['kappa'],
        parameter_lists['theta'],
        parameter_lists['xi'],
        parameter_lists['rho'],
        parameter_lists['lam'],
        parameter_lists['m'],
        parameter_lists['v'],
        regime_change,
        random_state=random_state
    )
    # Define the starting day and N
    start_day = '2023-10-20'  # Example starting day
    # Create DatetimeIndex of the next N business days
    business_days = pd.date_range(start=start_day, periods=n_steps + 1, freq='B')[1:]  # Skip the start day

    prices = np.exp(np.cumsum(pd.Series(log_returns, business_days).ffill()))

    return prices, simulated_regimes

def parallel_generate_prices(
        number_of_paths: int,
        regimes: Dict[str, Dict[str, Union[float, List[float]]]],
        transition_matrix: np.ndarray,
        total_time: float,
        number_of_steps: int,
        random_state: Union[int, None] = None,
        n_jobs: int = 1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parallel generation of prices using the provided regimes.

    :param number_of_paths: The number of paths to generate.
    :param regimes: Dictionary containing regime names and their respective parameters.
    :param transition_matrix: Markov Chain transition matrix.
    :param total_time: Total time for the simulation.
    :param number_of_steps: Number of discrete steps in the simulation.
    :param random_state: Seed for random number generation.
    :param n_jobs: Number of parallel jobs to run.
    :return: A tuple containing the generated prices and simulated regimes as pandas DataFrames.
    """
    random_generator = np.random.default_rng(random_state)
    random_states = random_generator.choice(10 * number_of_paths, size=number_of_paths, replace=False)
    results = Parallel(n_jobs=n_jobs)(
        delayed(generate_prices_from_regimes)(regimes, transition_matrix, total_time, number_of_steps, random_state_value)
        for random_state_value in random_states
    )

    prices, simulated_regimes = zip(*results)

    prices_df = pd.DataFrame(prices)
    simulated_regimes_df = pd.DataFrame(simulated_regimes)

    return prices_df.T, simulated_regimes_df.T
