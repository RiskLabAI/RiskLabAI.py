"""
Generates synthetic price data using a Heston-Merton model
with Markov-switching regimes.
"""

from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import pandas as pd
import quantecon.markov as qe
from numba import jit
from joblib import Parallel, delayed

# Type hint for regime parameters
RegimeParams = Dict[str, Union[float, List[float]]]

@jit(nopython=True)
def compute_log_returns(
    n_steps: int,
    mu_vector: np.ndarray,
    kappa_vector: np.ndarray,
    theta_vector: np.ndarray,
    xi_vector: np.ndarray,
    dw_stock: np.ndarray,
    dw_vol: np.ndarray,
    jump_comp: np.ndarray,
    poisson_var: np.ndarray,
    dt: float,
    sqrt_dt: float,
    lambda_vector: np.ndarray,
    m_vector: np.ndarray,
    v_vector: np.ndarray,
    regime_change: np.ndarray,
) -> np.ndarray:
    r"""
    Compute log returns using the Heston-Merton model (Numba-optimized).

    Heston Model (Volatility):
    .. math::
       v_{i+1} = v_i + \kappa_i (\theta_i - \max(v_i, 0)) dt +
                 \xi_i \sqrt{\max(v_i, 0)} dwV_i \sqrt{dt}

    Merton Jump-Diffusion (Price):
    .. math::
       \log(S_{i+1}/S_i) = \left(\mu_i - 0.5 v_i - \lambda_i (m_i +
                          \frac{v^2_i}{2})\right) dt +
                          \sqrt{v_i} dwS_i \sqrt{dt} + dJ_i

    Parameters
    ----------
    n_steps : int
        Number of steps (N).
    mu_vector : np.ndarray
        Drift vector.
    kappa_vector : np.ndarray
        Mean-reversion speed vector.
    theta_vector : np.ndarray
        Long-term mean vector for volatility.
    xi_vector : np.ndarray
        Volatility of volatility vector.
    dw_stock : np.ndarray
        Wiener process for stock.
    dw_vol : np.ndarray
        Wiener process for volatility.
    jump_comp : np.ndarray
        Jump component (Y).
    poisson_var : np.ndarray
        Poisson random variable vector (n).
    dt : float
        Time step.
    sqrt_dt : float
        Square root of the time step.
    lambda_vector : np.ndarray
        Jump intensity vector.
    m_vector : np.ndarray
        Mean of jump size vector.
    v_vector : np.ndarray
        Variance of jump size vector.
    regime_change : np.ndarray
        Boolean array, True if regime changed at this step.

    Returns
    -------
    np.ndarray
        Array of log returns.
    """
    v = np.empty(n_steps + 1)
    jump_events = np.multiply(poisson_var, jump_comp)
    v[0] = theta_vector[0]  # Initial volatility
    log_returns = np.empty(n_steps)

    for i in range(n_steps):
        if regime_change[i]:
            v[i] = theta_vector[i]  # Reset vol to new regime's long-term mean

        # Ensure v[i] is non-negative for sqrt
        v_i_safe = max(v[i], 0.0)
        
        # Volatility process (Heston)
        v[i + 1] = (
            v[i]
            + kappa_vector[i] * (theta_vector[i] - v_i_safe) * dt
            + xi_vector[i] * np.sqrt(v_i_safe) * dw_vol[i] * sqrt_dt
        )

        # Log return (Merton Jump-Diffusion)
        log_returns[i] = (
            (
                mu_vector[i]
                - 0.5 * v_i_safe
                - lambda_vector[i] * (m_vector[i] + (v_vector[i] ** 2) / 2.0)
            )
            * dt
            + np.sqrt(v_i_safe) * dw_stock[i] * sqrt_dt
            + jump_events[i]
        )

    return log_returns


def heston_merton_log_returns(
    total_time: float,
    n_steps: int,
    mu_vector: np.ndarray,
    kappa_vector: np.ndarray,
    theta_vector: np.ndarray,
    xi_vector: np.ndarray,
    rho_vector: np.ndarray,
    lambda_vector: np.ndarray,
    m_vector: np.ndarray,
    v_vector: np.ndarray,
    regime_change: np.ndarray,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Generate Heston-Merton log returns.

    This function sets up the correlated random numbers and calls
    the Numba-jitted `compute_log_returns`.

    Parameters
    ----------
    total_time : float
        Total time (T).
    n_steps : int
        Number of steps (N).
    *..._vector : np.ndarray
        Arrays of model parameters (see `compute_log_returns`).
    regime_change : np.ndarray
        Boolean array, True if regime changed at this step.
    random_state : int, optional
        Seed for the random number generator.

    Returns
    -------
    np.ndarray
        Array of log returns.
    """
    # Ensure vectors are of length n_steps
    params = [
        mu_vector, kappa_vector, theta_vector, xi_vector, rho_vector,
        lambda_vector, m_vector, v_vector
    ]
    if not all(len(p) == n_steps for p in params):
        raise ValueError("All parameter vectors must have length `n_steps`")
    
    rng = np.random.default_rng(random_state)
    dt = total_time / n_steps
    sqrt_dt = np.sqrt(dt)

    # Generate correlated random numbers
    z = np.zeros((n_steps, 3))
    n = np.zeros(n_steps)
    for i in range(n_steps):
        cov_matrix = np.array([
            [1.0,           rho_vector[i], 0.0],
            [rho_vector[i], 1.0,           0.0],
            [0.0,           0.0,           v_vector[i] ** 2],
        ])
        z[i] = rng.multivariate_normal(
            [0.0, 0.0, m_vector[i]], cov_matrix
        )
        n[i] = rng.poisson(lambda_vector[i] * dt)

    dw_stock = z[:, 0]
    dw_vol = z[:, 1]
    jump_comp = z[:, 2]

    return compute_log_returns(
        n_steps,
        mu_vector,
        kappa_vector,
        theta_vector,
        xi_vector,
        dw_stock,
        dw_vol,
        jump_comp,
        n,
        dt,
        sqrt_dt,
        lambda_vector,
        m_vector,
        v_vector,
        regime_change,
    )


def align_params_length(
    regime_params: RegimeParams,
) -> Tuple[Dict[str, List[float]], int]:
    """
    Align the parameter lists within a regime to be the same length.

    If a parameter is a float, it's broadcast to a list.
    If a parameter is a list shorter than the max, it's extended
    by repeating its last value.

    Parameters
    ----------
    regime_params : RegimeParams
        Dictionary of regime parameters.

    Returns
    -------
    Tuple[Dict[str, List[float]], int]
        - The aligned regime parameter dictionary.
        - The maximum length (number of steps) for this regime.
    """
    max_len = max(
        len(v) if isinstance(v, list) else 1 for v in regime_params.values()
    )

    aligned_params: Dict[str, List[float]] = {}
    for key, value in regime_params.items():
        if isinstance(value, list):
            if len(value) < max_len:
                # Extend list by repeating last value
                aligned_params[key] = value + [value[-1]] * (max_len - len(value))
            else:
                aligned_params[key] = value[:max_len] # Truncate if too long
        else:
            # Broadcast float to list
            aligned_params[key] = [value] * max_len

    return aligned_params, max_len


def generate_prices_from_regimes(
    regimes: Dict[str, RegimeParams],
    transition_matrix: np.ndarray,
    total_time: float,
    n_steps: int,
    random_state: Optional[int] = None,
) -> Tuple[pd.Series, np.ndarray]:
    """
    Generate a price series from a Markov-switching regime model.

    Parameters
    ----------
    regimes : Dict[str, RegimeParams]
        Dictionary mapping regime names to their parameter sets.
    transition_matrix : np.ndarray
        The Markov Chain transition matrix.
    total_time : float
        Total time for the simulation (e.g., 1.0 for 1 year).
    n_steps : int
        Number of discrete steps in the simulation.
    random_state : int, optional
        Seed for random number generation.

    Returns
    -------
    Tuple[pd.Series, np.ndarray]
        - The generated price series, indexed by business days.
        - The array of simulated regime names for each step.
    """
    rng = np.random.default_rng(random_state)
    
    # 1. Simulate the Markov Chain
    regime_names = list(regimes.keys())
    markov_chain = qe.MarkovChain(transition_matrix, state_values=regime_names)
    simulated_regimes = markov_chain.simulate(
        ts_length=n_steps, random_state=rng
    )

    # 2. Unpack parameters based on simulated regimes
    param_lists: Dict[str, List[float]] = {
        "mu": [], "kappa": [], "theta": [], "xi": [],
        "rho": [], "lam": [], "m": [], "v": [],
    }
    
    regime_path_expanded = []
    
    current_step = 0
    while current_step < n_steps:
        regime_name = simulated_regimes[current_step]
        params, regime_len = align_params_length(regimes[regime_name].copy())
        
        steps_to_take = min(regime_len, n_steps - current_step)
        
        for key in param_lists:
            param_lists[key].extend(params[key][:steps_to_take])
            
        regime_path_expanded.extend([regime_name] * steps_to_take)
        current_step += steps_to_take

    # 3. Finalize parameter arrays and regime path
    simulated_regimes_final = np.array(regime_path_expanded)
    param_arrays: Dict[str, np.ndarray] = {
        key: np.array(val) for key, val in param_lists.items()
    }

    # 4. Identify regime change points
    regime_change = simulated_regimes_final[1:] != simulated_regimes_final[:-1]
    regime_change = np.pad(
        regime_change, (1, 0), mode="constant", constant_values=False
    )

    # 5. Generate log returns
    log_returns = heston_merton_log_returns(
        total_time,
        n_steps,
        param_arrays["mu"],
        param_arrays["kappa"],
        param_arrays["theta"],
        param_arrays["xi"],
        param_arrays["rho"],
        param_arrays["lam"],
        param_arrays["m"],
        param_arrays["v"],
        regime_change,
        random_state=rng.integers(0, 1e6),
    )

    # 6. Create price series with a Business Day index
    start_day = "2000-01-01"
    business_days = pd.date_range(
        start=start_day, periods=n_steps, freq="B"
    )
    
    price_series = pd.Series(log_returns, index=business_days).ffill()
    prices = 100 * np.exp(price_series.cumsum())
    prices.name = "Price"

    return prices, simulated_regimes_final

def parallel_generate_prices(
    number_of_paths: int,
    regimes: Dict[str, RegimeParams],
    transition_matrix: np.ndarray,
    total_time: float,
    n_steps: int,
    random_state: Optional[int] = None,
    n_jobs: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parallel generation of price paths.

    Parameters
    ----------
    number_of_paths : int
        The number of paths to generate.
    regimes : Dict[str, RegimeParams]
        Dictionary of regime parameters.
    transition_matrix : np.ndarray
        Markov Chain transition matrix.
    total_time : float
        Total time for the simulation.
    number_of_steps : int
        Number of steps in the simulation.
    random_state : int, optional
        Seed for *main* random number generator (which generates seeds).
    n_jobs : int, default=1
        Number of parallel jobs to run.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - prices_df: DataFrame of (n_steps x number_of_paths) prices.
        - regimes_df: DataFrame of (n_steps x number_of_paths) regimes.
    """
    rng = np.random.default_rng(random_state)
    # Generate unique seeds for each parallel job
    random_states = rng.integers(0, 10 * number_of_paths, size=number_of_paths)
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(generate_prices_from_regimes)(
            regimes,
            transition_matrix,
            total_time,
            n_steps,
            seed,
        )
        for seed in random_states
    )

    prices, simulated_regimes = zip(*results)

    prices_df = pd.concat(prices, axis=1)
    prices_df.columns = range(number_of_paths)
    
    simulated_regimes_df = pd.DataFrame(simulated_regimes).T
    simulated_regimes_df.columns = range(number_of_paths)
    simulated_regimes_df.index = prices_df.index

    return prices_df, simulated_regimes_df