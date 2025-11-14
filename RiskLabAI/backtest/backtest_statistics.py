"""
Calculates various backtest statistics like holding period,
concentration, and drawdowns.

## TODO:
- [ ] Optimize `calculate_holding_period` with Numba (@jit)
      to improve performance, similar to the `sharpe_ratio` function.
- [ ] Add an `annualized_sharpe_ratio` helper function that
      wraps the Numba `sharpe_ratio` and scales it by sqrt(freq).
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd
from numba import jit

@jit(nopython=True)
def sharpe_ratio(
    returns: np.ndarray, risk_free_rate: float = 0.0
) -> float:
    """
    Calculate the Sharpe Ratio (Numba-optimized).

    Parameters
    ----------
    returns : np.ndarray
        An array of returns.
    risk_free_rate : float, default=0.0
        The risk-free rate.

    Returns
    -------
    float
        The calculated Sharpe Ratio.
    """
    excess_returns = returns - risk_free_rate
    std_dev = np.std(excess_returns)
    
    if std_dev == 0.0:
        return 0.0
        
    return np.mean(excess_returns) / std_dev

def bet_timing(target_positions: pd.Series) -> pd.Index:
    """
    Determine the timestamps of bets, defined as when positions
    are closed (return to zero) or flipped (sign change).

    Parameters
    ----------
    target_positions : pd.Series
        Time series of target positions.

    Returns
    -------
    pd.Index
        A DatetimeIndex of timestamps when bets are timed.

    Example
    -------
    >>> dates = pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03', \
                                '2020-01-04', '2020-01-05', '2020-01-06'])
    >>> positions = pd.Series([0, 1, 1, 0, -1, 0], index=dates)
    >>> bet_timing(positions)
    DatetimeIndex(['2020-01-04', '2020-01-05', '2020-01-06'], \
dtype='datetime64[ns]', freq=None)
    """
    # 1. Bets at points where position returns to 0
    zero_positions = target_positions[target_positions == 0].index

    # 2. Bets at points where position flips sign
    # Find timestamps where t=0 and t-1 != 0
    lagged_non_zero = target_positions.shift(1).fillna(0)
    lagged_non_zero = lagged_non_zero[lagged_non_zero != 0].index
    bets = zero_positions.intersection(lagged_non_zero)

    # Find timestamps where sign flips, e.g., 1 to -1
    sign_flips = target_positions.iloc[1:] * target_positions.iloc[:-1].values
    bets = bets.union(sign_flips[sign_flips < 0].index).sort_values()

    # 3. Add last timestamp if not already included
    if target_positions.index[-1] not in bets:
        bets = bets.append(pd.Index([target_positions.index[-1]]))

    return bets

def calculate_holding_period(
    target_positions: pd.Series,
) -> Tuple[pd.DataFrame, float]:
    """
    Derive the average holding period in days.

    Uses the average entry time pairing algorithm.

    Parameters
    ----------
    target_positions : pd.Series
        Time series of target positions.

    Returns
    -------
    Tuple[pd.DataFrame, float]
        - hold_period_df: DataFrame with holding time ('dT') and
                          weight ('w') for each bet.
        - mean_holding_period: The weighted-average holding period in days.
    """
    hold_period_data = []
    time_entry = 0.0  # Average entry time
    position_diff = target_positions.diff()
    # Time difference in fractional days
    time_diff = (
        target_positions.index - target_positions.index[0]
    ) / np.timedelta64(1, "D")

    for i in range(1, target_positions.shape[0]):
        current_pos = target_positions.iloc[i]
        prev_pos = target_positions.iloc[i - 1]
        diff = position_diff.iloc[i]

        if diff * prev_pos >= 0:  # Position increase or flat
            if current_pos != 0:
                # Update average entry time
                time_entry = (
                    time_entry * prev_pos + time_diff[i] * diff
                ) / current_pos
        else:  # Position decrease or flip
            if current_pos * prev_pos < 0:  # Position flip
                # Close old position
                hold_period_data.append(
                    {
                        "index": target_positions.index[i],
                        "dT": time_diff[i] - time_entry,
                        "w": abs(prev_pos),
                    }
                )
                time_entry = time_diff[i]  # Reset entry time
            else:  # Position decrease (but not flip)
                hold_period_data.append(
                    {
                        "index": target_positions.index[i],
                        "dT": time_diff[i] - time_entry,
                        "w": abs(diff),
                    }
                )
    
    if not hold_period_data:
        return pd.DataFrame(columns=['dT', 'w']), np.nan
        
    hold_period_df = pd.DataFrame(hold_period_data).set_index('index')
    
    if hold_period_df["w"].sum() > 0:
        mean_holding_period = (
            (hold_period_df["dT"] * hold_period_df["w"]).sum()
            / hold_period_df["w"].sum()
        )
    else:
        mean_holding_period = np.nan

    return hold_period_df, mean_holding_period

def calculate_hhi(bet_returns: pd.Series) -> float:
    """
    Calculate the Herfindahl-Hirschman Index (HHI) for concentration.

    HHI is normalized to be between 0 and 1.
    - 0 indicates a perfectly diversified portfolio.
    - 1 indicates a fully concentrated portfolio.

    Parameters
    ----------
    bet_returns : pd.Series
        Series of returns for each bet.

    Returns
    -------
    float
        The normalized HHI value, or np.nan if undefined.
    """
    if bet_returns.shape[0] <= 2:
        return np.nan

    total_return = bet_returns.sum()
    if total_return == 0:
        return np.nan

    weights = bet_returns / total_return
    hhi = (weights**2).sum()
    
    # Normalize HHI
    n = bet_returns.shape[0]
    hhi_normalized = (hhi - 1.0 / n) / (1.0 - 1.0 / n)

    return hhi_normalized

def calculate_hhi_concentration(returns: pd.Series) -> Tuple[float, float, float]:
    """
    Calculate HHI concentration for positive, negative, and monthly returns.

    Parameters
    ----------
    returns : pd.Series
        Time series of returns.

    Returns
    -------
    Tuple[float, float, float]
        - hhi_positive: HHI for positive returns.
        - hhi_negative: HHI for negative returns.
        - hhi_time: HHI for time-based (monthly) concentration.
    """
    hhi_positive = calculate_hhi(returns[returns >= 0])
    hhi_negative = calculate_hhi(returns[returns < 0])
    
    # Calculate time concentration (by month)
    time_concentration = returns.groupby(pd.Grouper(freq="M")).count()
    hhi_time = calculate_hhi(time_concentration)

    return hhi_positive, hhi_negative, hhi_time

def compute_drawdowns_time_under_water(
    pnl_series: pd.Series, dollars: bool = False
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute series of drawdowns and the time under water.

    Parameters
    ----------
    pnl_series : pd.Series
        Time series of returns or dollar PnL.
    dollars : bool, default=False
        If True, treat `pnl_series` as dollar PnL.
        If False, treat as returns (e.g., 1.01, 0.99) and
        calculate drawdowns as percentages.

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        - drawdown: pd.Series indexed by start date, showing max drawdown.
        - time_under_water: pd.Series indexed by start date, showing
                            duration of drawdown in fractional years.
    """
    series_df = pnl_series.to_frame("PnL")
    series_df["HWM"] = series_df["PnL"].expanding().max()

    # Group by High Water Mark (HWM)
    groups = series_df.groupby("HWM")
    drawdown_analysis_data = []

    for hwm, group in groups:
        if len(group) <= 1 or hwm == group["PnL"].min():
            continue  # Skip groups with no drawdown

        drawdown_analysis_data.append(
            {
                "Start": group.index[0],
                "Stop": group.index[-1],
                "HWM": hwm,
                "Min": group["PnL"].min(),
            }
        )

    if not drawdown_analysis_data:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    drawdown_analysis_df = pd.DataFrame(drawdown_analysis_data)
    drawdown_analysis_df = drawdown_analysis_df.set_index("Start")
    
    if dollars:
        drawdown = drawdown_analysis_df["HWM"] - drawdown_analysis_df["Min"]
    else:
        # Percentage drawdown
        drawdown = 1.0 - (drawdown_analysis_df["Min"] / drawdown_analysis_df["HWM"])

    # Time under water in fractional years
    time_under_water = (
        drawdown_analysis_df["Stop"] - drawdown_analysis_df.index
    ) / np.timedelta64(1, "D") / 365.25


    drawdown.index.name = 'Datetime'
    time_under_water.index.name = 'Datetime'

    return drawdown, time_under_water