"""
Determines strategy side (long/short) based on moving average crossovers.
"""

import pandas as pd

def determine_strategy_side(
    prices: pd.Series,
    fast_window: int = 20,
    slow_window: int = 50,
    exponential: bool = False,
    mean_reversion: bool = False,
) -> pd.Series:
    r"""
    Determines the trading side based on moving average crossovers.

    .. math::
        \text{Momentum:}
        \begin{cases}
        1 & \text{if } MA_{fast} \geq MA_{slow} \\
        -1 & \text{otherwise}
        \end{cases}

        \text{Mean Reversion:}
        \begin{cases}
        1 & \text{if } MA_{fast} < MA_{slow} \\
        -1 & \text{otherwise}
        \end{cases}

    Parameters
    ----------
    prices : pd.Series
        Time series of prices.
    fast_window : int, default=20
        Window size for the fast moving average.
    slow_window : int, default=50
        Window size for the slow moving average.
    exponential : bool, default=False
        If True, use Exponential Moving Averages (EWMA).
        If False, use Simple Moving Averages (SMA).
    mean_reversion : bool, default=False
        If True, strategy is mean-reverting (short fast > slow).
        If False, strategy is momentum-based (long fast > slow).

    Returns
    -------
    pd.Series
        A Series of sides (1 for long, -1 for short).
    """
    if fast_window >= slow_window:
        raise ValueError("fast_window must be smaller than slow_window.")

    if exponential:
        fast_ma = prices.ewm(span=fast_window, adjust=False, min_periods=1).mean()
        slow_ma = prices.ewm(span=slow_window, adjust=False, min_periods=1).mean()
    else:
        fast_ma = prices.rolling(window=fast_window, min_periods=1).mean()
        slow_ma = prices.rolling(window=slow_window, min_periods=1).mean()

    # Create signal: 1 if fast > slow, -1 if fast < slow
    signal = (fast_ma >= slow_ma).astype(int) * 2 - 1
    
    if mean_reversion:
        # Invert the signal for mean reversion
        return -signal
    
    return signal