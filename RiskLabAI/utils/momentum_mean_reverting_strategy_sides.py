import pandas as pd


def determine_strategy_side(
    prices: pd.Series,
    fast_window: int = 20,
    slow_window: int = 50,
    exponential : bool = False,
    mean_reversion: bool = False
) -> pd.Series:
    """
    Determines the trading side (long or short) based on moving average crossovers and 
    the nature of the strategy (momentum or mean reversion).

    This function computes the fast and slow moving averages of the provided price series. 
    The trading side is decided based on the relationship between these averages and 
    the chosen strategy type (momentum or mean reversion).

    .. math::
        \text{Momentum:}
        \begin{cases}
        1 & \text{if } \text{fast\_moving\_average} \geq \text{slow\_moving\_average} \\
        -1 & \text{otherwise}
        \end{cases}

        \text{Mean Reversion:}
        \begin{cases}
        1 & \text{if } \text{fast\_moving\_average} < \text{slow\_moving\_average} \\
        -1 & \text{otherwise}
        \end{cases}

    :param prices: Series containing the prices.
    :param fast_window: Window size for the fast moving average.
    :param slow_window: Window size for the slow moving average.
    :param exponential: If True, compute exponential moving averages. Otherwise, compute simple moving averages.
    :param mean_reversion: If True, strategy is mean reverting. If False, strategy is momentum-based.
    :return: Series containing strategy sides.
    """
    # Check for invalid window sizes
    if fast_window >= slow_window:
        raise ValueError("The fast window should be smaller than the slow window.")

    if exponential:
        fast_moving_average = prices.ewm(span=fast_window, adjust=False, min_periods=1).mean()
        slow_moving_average = prices.ewm(span=slow_window, adjust=False, min_periods=1).mean()
    else:
        fast_moving_average = prices.rolling(window=fast_window, min_periods=1).mean()
        slow_moving_average = prices.rolling(window=slow_window, min_periods=1).mean()

    if mean_reversion:
        return (fast_moving_average < slow_moving_average).astype(int) * 2 - 1
    else:
        return (fast_moving_average >= slow_moving_average).astype(int) * 2 - 1
