import numpy as np
import pandas as pd

def bet_timing(target_positions: pd.Series) -> pd.Index:
    """
    Determine the timing of bets when positions flatten or flip.

    :param target_positions: Series of target positions.
    :return: Index of bet timing.
    """
    zero_positions = target_positions[target_positions == 0].index

    lagged_non_zero_positions = target_positions.shift(1)
    lagged_non_zero_positions = lagged_non_zero_positions[lagged_non_zero_positions != 0].index

    bets = zero_positions.intersection(lagged_non_zero_positions)
    zero_positions = target_positions.iloc[1:] * target_positions.iloc[:-1].values
    bets = bets.union(zero_positions[zero_positions < 0].index).sort_values()

    if target_positions.index[-1] not in bets:
        bets = bets.append(target_positions.index[-1:])

    return bets

def calculate_holding_period(target_positions: pd.Series) -> tuple:
    """
    Derive average holding period (in days) using the average entry time pairing algorithm.

    :param target_positions: Series of target positions.
    :return: Tuple containing holding period DataFrame and mean holding period.
    """
    hold_period, time_entry = pd.DataFrame(columns=['dT', 'w']), 0.0
    position_difference = target_positions.diff()
    time_difference = (target_positions.index - target_positions.index[0]) / np.timedelta64(1, 'D')

    for i in range(1, target_positions.shape[0]):
        if position_difference.iloc[i] * target_positions.iloc[i - 1] >= 0:
            if target_positions.iloc[i] != 0:
                time_entry = (time_entry * target_positions.iloc[i - 1] + time_difference[i] * position_difference.iloc[i]) / target_positions.iloc[i]
        else:
            if target_positions.iloc[i] * target_positions.iloc[i - 1] < 0:
                hold_period.loc[target_positions.index[i], ['dT', 'w']] = (time_difference[i] - time_entry, abs(target_positions.iloc[i - 1]))
                time_entry = time_difference[i]
            else:
                hold_period.loc[target_positions.index[i], ['dT', 'w']] = (time_difference[i] - time_entry, abs(position_difference.iloc[i]))

    if hold_period['w'].sum() > 0:
        mean_holding_period = (hold_period['dT'] * hold_period['w']).sum() / hold_period['w'].sum()
    else:
        mean_holding_period = np.nan

    return hold_period, mean_holding_period

def calculate_hhi_concentration(returns: pd.Series) -> tuple:
    """
    Calculate the HHI concentration measures.

    :param returns: Series of returns.
    :return: Tuple containing positive returns HHI, negative returns HHI, and time-concentrated HHI.
    """
    returns_hhi_positive = calculate_hhi(returns[returns >= 0])
    returns_hhi_negative = calculate_hhi(returns[returns < 0])
    time_concentrated_hhi = calculate_hhi(returns.groupby(pd.Grouper(freq='M')).count())

    return returns_hhi_positive, returns_hhi_negative, time_concentrated_hhi

def calculate_hhi(bet_returns: pd.Series) -> float:
    """
    Calculate the Herfindahl-Hirschman Index (HHI) concentration measure.

    :param bet_returns: Series of bet returns.
    :return: Calculated HHI value.
    """
    if bet_returns.shape[0] <= 2:
        return np.nan

    weight = bet_returns / bet_returns.sum()
    hhi_ = (weight ** 2).sum()
    hhi_ = (hhi_ - bet_returns.shape[0] ** -1) / (1.0 - bet_returns.shape[0] ** -1)

    return hhi_

def compute_drawdowns_time_under_water(series: pd.Series, dollars: bool = False) -> tuple:
    """
    Compute series of drawdowns and the time under water associated with them.

    :param series: Series of returns or dollar performance.
    :param dollars: Whether the input series represents returns or dollar performance.
    :return: Tuple containing drawdown series, time under water series, and drawdown analysis DataFrame.
    """
    series_df = series.to_frame('PnL').reset_index(names='Datetime')
    series_df['HWM'] = series.expanding().max().values

    def process_groups(group):
        if len(group) <= 1:
            return None

        result = pd.Series()
        result.loc['Start'] = group['Datetime'].iloc[0]
        result.loc['Stop'] = group['Datetime'].iloc[-1]
        result.loc['HWM'] = group['HWM'].iloc[0]
        result.loc['Min'] = group['PnL'].min()
        result.loc['Min. Time'] = group['Datetime'][group['PnL'] == group['PnL'].min()].iloc[0]

        return result

    groups = series_df.groupby('HWM')
    drawdown_analysis = pd.DataFrame()

    for _, group in groups:
        drawdown_analysis = drawdown_analysis.append(process_groups(group), ignore_index=True)

    if dollars:
        drawdown = drawdown_analysis['HWM'] - drawdown_analysis['Min']
    else:
        drawdown = 1 - drawdown_analysis['Min'] / drawdown_analysis['HWM']

    drawdown.index = drawdown_analysis['Start']
    drawdown.index.name = 'Datetime'

    time_under_water = ((drawdown_analysis['Stop'] - drawdown_analysis['Start']) / np.timedelta64(1, 'Y')).values
    time_under_water = pd.Series(time_under_water, index=drawdown_analysis['Start'])

    return drawdown, time_under_water, drawdown_analysis
