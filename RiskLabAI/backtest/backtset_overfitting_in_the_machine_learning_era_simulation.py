import numpy as np
import pandas as pd
from math import ceil
from scipy import stats as ss
import ta
import itertools
import warnings
from typing import Dict, Union, Tuple, List, Any

from RiskLabAI.data.differentiation import fractionally_differentiated_log_price
from RiskLabAI.data.labeling import daily_volatility_with_log_returns, cusum_filter_events_dynamic_threshold, vertical_barrier, meta_events, meta_labeling
from RiskLabAI.data.weights import sample_weight_absolute_return_meta_labeling
from RiskLabAI.utils import determine_strategy_side
from RiskLabAI.backtest.validation import CrossValidatorController
from RiskLabAI.backtest import probability_of_backtest_overfitting, probabilistic_sharpe_ratio, benchmark_sharpe_ratio, sharpe_ratio, strategy_bet_sizing

def backtest_overfitting_simulation(
    prices: pd.Series, 
    strategy_parameters: Dict[str, Union[List[int], List[float], List[bool]]],
    models: Dict[str, Dict[str, Any]],
    step_risk_free_rate: float,
    overfitting_partitions_length: int,
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """
    Conducts a simulation to evaluate the performance of trading strategies and models, assessing the risk of overfitting.

    This function simulates a trading environment to assess various cross-validation methods in the context of financial analytics. It uses a set of market regime parameters and machine learning models to backtest trading strategies and compute metrics indicative of overfitting.

    Args:
        prices (pd.Series): Time series of asset prices.
        strategy_parameters (dict): Parameters dictating trading strategy behavior, including window sizes and flags for mean reversion.
        models (dict): A collection of machine learning models and their associated parameters.
        step_risk_free_rate (float): The risk-free rate used in the simulation for Sharpe ratio calculations.
        overfitting_partitions_length (int): The number of partitions to divide the dataset into for temporal overfitting analysis.

    Returns:
        Tuple[Dict[str, List[float]], Dict[str, List[float]]]: A tuple containing two dictionaries, one for the Probability of Backtest Overfitting (PBO) and the other for the Deflated Sharpe Ratio (DSR), for each cross-validation method tested.
    """

    fd_log_price = fractionally_differentiated_log_price(prices)
    volatility = daily_volatility_with_log_returns(prices, 100)
    filter_threshold = 1.8
    moelcules = cusum_filter_events_dynamic_threshold(np.log(prices), filter_threshold * volatility)
    vertical_barriers = vertical_barrier(prices, moelcules, 20)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # Features
        features = pd.DataFrame()
        features['FracDiff'] = fd_log_price
        features['Volatility'] = volatility
        features['Z-Score'] = (prices - prices.rolling(20).mean()) / prices.rolling(20).std()
        macd_line = np.log(prices.ewm(span=12).mean() / prices.ewm(span=26).mean())
        signal_line = macd_line.ewm(span=9).mean()
        features["Log MACD Histogram"] = macd_line - signal_line
        features["ADX"] = ta.trend.ADXIndicator(prices, prices, prices, fillna=True).adx()
        features["RSI"] = ta.momentum.RSIIndicator(prices, fillna=True).rsi()
        features["CCI"] = ta.trend.CCIIndicator(prices, prices, prices, fillna=True).cci()
        stochastic = ta.momentum.StochasticOscillator(prices, prices, prices, fillna=True)
        features["Stochastic"] = stochastic.stoch()
        features["ROC"] = ta.momentum.ROCIndicator(prices, fillna=True).roc()
        features["ATR"] = ta.volatility.AverageTrueRange(prices, prices, prices, fillna=True).average_true_range()
        features["Log DPO"] = np.log(prices.rolling(11).mean() / prices.rolling(20).mean())
        
        # 1. MACD Crossovers:
        features["MACD Position"] = 0  # default to no crossover
        features.loc[features["Log MACD Histogram"] >= 0, "MACD Position"] = 1
        features.loc[features["Log MACD Histogram"] < 0, "MACD Position"] = -1

        # 2. ADX Trend Strength:
        features["ADX Strength"] = 0
        features.loc[features["ADX"] > 25, "ADX Strength"] = 1
        features.loc[features["ADX"] < 25, "ADX Strength"] = -1

        # 3. RSI Overbought/Oversold:
        features["RSI Signal"] = 0
        features.loc[features["RSI"] > 70, "RSI Signal"] = 1
        features.loc[features["RSI"] < 30, "RSI Signal"] = -1

        # 4. CCI Overbought/Oversold:
        features["CCI Signal"] = 0
        features.loc[features["CCI"] > 100, "CCI Signal"] = 1
        features.loc[features["CCI"] < -100, "CCI Signal"] = -1

        # 5. Stochastic Oscillator Overbought/Oversold:
        stochastic_signal = stochastic.stoch_signal()
        features["Stochastic Signal"] = 0
        features.loc[stochastic_signal > 80, "Stochastic Signal"] = 1
        features.loc[stochastic_signal < 20, "Stochastic Signal"] = -1

        # 6. ROC Momentum Shifts:
        features["ROC Momentum"] = 0
        features.loc[features["ROC"] > 0, "ROC Momentum"] = 1  # Positive momentum
        features.loc[features["ROC"] < 0, "ROC Momentum"] = -1  # Negative momentum
        ichimoku = ta.trend.IchimokuIndicator(prices, prices, visual=False, fillna=True)

        # 1. Kumo Breakouts
        features["Kumo Breakout"] = 0
        senkou_span_a = ichimoku.ichimoku_a()
        senkou_span_b = ichimoku.ichimoku_b()
        features.loc[(prices > senkou_span_a) & (prices > senkou_span_b), "Kumo Breakout"] = 1
        features.loc[(prices < senkou_span_a) & (prices < senkou_span_b), "Kumo Breakout"] = -1

        # 2. TK Crosses
        features["TK Position"] = 0
        tenkan_sen = ichimoku.ichimoku_conversion_line()
        kijun_sen = ichimoku.ichimoku_base_line()
        features.loc[tenkan_sen >= kijun_sen, "TK Position"] = 1
        features.loc[tenkan_sen < kijun_sen, "TK Position"] = -1

        # 4. Price Relative to Kumo
        features["Price Kumo Position"] = 0
        features.loc[(prices > senkou_span_a) & (prices > senkou_span_b), "Price Kumo Position"] = 1
        features.loc[(prices < senkou_span_a) & (prices < senkou_span_b), "Price Kumo Position"] = -1

        # 5. Cloud Thickness
        features["Cloud Thickness"] = np.log(senkou_span_a / senkou_span_b)

        # 6. Momentum Confirmation
        features["Momentum Confirmation"] = 0
        features.loc[(tenkan_sen > senkou_span_a) & (prices > senkou_span_a), "Momentum Confirmation"] = 1
        features.loc[(tenkan_sen < senkou_span_a) & (prices < senkou_span_a), "Momentum Confirmation"] = -1

    results = {
        'Walk-Forward' : [],
        'K-Fold' : [],
        'Purged K-Fold' : [],
        'Combinatorial Purged' : [],
    }

    # Iterate over each strategy parameter combination
    strategy_parameters_keys, strategy_parameters_values = zip(*strategy_parameters.items())
    for strategy_parameters_value in itertools.product(*strategy_parameters_values):
        strategy_params = dict(zip(strategy_parameters_keys, strategy_parameters_value))
        if (strategy_params['fast_window'] == strategy_parameters['fast_window'][0] and strategy_params['slow_window'] == strategy_parameters['slow_window'][0]) or \
        (strategy_params['fast_window'] == strategy_parameters['fast_window'][1] and strategy_params['slow_window'] == strategy_parameters['slow_window'][1]) or \
        (strategy_params['fast_window'] == strategy_parameters['fast_window'][2] and strategy_params['slow_window'] == strategy_parameters['slow_window'][2]) or \
        (strategy_params['fast_window'] == strategy_parameters['fast_window'][3] and strategy_params['slow_window'] == strategy_parameters['slow_window'][3]):
            
            strategy_sides = determine_strategy_side(prices, **strategy_params)
        else:
            continue    

        triple_barrier_events = meta_events(prices, moelcules, [0.5, 1.5], volatility, 0, 1, vertical_barriers, strategy_sides)
        # triple_barrier_events = get_meta_events(prices, moelcules, [1.5, 0.5], volatility, 0, 1, vertical_barriers, strategy_sides)
        labels = meta_labeling(triple_barrier_events, prices)
        sample_weights = sample_weight_absolute_return_meta_labeling(triple_barrier_events['End Time'], prices, moelcules)

        index = features.loc[moelcules].dropna().index.intersection(labels.dropna().index).intersection(sample_weights.dropna().index)
        data = features.loc[index]
        target = labels.loc[index]['Label']
        weights = sample_weights.loc[index]
        times = labels.loc[index]['End Time']

        cross_validators = {
            'Walk-Forward' : CrossValidatorController(
                'walkforward',
                n_splits=4,
            ).cross_validator,
            'K-Fold' : CrossValidatorController(
                'kfold',
                n_splits=4,
            ).cross_validator,
            'Purged K-Fold' : CrossValidatorController(
                'purgedkfold',
                n_splits=4,
                times=times,
                embargo=0.02
            ).cross_validator,
            'Combinatorial Purged' : CrossValidatorController(
                'combinatorialpurged',
                n_splits=8,
                n_test_groups=2,
                times=times,
                embargo=0.02
            ).cross_validator,
        }

        # Iterate over each model and hyperparameter configuration
        for model_name, model_details in models.items():
            model = model_details['Model']
            param_grid = model_details['Parameters']

            # Generate all combinations of hyperparameters
            model_keys, model_values = zip(*param_grid.items())
            for model_value in itertools.product(*model_values):
                params = dict(zip(model_keys, model_value))
                model.set_params(**params)

                for cross_validator_type, cross_validator in cross_validators.items():
                    predictions = cross_validator.backtest_predictions(model, data, target, weights, predict_probability=True, n_jobs=1)
                    probabilities = pd.Series(np.vstack(list(map(lambda x: x[:, 1], predictions.values()))).mean(axis=0), times.index).dropna()
                    positions = strategy_bet_sizing(prices.index, times.loc[probabilities.index], strategy_sides[probabilities.index], probabilities)
                    strategy_log_returns = (np.log(prices).diff() * positions.shift()).dropna()
                    results[cross_validator_type].append({
                        'Trial Info.' : {
                            'Strategy Parameters' : strategy_params,
                            'Model Name': model_name,
                            'Model Parameters': params,
                        },
                        'Returns': strategy_log_returns
                    })

    cv_deflated_sr = {cv: [] for cv in results.keys()}
    cv_pbo = {cv: [] for cv in results.keys()}

    for cv, trials in results.items():
        performances = pd.concat([trial['Returns'] for trial in trials], axis=1)
        
        # Calculate the number of chunks using ceil
        n_chunks = ceil(performances.shape[0] / overfitting_partitions_length)
        for chunk in np.array_split(performances, n_chunks):
            sharpe_ratios = chunk.apply(lambda y: sharpe_ratio(y.values))
            benchmark_sr = benchmark_sharpe_ratio(sharpe_ratios)
            best_strategy_index = sharpe_ratios.idxmax()
            
            deflated_sr = probabilistic_sharpe_ratio(
                sharpe_ratios.loc[best_strategy_index], 
                benchmark_sr, len(chunk), 
                ss.skew(chunk[best_strategy_index]), 
                ss.kurtosis(chunk[best_strategy_index]),
                return_test_statistic=True
            )
            cv_deflated_sr[cv].append(deflated_sr)
            
            pbo, logit_values = probability_of_backtest_overfitting(chunk.values, risk_free_return=step_risk_free_rate)
            cv_pbo[cv].append(pbo)

    return cv_pbo, cv_deflated_sr
