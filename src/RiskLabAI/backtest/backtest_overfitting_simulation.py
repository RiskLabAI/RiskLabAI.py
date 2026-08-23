"""
Orchestrates complex backtest overfitting and simulation scenarios,
including PBO, DSR, and hardware performance profiling.

## TODO:
- [ ] **Refactor Feature Generation:** Move hardcoded parameters in
      `financial_features_backtest_overfitting_simulation`
      (e.g., volatility=100, rolling=20, indicator windows)
      to a configuration dictionary or function arguments.
- [ ] **Refactor Profiling:** Move hardware/performance functions
      (`get_cpu_info`, `format_cpu_info`,
      `measure_computational_requirements`, etc.)
      to a separate `RiskLabAI.utils.profiling` module
      to reduce this module's responsibilities.
"""

import itertools
import subprocess
import time
import warnings
from math import ceil
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats as ss
from scipy.stats import kendalltau
from sklearn.linear_model import LogisticRegression

from RiskLabAI.core import CROSS_VALIDATORS
from RiskLabAI.data.differentiation import fractionally_differentiated_log_price
from RiskLabAI.data.labeling import (
    cusum_filter_events_dynamic_threshold,
    daily_volatility_with_log_returns,
    meta_events,
    meta_labeling,
    vertical_barrier,
)
from RiskLabAI.data.weights import sample_weight_absolute_return_meta_labeling
from RiskLabAI.utils import determine_strategy_side

from .bet_sizing import strategy_bet_sizing
from .probabilistic_sharpe_ratio import (
    benchmark_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from .probability_of_backtest_overfitting import probability_of_backtest_overfitting


def _progress(iterable):
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable)


def _wilder_change_average(values: np.ndarray, window: int) -> np.ndarray:
    """Return Wilder's recursive average for quantities defined on price changes."""
    if not isinstance(window, int) or isinstance(window, bool) or window < 1:
        raise ValueError("window must be a positive integer")

    array = np.asarray(values, dtype=float)
    result = np.full(array.shape, np.nan, dtype=float)
    if array.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if array.size <= window:
        return result

    seed = array[1 : window + 1]
    if not np.isfinite(seed).all():
        return result
    result[window] = float(seed.mean())
    for position in range(window + 1, array.size):
        value = array[position]
        if not np.isfinite(value):
            result[position] = np.nan
        elif np.isfinite(result[position - 1]):
            result[position] = (
                result[position - 1] * (window - 1) + value
            ) / window
    return result


def _relative_strength_index(prices: pd.Series, window: int = 14) -> pd.Series:
    values = prices.to_numpy(dtype=float, copy=False)
    changes = np.empty(values.shape, dtype=float)
    changes[0] = 0.0
    changes[1:] = np.diff(values)
    average_gain = _wilder_change_average(np.maximum(changes, 0.0), window)
    average_loss = _wilder_change_average(np.maximum(-changes, 0.0), window)

    denominator = average_gain + average_loss
    result = np.full(values.shape, np.nan, dtype=float)
    np.divide(100.0 * average_gain, denominator, out=result, where=denominator > 0.0)
    result[(denominator == 0.0) & np.isfinite(denominator)] = 50.0
    return pd.Series(result, index=prices.index, dtype=float)


def _average_true_range(prices: pd.Series, window: int = 14) -> pd.Series:
    values = prices.to_numpy(dtype=float, copy=False)
    true_range = np.empty(values.shape, dtype=float)
    true_range[0] = 0.0
    true_range[1:] = np.abs(np.diff(values))
    return pd.Series(
        _wilder_change_average(true_range, window),
        index=prices.index,
        dtype=float,
    )


def _average_directional_index(prices: pd.Series, window: int = 14) -> pd.Series:
    values = prices.to_numpy(dtype=float, copy=False)
    changes = np.empty(values.shape, dtype=float)
    changes[0] = 0.0
    changes[1:] = np.diff(values)
    true_range = np.abs(changes)
    positive_movement = np.maximum(changes, 0.0)
    negative_movement = np.maximum(-changes, 0.0)

    average_range = _wilder_change_average(true_range, window)
    average_positive = _wilder_change_average(positive_movement, window)
    average_negative = _wilder_change_average(negative_movement, window)
    positive_index = np.zeros(values.shape, dtype=float)
    negative_index = np.zeros(values.shape, dtype=float)
    np.divide(
        100.0 * average_positive,
        average_range,
        out=positive_index,
        where=average_range > 0.0,
    )
    np.divide(
        100.0 * average_negative,
        average_range,
        out=negative_index,
        where=average_range > 0.0,
    )
    direction_sum = positive_index + negative_index
    directional_index = np.zeros(values.shape, dtype=float)
    np.divide(
        100.0 * np.abs(positive_index - negative_index),
        direction_sum,
        out=directional_index,
        where=direction_sum > 0.0,
    )

    result = np.full(values.shape, np.nan, dtype=float)
    first = 2 * window - 1
    if values.size > first:
        result[first] = float(directional_index[window : first + 1].mean())
        for position in range(first + 1, values.size):
            result[position] = (
                result[position - 1] * (window - 1)
                + directional_index[position]
            ) / window
    return pd.Series(result, index=prices.index, dtype=float)


def _commodity_channel_index(prices: pd.Series, window: int = 20) -> pd.Series:
    rolling_mean = prices.rolling(window, min_periods=window).mean()
    mean_deviation = prices.rolling(window, min_periods=window).apply(
        lambda values: float(np.mean(np.abs(values - values.mean()))), raw=True
    )
    denominator = 0.015 * mean_deviation
    result = (prices - rolling_mean) / denominator
    return result.where(denominator != 0.0, 0.0).astype(float)


def _stochastic_oscillator(
    prices: pd.Series, window: int = 14, smooth_window: int = 3
) -> tuple[pd.Series, pd.Series]:
    rolling_low = prices.rolling(window, min_periods=window).min()
    rolling_high = prices.rolling(window, min_periods=window).max()
    price_range = rolling_high - rolling_low
    oscillator = 100.0 * (prices - rolling_low) / price_range
    oscillator = oscillator.where(price_range != 0.0, 50.0).astype(float)
    signal = oscillator.rolling(
        smooth_window, min_periods=smooth_window
    ).mean()
    return oscillator, signal


def _ichimoku_lines(
    prices: pd.Series,
    conversion_window: int = 9,
    base_window: int = 26,
    span_window: int = 52,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    def midpoint(window: int) -> pd.Series:
        rolling = prices.rolling(window, min_periods=window)
        return (rolling.max() + rolling.min()) / 2.0

    conversion_line = midpoint(conversion_window)
    base_line = midpoint(base_window)
    span_a = (conversion_line + base_line) / 2.0
    span_b = midpoint(span_window)
    return conversion_line, base_line, span_a, span_b


def _create_cross_validator(validator_type, **kwargs):
    """
    Build a cross-validator from the core registry (the 2.0.0 single source).

    Mirrors the old ``_create_cross_validator(...)`` behaviour,
    including dropping keyword arguments a given validator does not accept.
    """
    return CROSS_VALIDATORS.create(validator_type, filter_unknown_kwargs=True, **kwargs)


def financial_features_backtest_overfitting_simulation(
    prices: pd.Series, noise_scale: float = 0.0, random_state: Optional[int] = None
) -> pd.DataFrame:
    """
    Create a DataFrame of financial features from the given price series.

    Args:
        prices (pd.Series): Time series of asset prices.
        noise_scale (float): Scale of Gaussian noise to be added to the features. Default is 0.0.
        random_state (Optional[int]): Seed for random number generator. Default is None.

    Returns:
        pd.DataFrame: DataFrame containing the computed financial features.

    Notes:
        RSI, ATR, and ADX use Wilder smoothing. CCI uses the standard 0.015
        scaling constant; stochastic momentum uses a 14-period range and a
        3-period signal; ROC uses 12 periods; and Ichimoku uses 9, 26, and 52
        periods without a visual displacement. These formulas are implemented
        locally so this preserved API does not depend on an unmaintained
        technical-analysis package.
    """
    rng = np.random.default_rng(random_state)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # Features
        features = pd.DataFrame()
        features["FracDiff"] = fractionally_differentiated_log_price(prices)
        features["Volatility"] = daily_volatility_with_log_returns(prices, 100)
        features["Z-Score"] = (prices - prices.rolling(20).mean()) / prices.rolling(
            20
        ).std()
        macd_line = np.log(prices.ewm(span=12).mean() / prices.ewm(span=26).mean())
        signal_line = macd_line.ewm(span=9).mean()
        features["Log MACD Histogram"] = macd_line - signal_line
        features["ADX"] = _average_directional_index(prices)
        features["RSI"] = _relative_strength_index(prices)
        features["CCI"] = _commodity_channel_index(prices)
        stochastic, stochastic_signal = _stochastic_oscillator(prices)
        features["Stochastic"] = stochastic
        features["ROC"] = prices.pct_change(12) * 100.0
        features["ATR"] = _average_true_range(prices)
        features["Log DPO"] = np.log(
            prices.rolling(11).mean() / prices.rolling(20).mean()
        )

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
        features["Stochastic Signal"] = 0
        features.loc[stochastic_signal > 80, "Stochastic Signal"] = 1
        features.loc[stochastic_signal < 20, "Stochastic Signal"] = -1

        # 6. ROC Momentum Shifts:
        features["ROC Momentum"] = 0
        features.loc[features["ROC"] > 0, "ROC Momentum"] = 1  # Positive momentum
        features.loc[features["ROC"] < 0, "ROC Momentum"] = -1  # Negative momentum
        tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b = _ichimoku_lines(
            prices
        )

        # 1. Kumo Breakouts
        features["Kumo Breakout"] = 0
        features.loc[
            (prices > senkou_span_a) & (prices > senkou_span_b), "Kumo Breakout"
        ] = 1
        features.loc[
            (prices < senkou_span_a) & (prices < senkou_span_b), "Kumo Breakout"
        ] = -1

        # 2. TK Crosses
        features["TK Position"] = 0
        features.loc[tenkan_sen >= kijun_sen, "TK Position"] = 1
        features.loc[tenkan_sen < kijun_sen, "TK Position"] = -1

        # 4. Price Relative to Kumo
        features["Price Kumo Position"] = 0
        features.loc[
            (prices > senkou_span_a) & (prices > senkou_span_b), "Price Kumo Position"
        ] = 1
        features.loc[
            (prices < senkou_span_a) & (prices < senkou_span_b), "Price Kumo Position"
        ] = -1

        # 5. Cloud Thickness
        features["Cloud Thickness"] = np.log(senkou_span_a / senkou_span_b)

        # 6. Momentum Confirmation
        features["Momentum Confirmation"] = 0
        features.loc[
            (tenkan_sen > senkou_span_a) & (prices > senkou_span_a),
            "Momentum Confirmation",
        ] = 1
        features.loc[
            (tenkan_sen < senkou_span_a) & (prices < senkou_span_a),
            "Momentum Confirmation",
        ] = -1

    if noise_scale > 0.0:
        for col in features.columns:
            noise = rng.normal(
                loc=0, scale=noise_scale * features[col].std(), size=features[col].shape
            )
            features[col] += noise

    return features


def backtest_overfitting_simulation_results(
    prices: pd.Series,
    strategy_parameters: dict[str, Union[list[int], list[float], list[bool]]],
    models: dict[str, dict[str, Any]],
    cross_validators: dict[str, Any],
    noise_scale: float = 0.0,
    random_state: int = None,
    n_jobs: int = 1,
) -> dict[str, list[dict[str, Any]]]:
    """
    Conducts a simulation to evaluate the performance of trading strategies and models.

    This function simulates a trading environment to assess various cross-validation methods in the context of financial analytics. It uses a set of market regime parameters and machine learning models to backtest trading strategies and compute metrics indicative of overfitting.

    Args:
        prices (pd.Series): Time series of asset prices.
        strategy_parameters (dict): Parameters dictating trading strategy behavior, including window sizes and flags for mean reversion.
        models (dict): A collection of machine learning models and their associated parameters.
        cross_validators (dict): A dictionary of cross-validation methods.
        noise_scale (float): Scale of Gaussian noise to be added to the features.
        random_state (int): Seed for random number generator.
        n_jobs (int): The number of jobs to run in parallel for cross_validator.backtest_predictions.

    Returns:
        Dict[str, List[Dict[str, Any]]]: A dictionary containing the results of the backtest for each cross-validation method tested.
    """

    volatility = daily_volatility_with_log_returns(prices, 100)
    filter_threshold = 1.8
    moelcules = cusum_filter_events_dynamic_threshold(
        np.log(prices), filter_threshold * volatility
    )
    vertical_barriers = vertical_barrier(prices, moelcules, 20)
    features = financial_features_backtest_overfitting_simulation(
        prices, noise_scale=noise_scale, random_state=random_state
    )

    results = {cv: [] for cv in cross_validators.keys()}

    # Iterate over each strategy parameter combination
    strategy_parameters_keys, strategy_parameters_values = zip(
        *strategy_parameters.items()
    )
    for strategy_parameters_value in itertools.product(*strategy_parameters_values):
        strategy_params = dict(zip(strategy_parameters_keys, strategy_parameters_value))
        if (
            (
                strategy_params["fast_window"] == strategy_parameters["fast_window"][0]
                and strategy_params["slow_window"]
                == strategy_parameters["slow_window"][0]
            )
            or (
                strategy_params["fast_window"] == strategy_parameters["fast_window"][1]
                and strategy_params["slow_window"]
                == strategy_parameters["slow_window"][1]
            )
            or (
                strategy_params["fast_window"] == strategy_parameters["fast_window"][2]
                and strategy_params["slow_window"]
                == strategy_parameters["slow_window"][2]
            )
            or (
                strategy_params["fast_window"] == strategy_parameters["fast_window"][3]
                and strategy_params["slow_window"]
                == strategy_parameters["slow_window"][3]
            )
        ):

            strategy_sides = determine_strategy_side(prices, **strategy_params)
        else:
            continue

        triple_barrier_events = meta_events(
            prices,
            moelcules,
            [0.5, 1.5],
            volatility,
            0,
            1,
            vertical_barriers,
            strategy_sides,
        )
        labels = meta_labeling(triple_barrier_events, prices)
        sample_weights = sample_weight_absolute_return_meta_labeling(
            triple_barrier_events["End Time"], prices, moelcules
        )

        index = (
            features.loc[moelcules]
            .dropna()
            .index.intersection(labels.dropna().index)
            .intersection(sample_weights.dropna().index)
        )
        data = features.loc[index]
        target = labels.loc[index]["Label"]
        weights = sample_weights.loc[index]
        times = labels.loc[index]["End Time"]
        external_feature = volatility.loc[index]

        # Iterate over each model and hyperparameter configuration
        for model_name, model_details in models.items():
            model = model_details["Model"]
            param_grid = model_details["Parameters"]

            # Generate all combinations of hyperparameters
            model_keys, model_values = zip(*param_grid.items())
            for model_value in itertools.product(*model_values):
                params = dict(zip(model_keys, model_value))
                model.set_params(**params)

                for cross_validator_type, cross_validator in cross_validators.items():
                    if "times" in cross_validator.__dict__:
                        cross_validator.times = times
                    if "external_feature" in cross_validator.__dict__:
                        cross_validator.external_feature = external_feature
                    predictions = cross_validator.backtest_predictions(
                        model,
                        data,
                        target,
                        weights,
                        predict_probability=True,
                        n_jobs=n_jobs,
                    )
                    probabilities = pd.Series(
                        np.vstack(
                            list(map(lambda x: x[:, 1], predictions.values()))
                        ).mean(axis=0),
                        times.index,
                    ).dropna()
                    positions = strategy_bet_sizing(
                        prices.index,
                        times.loc[probabilities.index],
                        strategy_sides[probabilities.index],
                        probabilities,
                    )
                    strategy_log_returns = (
                        np.log(prices).diff() * positions.shift()
                    ).dropna()
                    results[cross_validator_type].append(
                        {
                            "Trial Info.": {
                                "Strategy Parameters": strategy_params,
                                "Model Name": model_name,
                                "Model Parameters": params,
                            },
                            "Returns": strategy_log_returns,
                        }
                    )

    return results


def overall_backtest_overfitting_simulation(
    prices: pd.Series,
    strategy_parameters: dict[str, Union[list[int], list[float], list[bool]]],
    models: dict[str, dict[str, Any]],
    step_risk_free_rate: float,
    noise_scale: float = 0.0,
    random_state: int = None,
    n_jobs: int = 1,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Conducts an overall backtest overfitting simulation to calculate the metrics.

    Args:
        prices (pd.Series): Time series of asset prices.
        strategy_parameters (dict): Parameters dictating trading strategy behavior.
        models (dict): A collection of machine learning models and their associated parameters.
        step_risk_free_rate (float): The risk-free rate used in the simulation for Sharpe ratio calculations.
        noise_scale (float): Scale of Gaussian noise to be added to the features.
        random_state (int): Seed for random number generator.
        n_jobs (int): The number of jobs to run in parallel for cross_validator.backtest_predictions.

    Returns:
        Tuple[Dict[str, float], Dict[str, float]]: A tuple containing two dictionaries, one for the Probability of Backtest Overfitting (PBO) and the other for the Deflated Sharpe Ratio (DSR), for each cross-validation method tested.
    """

    cross_validators = {
        "Walk-Forward": _create_cross_validator(
            "walkforward",
            n_splits=4,
        ),
        "K-Fold": _create_cross_validator(
            "kfold",
            n_splits=4,
        ),
        "Purged K-Fold": _create_cross_validator(
            "purgedkfold", n_splits=4, times=None, embargo=0.02
        ),
        "Combinatorial Purged": _create_cross_validator(
            "combinatorialpurged", n_splits=8, n_test_groups=2, times=None, embargo=0.02
        ),
    }

    results = backtest_overfitting_simulation_results(
        prices,
        strategy_parameters,
        models,
        cross_validators,
        noise_scale,
        random_state,
        n_jobs=n_jobs,
    )

    cv_deflated_sr = {}
    cv_pbo = {}

    for cv, trials in results.items():
        performances = pd.concat([trial["Returns"] for trial in trials], axis=1)

        sharpe_ratios = performances.apply(lambda y: sharpe_ratio(y.values))
        benchmark_sr = benchmark_sharpe_ratio(sharpe_ratios)
        best_strategy_index = sharpe_ratios.idxmax()
        cv_deflated_sr[cv] = probabilistic_sharpe_ratio(
            sharpe_ratios.loc[best_strategy_index],
            benchmark_sr,
            len(performances),
            ss.skew(performances[best_strategy_index]),
            ss.kurtosis(performances[best_strategy_index]),
            return_test_statistic=True,
        )

        pbo, logit_values = probability_of_backtest_overfitting(
            performances.values, risk_free_return=step_risk_free_rate
        )
        cv_pbo[cv] = pbo

    return cv_pbo, cv_deflated_sr


def temporal_backtest_overfitting_simulation(
    prices: pd.Series,
    strategy_parameters: dict[str, Union[list[int], list[float], list[bool]]],
    models: dict[str, dict[str, Any]],
    step_risk_free_rate: float,
    overfitting_partitions_length: int,
    n_jobs: int = 1,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """
    Conducts a temporal backtest overfitting simulation to calculate the metrics in chunks.

    Args:
        prices (pd.Series): Time series of asset prices.
        strategy_parameters (dict): Parameters dictating trading strategy behavior.
        models (dict): A collection of machine learning models and their associated parameters.
        step_risk_free_rate (float): The risk-free rate used in the simulation for Sharpe ratio calculations.
        overfitting_partitions_length (int): The number of partitions to divide the dataset into for temporal overfitting analysis.
        n_jobs (int): The number of jobs to run in parallel for cross_validator.backtest_predictions.

    Returns:
        Tuple[Dict[str, List[float]], Dict[str, List[float]]]: A tuple containing two dictionaries, one for the Probability of Backtest Overfitting (PBO) and the other for the Deflated Sharpe Ratio (DSR), for each cross-validation method tested.
    """
    cross_validators = {
        "Walk-Forward": _create_cross_validator(
            "walkforward",
            n_splits=4,
        ),
        "K-Fold": _create_cross_validator(
            "kfold",
            n_splits=4,
        ),
        "Purged K-Fold": _create_cross_validator(
            "purgedkfold", n_splits=4, times=None, embargo=0.02
        ),
        "Combinatorial Purged": _create_cross_validator(
            "combinatorialpurged", n_splits=8, n_test_groups=2, times=None, embargo=0.02
        ),
    }

    results = backtest_overfitting_simulation_results(
        prices, strategy_parameters, models, cross_validators, n_jobs=n_jobs
    )

    cv_deflated_sr = {cv: [] for cv in results.keys()}
    cv_pbo = {cv: [] for cv in results.keys()}

    for cv, trials in results.items():
        performances = pd.concat([trial["Returns"] for trial in trials], axis=1)

        # Calculate the number of chunks using ceil
        n_chunks = ceil(performances.shape[0] / overfitting_partitions_length)
        for chunk in np.array_split(performances, n_chunks):
            sharpe_ratios = chunk.apply(lambda y: sharpe_ratio(y.values))
            benchmark_sr = benchmark_sharpe_ratio(sharpe_ratios)
            best_strategy_index = sharpe_ratios.idxmax()

            deflated_sr = probabilistic_sharpe_ratio(
                sharpe_ratios.loc[best_strategy_index],
                benchmark_sr,
                len(chunk),
                ss.skew(chunk[best_strategy_index]),
                ss.kurtosis(chunk[best_strategy_index]),
                return_test_statistic=True,
            )
            cv_deflated_sr[cv].append(deflated_sr)

            pbo, logit_values = probability_of_backtest_overfitting(
                chunk.values, risk_free_return=step_risk_free_rate, n_jobs=n_jobs
            )
            cv_pbo[cv].append(pbo)

    return cv_pbo, cv_deflated_sr


def time_temporal_backtest_overfitting_simulation(
    prices: pd.Series,
    strategy_parameters: dict[str, Union[list[int], list[float], list[bool]]],
    models: dict[str, dict[str, Any]],
    step_risk_free_rate: float,
    overfitting_partitions_duration: str = "A",  # Annual grouping by default
    n_jobs: int = 1,
) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    """
    Conducts a time-temporal backtest overfitting simulation to calculate the metrics in time-indexed chunks.

    Args:
        prices (pd.Series): Time series of asset prices.
        strategy_parameters (dict): Parameters dictating trading strategy behavior.
        models (dict): A collection of machine learning models and their associated parameters.
        step_risk_free_rate (float): The risk-free rate used in the simulation for Sharpe ratio calculations.
        overfitting_partitions_duration (str): The frequency for time-based grouping to divide the dataset into for temporal overfitting analysis.
        n_jobs (int): The number of jobs to run in parallel for cross_validator.backtest_predictions.

    Returns:
        Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]: A tuple containing two dictionaries, one for the Probability of Backtest Overfitting (PBO) and the other for the Deflated Sharpe Ratio (DSR), for each cross-validation method tested, indexed by time.
    """
    cross_validators = {
        "Walk-Forward": _create_cross_validator(
            "walkforward",
            n_splits=4,
        ),
        "K-Fold": _create_cross_validator(
            "kfold",
            n_splits=4,
        ),
        "Purged K-Fold": _create_cross_validator(
            "purgedkfold", n_splits=4, times=None, embargo=0.02
        ),
        "Combinatorial Purged": _create_cross_validator(
            "combinatorialpurged", n_splits=8, n_test_groups=2, times=None, embargo=0.02
        ),
    }

    results = backtest_overfitting_simulation_results(
        prices, strategy_parameters, models, cross_validators, n_jobs=n_jobs
    )

    cv_deflated_sr = {cv: pd.Series(dtype=float) for cv in results.keys()}
    cv_pbo = {cv: pd.Series(dtype=float) for cv in results.keys()}

    for cv, trials in results.items():
        performances = pd.concat([trial["Returns"] for trial in trials], axis=1)

        # Group by the specified duration and calculate metrics
        grouped = performances.groupby(pd.Grouper(freq=overfitting_partitions_duration))
        for timestamp, chunk in grouped:
            sharpe_ratios = chunk.apply(lambda y: sharpe_ratio(y.values))
            benchmark_sr = benchmark_sharpe_ratio(sharpe_ratios)
            best_strategy_index = sharpe_ratios.idxmax()

            deflated_sr = probabilistic_sharpe_ratio(
                sharpe_ratios.loc[best_strategy_index],
                benchmark_sr,
                len(chunk),
                ss.skew(chunk[best_strategy_index]),
                ss.kurtosis(chunk[best_strategy_index]),
                return_test_statistic=True,
            )
            cv_deflated_sr[cv].at[timestamp] = deflated_sr

            pbo, logit_values = probability_of_backtest_overfitting(
                chunk.values, risk_free_return=step_risk_free_rate, n_jobs=1
            )
            cv_pbo[cv].at[timestamp] = pbo

    return cv_pbo, cv_deflated_sr


def varying_embargo_backtest_overfitting_simulation(
    prices: pd.Series,
    strategy_parameters: dict[str, Union[list[int], list[float], list[bool]]],
    models: dict[str, dict[str, Any]],
    step_risk_free_rate: float,
    embargo_values: list[float],
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Conducts a backtest overfitting simulation with varying embargo values to calculate the metrics.

    Args:
        prices (pd.Series): Time series of asset prices.
        strategy_parameters (dict): Parameters dictating trading strategy behavior.
        models (dict): A collection of machine learning models and their associated parameters.
        step_risk_free_rate (float): The risk-free rate used in the simulation for Sharpe ratio calculations.
        embargo_values (List[float]): List of embargo values to test.
        n_jobs (int): The number of jobs to run in parallel for cross_validator.backtest_predictions.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Two DataFrames containing PBO and DSR values for each embargo value and cross-validation method.
    """

    volatility = daily_volatility_with_log_returns(prices, 100)
    filter_threshold = 1.8
    moelcules = cusum_filter_events_dynamic_threshold(
        np.log(prices), filter_threshold * volatility
    )
    vertical_barriers = vertical_barrier(prices, moelcules, 20)
    features = financial_features_backtest_overfitting_simulation(prices)

    cv_pbo_embargo = pd.DataFrame(
        index=embargo_values, columns=["Purged K-Fold", "Combinatorial Purged"]
    )
    cv_deflated_sr_embargo = pd.DataFrame(
        index=embargo_values, columns=["Purged K-Fold", "Combinatorial Purged"]
    )

    for embargo in embargo_values:
        results = {
            "Purged K-Fold": [],
            "Combinatorial Purged": [],
        }

        cross_validators = {
            "Purged K-Fold": _create_cross_validator(
                "purgedkfold", n_splits=4, times=None, embargo=embargo
            ),
            "Combinatorial Purged": _create_cross_validator(
                "combinatorialpurged",
                n_splits=8,
                n_test_groups=2,
                times=None,
                embargo=embargo,
            ),
        }

        # Iterate over each strategy parameter combination
        strategy_parameters_keys, strategy_parameters_values = zip(
            *strategy_parameters.items()
        )
        for strategy_parameters_value in itertools.product(*strategy_parameters_values):
            strategy_params = dict(
                zip(strategy_parameters_keys, strategy_parameters_value)
            )
            if (
                (
                    strategy_params["fast_window"]
                    == strategy_parameters["fast_window"][0]
                    and strategy_params["slow_window"]
                    == strategy_parameters["slow_window"][0]
                )
                or (
                    strategy_params["fast_window"]
                    == strategy_parameters["fast_window"][1]
                    and strategy_params["slow_window"]
                    == strategy_parameters["slow_window"][1]
                )
                or (
                    strategy_params["fast_window"]
                    == strategy_parameters["fast_window"][2]
                    and strategy_params["slow_window"]
                    == strategy_parameters["slow_window"][2]
                )
                or (
                    strategy_params["fast_window"]
                    == strategy_parameters["fast_window"][3]
                    and strategy_params["slow_window"]
                    == strategy_parameters["slow_window"][3]
                )
            ):

                strategy_sides = determine_strategy_side(prices, **strategy_params)
            else:
                continue

            triple_barrier_events = meta_events(
                prices,
                moelcules,
                [0.5, 1.5],
                volatility,
                0,
                1,
                vertical_barriers,
                strategy_sides,
            )
            labels = meta_labeling(triple_barrier_events, prices)
            sample_weights = sample_weight_absolute_return_meta_labeling(
                triple_barrier_events["End Time"], prices, moelcules
            )

            index = (
                features.loc[moelcules]
                .dropna()
                .index.intersection(labels.dropna().index)
                .intersection(sample_weights.dropna().index)
            )
            data = features.loc[index]
            target = labels.loc[index]["Label"]
            weights = sample_weights.loc[index]
            times = labels.loc[index]["End Time"]

            # Iterate over each model and hyperparameter configuration
            for model_name, model_details in models.items():
                model = model_details["Model"]
                param_grid = model_details["Parameters"]

                # Generate all combinations of hyperparameters
                model_keys, model_values = zip(*param_grid.items())
                for model_value in itertools.product(*model_values):
                    params = dict(zip(model_keys, model_value))
                    model.set_params(**params)

                    for (
                        cross_validator_type,
                        cross_validator,
                    ) in cross_validators.items():
                        if "times" in cross_validator.__dict__:
                            cross_validator.times = times
                        predictions = cross_validator.backtest_predictions(
                            model,
                            data,
                            target,
                            weights,
                            predict_probability=True,
                            n_jobs=n_jobs,
                        )
                        probabilities = pd.Series(
                            np.vstack(
                                list(map(lambda x: x[:, 1], predictions.values()))
                            ).mean(axis=0),
                            times.index,
                        ).dropna()
                        positions = strategy_bet_sizing(
                            prices.index,
                            times.loc[probabilities.index],
                            strategy_sides[probabilities.index],
                            probabilities,
                        )
                        strategy_log_returns = (
                            np.log(prices).diff() * positions.shift()
                        ).dropna()
                        results[cross_validator_type].append(
                            {
                                "Trial Info.": {
                                    "Strategy Parameters": strategy_params,
                                    "Model Name": model_name,
                                    "Model Parameters": params,
                                },
                                "Returns": strategy_log_returns,
                            }
                        )

        cv_deflated_sr = {}
        cv_pbo = {}

        for cv, trials in results.items():
            performances = pd.concat([trial["Returns"] for trial in trials], axis=1)

            sharpe_ratios = performances.apply(lambda y: sharpe_ratio(y.values))
            benchmark_sr = benchmark_sharpe_ratio(sharpe_ratios)
            best_strategy_index = sharpe_ratios.idxmax()
            cv_deflated_sr[cv] = probabilistic_sharpe_ratio(
                sharpe_ratios.loc[best_strategy_index],
                benchmark_sr,
                len(performances),
                ss.skew(performances[best_strategy_index]),
                ss.kurtosis(performances[best_strategy_index]),
                return_test_statistic=True,
            )

            pbo, logit_values = probability_of_backtest_overfitting(
                performances.values, risk_free_return=step_risk_free_rate, n_jobs=1
            )
            cv_pbo[cv] = pbo

        for cv in results.keys():
            cv_pbo_embargo.loc[embargo, cv] = cv_pbo[cv]
            cv_deflated_sr_embargo.loc[embargo, cv] = cv_deflated_sr[cv]

    return cv_pbo_embargo, cv_deflated_sr_embargo


def sharpe_ratio(returns, risk_free_rate=0):
    """Calculate the Sharpe ratio of the given returns."""
    std = returns.std()
    if std == 0:
        return 0.0
    return (returns.mean() - risk_free_rate) / std


def sortino_ratio(returns, risk_free_rate=0):
    """Calculate the Sortino ratio of the given returns."""
    downside_returns = returns[returns < risk_free_rate]
    expected_return = returns.mean() - risk_free_rate

    if downside_returns.empty:
        return np.inf if expected_return > 0 else 0.0

    downside_risk = downside_returns.std()

    if downside_risk == 0:
        return np.inf if expected_return > 0 else 0.0

    return expected_return / downside_risk


def expected_shortfall(returns, step_risk_free_rate, confidence_level=0.05):
    """Calculate the expected shortfall (conditional VaR) of the given returns."""
    var = np.percentile(returns, 100 * confidence_level)
    es = returns[returns <= var].mean()
    return es


def backtest_overfitting_simulation_financial_metrics_rank_correlation(
    prices: pd.Series,
    strategy_parameters: dict[str, Union[list[int], list[float], list[bool]]],
    models: dict[str, dict[str, Any]],
    step_risk_free_rate: float,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Conducts a backtest overfitting simulation and calculates the rank correlation of financial metrics.

    Args:
        prices (pd.Series): Time series of asset prices.
        strategy_parameters (dict): Parameters dictating trading strategy behavior.
        models (dict): A collection of machine learning models and their associated parameters.
        step_risk_free_rate (float): The risk-free rate used in the simulation for Sharpe ratio calculations.
        n_jobs (int): The number of jobs to run in parallel for cross_validator.backtest_predictions.

    Returns:
        pd.DataFrame: DataFrame containing the rank correlations for each cross-validation method and each metric.
    """
    # Run the backtest overfitting simulation to get results
    cross_validators = {
        "Walk-Forward": _create_cross_validator(
            "walkforward",
            n_splits=4,
        ),
        "K-Fold": _create_cross_validator(
            "kfold",
            n_splits=4,
        ),
        "Purged K-Fold": _create_cross_validator(
            "purgedkfold", n_splits=4, times=None, embargo=0.02
        ),
        "Combinatorial Purged": _create_cross_validator(
            "combinatorialpurged", n_splits=8, n_test_groups=2, times=None, embargo=0.02
        ),
    }

    results = backtest_overfitting_simulation_results(
        prices, strategy_parameters, models, cross_validators, n_jobs=n_jobs
    )

    metrics = {
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Expected Shortfall": expected_shortfall,
    }

    rank_correlations = {cv: {} for cv in results.keys()}

    for cv, trials in results.items():
        performances = pd.concat([trial["Returns"] for trial in trials], axis=1)

        # Split the data into two halves
        midpoint = len(performances) // 2
        first_half = performances.iloc[:midpoint]
        second_half = performances.iloc[midpoint:]

        # Calculate the metrics for each half
        first_half_metrics = {
            metric_name: first_half.apply(
                lambda x, fn=metric_func: fn(x, step_risk_free_rate), axis=0
            )
            for metric_name, metric_func in metrics.items()
        }
        second_half_metrics = {
            metric_name: second_half.apply(
                lambda x, fn=metric_func: fn(x, step_risk_free_rate), axis=0
            )
            for metric_name, metric_func in metrics.items()
        }

        # Rank the trials in each half
        first_half_ranks = {
            metric_name: metrics_values.rank()
            for metric_name, metrics_values in first_half_metrics.items()
        }
        second_half_ranks = {
            metric_name: metrics_values.rank()
            for metric_name, metrics_values in second_half_metrics.items()
        }

        # Calculate the rank correlation for each metric
        for metric_name in metrics.keys():
            rank_corr, _ = kendalltau(
                first_half_ranks[metric_name], second_half_ranks[metric_name]
            )
            rank_correlations[cv][metric_name] = rank_corr

    # Create the final DataFrame
    rank_corr_df = pd.DataFrame(rank_correlations)

    return rank_corr_df


def backtest_overfitting_simulation_model_complexity(
    prices: pd.Series,
    strategy_parameters: dict[str, Union[list[int], list[float], list[bool]]],
    models: dict[str, Any],
    step_risk_free_rate: float,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Conducts a backtest overfitting simulation to compare the PBO and DSR values of each CV method for simple and complex models.

    Args:
        prices (pd.Series): Time series of asset prices.
        strategy_parameters (dict): Parameters dictating trading strategy behavior.
        models (dict): A collection of machine learning models.
        step_risk_free_rate (float): The risk-free rate used in the simulation for Sharpe ratio calculations.
        n_jobs (int): The number of jobs to run in parallel for cross_validator.backtest_predictions.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Two DataFrames containing PBO and DSR values for each model and each CV method.
    """
    # Initialize DataFrames to store results
    pbo_df = pd.DataFrame(columns=["Combinatorial Purged"], index=models.keys())
    dsr_df = pd.DataFrame(columns=["Combinatorial Purged"], index=models.keys())

    # Create features
    volatility = daily_volatility_with_log_returns(prices, 100)
    filter_threshold = 1.8
    moelcules = cusum_filter_events_dynamic_threshold(
        np.log(prices), filter_threshold * volatility
    )
    vertical_barriers = vertical_barrier(prices, moelcules, 20)
    features = financial_features_backtest_overfitting_simulation(prices)

    for model_name, model in models.items():
        results = {
            "Combinatorial Purged": [],
        }

        # Iterate over each strategy parameter combination
        strategy_parameters_keys, strategy_parameters_values = zip(
            *strategy_parameters.items()
        )
        for strategy_parameters_value in itertools.product(*strategy_parameters_values):
            strategy_params = dict(
                zip(strategy_parameters_keys, strategy_parameters_value)
            )
            try:
                strategy_sides = determine_strategy_side(prices, **strategy_params)

            except ValueError:
                continue

            triple_barrier_events = meta_events(
                prices,
                moelcules,
                [0.5, 1.5],
                volatility,
                0,
                1,
                vertical_barriers,
                strategy_sides,
            )
            labels = meta_labeling(triple_barrier_events, prices)
            sample_weights = sample_weight_absolute_return_meta_labeling(
                triple_barrier_events["End Time"], prices, moelcules
            )

            index = (
                features.loc[moelcules]
                .dropna()
                .index.intersection(labels.dropna().index)
                .intersection(sample_weights.dropna().index)
            )
            data = features.loc[index]
            target = labels.loc[index]["Label"]
            weights = sample_weights.loc[index]
            times = labels.loc[index]["End Time"]

            cross_validators = {
                "Combinatorial Purged": _create_cross_validator(
                    "combinatorialpurged",
                    n_splits=8,
                    n_test_groups=2,
                    times=times,
                    embargo=0.02,
                ),
            }

            for cross_validator_type, cross_validator in cross_validators.items():
                predictions = cross_validator.backtest_predictions(
                    model,
                    data,
                    target,
                    weights,
                    predict_probability=True,
                    n_jobs=n_jobs,
                )
                probabilities = pd.Series(
                    np.vstack(list(map(lambda x: x[:, 1], predictions.values()))).mean(
                        axis=0
                    ),
                    times.index,
                ).dropna()
                positions = strategy_bet_sizing(
                    prices.index,
                    times.loc[probabilities.index],
                    strategy_sides[probabilities.index],
                    probabilities,
                )
                strategy_log_returns = (
                    np.log(prices).diff() * positions.shift()
                ).dropna()
                results[cross_validator_type].append(
                    {
                        "Trial Info.": {
                            "Strategy Parameters": strategy_params,
                            "Model Name": model_name,
                            "Model Parameters": {},
                        },
                        "Returns": strategy_log_returns,
                    }
                )

        cv_deflated_sr = {}
        cv_pbo = {}

        for cv, trials in results.items():
            performances = pd.concat([trial["Returns"] for trial in trials], axis=1)

            sharpe_ratios = performances.apply(lambda y: sharpe_ratio(y.values))
            benchmark_sr = benchmark_sharpe_ratio(sharpe_ratios)
            best_strategy_index = sharpe_ratios.idxmax()
            cv_deflated_sr[cv] = probabilistic_sharpe_ratio(
                sharpe_ratios.loc[best_strategy_index],
                benchmark_sr,
                len(performances),
                ss.skew(performances[best_strategy_index]),
                ss.kurtosis(performances[best_strategy_index]),
                return_test_statistic=True,
            )

            pbo, logit_values = probability_of_backtest_overfitting(
                performances.values, risk_free_return=step_risk_free_rate, n_jobs=1
            )
            cv_pbo[cv] = pbo

        for cv in results.keys():
            pbo_df.loc[model_name, cv] = cv_pbo[cv]
            dsr_df.loc[model_name, cv] = cv_deflated_sr[cv]

    return pbo_df, dsr_df


def noised_backtest_overfitting_simulation(
    prices: pd.Series,
    strategy_parameters: dict[str, Union[list[int], list[float], list[bool]]],
    models: dict[str, dict[str, Any]],
    step_risk_free_rate: float,
    noise_scales: list[float],
    random_state: int = None,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Conducts a noised backtest overfitting simulation to compare the new PBO/DSR values for different noise scales.

    Args:
        prices (pd.Series): Time series of asset prices.
        strategy_parameters (dict): Parameters dictating trading strategy behavior.
        models (dict): A collection of machine learning models and their associated parameters.
        step_risk_free_rate (float): The risk-free rate used in the simulation for Sharpe ratio calculations.
        noise_scales (List[float]): List of noise scale values to test.
        random_state (int): Seed for random number generator.
        n_jobs (int): The number of jobs to run in parallel for cross_validator.backtest_predictions.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Two DataFrames containing PBO and DSR values for each noise scale and each CV method.
    """
    # Initialize DataFrames to store results
    pbo_df = pd.DataFrame(
        columns=["Walk-Forward", "K-Fold", "Purged K-Fold", "Combinatorial Purged"],
        index=noise_scales,
    )
    dsr_df = pd.DataFrame(
        columns=["Walk-Forward", "K-Fold", "Purged K-Fold", "Combinatorial Purged"],
        index=noise_scales,
    )

    for noise_scale in noise_scales:
        # Perform the overall backtest overfitting simulation with the given noise scale
        cv_pbo, cv_deflated_sr = overall_backtest_overfitting_simulation(
            prices=prices,
            strategy_parameters=strategy_parameters,
            models=models,
            step_risk_free_rate=step_risk_free_rate,
            noise_scale=noise_scale,
            random_state=random_state,
            n_jobs=n_jobs,
        )

        # Store the results in the DataFrames
        for cv in cv_pbo.keys():
            pbo_df.loc[noise_scale, cv] = cv_pbo[cv]
            dsr_df.loc[noise_scale, cv] = cv_deflated_sr[cv]

    return pbo_df, dsr_df


def overall_novel_methods_backtest_overfitting_simulation(
    prices: pd.Series,
    strategy_parameters: dict[str, Union[list[int], list[float], list[bool]]],
    models: dict[str, dict[str, Any]],
    step_risk_free_rate: float,
    noise_scale: float = 0.0,
    random_state: int = None,
    n_jobs: int = 1,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Conducts an overall backtest overfitting simulation to calculate the metrics for the novel CPCV methods.

    Args:
        prices (pd.Series): Time series of asset prices.
        strategy_parameters (dict): Parameters dictating trading strategy behavior.
        models (dict): A collection of machine learning models and their associated parameters.
        step_risk_free_rate (float): The risk-free rate used in the simulation for Sharpe ratio calculations.
        noise_scale (float): Scale of Gaussian noise to be added to the features.
        random_state (int): Seed for random number generator.
        n_jobs (int): The number of jobs to run in parallel for cross_validator.backtest_predictions.

    Returns:
        Tuple[Dict[str, float], Dict[str, float]]: A tuple containing two dictionaries, one for the Probability of Backtest Overfitting (PBO) and the other for the Deflated Sharpe Ratio (DSR), for each cross-validation method tested.
    """

    cross_validators = {
        "Combinatorial Purged": _create_cross_validator(
            "combinatorialpurged", n_splits=8, n_test_groups=2, times=None, embargo=0.02
        ),
        "Bagged Combinatorial Purged": _create_cross_validator(
            "baggedcombinatorialpurged",
            n_splits=8,
            n_test_groups=2,
            times=None,
            embargo=0.02,
            random_state=random_state,
        ),
        "Adaptive Combinatorial Purged": _create_cross_validator(
            "adaptivecombinatorialpurged",
            n_splits=8,
            n_test_groups=2,
            times=None,
            embargo=0.02,
        ),
    }

    results = backtest_overfitting_simulation_results(
        prices,
        strategy_parameters,
        models,
        cross_validators,
        noise_scale,
        random_state,
        n_jobs=n_jobs,
    )

    cv_deflated_sr = {}
    cv_pbo = {}

    for cv, trials in results.items():
        performances = pd.concat([trial["Returns"] for trial in trials], axis=1)

        sharpe_ratios = performances.apply(lambda y: sharpe_ratio(y.values))
        benchmark_sr = benchmark_sharpe_ratio(sharpe_ratios)
        best_strategy_index = sharpe_ratios.idxmax()
        cv_deflated_sr[cv] = probabilistic_sharpe_ratio(
            sharpe_ratios.loc[best_strategy_index],
            benchmark_sr,
            len(performances),
            ss.skew(performances[best_strategy_index]),
            ss.kurtosis(performances[best_strategy_index]),
            return_test_statistic=True,
        )

        pbo, logit_values = probability_of_backtest_overfitting(
            performances.values, risk_free_return=step_risk_free_rate
        )
        cv_pbo[cv] = pbo

    return cv_pbo, cv_deflated_sr


def get_cpu_info():
    # Run the lscpu command
    result = subprocess.run(["lscpu"], stdout=subprocess.PIPE)
    # Decode the output from bytes to string
    lscpu_output = result.stdout.decode("utf-8")

    # Parse the lscpu output
    cpu_info = {}
    for line in lscpu_output.split("\n"):
        if line.strip():
            parts = line.split(":", 1)
            if len(parts) == 2:
                key, value = parts
                cpu_info[key.strip()] = value.strip()

    # Extract useful information
    useful_info = {
        "Architecture": cpu_info.get("Architecture"),
        "CPU op-mode(s)": cpu_info.get("CPU op-mode(s)"),
        "CPU(s)": cpu_info.get("CPU(s)"),
        "Thread(s) per core": cpu_info.get("Thread(s) per core"),
        "Core(s) per socket": cpu_info.get("Core(s) per socket"),
        "Socket(s)": cpu_info.get("Socket(s)"),
        "Vendor ID": cpu_info.get("Vendor ID"),
        "Model name": cpu_info.get("Model name"),
        "CPU MHz": cpu_info.get("CPU MHz"),
        "L1d cache": cpu_info.get("L1d cache"),
        "L1i cache": cpu_info.get("L1i cache"),
        "L2 cache": cpu_info.get("L2 cache"),
        "L3 cache": cpu_info.get("L3 cache"),
    }

    return useful_info


def format_cpu_info(cpu_info):
    report = (
        f"Architecture: {cpu_info['Architecture']}\n"
        f"CPU Operational Modes: {cpu_info['CPU op-mode(s)']}\n"
        f"Total CPUs: {cpu_info['CPU(s)']}\n"
        f"Threads per Core: {cpu_info['Thread(s) per core']}\n"
        f"Cores per Socket: {cpu_info['Core(s) per socket']}\n"
        f"Sockets: {cpu_info['Socket(s)']}\n"
        f"Vendor ID: {cpu_info['Vendor ID']}\n"
        f"Model Name: {cpu_info['Model name']}\n"
        f"CPU MHz: {cpu_info['CPU MHz']}\n"
        f"L1d Cache: {cpu_info['L1d cache']}\n"
        f"L1i Cache: {cpu_info['L1i cache']}\n"
        f"L2 Cache: {cpu_info['L2 cache']}\n"
        f"L3 Cache: {cpu_info['L3 cache']}\n"
    )
    return report


# Function to generate random data, target, weights, and times
def generate_random_data(
    n_samples: int, n_features: int
) -> tuple[pd.DataFrame, pd.Series, np.ndarray, pd.Series]:
    date_range = pd.date_range(start="1980-01-01", periods=n_samples, freq="1h")
    data = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
        index=date_range,
    )
    target = pd.Series(np.random.randint(0, 2, n_samples), index=date_range)
    weights = pd.Series(np.random.rand(n_samples), index=date_range)
    weights = weights / weights.sum()
    times = pd.Series(date_range + pd.DateOffset(hours=3), index=date_range)
    return data, target, weights, times


# Function to measure computational requirements
def measure_computational_requirements(
    cross_validator, model, data, target, weights, n_jobs: int = 1
) -> dict[str, Any]:
    from memory_profiler import memory_usage  # optional dependency: RiskLabAI[dev]

    start_time = time.time()
    mem_usage = memory_usage(
        (
            cross_validator.backtest_predictions,
            (model, data, target, weights),
            {"predict_probability": True, "n_jobs": n_jobs},
        ),
        interval=0.1,
    )
    end_time = time.time()
    return {"execution_time": end_time - start_time, "memory_usage": max(mem_usage)}


# Main function to measure computational requirements for all CV methods
def measure_all_cv_computational_requirements(
    cross_validators: dict[str, Any],
    n_samples: int = 40 * 252,
    n_features: int = 22,
    n_jobs: int = 1,
    n_repeats: int = 30,
) -> pd.DataFrame:
    # Generate random data, target, weights, and times
    data, target, weights, times = generate_random_data(n_samples, n_features)

    # Define the logistic regression model without regularization
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=10000)

    # Measure computational requirements for each CV method
    results = {
        cv_name: {"execution_time": [], "memory_usage": []}
        for cv_name in cross_validators.keys()
    }

    for _ in _progress(range(n_repeats)):
        for cv_name, cross_validator in cross_validators.items():
            # print(f"Measuring computational requirements for {cv_name} (repeat {_ + 1}/{n_repeats})...")
            # Update cross-validator with times if required
            if "times" in cross_validator.__dict__:
                cross_validator.times = times
            result = measure_computational_requirements(
                cross_validator, model, data, target, weights, n_jobs=n_jobs
            )
            results[cv_name]["execution_time"].append(result["execution_time"])
            results[cv_name]["memory_usage"].append(result["memory_usage"])

    # Calculate mean and standard deviation for each CV method
    results_summary = {}
    for cv_name, metrics in results.items():
        results_summary[cv_name] = {
            "execution_time_mean": np.mean(metrics["execution_time"]),
            "execution_time_std": np.std(metrics["execution_time"]),
            "memory_usage_mean": np.mean(metrics["memory_usage"]),
            "memory_usage_std": np.std(metrics["memory_usage"]),
        }

    # Convert results to DataFrame
    results_df = pd.DataFrame(results_summary).T

    return results_df


_DEFAULT_N_JOBS_LIST = list(range(1, 9))


def measure_cpcv_parallelization(
    n_samples: int = 40 * 252,
    n_features: int = 22,
    n_repeats: int = 30,
    n_jobs_list: list[int] = _DEFAULT_N_JOBS_LIST,
) -> pd.DataFrame:
    # Generate random data, target, weights, and times
    data, target, weights, times = generate_random_data(n_samples, n_features)

    # Define the logistic regression model without regularization
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=10000)

    # Define the CPCV cross-validator
    cpcv_cross_validator = _create_cross_validator(
        "combinatorialpurged", n_splits=8, n_test_groups=2, times=times, embargo=0.02
    )

    # Measure computational requirements with and without parallelization
    results = {}

    for n_jobs in _progress(n_jobs_list):
        key = f"n_jobs_{n_jobs}"
        results[key] = {"execution_time": [], "memory_usage": []}
        for _ in range(n_repeats):
            result = measure_computational_requirements(
                cpcv_cross_validator, model, data, target, weights, n_jobs=n_jobs
            )
            results[key]["execution_time"].append(result["execution_time"])
            results[key]["memory_usage"].append(result["memory_usage"])

    # Calculate mean and standard deviation for each setting
    results_summary = {}
    for key, metrics in results.items():
        results_summary[key] = {
            "execution_time_mean": np.mean(metrics["execution_time"]),
            "execution_time_std": np.std(metrics["execution_time"]),
            "memory_usage_mean": np.mean(metrics["memory_usage"]),
            "memory_usage_std": np.std(metrics["memory_usage"]),
        }

    # Convert results to DataFrame
    results_df = pd.DataFrame(results_summary).T

    return results_df


def measure_cpcv_scalability(
    sample_sizes: list[int],
    feature_sizes: list[int],
    n_repeats: int = 1,
    n_jobs: int = 1,
) -> pd.DataFrame:
    # Define the logistic regression model without regularization
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=10000)

    execution_times = pd.DataFrame(index=sample_sizes, columns=feature_sizes)
    memory_usages = pd.DataFrame(index=sample_sizes, columns=feature_sizes)

    for n_samples, n_features in itertools.product(sample_sizes, feature_sizes):
        execution_time_list = []
        memory_usage_list = []

        for _ in range(n_repeats):
            # Generate random data, target, weights, and times
            data, target, weights, times = generate_random_data(n_samples, n_features)

            # Define the CPCV cross-validator
            cpcv_cross_validator = _create_cross_validator(
                "combinatorialpurged",
                n_splits=8,
                n_test_groups=2,
                times=times,
                embargo=0.02,
            )

            # Measure computational requirements
            result = measure_computational_requirements(
                cpcv_cross_validator, model, data, target, weights, n_jobs=n_jobs
            )
            execution_time_list.append(result["execution_time"])
            memory_usage_list.append(result["memory_usage"])

        # Calculate mean execution time and memory usage
        execution_times.loc[n_samples, n_features] = np.mean(execution_time_list)
        memory_usages.loc[n_samples, n_features] = np.mean(memory_usage_list)

    return execution_times, memory_usages
