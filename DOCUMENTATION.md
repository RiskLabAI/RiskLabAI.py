# Documentation for `RiskLabAI` Library

## 🌳 File Structure

```
📁 RiskLabAI/
├── 📁 backtest/
│   ├── 📁 validation/
│   │   ├── 📄 __init__.py
│   │   ├── 📄 adaptive_combinatorial_purged.py
│   │   ├── 📄 bagged_combinatorial_purged.py
│   │   ├── 📄 combinatorial_purged.py
│   │   ├── 📄 cross_validator_controller.py
│   │   ├── 📄 cross_validator_factory.py
│   │   ├── 📄 cross_validator_interface.py
│   │   ├── 📄 kfold.py
│   │   ├── 📄 purged_kfold.py
│   │   └── 📄 walk_forward.py
│   ├── 📄 __init__.py
│   ├── 📄 backtest_statistics.py
│   ├── 📄 backtest_synthetic_data.py
│   ├── 📄 backtset_overfitting_in_the_machine_learning_era_simulation.py
│   ├── 📄 bet_sizing.py
│   ├── 📄 probabilistic_sharpe_ratio.py
│   ├── 📄 probability_of_backtest_overfitting.py
│   ├── 📄 strategy_risk.py
│   └── 📄 test_set_overfitting.py
├── 📁 cluster/
│   ├── 📄 __init__.py
│   └── 📄 clustering.py
├── 📁 controller/
│   ├── 📄 __init__.py
│   ├── 📄 bars_initializer.py
│   └── 📄 data_structure_controller.py
├── 📁 data/
│   ├── 📁 denoise/
│   │   ├── 📄 __init__.py
│   │   └── 📄 denoising.py
│   ├── 📁 differentiation/
│   │   ├── 📄 __init__.py
│   │   └── 📄 differentiation.py
│   ├── 📁 distance/
│   │   ├── 📄 __init__.py
│   │   └── 📄 distance_metric.py
│   ├── 📁 labeling/
│   │   ├── 📄 __init__.py
│   │   ├── 📄 financial_labels.py
│   │   └── 📄 labeling.py
│   ├── 📁 structures/
│   │   ├── 📄 __init__.py
│   │   ├── 📄 abstract_bars.py
│   │   ├── 📄 abstract_imbalance_bars.py
│   │   ├── 📄 abstract_information_driven_bars.py
│   │   ├── 📄 abstract_run_bars.py
│   │   ├── 📄 data_structures_lopez.py
│   │   ├── 📄 filtering_lopez.py
│   │   ├── 📄 hedging.py
│   │   ├── 📄 imbalance_bars.py
│   │   ├── 📄 infomation_driven_bars.py
│   │   ├── 📄 run_bars.py
│   │   ├── 📄 standard_bars.py
│   │   ├── 📄 standard_bars_lopez.py
│   │   ├── 📄 time_bars.py
│   │   └── 📄 utilities_lopez.py
│   ├── 📁 synthetic_data/
│   │   ├── 📄 __init__.py
│   │   ├── 📄 drift_burst_hypothesis.py
│   │   └── 📄 synthetic_controlled_environment.py
│   ├── 📁 weights/
│   │   ├── 📄 __init__.py
│   │   └── 📄 sample_weights.py
│   └── 📄 __init__.py
├── 📁 ensemble/
│   ├── 📄 __init__.py
│   └── 📄 bagging_classifier_accuracy.py
├── 📁 features/
│   ├── 📁 entropy_features/
│   │   ├── 📄 __init__.py
│   │   ├── 📄 entropy.py
│   │   ├── 📄 kontoyiannis.py
│   │   ├── 📄 lempel_ziv.py
│   │   ├── 📄 plug_in.py
│   │   ├── 📄 pmf.py
│   │   └── 📄 shannon.py
│   ├── 📁 feature_importance/
│   │   ├── 📄 __init__.py
│   │   ├── 📄 clustered_feature_importance_mda.py
│   │   ├── 📄 clustered_feature_importance_mdi.py
│   │   ├── 📄 clustering.py
│   │   ├── 📄 feature_importance_controller.py
│   │   ├── 📄 feature_importance_factory.py
│   │   ├── 📄 feature_importance_mda.py
│   │   ├── 📄 feature_importance_mdi.py
│   │   ├── 📄 feature_importance_sfi.py
│   │   ├── 📄 feature_importance_strategy.py
│   │   ├── 📄 FeatureImportance.ipynb
│   │   ├── 📄 generate_synthetic_data.py
│   │   ├── 📄 orthogonal_features.py
│   │   └── 📄 weighted_tau.py
│   ├── 📁 microstructural_features/
│   │   ├── 📄 __init__.py
│   │   ├── 📄 bekker_parkinson_volatility_estimator.py
│   │   └── 📄 corwin_schultz.py
│   ├── 📁 structural_breaks/
│   │   ├── 📄 __init__.py
│   │   └── 📄 structural_breaks.py
│   ├── 📄 __init__.py
│   └── 📄 test.ipynb
├── 📁 hpc/
│   ├── 📄 __init__.py
│   └── 📄 hpc.py
├── 📁 optimization/
│   ├── 📄 __init__.py
│   ├── 📄 hrp.py
│   ├── 📄 hyper_parameter_tuning.py
│   └── 📄 nco.py
├── 📁 pde/
│   ├── 📄 __init__.py
│   ├── 📄 equation.py
│   ├── 📄 model.py
│   └── 📄 solver.py
├── 📁 utils/
│   ├── 📄 __init__.py
│   ├── 📄 constants.py
│   ├── 📄 ewma.py
│   ├── 📄 momentum_mean_reverting_strategy_sides.py
│   ├── 📄 progress.py
│   ├── 📄 smoothing_average.py
│   └── 📄 update_figure_layout.py
└── 📄 __init__.py
```

## 📄 Module & Function Reference

### 📄 `RiskLabAI\backtest\backtest_statistics.py`

#### `function bet_timing`

```python
def bet_timingtarget_positions: pd.Series:
```

> Determine the timing of bets when positions flatten or flip.

:param target_positions: Series of target positions.
:return: Index of bet timing.

#### `function calculate_holding_period`

```python
def calculate_holding_periodtarget_positions: pd.Series:
```

> Derive average holding period (in days) using the average entry time pairing algorithm.

:param target_positions: Series of target positions.
:return: Tuple containing holding period DataFrame and mean holding period.

#### `function calculate_hhi_concentration`

```python
def calculate_hhi_concentrationreturns: pd.Series:
```

> Calculate the HHI concentration measures.

:param returns: Series of returns.
:return: Tuple containing positive returns HHI, negative returns HHI, and time-concentrated HHI.

#### `function calculate_hhi`

```python
def calculate_hhibet_returns: pd.Series:
```

> Calculate the Herfindahl-Hirschman Index (HHI) concentration measure.

:param bet_returns: Series of bet returns.
:return: Calculated HHI value.

#### `function compute_drawdowns_time_under_water`

```python
def compute_drawdowns_time_under_waterseries: pd.Series, dollars: bool=False:
```

> Compute series of drawdowns and the time under water associated with them.

:param series: Series of returns or dollar performance.
:param dollars: Whether the input series represents returns or dollar performance.
:return: Tuple containing drawdown series, time under water series, and drawdown analysis DataFrame.


### 📄 `RiskLabAI\backtest\backtest_synthetic_data.py`

#### `function synthetic_back_testing`

```python
def synthetic_back_testingforecast: float, half_life: float, sigma: float, n_iteration: int=100000, maximum_holding_period: int=100, profit_taking_range: np.ndarray=np.linspace(0.5, 10, 20), stop_loss_range: np.ndarray=np.linspace(0.5, 10, 20), seed: int=0:
```

> Perform backtesting on synthetic price data generated using the Ornstein-Uhlenbeck process.

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


### 📄 `RiskLabAI\backtest\backtset_overfitting_in_the_machine_learning_era_simulation.py`

#### `function financial_features_backtest_overfitting_simulation`

```python
def financial_features_backtest_overfitting_simulationprices: pd.Series, noise_scale: float=0.0, random_state: Optional[int]=None:
```

> Create a DataFrame of financial features from the given price series.

Args:
    prices (pd.Series): Time series of asset prices.
    noise_scale (float): Scale of Gaussian noise to be added to the features. Default is 0.0.
    random_state (Optional[int]): Seed for random number generator. Default is None.

Returns:
    pd.DataFrame: DataFrame containing the computed financial features.

#### `function backtest_overfitting_simulation_results`

```python
def backtest_overfitting_simulation_resultsprices: pd.Series, strategy_parameters: Dict[str, Union[List[int], List[float], List[bool]]], models: Dict[str, Dict[str, Any]], cross_validators: Dict[str, Any], noise_scale: float=0.0, random_state: int=None, n_jobs: int=1:
```

> Conducts a simulation to evaluate the performance of trading strategies and models.

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

#### `function overall_backtest_overfitting_simulation`

```python
def overall_backtest_overfitting_simulationprices: pd.Series, strategy_parameters: Dict[str, Union[List[int], List[float], List[bool]]], models: Dict[str, Dict[str, Any]], step_risk_free_rate: float, noise_scale: float=0.0, random_state: int=None, n_jobs: int=1:
```

> Conducts an overall backtest overfitting simulation to calculate the metrics.

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

#### `function temporal_backtest_overfitting_simulation`

```python
def temporal_backtest_overfitting_simulationprices: pd.Series, strategy_parameters: Dict[str, Union[List[int], List[float], List[bool]]], models: Dict[str, Dict[str, Any]], step_risk_free_rate: float, overfitting_partitions_length: int, n_jobs: int=1:
```

> Conducts a temporal backtest overfitting simulation to calculate the metrics in chunks.

Args:
    prices (pd.Series): Time series of asset prices.
    strategy_parameters (dict): Parameters dictating trading strategy behavior.
    models (dict): A collection of machine learning models and their associated parameters.
    step_risk_free_rate (float): The risk-free rate used in the simulation for Sharpe ratio calculations.
    overfitting_partitions_length (int): The number of partitions to divide the dataset into for temporal overfitting analysis.
    n_jobs (int): The number of jobs to run in parallel for cross_validator.backtest_predictions.

Returns:
    Tuple[Dict[str, List[float]], Dict[str, List[float]]]: A tuple containing two dictionaries, one for the Probability of Backtest Overfitting (PBO) and the other for the Deflated Sharpe Ratio (DSR), for each cross-validation method tested.

#### `function time_temporal_backtest_overfitting_simulation`

```python
def time_temporal_backtest_overfitting_simulationprices: pd.Series, strategy_parameters: Dict[str, Union[List[int], List[float], List[bool]]], models: Dict[str, Dict[str, Any]], step_risk_free_rate: float, overfitting_partitions_duration: str='A', n_jobs: int=1:
```

> Conducts a time-temporal backtest overfitting simulation to calculate the metrics in time-indexed chunks.

Args:
    prices (pd.Series): Time series of asset prices.
    strategy_parameters (dict): Parameters dictating trading strategy behavior.
    models (dict): A collection of machine learning models and their associated parameters.
    step_risk_free_rate (float): The risk-free rate used in the simulation for Sharpe ratio calculations.
    overfitting_partitions_duration (str): The frequency for time-based grouping to divide the dataset into for temporal overfitting analysis.
    n_jobs (int): The number of jobs to run in parallel for cross_validator.backtest_predictions.

Returns:
    Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]: A tuple containing two dictionaries, one for the Probability of Backtest Overfitting (PBO) and the other for the Deflated Sharpe Ratio (DSR), for each cross-validation method tested, indexed by time.

#### `function varying_embargo_backtest_overfitting_simulation`

```python
def varying_embargo_backtest_overfitting_simulationprices: pd.Series, strategy_parameters: Dict[str, Union[List[int], List[float], List[bool]]], models: Dict[str, Dict[str, Any]], step_risk_free_rate: float, embargo_values: List[float], n_jobs: int=1:
```

> Conducts a backtest overfitting simulation with varying embargo values to calculate the metrics.

Args:
    prices (pd.Series): Time series of asset prices.
    strategy_parameters (dict): Parameters dictating trading strategy behavior.
    models (dict): A collection of machine learning models and their associated parameters.
    step_risk_free_rate (float): The risk-free rate used in the simulation for Sharpe ratio calculations.
    embargo_values (List[float]): List of embargo values to test.
    n_jobs (int): The number of jobs to run in parallel for cross_validator.backtest_predictions.

Returns:
    Tuple[pd.DataFrame, pd.DataFrame]: Two DataFrames containing PBO and DSR values for each embargo value and cross-validation method.

#### `function sharpe_ratio`

```python
def sharpe_ratioreturns, risk_free_rate=0:
```

> Calculate the Sharpe ratio of the given returns.

#### `function sortino_ratio`

```python
def sortino_ratioreturns, risk_free_rate=0:
```

> Calculate the Sortino ratio of the given returns.

#### `function expected_shortfall`

```python
def expected_shortfallreturns, step_risk_free_rate, confidence_level=0.05:
```

> Calculate the expected shortfall (conditional VaR) of the given returns.

#### `function backtest_overfitting_simulation_financial_metrics_rank_correlation`

```python
def backtest_overfitting_simulation_financial_metrics_rank_correlationprices: pd.Series, strategy_parameters: Dict[str, Union[List[int], List[float], List[bool]]], models: Dict[str, Dict[str, Any]], step_risk_free_rate: float, n_jobs: int=1:
```

> Conducts a backtest overfitting simulation and calculates the rank correlation of financial metrics.

Args:
    prices (pd.Series): Time series of asset prices.
    strategy_parameters (dict): Parameters dictating trading strategy behavior.
    models (dict): A collection of machine learning models and their associated parameters.
    step_risk_free_rate (float): The risk-free rate used in the simulation for Sharpe ratio calculations.
    n_jobs (int): The number of jobs to run in parallel for cross_validator.backtest_predictions.

Returns:
    pd.DataFrame: DataFrame containing the rank correlations for each cross-validation method and each metric.

#### `function backtest_overfitting_simulation_model_complexity`

```python
def backtest_overfitting_simulation_model_complexityprices: pd.Series, strategy_parameters: Dict[str, Union[List[int], List[float], List[bool]]], models: Dict[str, Any], step_risk_free_rate: float, n_jobs: int=1:
```

> Conducts a backtest overfitting simulation to compare the PBO and DSR values of each CV method for simple and complex models.

Args:
    prices (pd.Series): Time series of asset prices.
    strategy_parameters (dict): Parameters dictating trading strategy behavior.
    models (dict): A collection of machine learning models.
    step_risk_free_rate (float): The risk-free rate used in the simulation for Sharpe ratio calculations.
    n_jobs (int): The number of jobs to run in parallel for cross_validator.backtest_predictions.

Returns:
    Tuple[pd.DataFrame, pd.DataFrame]: Two DataFrames containing PBO and DSR values for each model and each CV method.

#### `function noised_backtest_overfitting_simulation`

```python
def noised_backtest_overfitting_simulationprices: pd.Series, strategy_parameters: Dict[str, Union[List[int], List[float], List[bool]]], models: Dict[str, Dict[str, Any]], step_risk_free_rate: float, noise_scales: List[float], random_state: int=None, n_jobs: int=1:
```

> Conducts a noised backtest overfitting simulation to compare the new PBO/DSR values for different noise scales.

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

#### `function overall_novel_methods_backtest_overfitting_simulation`

```python
def overall_novel_methods_backtest_overfitting_simulationprices: pd.Series, strategy_parameters: Dict[str, Union[List[int], List[float], List[bool]]], models: Dict[str, Dict[str, Any]], step_risk_free_rate: float, noise_scale: float=0.0, random_state: int=None, n_jobs: int=1:
```

> Conducts an overall backtest overfitting simulation to calculate the metrics for the novel CPCV methods.

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

#### `function get_cpu_info`

```python
def get_cpu_info:
```
#### `function format_cpu_info`

```python
def format_cpu_infocpu_info:
```
#### `function generate_random_data`

```python
def generate_random_datan_samples: int, n_features: int:
```
#### `function measure_computational_requirements`

```python
def measure_computational_requirementscross_validator, model, data, target, weights, n_jobs: int=1:
```
#### `function measure_all_cv_computational_requirements`

```python
def measure_all_cv_computational_requirementscross_validators: Dict[str, Any], n_samples: int=40 * 252, n_features: int=22, n_jobs: int=1, n_repeats: int=30:
```
#### `function measure_cpcv_parallelization`

```python
def measure_cpcv_parallelizationn_samples: int=40 * 252, n_features: int=22, n_repeats: int=30, n_jobs_list: List[int]=range(1, 9):
```
#### `function measure_cpcv_scalability`

```python
def measure_cpcv_scalabilitysample_sizes: List[int], feature_sizes: List[int], n_repeats: int=1, n_jobs: int=1:
```

### 📄 `RiskLabAI\backtest\bet_sizing.py`

#### `function probability_bet_size`

```python
def probability_bet_sizeprobabilities: np.ndarray, sides: np.ndarray:
```

> Calculate the bet size based on probabilities and side.

:param probabilities: array of probabilities
:param sides: array indicating the side of the bet (e.g., long/short or buy/sell)
:return: array of bet sizes

.. math::

    ext{bet size} =         ext{side}       imes (2         imes    ext{CDF}(       ext{probabilities}) - 1)

#### `function average_bet_sizes`

```python
def average_bet_sizesprice_dates: np.ndarray, start_dates: np.ndarray, end_dates: np.ndarray, bet_sizes: np.ndarray:
```

> Compute average bet sizes for each date.

:param price_dates: array of price dates
:param start_dates: array of start dates for bets
:param end_dates: array of end dates for bets
:param bet_sizes: array of bet sizes for each date range
:return: array of average bet sizes for each price date

#### `function strategy_bet_sizing`

```python
def strategy_bet_sizingprice_timestamps: pd.Series, times: pd.Series, sides: pd.Series, probabilities: pd.Series:
```

> Calculate the average bet size for a trading strategy given price timestamps.

:param price_timestamps: series of price timestamps
:param times: series with start times as indices and end times as values
:param sides: series indicating the side of the position (e.g., long/short)
:param probabilities: series of probabilities associated with each position
:return: series of average bet sizes for each price timestamp

#### `function avgActiveSignals`

```python
def avgActiveSignalssignals, nThreads:
```
#### `function mpAvgActiveSignals`

```python
def mpAvgActiveSignalssignals, molecule:
```
#### `function discreteSignal`

```python
def discreteSignalsignal, stepSize:
```
#### `function Signal`

```python
def Signalevents, stepSize, probability, prediction, nClasses, nThreads:
```
#### `function betSize`

```python
def betSizew, x:
```
#### `function TPos`

```python
def TPosw, f, acctualPrice, maximumPositionSize:
```
#### `function inversePrice`

```python
def inversePricef, w, m:
```
#### `function limitPrice`

```python
def limitPricetargetPositionSize, cPosition, f, w, maximumPositionSize:
```
#### `function getW`

```python
def getWx, m:
```

### 📄 `RiskLabAI\backtest\probabilistic_sharpe_ratio.py`

#### `function probabilistic_sharpe_ratio`

```python
def probabilistic_sharpe_ratioobserved_sharpe_ratio: float, benchmark_sharpe_ratio: float, number_of_returns: int, skewness_of_returns: float=0, kurtosis_of_returns: float=3, return_test_statistic: bool=False:
```

> Calculates the Probabilistic Sharpe Ratio (PSR) based on observed and benchmark Sharpe ratios.

The PSR provides a means to test whether a track record would have achieved an observed 
level of outperformance due to skill or luck. It is calculated using:

.. math::
    \frac{(\hat{SR} - SR^*) \sqrt{T-1}}{\sqrt{1 - S \hat{SR} + \frac{K-1}{4} \hat{SR}^2}}

Where:
- \(\hat{SR}\) is the observed Sharpe ratio
- \(SR^*\) is the benchmark Sharpe ratio
- \(T\) is the number of returns
- \(S\) is the skewness of returns
- \(K\) is the kurtosis of returns

:param observed_sharpe_ratio: The observed Sharpe ratio.
:param benchmark_sharpe_ratio: The benchmark Sharpe ratio.
:param number_of_returns: The number of return observations.
:param skewness_of_returns: The skewness of the returns (default = 0).
:param kurtosis_of_returns: The kurtosis of the returns (default = 3).
:param return_test_statistic: Return the test statistic instead of the CDF value.
:return: The Probabilistic Sharpe Ratio.

#### `function benchmark_sharpe_ratio`

```python
def benchmark_sharpe_ratiosharpe_ratio_estimates: list:
```

> Calculates the Benchmark Sharpe Ratio based on Sharpe ratio estimates.

The benchmark Sharpe ratio is computed using:

.. math::
    \sigma_{SR} \left[ (1 - \gamma) \Phi^{-1}(1 - \frac{1}{N}) + \gamma \Phi^{-1}(1 - \frac{1}{N} e^{-1}) \right]

Where:
- \(\sigma_{SR}\) is the standard deviation of Sharpe ratio estimates
- \(\gamma\) is the Euler's constant
- \(\Phi^{-1}\) is the inverse of the cumulative distribution function (CDF) of a standard normal distribution
- \(N\) is the number of Sharpe ratio estimates

:param sharpe_ratio_estimates: List of Sharpe ratio estimates.
:return: The Benchmark Sharpe Ratio.


### 📄 `RiskLabAI\backtest\probability_of_backtest_overfitting.py`

#### `function sharpe_ratio`

```python
def sharpe_ratioreturns: np.ndarray, risk_free_rate: float=0.0:
```

> Calculate the Sharpe Ratio for a given set of returns.

:param returns: An array of returns for a portfolio.
:param risk_free_rate: The risk-free rate.
:return: The calculated Sharpe Ratio.

.. math::

            ext{Sharpe Ratio} = rac{       ext{Mean Portfolio Return} -    ext{Risk-Free Rate}}
                              {     ext{Standard Deviation of Portfolio Returns}}

#### `function performance_evaluation`

```python
def performance_evaluationtrain_partition: np.ndarray, test_partition: np.ndarray, n_strategies: int, metric: Callable, risk_free_return: float:
```

> Evaluate the performance of various strategies on given train and test partitions and 
compute the logit value to determine if the best in-sample strategy is overfitting.

:param train_partition: Training data partition used for evaluating in-sample performance.
:type train_partition: np.ndarray
:param test_partition: Testing data partition used for evaluating out-of-sample performance.
:type test_partition: np.ndarray
:param n_strategies: Number of strategies to evaluate.
:type n_strategies: int
:param metric: Metric function for evaluating strategy performance. 
               The function should accept a data array and risk_free_return as arguments.
:type metric: Callable
:param risk_free_return: Risk-free return used in the metric function, often used for Sharpe ratio.
:type risk_free_return: float

:return: Tuple where the first value indicates if the best in-sample strategy is overfitting 
         (True if overfitting, False otherwise) and the second value is the logit value computed.
:rtype: Tuple[bool, float]

#### `function probability_of_backtest_overfitting`

```python
def probability_of_backtest_overfittingperformances: np.ndarray, n_partitions: int=16, risk_free_return: float=0.0, metric: Callable=None, n_jobs: int=1:
```

> Computes the Probability Of Backtest Overfitting.

For instance, if \(S=16\), we will form 12,780 combinations.

.. math::
    \left(\begin{array}{c}
    S \\
    S / 2
    \end{array}\right) = \prod_{i=0}^{S / 2^{-1}} \frac{S-i}{S / 2-i}

:param performances: Matrix of T×N for T observations on N strategies.
:type performances: np.ndarray
:param n_partitions: Number of partitions (must be even).
:type n_partitions: int
:param metric: Metric function for evaluating strategy.
:type metric: Callable
:param risk_free_return: Risk-free return for calculating Sharpe ratio.
:type risk_free_return: float
:param n_jobs: Number of parallel jobs.
:type n_jobs: int

:return: Tuple containing Probability Of Backtest Overfitting and an array of logit values.
:rtype: Tuple[float, List[float]]


### 📄 `RiskLabAI\backtest\strategy_risk.py`

#### `function sharpe_ratio_trials`

```python
def sharpe_ratio_trialsp: float, n_run: int:
```

> Simulate trials to calculate the mean, standard deviation, and Sharpe ratio.

The Sharpe ratio is calculated as follows:

.. math:: S = \\frac{\\mu}{\\sigma}

where:
- \(\\mu\) is the mean of the returns
- \(\\sigma\) is the standard deviation of the returns

Args:
    p (float): Probability of success.
    n_run (int): Number of runs.

Returns:
    tuple[float, float, float]: Tuple containing mean, standard deviation, and Sharpe ratio.

#### `function target_sharpe_ratio_symbolic`

```python
def target_sharpe_ratio_symbolic:
```

> Calculate the target Sharpe ratio using symbolic operations.

The Sharpe ratio is calculated using the following formula:

.. math:: S = \\frac{p \\cdot u^2 + (1 - p) \\cdot d^2 - (p \\cdot u + (1 - p) \\cdot d)^2}{\\sigma}

where:
- \(p\) is the probability of success
- \(u\) is the upward movement
- \(d\) is the downward movement
- \(\\sigma\) is the standard deviation of the returns

Returns:
    sympy.Add: Symbolic expression for target Sharpe ratio.

#### `function implied_precision`

```python
def implied_precisionstop_loss: float, profit_taking: float, frequency: float, target_sharpe_ratio: float:
```

> Calculate the implied precision for given parameters.

The implied precision is calculated as follows:

.. math::
    a = (f + S^2) * (p - s)^2
    b = (2 * f * s - S^2 * (p - s)) * (p - s)
    c = f * s^2
    precision = (-b + \\sqrt{b^2 - 4 * a * c}) / (2 * a)

where:
- \(f\) is the frequency of bets per year
- \(S\) is the target annual Sharpe ratio
- \(p\) is the profit-taking threshold
- \(s\) is the stop-loss threshold

Args:
    stop_loss (float): Stop-loss threshold.
    profit_taking (float): Profit-taking threshold.
    frequency (float): Number of bets per year.
    target_sharpe_ratio (float): Target annual Sharpe ratio.

Returns:
    float: Calculated implied precision.

#### `function bin_frequency`

```python
def bin_frequencystop_loss: float, profit_taking: float, precision: float, target_sharpe_ratio: float:
```

> Calculate the number of bets per year needed to achieve a target Sharpe ratio with a certain precision.

The frequency of bets is calculated as follows:

.. math::
    frequency = \\frac{S^2 * (p - s)^2 * precision * (1 - precision)}{((p - s) * precision + s)^2}

where:
- \(S\) is the target annual Sharpe ratio
- \(p\) is the profit-taking threshold
- \(s\) is the stop-loss threshold
- \(precision\) is the precision rate

Args:
    stop_loss (float): Stop-loss threshold.
    profit_taking (float): Profit-taking threshold.
    precision (float): Precision rate p.
    target_sharpe_ratio (float): Target annual Sharpe ratio.

Returns:
    float: Calculated frequency of bets.

#### `function binomial_sharpe_ratio`

```python
def binomial_sharpe_ratiostop_loss: float, profit_taking: float, frequency: float, probability: float:
```

> Calculate the Sharpe Ratio for a binary outcome.

The Sharpe ratio is calculated as follows:

.. math::
    SR = \\frac{(p - s) * p + s}{(p - s) * \\sqrt{p * (1 - p)}} * \\sqrt{f}

where:
- \(p\) is the profit-taking threshold
- \(s\) is the stop-loss threshold
- \(f\) is the frequency of bets per year

Args:
    stop_loss (float): Stop loss threshold.
    profit_taking (float): Profit taking threshold.
    frequency (float): Frequency of bets per year.
    probability (float): Probability of success.

Returns:
    float: Calculated Sharpe Ratio.

#### `function mix_gaussians`

```python
def mix_gaussiansmu1: float, mu2: float, sigma1: float, sigma2: float, probability: float, n_obs: int:
```

> Generate a mixture of Gaussian-distributed bet outcomes.

Args:
    mu1 (float): Mean of the first Gaussian distribution.
    mu2 (float): Mean of the second Gaussian distribution.
    sigma1 (float): Standard deviation of the first Gaussian distribution.
    sigma2 (float): Standard deviation of the second Gaussian distribution.
    probability (float): Probability of success.
    n_obs (int): Number of observations.

Returns:
    np.ndarray: Array of generated bet outcomes.

#### `function failure_probability`

```python
def failure_probabilityreturns: np.ndarray, frequency: float, target_sharpe_ratio: float:
```

> Calculate the probability that the strategy may fail.

Args:
    returns (np.ndarray): Array of returns.
    frequency (float): Number of bets per year.
    target_sharpe_ratio (float): Target annual Sharpe ratio.

Returns:
    float: Calculated failure probability.

#### `function calculate_strategy_risk`

```python
def calculate_strategy_riskmu1: float, mu2: float, sigma1: float, sigma2: float, probability: float, n_obs: int, frequency: float, target_sharpe_ratio: float:
```

> Calculate the strategy risk in practice.

Args:
    mu1 (float): Mean of the first Gaussian distribution.
    mu2 (float): Mean of the second Gaussian distribution.
    sigma1 (float): Standard deviation of the first Gaussian distribution.
    sigma2 (float): Standard deviation of the second Gaussian distribution.
    probability (float): Probability of success.
    n_obs (int): Number of observations.
    frequency (float): Number of bets per year.
    target_sharpe_ratio (float): Target annual Sharpe ratio.

Returns:
    float: Calculated probability of strategy failure.


### 📄 `RiskLabAI\backtest\test_set_overfitting.py`

#### `function expected_max_sharpe_ratio`

```python
def expected_max_sharpe_ration_trials: int, mean_sharpe_ratio: float, std_sharpe_ratio: float:
```

> Calculate the expected maximum Sharpe Ratio.

Uses the formula:
.. math::
    \text{sharpe\_ratio} = (\text{mean\_sharpe\_ratio} - \gamma) \times \Phi^{-1}(1 - \frac{1}{n\_trials}) + 
                           \gamma \times \Phi^{-1}(1 - n\_trials \times e^{-1})

where:
- \(\gamma\) is the Euler's gamma constant
- \(\Phi^{-1}\) is the inverse of the cumulative distribution function of the standard normal distribution

:param n_trials: Number of trials.
:param mean_sharpe_ratio: Mean Sharpe Ratio.
:param std_sharpe_ratio: Standard deviation of Sharpe Ratios.

:return: Expected maximum Sharpe Ratio.

#### `function generate_max_sharpe_ratios`

```python
def generate_max_sharpe_ratiosn_sims: int, n_trials_list: list, std_sharpe_ratio: float, mean_sharpe_ratio: float:
```

> Generate maximum Sharpe Ratios from simulations.

:param n_sims: Number of simulations.
:param n_trials_list: List of numbers of trials.
:param std_sharpe_ratio: Standard deviation of Sharpe Ratios.
:param mean_sharpe_ratio: Mean of Sharpe Ratios.

:return: DataFrame containing generated maximum Sharpe Ratios.

#### `function mean_std_error`

```python
def mean_std_errorn_sims0: int, n_sims1: int, n_trials: List[int], std_sharpe_ratio: float=1, mean_sharpe_ratio: float=0:
```

> Calculate mean and standard deviation of the predicted errors.

:param n_sims0: Number of max{SR} used to estimate E[max{SR}].
:param n_sims1: Number of errors on which std is computed.
:param n_trials: List of numbers of trials.
:param std_sharpe_ratio: Standard deviation of Sharpe Ratios.
:param mean_sharpe_ratio: Mean of Sharpe Ratios.

:return: DataFrame containing mean and standard deviation of errors.

#### `function estimated_sharpe_ratio_z_statistics`

```python
def estimated_sharpe_ratio_z_statisticssharpe_ratio: float, t: int, true_sharpe_ratio: float=0, skew: float=0, kurt: int=3:
```

> Calculate z statistics for the estimated Sharpe Ratios.

Uses the formula:
.. math::
    z = \frac{(sharpe\_ratio - true\_sharpe\_ratio) \times \sqrt{t - 1}}{\sqrt{1 - skew \times sharpe\_ratio + \frac{kurt - 1}{4} \times sharpe\_ratio^2}}

:param sharpe_ratio: Estimated Sharpe Ratio.
:param t: Number of observations.
:param true_sharpe_ratio: True Sharpe Ratio.
:param skew: Skewness of returns.
:param kurt: Kurtosis of returns.

:return: Calculated z statistics.

#### `function strategy_type1_error_probability`

```python
def strategy_type1_error_probabilityz: float, k: int=1:
```

> Calculate type I error probability of strategies.

.. math::
    \alpha_k = 1 - (1 - \alpha)^k

:param z: Z statistic for the estimated Sharpe Ratios.
:param k: Number of tests.

:return: Calculated type I error probability.

#### `function theta_for_type2_error`

```python
def theta_for_type2_errorsharpe_ratio: float, t: int, true_sharpe_ratio: float=0, skew: float=0, kurt: int=3:
```

> Calculate θ parameter for type II error probability.

.. math::
    \\theta = \\frac{\\text{true\_sharpe\_ratio} \cdot \\sqrt{t - 1}}{\\sqrt{1 - \\text{skew} \cdot \\text{sharpe\_ratio} + \\frac{\\text{kurt} - 1}{4} \cdot \\text{sharpe\_ratio}^2}}

:param sharpe_ratio: Estimated Sharpe Ratio.
:param t: Number of observations.
:param true_sharpe_ratio: True Sharpe Ratio.
:param skew: Skewness of returns.
:param kurt: Kurtosis of returns.

:return: Calculated θ parameter.

#### `function strategy_type2_error_probability`

```python
def strategy_type2_error_probabilityα_k: float, k: int, θ: float:
```

> Calculate type II error probability of strategies.

.. math::
    z = \text{ss.norm.ppf}((1 - \alpha_k)^{1.0 / k})
    \beta = \text{ss.norm.cdf}(z - \theta)

:param α_k: Type I error.
:param k: Number of tests.
:param θ: Calculated θ parameter.

:return: Calculated type II error probability.


### 📄 `RiskLabAI\backtest\validation\adaptive_combinatorial_purged.py`

#### `class AdaptiveCombinatorialPurged`

##### `method __init__`

```python
def __init__self, n_splits: int, n_test_groups: int, times: Union[pd.Series, Dict[str, pd.Series]], embargo: float=0, n_subsplits: int=3, external_feature: Union[pd.Series, Dict[str, pd.Series]]=None, lower_quantile: float=0.25, upper_quantile: float=0.75, subtract_border_adjustments: bool=True:
```

> Initialize the AdaptiveCombinatorialPurged class.

Parameters
----------
n_splits : int
    Number of splits/groups to partition the data into.
n_test_groups : int
    Size of the testing set in terms of groups.
times : Union[pd.Series, Dict[str, pd.Series]]
    The timestamp series associated with the labels.
embargo : float
    The embargo rate for purging.
n_subsplits : int
    Number of subsplits within each split segment.
external_feature : Union[pd.Series, Dict[str, pd.Series]]
    The external feature based on which the adaptive splitting is performed.
lower_quantile : float
    The lower quantile threshold for adjusting the split segments.
upper_quantile : float
    The upper quantile threshold for adjusting the split segments.
subtract_border_adjustments : bool
    Flag to determine whether to subtract border adjustments instead of adding.

##### `method _validate_input`

```python
def _validate_inputself, single_times: pd.Series, single_data: pd.DataFrame, single_external_feature: pd.Series:
```

> Validate that the input data, times, and external feature share the same index.

This function checks if the provided data, times, and external feature have the same index.
If they do not match, it raises a `ValueError`.

:param single_times: Time series data to be validated.
:type single_times: pd.Series
:param single_data: Dataset with which the times should align.
:type single_data: pd.DataFrame
:param single_external_feature: External feature series to be validated.
:type single_external_feature: pd.Series
:raises ValueError: If the indices of the data, times, and external feature do not match.
:return: None

##### `method _single_adaptive_split_segments`

```python
def _single_adaptive_split_segmentsself, indices: np.ndarray, single_external_feature: pd.Series:
```

> Adaptively split data indices based on the external feature's values and quantile thresholds.

Parameters
----------
indices : np.ndarray
    Array of data indices to be split.
single_external_feature : pd.Series
    The external feature based on which the adaptive splitting is performed.

Returns
-------
split_segments : List[np.ndarray]
    List of adaptively split data indices.

##### `method _single_split`

```python
def _single_splitself, single_times: pd.Series, single_data: pd.DataFrame, single_external_feature: pd.Series:
```

> Splits data into train and test indices based on the defined combinatorial splits.

This function is used to generate multiple train-test splits based on the combinatorial
cross-validation method. It ensures that each train-test split is properly purged and
embargoed to prevent data leakage.

:param single_times: Timestamp series associated with the labels.
:param single_data: The input data to be split.
:param single_external_feature: External feature series used for adaptive splitting.

:return: Generator that yields tuples of (train indices, test indices).

.. note:: The function validates the input, and uses combinatorial cross-validation method to
        produce the train-test splits.

##### `method split`

```python
def splitself, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], labels: Optional[Union[pd.Series, Dict[str, pd.Series]]]=None, groups: Optional[np.ndarray]=None:
```

> Split multiple datasets into train and test sets.

This function either splits a single dataset or multiple datasets considering
purging and embargo.

:param data: Dataset or dictionary of datasets.
:type data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
:param labels: Labels corresponding to the datasets, if available.
:type labels: Optional[Union[pd.Series, Dict[str, pd.Series]]]
:param groups: Group information, if available.
:type groups: Optional[np.ndarray]
:return: Train and test indices or key with train and test indices for multiple datasets.
:rtype: Union[Generator[Tuple[np.ndarray, np.ndarray], None, None],
            Generator[Tuple[str, Tuple[np.ndarray, np.ndarray]], None, None]]

##### `method _combinations_and_path_locations_and_split_segments`

```python
def _combinations_and_path_locations_and_split_segmentsself, data: pd.DataFrame, single_external_feature: pd.Series:
```

> Generate combinations, path locations, and split segments for the data.

This function is a helper that computes necessary components for combinatorial cross-validation.

:param data: The input dataframe to generate combinations, path locations, and split segments.
:param single_external_feature: External feature series used for adaptive splitting.

:return: Tuple containing combinations, path locations, and split segments.

.. math::
\text{combinations} = \binom{n}{k}

##### `method _single_backtest_paths`

```python
def _single_backtest_pathsself, single_times: pd.Series, single_data: pd.DataFrame, single_external_feature: pd.Series:
```

> Generate the backtest paths for given input data.

This function creates multiple backtest paths based on combinatorial splits, where
each path represents a sequence of train-test splits. It ensures that data leakage
is prevented by purging and applying embargo to the train-test splits.

:param single_times: Timestamp series associated with the data.
:param single_data: Input data on which the backtest paths are based.
:param single_external_feature: External feature series used for adaptive splitting.

:return: A dictionary where each key is a backtest path name, and the value is
        a list of dictionaries with train and test index arrays.

.. note:: This function relies on combinatorial cross-validation for backtesting to
        generate multiple paths of train-test splits.

##### `method backtest_paths`

```python
def backtest_pathsself, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
```

> Generate backtest paths for single or multiple datasets.

This function checks whether multiple datasets are being used. If so, it iterates through each
dataset, generating backtest paths using the `_single_backtest_paths` method. Otherwise, it directly
returns the backtest paths for the single dataset.

:param data: Input data on which the backtest paths are based.
            Can be either a single DataFrame or a dictionary of DataFrames for multiple datasets.
:type data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]

:return: A dictionary where each key is a backtest path name, and the value is
        a list of dictionaries with train and test index arrays. For multiple datasets,
        a nested dictionary structure is returned.
:rtype: Union[
    Dict[str, List[Dict[str, np.ndarray]]],
    Dict[str, Dict[str, List[Dict[str, List[np.ndarray]]]]]
]

##### `method _single_backtest_predictions`

```python
def _single_backtest_predictionsself, single_estimator: Any, single_times: pd.Series, single_data: pd.DataFrame, single_labels: pd.Series, single_weights: np.ndarray, single_external_feature: pd.Series, predict_probability: bool=False, n_jobs: int=1:
```

> Generate predictions for a single backtest using combinatorial splits.

This method calculates predictions across various paths created by combinatorial splits
of the data. For each combinatorial split, a separate estimator is trained and then used
to predict on the corresponding test set.

:param single_estimator: The machine learning model or estimator to be trained.
:param single_times: Timestamps corresponding to the data points.
:param single_data: Input data on which the model is trained and predictions are made.
:param single_labels: Labels corresponding to the input data.
:param single_weights: Weights for each data point.
:param single_external_feature: External feature series used for adaptive splitting.
:param predict_probability: If True, predict the probability of forecasts.
:type predict_probability: bool
:param n_jobs: Number of CPU cores to use for parallelization. Default is 1.

:return: A dictionary where keys are path names and values are arrays of predictions.

.. note:: This function relies on internal methods (e.g., `_get_train_indices`)
        to manage data splits and training.

.. note:: Parallelization is used to speed up the training of models for different splits.

##### `method backtest_predictions`

```python
def backtest_predictionsself, estimator: Union[Any, Dict[str, Any]], data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], labels: Union[pd.Series, Dict[str, pd.Series]], sample_weights: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]=None, predict_probability: bool=False, n_jobs: int=1:
```

> Generate backtest predictions for single or multiple datasets.

For each dataset, this function leverages the `_single_backtest_predictions` method to obtain
predictions for different train-test splits using the given estimator.

:param estimator: Model or estimator to be trained and used for predictions.
                Can be a single estimator or a dictionary of estimators for multiple datasets.
:type estimator: Union[Any, Dict[str, Any]]
:param data: Input data for training and testing. Can be a single dataset or
            a dictionary of datasets for multiple datasets.
:type data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
:param labels: Target labels for training and testing. Can be a single series or
            a dictionary of series for multiple datasets.
:type labels: Union[pd.Series, Dict[str, pd.Series]]
:param sample_weights: Weights for the observations in the dataset(s).
                    Can be a single array or a dictionary of arrays for multiple datasets.
                    Defaults to None, which means equal weights for all observations.
:type sample_weights: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]
:param predict_probability: If True, predict the probability of forecasts.
:type predict_probability: bool
:param n_jobs: The number of jobs to run in parallel. Default is 1.
:type n_jobs: int, optional
:return: Backtest predictions structured in a dictionary (or nested dictionaries for multiple datasets).
:rtype: Union[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]


### 📄 `RiskLabAI\backtest\validation\bagged_combinatorial_purged.py`

#### `class BaggedCombinatorialPurged`

##### `method __init__`

```python
def __init__self, n_splits: int, n_test_groups: int, times: Union[pd.Series, Dict[str, pd.Series]], embargo: float=0, classifier: bool=True, n_estimators: int=10, max_samples: float=1.0, max_features: float=1.0, bootstrap: bool=True, bootstrap_features: bool=False, random_state: int=None:
```

> Initialize the BaggedCombinatorialPurged class.

Parameters
----------
n_splits : int
    Number of splits/groups to partition the data into.
n_test_groups : int
    Size of the testing set in terms of groups.
times : Union[pd.Series, Dict[str, pd.Series]]
    The timestamp series associated with the labels.
embargo : float
    The embargo rate for purging.
classifier : bool
    Determines whether to use a BaggingClassifier or BaggingRegressor.
n_estimators : int
    The number of base estimators in the ensemble.
max_samples : float
    The number of samples to draw from X to train each base estimator.
max_features : float
    The number of features to draw from X to train each base estimator.
bootstrap : bool
    Whether samples are drawn with replacement.
bootstrap_features : bool
    Whether features are drawn with replacement.
random_state : int
    The seed used by the random number generator.

##### `method _single_backtest_predictions`

```python
def _single_backtest_predictionsself, single_estimator: Any, single_times: pd.Series, single_data: pd.DataFrame, single_labels: pd.Series, single_weights: np.ndarray, predict_probability: bool=False, n_jobs: int=1:
```

> Generate predictions for a single backtest using combinatorial splits with bagging.

This method calculates predictions across various paths created by combinatorial splits
of the data. For each combinatorial split, a bagged estimator is trained and then used
to predict on the corresponding test set.

:param single_estimator: The machine learning model or estimator to be trained.
:param single_times: Timestamps corresponding to the data points.
:param single_data: Input data on which the model is trained and predictions are made.
:param single_labels: Labels corresponding to the input data.
:param single_weights: Weights for each data point.
:param predict_probability: If True, predict the probability of forecasts.
:type predict_probability: bool
:param n_jobs: Number of CPU cores to use for parallelization. Default is 1.

:return: A dictionary where keys are path names and values are arrays of predictions.

.. note:: This function relies on internal methods (e.g., `_get_train_indices`)
        to manage data splits and training.

.. note:: Parallelization is used to speed up the training of models for different splits.


### 📄 `RiskLabAI\backtest\validation\combinatorial_purged.py`

#### `class CombinatorialPurged`

> Combinatorial Purged Cross-Validation (CPCV) implementation based on Marcos Lopez de Prado's method.

This class provides a cross-validation scheme that aims to address the main drawback of the Walk Forward
and traditional Cross-Validation methods by testing multiple paths. Given a number of backtest paths,
CPCV generates the precise number of combinations of training/testing sets needed to generate those paths,
while purging training observations that might contain leaked information.

Parameters
----------
n_splits : int
    Number of splits/groups to partition the data into.
n_test_groups : int
    Size of the testing set in terms of groups.
times : Union[pd.Series, Dict[str, pd.Series]]
    The timestamp series associated with the labels.
embargo : float
    The embargo rate for purging.

##### `method _path_locations`

```python
def _path_locationsn_splits: int, combinations_: List[Tuple[int]]:
```

> Generate a labeled path matrix and return path locations for N choose K.

This method generates a matrix where each entry corresponds to a specific combination of
training/testing sets, and helps in mapping these combinations to specific backtest paths.

Parameters
----------
n_splits : int
    Number of splits/groups to partition the data into.
combinations_ : list
    List of combinations for training/testing sets.

Returns
-------
dict
    A dictionary mapping each backtest path to its corresponding train/test combination.

##### `method _combinatorial_splits`

```python
def _combinatorial_splitscombinations_: List[Tuple[int]], split_segments: np.ndarray:
```

> Generate combinatorial test sets based on the number of test groups (n_test_groups).

This method creates test sets by considering all possible combinations of group splits, allowing
for the creation of multiple test paths, as described in the CPCV methodology.

Parameters
----------
combinations_ : list
    List of combinations for training/testing sets.
split_segments : np.ndarray
    Array of data split segments.

Returns
-------
Generator[np.ndarray]
    A generator yielding the combinatorial test sets.

##### `method __init__`

```python
def __init__self, n_splits: int, n_test_groups: int, times: Union[pd.Series, Dict[str, pd.Series]], embargo: float=0:
```

> Initialize the CombinatorialPurged class.

Parameters
----------
n_splits : int
    Number of splits/groups to partition the data into.
n_test_groups : int
    Size of the testing set in terms of groups.
times : Union[pd.Series, Dict[str, pd.Series]]
    The timestamp series associated with the labels.
embargo : float
    The embargo rate for purging.

##### `method get_n_splits`

```python
def get_n_splitsself, data: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]]=None, labels: Optional[Union[pd.Series, Dict[str, pd.Series]]]=None, groups: Optional[np.ndarray]=None:
```

> Return number of splits.

:param data: Dataset or dictionary of datasets.
:type data: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]]

:param labels: Labels or dictionary of labels.
:type labels: Optional[Union[pd.Series, Dict[str, pd.Series]]]

:param groups: Group labels for the samples.
:type groups: Optional[np.ndarray]

:return: Number of splits.
:rtype: int

##### `method _single_split`

```python
def _single_splitself, single_times: np.ndarray, single_data: np.ndarray:
```

> Splits data into train and test indices based on the defined combinatorial splits.

This function is used to generate multiple train-test splits based on the combinatorial
cross-validation method. It ensures that each train-test split is properly purged and
embargoed to prevent data leakage.

:param single_times: Timestamp series associated with the labels.
:param single_data: The input data to be split.

:return: Generator that yields tuples of (train indices, test indices).

.. note:: The function validates the input, and uses combinatorial cross-validation method to
        produce the train-test splits.

##### `method _combinations_and_path_locations_and_split_segments`

```python
def _combinations_and_path_locations_and_split_segmentsself, data: pd.DataFrame:
```

> Generate combinations, path locations, and split segments for the data.

This function is a helper that computes necessary components for combinatorial cross-validation.

:param data: The input dataframe to generate combinations, path locations, and split segments.

:return: Tuple containing combinations, path locations, and split segments.

.. math::
\text{combinations} = \binom{n}{k}

##### `method _single_backtest_paths`

```python
def _single_backtest_pathsself, single_times: pd.Series, single_data: pd.DataFrame:
```

> Generate the backtest paths for given input data.

This function creates multiple backtest paths based on combinatorial splits, where
each path represents a sequence of train-test splits. It ensures that data leakage
is prevented by purging and applying embargo to the train-test splits.

:param single_times: Timestamp series associated with the data.
:param single_data: Input data on which the backtest paths are based.

:return: A dictionary where each key is a backtest path name, and the value is
        a list of dictionaries with train and test index arrays.

.. note:: This function relies on combinatorial cross-validation for backtesting to
        generate multiple paths of train-test splits.

##### `method _single_backtest_predictions`

```python
def _single_backtest_predictionsself, single_estimator: Any, single_times: pd.Series, single_data: pd.DataFrame, single_labels: pd.Series, single_weights: np.ndarray, predict_probability: bool=False, n_jobs: int=1:
```

> Generate predictions for a single backtest using combinatorial splits.

This method calculates predictions across various paths created by combinatorial splits
of the data. For each combinatorial split, a separate estimator is trained and then used
to predict on the corresponding test set.

:param single_estimator: The machine learning model or estimator to be trained.
:param single_times: Timestamps corresponding to the data points.
:param single_data: Input data on which the model is trained and predictions are made.
:param single_labels: Labels corresponding to the input data.
:param single_weights: Weights for each data point.
:param predict_probability: If True, predict the probability of forecasts.
:type predict_probability: bool
:param n_jobs: Number of CPU cores to use for parallelization. Default is 1.

:return: A dictionary where keys are path names and values are arrays of predictions.

.. note:: This function relies on internal methods (e.g., `_get_train_indices`)
        to manage data splits and training.

.. note:: Parallelization is used to speed up the training of models for different splits.


### 📄 `RiskLabAI\backtest\validation\cross_validator_controller.py`

#### `class CrossValidatorController`

> Controller class to handle the cross-validation process.

##### `method __init__`

```python
def __init__self, validator_type: str, **kwargs:
```

> Initializes the CrossValidatorController.

:param validator_type: Type of cross-validator to create and use.
    This is passed to the factory to instantiate the appropriate cross-validator.
:type validator_type: str

:param kwargs: Additional keyword arguments to be passed to the cross-validator's constructor.
:type kwargs: Type


### 📄 `RiskLabAI\backtest\validation\cross_validator_factory.py`

#### `class CrossValidatorFactory`

> Factory class for creating cross-validator objects.

##### `method create_cross_validator`

```python
def create_cross_validatorvalidator_type: str, **kwargs:
```

> Factory method to create and return an instance of a cross-validator
based on the provided type.

:param validator_type: Type of cross-validator to create. Options include
    'kfold', 'walkforward', 'purgedkfold', and 'combinatorialpurged'.
:type validator_type: str

:param kwargs: Additional keyword arguments to be passed to the cross-validator's constructor.
:type kwargs: Type

:return: An instance of the specified cross-validator.
:rtype: CrossValidator

:raises ValueError: If an invalid validator type is provided.


### 📄 `RiskLabAI\backtest\validation\cross_validator_interface.py`

#### `class CrossValidator`

> Abstract Base Class (ABC) for cross-validation strategies.
Handles both single data inputs and dictionary inputs.

:param data: The input data, either as a single DataFrame or a dictionary of DataFrames.
:type data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]

:param labels: The labels corresponding to the data, either as a single Series or a dictionary of Series.
:type labels: Optional[Union[pd.Series, Dict[str, pd.Series]]]

:param groups: Optional group labels for stratified splitting.
:type groups: Optional[np.ndarray]

##### `method get_n_splits`

```python
def get_n_splitsself, data: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]]=None, labels: Optional[Union[pd.Series, Dict[str, pd.Series]]]=None, groups: Optional[np.ndarray]=None:
```

> Return number of splits.

:param data: Dataset or dictionary of datasets.
:type data: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]]

:param labels: Labels or dictionary of labels.
:type labels: Optional[Union[pd.Series, Dict[str, pd.Series]]]

:param groups: Group labels for the samples.
:type groups: Optional[np.ndarray]

:return: Number of splits.
:rtype: int

##### `method _single_split`

```python
def _single_splitself, single_data: pd.DataFrame:
```

> Splits a single data set into train-test indices.

This function provides train-test indices to split the data into train/test sets
by respecting the time order (if applicable) and the specified number of splits.

:param single_data: Input dataset.
:type single_data: pd.DataFrame

:return: Generator yielding train-test indices.
:rtype: Generator[Tuple[np.ndarray, np.ndarray], None, None]

##### `method split`

```python
def splitself, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], labels: Optional[Union[pd.Series, Dict[str, pd.Series]]]=None, groups: Optional[np.ndarray]=None:
```

> Splits data or a dictionary of data into train-test indices.

This function returns a generator that yields train-test indices. If a dictionary
of data is provided, the generator yields a key followed by the train-test indices.

:param data: Dataset or dictionary of datasets.
:type data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
:param labels: Labels or dictionary of labels.
:type labels: Optional[Union[pd.Series, Dict[str, pd.Series]]]
:param groups: Group labels for the samples.
:type groups: Optional[np.ndarray]

:return: Generator yielding either train-test indices directly or a key
        followed by train-test indices.
:rtype: Union[
    Generator[Tuple[np.ndarray, np.ndarray], None, None],
    Generator[Tuple[str, Tuple[np.ndarray, np.ndarray]], None, None]
]

##### `method _single_backtest_paths`

```python
def _single_backtest_pathsself, single_data: pd.DataFrame:
```

> Generates backtest paths for a single dataset.

This function creates and returns backtest paths (i.e., combinations of training and test sets)
for a single dataset by applying k-fold splitting or any other splitting strategy defined
by the `_single_split` function.

:param single_data: Input dataset.
:type single_data: pd.DataFrame

:return: Dictionary of backtest paths.
:rtype: Dict[str, List[Dict[str, List[np.ndarray]]]]

##### `method backtest_paths`

```python
def backtest_pathsself, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
```

> Generates backtest paths for data.

This function returns backtest paths for either a single dataset or a dictionary
of datasets. Each backtest path consists of combinations of training and test sets.

:param data: Dataset or dictionary of datasets.
:type data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
:param labels: Labels or dictionary of labels.
:type labels: Union[pd.Series, Dict[str, pd.Series]]

:return: Dictionary of backtest paths or dictionary of dictionaries for multiple datasets.
:rtype: Union[
    Dict[str, List[Dict[str, np.ndarray]]],
    Dict[str, Dict[str, List[Dict[str, List[np.ndarray]]]]]
]

##### `method _single_backtest_predictions`

```python
def _single_backtest_predictionsself, single_estimator: Any, single_data: pd.DataFrame, single_labels: pd.Series, single_weights: Optional[np.ndarray]=None, n_jobs: int=1:
```

> Obtain predictions for a single dataset during backtesting.

This function leverages parallel computation to train and predict on different train-test splits
of a single dataset using a given estimator. It utilizes the `_single_split` method to generate
the train-test splits.

:param single_estimator: Estimator or model to be trained and used for predictions.
:type single_estimator: Any
:param single_data: Data of the single dataset.
:type single_data: pd.DataFrame
:param single_labels: Labels corresponding to the single dataset.
:type single_labels: pd.Series
:param single_weights: Weights for the observations in the single dataset.
                    Defaults to equally weighted if not provided.
:type single_weights: np.ndarray, optional
:param n_jobs: The number of jobs to run in parallel. Default is 1.
:type n_jobs: int, optional
:return: Predictions structured in a dictionary for the backtest paths.
:rtype: Dict[str, np.ndarray]

##### `method backtest_predictions`

```python
def backtest_predictionsself, estimator: Union[Any, Dict[str, Any]], data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], labels: Union[pd.Series, Dict[str, pd.Series]], sample_weights: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]=None, predict_probability: bool=False, n_jobs: int=1:
```

> Generate backtest predictions for single or multiple datasets.

For each dataset, this function leverages the `_single_backtest_predictions` method to obtain
predictions for different train-test splits using the given estimator.

:param estimator: Model or estimator to be trained and used for predictions.
                Can be a single estimator or a dictionary of estimators for multiple datasets.
:type estimator: Union[Any, Dict[str, Any]]
:param data: Input data for training and testing. Can be a single dataset or
            a dictionary of datasets for multiple datasets.
:type data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
:param labels: Target labels for training and testing. Can be a single series or
            a dictionary of series for multiple datasets.
:type labels: Union[pd.Series, Dict[str, pd.Series]]
:param sample_weights: Weights for the observations in the dataset(s).
                    Can be a single array or a dictionary of arrays for multiple datasets.
                    Defaults to None, which means equal weights for all observations.
:type sample_weights: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]
:param n_jobs: The number of jobs to run in parallel. Default is 1.
:type n_jobs: int, optional
:return: Backtest predictions structured in a dictionary (or nested dictionaries for multiple datasets).
:rtype: Union[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]


### 📄 `RiskLabAI\backtest\validation\kfold.py`

#### `class KFold`

> K-Fold cross-validator.

This class implements the K-Fold cross-validation strategy, where the dataset is
divided into `k` consecutive folds. Each fold is then used once as a validation set
while the `k - 1` remaining folds form the training set.

##### `method __init__`

```python
def __init__self, n_splits: int, shuffle: bool=False, random_seed: int=None:
```

> Initialize the K-Fold cross-validator.

:param n_splits: Number of splits or folds for the cross-validation.
                 The dataset will be divided into `n_splits` consecutive parts.
:type n_splits: int
:param shuffle: Whether to shuffle the data before splitting it into folds.
                If `shuffle` is set to True, the data will be shuffled before splitting.
:type shuffle: bool, optional
:param random_seed: Seed used for random shuffling. Set this seed for reproducibility.
                    Only used when `shuffle` is True.
:type random_seed: int, optional

##### `method get_n_splits`

```python
def get_n_splitsself, data: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]]=None, labels: Optional[Union[pd.Series, Dict[str, pd.Series]]]=None, groups: Optional[np.ndarray]=None:
```

> Return number of splits.

:param data: Dataset or dictionary of datasets.
:type data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]

:param labels: Labels or dictionary of labels.
:type labels: Optional[Union[pd.Series, Dict[str, pd.Series]]]

:param groups: Group labels for the samples.
:type groups: Optional[np.ndarray]

:return: Number of splits.
:rtype: int

##### `method _single_split`

```python
def _single_splitself, single_data: pd.DataFrame:
```

> Splits a single data set into train-test indices.

This function provides train-test indices to split the data into train/test sets
by respecting the time order (if applicable) and the specified number of splits.

:param single_data: Input dataset.
:type single_data: pd.DataFrame

:return: Generator yielding train-test indices.
:rtype: Generator[Tuple[np.ndarray, np.ndarray], None, None]

##### `method split`

```python
def splitself, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], labels: Optional[Union[pd.Series, Dict[str, pd.Series]]]=None, groups: Optional[np.ndarray]=None:
```

> Splits data or a dictionary of data into train-test indices.

This function returns a generator that yields train-test indices. If a dictionary
of data is provided, the generator yields a key followed by the train-test indices.

:param data: Dataset or dictionary of datasets.
:type data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
:param labels: Labels or dictionary of labels.
:type labels: Optional[Union[pd.Series, Dict[str, pd.Series]]]
:param groups: Group labels for the samples.
:type groups: Optional[np.ndarray]

:return: Generator yielding either train-test indices directly or a key
        followed by train-test indices.
:rtype: Union[
    Generator[Tuple[np.ndarray, np.ndarray], None, None],
    Generator[Tuple[str, Tuple[np.ndarray, np.ndarray]], None, None]
]

##### `method _single_backtest_paths`

```python
def _single_backtest_pathsself, single_data: pd.DataFrame:
```

> Generates backtest paths for a single dataset.

This function creates and returns backtest paths (i.e., combinations of training and test sets)
for a single dataset by applying k-fold splitting or any other splitting strategy defined
by the `_single_split` function.

:param single_data: Input dataset.
:type single_data: pd.DataFrame

:return: Dictionary of backtest paths.
:rtype: Dict[str, List[Dict[str, List[np.ndarray]]]]

##### `method backtest_paths`

```python
def backtest_pathsself, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
```

> Generates backtest paths for data.

This function returns backtest paths for either a single dataset or a dictionary
of datasets. Each backtest path consists of combinations of training and test sets.

:param data: Dataset or dictionary of datasets.
:type data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]

:return: Dictionary of backtest paths or dictionary of dictionaries for multiple datasets.
:rtype: Union[
    Dict[str, List[Dict[str, np.ndarray]]],
    Dict[str, Dict[str, List[Dict[str, List[np.ndarray]]]]]
]

##### `method _single_backtest_predictions`

```python
def _single_backtest_predictionsself, single_estimator: Any, single_data: pd.DataFrame, single_labels: pd.Series, single_weights: Optional[np.ndarray]=None, predict_probability: bool=False, n_jobs: int=1:
```

> Obtain predictions for a single dataset during backtesting.

This function leverages parallel computation to train and predict on different train-test splits
of a single dataset using a given estimator. It utilizes the `_single_split` method to generate
the train-test splits.

:param single_estimator: Estimator or model to be trained and used for predictions.
:type single_estimator: Any
:param single_data: Data of the single dataset.
:type single_data: pd.DataFrame
:param single_labels: Labels corresponding to the single dataset.
:type single_labels: pd.Series
:param single_weights: Weights for the observations in the single dataset.
                    Defaults to equally weighted if not provided.
:type single_weights: np.ndarray, optional
:param predict_probability: If True, predict the probability of forecasts.
:type predict_probability: bool
:param n_jobs: The number of jobs to run in parallel. Default is 1.
:type n_jobs: int, optional
:return: Predictions structured in a dictionary for the backtest paths.
:rtype: Dict[str, np.ndarray]

##### `method backtest_predictions`

```python
def backtest_predictionsself, estimator: Union[Any, Dict[str, Any]], data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], labels: Union[pd.Series, Dict[str, pd.Series]], sample_weights: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]=None, predict_probability: bool=False, n_jobs: int=1:
```

> Generate backtest predictions for single or multiple datasets.

For each dataset, this function leverages the `_single_backtest_predictions` method to obtain
predictions for different train-test splits using the given estimator.

:param estimator: Model or estimator to be trained and used for predictions.
                Can be a single estimator or a dictionary of estimators for multiple datasets.
:type estimator: Union[Any, Dict[str, Any]]
:param data: Input data for training and testing. Can be a single dataset or
            a dictionary of datasets for multiple datasets.
:type data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
:param labels: Target labels for training and testing. Can be a single series or
            a dictionary of series for multiple datasets.
:type labels: Union[pd.Series, Dict[str, pd.Series]]
:param sample_weights: Weights for the observations in the dataset(s).
                    Can be a single array or a dictionary of arrays for multiple datasets.
                    Defaults to None, which means equal weights for all observations.
:type sample_weights: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]
:param predict_probability: If True, predict the probability of forecasts.
:type predict_probability: bool
:param n_jobs: The number of jobs to run in parallel. Default is 1.
:type n_jobs: int, optional

:return: Backtest predictions structured in a dictionary (or nested dictionaries for multiple datasets).
:rtype: Union[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]


### 📄 `RiskLabAI\backtest\validation\purged_kfold.py`

#### `class PurgedKFold`

##### `method filtered_training_indices_with_embargo`

```python
def filtered_training_indices_with_embargodata_info_range: pd.Series, test_time_range: pd.Series, embargo_fraction: float=0, continous_test_times: bool=False:
```

> Purge observations in the training set with embargo.

Finds the training set indices based on the information on each record
and the test set range. It purges the training set of observations that
overlap with the test set in the time dimension and adds an embargo period
to further prevent potential information leakage.

.. math::
    \text{embargo\_length} = \text{len(data\_info\_range)} \times \text{embargo\_fraction}

:param data_info_range: Series detailing the information range for each record.
    - *data_info_range.index*: Time when the information extraction started.
    - *data_info_range.value*: Time when the information extraction ended.
:type data_info_range: pd.Series
:param test_time_range: Series containing times for the test dataset.
:type test_time_range: pd.Series
:param embargo_fraction: Fraction of the dataset trailing the test observations to exclude from training.
:type embargo_fraction: float
:param continuous_test_times: If set to True, considers the test time range as continuous.
:type continuous_test_times: bool

:return: Series of filtered training data after applying embargo.
:rtype: pd.Series

##### `method __init__`

```python
def __init__self, n_splits: int, times: Union[pd.Series, Dict[str, pd.Series]], embargo: float=0:
```

> Purged k-fold cross-validation to prevent information leakage.

Implements a cross-validation strategy where each fold is purged
of observations overlapping with the training set in the time dimension.
An embargo period is also introduced to further prevent potential
information leakage.

Attributes:
    n_splits (int): Number of splits/folds.
    times (Union[pd.Series, Dict[str, pd.Series]]): Series or dict containing time data.
    embargo (float): The embargo period.
    is_multiple_datasets (bool): True if `times` is a dict, else False.

:param n_splits: Number of splits or folds.
:type n_splits: int

:param times: Series detailing the information range for each record.
    - *times.index*: Time when the information extraction started.
    - *times.value*: Time when the information extraction ended.
:type times: pd.Series

:param embargo: The embargo period to further prevent potential
                information leakage.
:type embargo: float

##### `method get_n_splits`

```python
def get_n_splitsself, data: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]]=None, labels: Optional[Union[pd.Series, Dict[str, pd.Series]]]=None, groups: Optional[np.ndarray]=None:
```

> Return number of splits.

:param data: Dataset or dictionary of datasets.
:type data: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]]

:param labels: Labels or dictionary of labels.
:type labels: Optional[Union[pd.Series, Dict[str, pd.Series]]]

:param groups: Group labels for the samples.
:type groups: Optional[np.ndarray]

:return: Number of splits.
:rtype: int

##### `method _validate_input`

```python
def _validate_inputself, single_times: pd.Series, single_data: pd.DataFrame:
```

> Validate that the input data and times share the same index.

This function checks if the provided data and its corresponding times
have the same index. If they do not match, it raises a `ValueError`.

:param single_times: Time series data to be validated.
:type single_times: pd.Series
:param single_data: Dataset with which the times should align.
:type single_data: pd.DataFrame
:raises ValueError: If the indices of the data and times do not match.
:return: None

##### `method _get_train_indices`

```python
def _get_train_indicesself, test_indices: np.ndarray, single_times: pd.Series, continous_test_times: bool=False:
```

> Obtain the training indices considering purging and embargo.

This function retrieves the training set indices based on the given test indices
while considering the purging and embargo strategy.

:param test_indices: Indices used for the test set.
:type test_indices: np.ndarray
:param single_times: Time series data used for purging and embargo.
:type single_times: pd.Series
:return: Training indices after applying purging and embargo.
:rtype: np.ndarray

##### `method _single_split`

```python
def _single_splitself, single_times: pd.Series, single_data: pd.DataFrame:
```

> Split the data into train and test sets.

This function splits the data for a single dataset considering purging and embargo.

:param single_times: Time series data used for purging and embargo.
:type single_times: pd.Series
:param single_data: Dataset to split.
:type single_data: pd.DataFrame
:return: Train and test indices.
:rtype: Generator[Tuple[np.ndarray, np.ndarray], None, None]

##### `method split`

```python
def splitself, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], labels: Optional[Union[pd.Series, Dict[str, pd.Series]]]=None, groups: Optional[np.ndarray]=None:
```

> Split multiple datasets into train and test sets.

This function either splits a single dataset or multiple datasets considering
purging and embargo.

:param data: Dataset or dictionary of datasets.
:type data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
:param labels: Labels corresponding to the datasets, if available.
:type labels: Optional[Union[pd.Series, Dict[str, pd.Series]]]
:param groups: Group information, if available.
:type groups: Optional[np.ndarray]
:return: Train and test indices or key with train and test indices for multiple datasets.
:rtype: Union[Generator[Tuple[np.ndarray, np.ndarray], None, None],
            Generator[Tuple[str, Tuple[np.ndarray, np.ndarray]], None, None]]

##### `method _single_backtest_paths`

```python
def _single_backtest_pathsself, single_times: pd.Series, single_data: pd.DataFrame:
```

> Generate backtest paths based on training and testing indices.

This function first validates the input data and times. Then, it generates
the training and testing indices for backtesting. These paths are organized
into a dictionary with a designated name for each backtest path.

:param single_times: Time series data for validation.
:type single_times: pd.Series
:param single_data: Dataset with which the times should align.
:type single_data: pd.DataFrame
:return: Dictionary containing the backtest paths with training and testing indices.
:rtype: Dict[str, List[Dict[str, np.ndarray]]]

##### `method backtest_paths`

```python
def backtest_pathsself, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
```

> Generate backtest paths for single or multiple datasets.

This function checks whether multiple datasets are being used. If so, it iterates through each
dataset, generating backtest paths using the `_single_backtest_paths` method. Otherwise, it directly
returns the backtest paths for the single dataset.

:param data: Input data on which the backtest paths are based.
            Can be either a single DataFrame or a dictionary of DataFrames for multiple datasets.
:type data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]

:return: A dictionary where each key is a backtest path name, and the value is
        a list of dictionaries with train and test index arrays. For multiple datasets,
        a nested dictionary structure is returned.
:rtype: Union[
    Dict[str, List[Dict[str, np.ndarray]]],
    Dict[str, Dict[str, List[Dict[str, List[np.ndarray]]]]]
]

##### `method _single_backtest_predictions`

```python
def _single_backtest_predictionsself, single_estimator: Any, single_times: pd.Series, single_data: pd.DataFrame, single_labels: pd.Series, single_weights: Optional[np.ndarray]=None, predict_probability: bool=False, n_jobs: int=1:
```

> Obtain predictions for a single dataset during backtesting.

This function leverages parallel computation to train and predict on different train-test splits
of a single dataset using a given estimator. It utilizes the `_single_split` method to generate
the train-test splits.

:param single_estimator: Estimator or model to be trained and used for predictions.
:type single_estimator: Any
:param single_times: Timestamps for the single dataset.
:type single_times: pd.Series
:param single_data: Data of the single dataset.
:type single_data: pd.DataFrame
:param single_labels: Labels corresponding to the single dataset.
:type single_labels: pd.Series
:param single_weights: Weights for the observations in the single dataset.
                    Defaults to equally weighted if not provided.
:type single_weights: np.ndarray, optional
:param predict_probability: If True, predict the probability of forecasts.
:type predict_probability: bool
:param n_jobs: The number of jobs to run in parallel. Default is 1.
:type n_jobs: int, optional
:return: Predictions structured in a dictionary for the backtest paths.
:rtype: Dict[str, np.ndarray]

##### `method backtest_predictions`

```python
def backtest_predictionsself, estimator: Union[Any, Dict[str, Any]], data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], labels: Union[pd.Series, Dict[str, pd.Series]], sample_weights: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]=None, predict_probability: bool=False, n_jobs: int=1:
```

> Generate backtest predictions for single or multiple datasets.

For each dataset, this function leverages the `_single_backtest_predictions` method to obtain
predictions for different train-test splits using the given estimator.

:param estimator: Model or estimator to be trained and used for predictions.
                Can be a single estimator or a dictionary of estimators for multiple datasets.
:type estimator: Union[Any, Dict[str, Any]]
:param data: Input data for training and testing. Can be a single dataset or
            a dictionary of datasets for multiple datasets.
:type data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
:param labels: Target labels for training and testing. Can be a single series or
            a dictionary of series for multiple datasets.
:type labels: Union[pd.Series, Dict[str, pd.Series]]
:param sample_weights: Weights for the observations in the dataset(s).
                    Can be a single array or a dictionary of arrays for multiple datasets.
                    Defaults to None, which means equal weights for all observations.
:type sample_weights: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]
:param predict_probability: If True, predict the probability of forecasts.
:type predict_probability: bool
:param n_jobs: The number of jobs to run in parallel. Default is 1.
:type n_jobs: int, optional
:return: Backtest predictions structured in a dictionary (or nested dictionaries for multiple datasets).
:rtype: Union[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]


### 📄 `RiskLabAI\backtest\validation\walk_forward.py`

#### `class WalkForward`

> WalkForward Cross-Validator for Time Series Data.

This cross-validator provides train/test indices meant to split time series data 
in a "walk-forward" manner, which is suitable for time series forecasting tasks. 
In each split, the training set progressively grows in size (subject to the optional
maximum size constraint) while the test set remains roughly constant in size. 
A gap can be optionally introduced between the training and test set to simulate 
forecasting on unseen future data after a certain interval.

The WalkForward cross-validator is inherently different from traditional K-Fold
cross-validation which shuffles and splits the dataset into train/test without 
considering the time order. In time series tasks, ensuring that the model is 
trained on past data and validated on future data is crucial. This cross-validator 
achieves that by progressively walking forward in time through the dataset.

##### `method __init__`

```python
def __init__self, n_splits: int=5, max_train_size: int=None, gap: int=0:
```

> Initialize the TimeSeriesWalkForward cross-validator.

Parameters:
-----------
n_splits : int, default=5
    Number of splits/folds. Must be at least 2.

max_train_size : int, optional
    Maximum number of observations allowed in the training dataset.
    If provided, the most recent `max_train_size` observations are used 
    for training.

gap : int, default=0
    Number of observations to skip between the end of the training data 
    and the start of the test data. Useful for simulating forecasting 
    scenarios where the test data is not immediately after the training data.

##### `method _single_split`

```python
def _single_splitself, single_data: pd.DataFrame:
```

> Splits a single data set into train-test indices.

This function provides train-test indices to split the data into train/test sets
by respecting the time order (if applicable) and the specified number of splits.

:param single_data: Input dataset.
:type single_data: pd.DataFrame

:return: Generator yielding train-test indices.
:rtype: Generator[Tuple[np.ndarray, np.ndarray], None, None]

##### `method _single_backtest_predictions`

```python
def _single_backtest_predictionsself, single_estimator: Any, single_data: pd.DataFrame, single_labels: pd.Series, single_weights: Optional[np.ndarray]=None, predict_probability: bool=False, n_jobs: int=1:
```

> Obtain predictions for a single dataset during backtesting.

This function leverages parallel computation to train and predict on different train-test splits
of a single dataset using a given estimator. It utilizes the `_single_split` method to generate
the train-test splits.

:param single_estimator: Estimator or model to be trained and used for predictions.
:type single_estimator: Any
:param single_data: Data of the single dataset.
:type single_data: pd.DataFrame
:param single_labels: Labels corresponding to the single dataset.
:type single_labels: pd.Series
:param single_weights: Weights for the observations in the single dataset.
                    Defaults to equally weighted if not provided.
:type single_weights: np.ndarray, optional
:param predict_probability: If True, predict the probability of forecasts.
:type predict_probability: bool
:param n_jobs: The number of jobs to run in parallel. Default is 1.
:type n_jobs: int, optional
:return: Predictions structured in a dictionary for the backtest paths.
:rtype: Dict[str, np.ndarray]


### 📄 `RiskLabAI\cluster\clustering.py`

#### `function covariance_to_correlation`

```python
def covariance_to_correlationcovariance: np.ndarray:
```

> Derive the correlation matrix from a covariance matrix.

.. math::
    \\text{correlation}_{ij} = \\frac{\\text{covariance}_{ij}}{\\sqrt{\\text{covariance}_{ii} \\text{covariance}_{jj}}}

Reference:
    De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    Snippet 2.3, Page 27

:param covariance: Covariance matrix.

:return: Correlation matrix.

#### `function cluster_k_means_base`

```python
def cluster_k_means_basecorrelation: pd.DataFrame, max_clusters: int=10, iterations: int=10:
```

> Clustering using K-Means.

Reference:
    De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    Snippet 4.1, Page 56

:param correlation: Correlation matrix.
:param max_clusters: Maximum number of clusters.
:param iterations: Number of iterations for clustering.

:return: Tuple containing the sorted correlation matrix, clusters, and silhouette scores.

#### `function make_new_outputs`

```python
def make_new_outputscorrelation: pd.DataFrame, clusters_1: dict, clusters_2: dict:
```

> Merge two clusters and produce new correlation matrix and silhouette scores.
Clusters are disjoint.

Reference:
    De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    Snippet 4.2, Page 58

:param correlation: Correlation matrix.
:param clusters_1: First cluster.
:param clusters_2: Second cluster.

:return: Tuple containing the new correlation matrix, new clusters, and new silhouette scores.

#### `function cluster_k_means_top`

```python
def cluster_k_means_topcorrelation: pd.DataFrame, max_clusters: int=None, iterations: int=10:
```

> Clustering using ONC method.

Reference:
    De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    Snippet 4.2, Page 58

:param correlation: Correlation matrix.
:param max_clusters: Maximum Number of clusters.
:param iterations: Number of iterations.

:return: Tuple containing the sorted correlation matrix, clusters, and silhouette scores.

#### `function random_covariance_sub`

```python
def random_covariance_subn_observations: int, n_columns: int, sigma: float, random_state: int=None:
```

> Generates covariance matrix for n_columns same normal random variables with a nomral noise scaled by sigma.
Variables have n_observations observations.

Reference:
    De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    Snippet 4.3, Page 61

:param n_observations: Number of observations.
:param n_columns: Number of columns.
:param sigma: Sigma for normal distribution.
:param random_state: Random state for reproducibility.

:return: Sub covariance matrix.

#### `function random_block_covariance`

```python
def random_block_covariancen_columns: int, n_blocks: int, block_size_min: int=1, sigma: float=1.0, random_state: int=None:
```

> Compute random block covariance matrix.

Reference:
    De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    Snippet 4.3, Page 61

:param n_columns: Number of columns.
:param n_blocks: Number of blocks.
:param block_size_min: Minimum size of block.
:param sigma: Sigma for normal distribution.
:param random_state: Random state for reproducibility.

:return: Random block covariance matrix.

#### `function random_block_correlation`

```python
def random_block_correlationn_columns: int, n_blocks: int, random_state: int=None, block_size_min: int=1:
```

> Compute random block correlation matrix.

Reference:
    De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    Snippet 4.3, Page 61

:param n_columns: Number of columns.
:param n_blocks: Number of blocks.
:param random_state: Random state for reproducibility.
:param block_size_min: Minimum size of block.

:return: Random block correlation matrix.


### 📄 `RiskLabAI\controller\bars_initializer.py`

#### `class BarsInitializerController`

> Controller for initializing various types of bars.

##### `method __init__`

```python
def __init__self:
```
##### `method initialize_expected_dollar_imbalance_bars`

```python
def initialize_expected_dollar_imbalance_barswindow_size_for_expected_n_ticks_estimation: int=3, window_size_for_expected_imbalance_estimation: int=10000, initial_estimate_of_expected_n_ticks_in_bar: int=20000, expected_ticks_number_bounds: Tuple[float]=None, analyze_thresholds: bool=False:
```

> Initialize expected dollar imbalance bars.

:param window_size_for_expected_n_ticks_estimation: The window size for estimating the expected number of ticks.
:param window_size_for_expected_imbalance_estimation: The window size for estimating the expected imbalance.
:param initial_estimate_of_expected_n_ticks_in_bar: The initial estimate for the expected number of ticks in a bar.
:param expected_ticks_number_bounds: Bounds for the expected number of ticks in a bar.
:param analyze_thresholds: Flag indicating whether to analyze thresholds.

:return: An instance of ExpectedImbalanceBars.

##### `method initialize_expected_volume_imbalance_bars`

```python
def initialize_expected_volume_imbalance_barswindow_size_for_expected_n_ticks_estimation: int=3, window_size_for_expected_imbalance_estimation: int=10000, initial_estimate_of_expected_n_ticks_in_bar: int=20000, expected_ticks_number_bounds: Tuple[float]=None, analyse_thresholds: bool=False:
```
##### `method initialize_expected_tick_imbalance_bars`

```python
def initialize_expected_tick_imbalance_barswindow_size_for_expected_n_ticks_estimation: int=3, window_size_for_expected_imbalance_estimation: int=10000, initial_estimate_of_expected_n_ticks_in_bar: int=20000, expected_ticks_number_bounds: Tuple[float]=None, analyse_thresholds: bool=False:
```
##### `method initialize_fixed_dollar_imbalance_bars`

```python
def initialize_fixed_dollar_imbalance_barswindow_size_for_expected_imbalance_estimation: int=10000, initial_estimate_of_expected_n_ticks_in_bar: int=20000, analyse_thresholds: bool=False:
```
##### `method initialize_fixed_volume_imbalance_bars`

```python
def initialize_fixed_volume_imbalance_barswindow_size_for_expected_imbalance_estimation: int=10000, initial_estimate_of_expected_n_ticks_in_bar: int=20000, analyse_thresholds: bool=False:
```
##### `method initialize_fixed_tick_imbalance_bars`

```python
def initialize_fixed_tick_imbalance_barswindow_size_for_expected_imbalance_estimation: int=10000, initial_estimate_of_expected_n_ticks_in_bar: int=20000, analyse_thresholds: bool=False:
```
##### `method initialize_expected_dollar_run_bars`

```python
def initialize_expected_dollar_run_barswindow_size_for_expected_n_ticks_estimation: int=3, window_size_for_expected_imbalance_estimation: int=10000, initial_estimate_of_expected_n_ticks_in_bar: int=20000, expected_ticks_number_bounds: Tuple[float]=None, analyse_thresholds: bool=False:
```
##### `method initialize_expected_volume_run_bars`

```python
def initialize_expected_volume_run_barswindow_size_for_expected_n_ticks_estimation: int=3, window_size_for_expected_imbalance_estimation: int=10000, initial_estimate_of_expected_n_ticks_in_bar: int=20000, expected_ticks_number_bounds: Tuple[float]=None, analyse_thresholds: bool=False:
```
##### `method initialize_expected_tick_run_bars`

```python
def initialize_expected_tick_run_barswindow_size_for_expected_n_ticks_estimation: int=3, window_size_for_expected_imbalance_estimation: int=10000, initial_estimate_of_expected_n_ticks_in_bar: int=20000, expected_ticks_number_bounds: Tuple[float]=None, analyse_thresholds: bool=False:
```
##### `method initialize_fixed_dollar_run_bars`

```python
def initialize_fixed_dollar_run_barswindow_size_for_expected_n_ticks_estimation: int=3, window_size_for_expected_imbalance_estimation: int=10000, initial_estimate_of_expected_n_ticks_in_bar: int=20000, analyse_thresholds: bool=False:
```
##### `method initialize_fixed_volume_run_bars`

```python
def initialize_fixed_volume_run_barswindow_size_for_expected_n_ticks_estimation: int=3, window_size_for_expected_imbalance_estimation: int=10000, initial_estimate_of_expected_n_ticks_in_bar: int=20000, analyse_thresholds: bool=False:
```
##### `method initialize_fixed_tick_run_bars`

```python
def initialize_fixed_tick_run_barswindow_size_for_expected_n_ticks_estimation: int=3, window_size_for_expected_imbalance_estimation: int=10000, initial_estimate_of_expected_n_ticks_in_bar: int=20000, analyse_thresholds: bool=False:
```
##### `method initialize_dollar_standard_bars`

```python
def initialize_dollar_standard_barsthreshold: Union[float, pd.Series]=70000000:
```
##### `method initialize_volume_standard_bars`

```python
def initialize_volume_standard_barsthreshold: Union[float, pd.Series]=30000:
```
##### `method initialize_tick_standard_bars`

```python
def initialize_tick_standard_barsthreshold: Union[float, pd.Series]=6000:
```
##### `method initialize_time_bars`

```python
def initialize_time_barsresolution_type: str='D', resolution_units: int=1:
```

### 📄 `RiskLabAI\controller\data_structure_controller.py`

#### `class Controller`

##### `method __init__`

```python
def __init__self:
```
##### `method handle_input_command`

```python
def handle_input_commandself, method_name: str, method_arguments: dict, input_data: Union[str, pd.DataFrame], output_path: Optional[str]=None, batch_size: int=1000000:
```

> Handles the input command to initialize bars and run on batches.

:param method_name: Name of the method to call
:param method_arguments: Arguments for the method
:param input_data: Input data as a DataFrame or string path
:param output_path: Optional path to save results as CSV
:param batch_size: Size of each batch to process
:return: DataFrame of aggregated bars

##### `method run_on_batches`

```python
def run_on_batchesself, initialized_bars: AbstractBars, input_data: Union[str, pd.DataFrame], batch_size: int, output_path: Optional[str]=None:
```

> Runs the initialized bars on batches of data.

:param initialized_bars: Initialized bars object
:param input_data: Input data as DataFrame or string path
:param batch_size: Size of each batch to process
:param output_path: Optional path to save results as CSV
:return: DataFrame of aggregated bars

##### `method construct_bars_from_batch`

```python
def construct_bars_from_batchbars: AbstractBars, data: pd.DataFrame:
```

> Construct bars from a single batch of data.

:param bars: Initialized bars object
:param data: Data for this batch as a DataFrame
:return: List of constructed bars

##### `method read_batches_from_string`

```python
def read_batches_from_stringinput_path: str, batch_size: int:
```

> Reads data in batches from a CSV file.

:param input_path: File path to read from
:param batch_size: Size of each batch
:return: Generator yielding batches of data

##### `method read_batches_from_dataframe`

```python
def read_batches_from_dataframeinput_data: pd.DataFrame, batch_size: int:
```

> Reads data in batches from a DataFrame.

:param input_data: DataFrame to read from
:param batch_size: Size of each batch
:return: Generator yielding batches of data


### 📄 `RiskLabAI\data\denoise\denoising.py`

#### `function marcenko_pastur_pdf`

```python
def marcenko_pastur_pdfvariance: float, q: float, num_points: int:
```

> Computes the Marcenko-Pastur probability density function (pdf).

:param variance: Variance of the observations
:type variance: float
:param q: Ratio T/N
:type q: float
:param num_points: Number of points in the pdf
:type num_points: int
:return: The Marcenko-Pastur pdf as a pandas Series
:rtype: pd.Series

The Marcenko-Pastur pdf is given by the formula:
.. math::
   \frac{q}{{2 \pi \sigma \lambda}} \sqrt{(\lambda_{max} - \lambda)(\lambda - \lambda_{min})}

where:
- :math:`\lambda` is the eigenvalue
- :math:`\sigma` is the variance of the observations
- :math:`q` is the ratio T/N
- :math:`\lambda_{max}` and :math:`\lambda_{min}` are the maximum and minimum eigenvalues respectively

#### `function pca`

```python
def pcamatrix: np.ndarray:
```

> Computes the principal component analysis of a Hermitian matrix.

:param matrix: Hermitian matrix
:type matrix: np.ndarray
:return: Eigenvalues and eigenvectors
:rtype: Tuple[np.ndarray, np.ndarray]

The principal component analysis is computed using the eigen decomposition of the Hermitian matrix.

#### `function fit_kde`

```python
def fit_kdeobservations: Union[np.ndarray, pd.Series], bandwidth: float=0.25, kernel: str='gaussian', x: Optional[Union[np.ndarray, pd.Series]]=None:
```

> Fit a kernel density estimator to a series of observations.

:param observations: Series of observations
:type observations: Union[np.ndarray, pd.Series]
:param bandwidth: Bandwidth of the kernel
:type bandwidth: float
:param kernel: Type of kernel to use (e.g., 'gaussian')
:type kernel: str
:param x: Array of values on which the fit KDE will be evaluated
:type x: Optional[Union[np.ndarray, pd.Series]]
:return: Kernel density estimate as a pandas Series
:rtype: pd.Series

#### `function random_cov`

```python
def random_covnum_columns: int, num_factors: int:
```

> Generate a random covariance matrix.

:param num_columns: Number of columns in the covariance matrix
:type num_columns: int
:param num_factors: Number of factors for random covariance matrix
:type num_factors: int
:return: Random covariance matrix
:rtype: np.ndarray

#### `function cov_to_corr`

```python
def cov_to_corrcov: np.ndarray:
```

> Convert a covariance matrix to a correlation matrix.

:param cov: Covariance matrix
:type cov: np.ndarray
:return: Correlation matrix
:rtype: np.ndarray

#### `function error_pdfs`

```python
def error_pdfsvariance: float, eigenvalues: np.ndarray, q: float, bandwidth: float, num_points: int=1000:
```

> Computes the sum of squared errors between the theoretical and empirical PDFs.

:param variance: Variance of the observations
:type variance: float
:param eigenvalues: Eigenvalues of the correlation matrix
:type eigenvalues: np.ndarray
:param q: Ratio T/N
:type q: float
:param bandwidth: Bandwidth of the kernel
:type bandwidth: float
:param num_points: Number of points in the PDF
:type num_points: int
:return: Sum of squared errors between the theoretical and empirical PDFs
:rtype: float

#### `function find_max_eval`

```python
def find_max_evaleigenvalues: np.ndarray, q: float, bandwidth: float:
```

> Find the maximum random eigenvalue by fitting the Marcenko-Pastur distribution.

:param eigenvalues: Eigenvalues of the correlation matrix
:type eigenvalues: np.ndarray
:param q: Ratio T/N
:type q: float
:param bandwidth: Bandwidth of the kernel
:type bandwidth: float
:return: Maximum random eigenvalue and its variance
:rtype: Tuple[float, float]

#### `function denoised_corr`

```python
def denoised_correigenvalues: np.ndarray, eigenvectors: np.ndarray, num_factors: int:
```

> Remove noise from the correlation matrix by fixing random eigenvalues.

:param eigenvalues: Eigenvalues of the correlation matrix
:type eigenvalues: np.ndarray
:param eigenvectors: Eigenvectors of the correlation matrix
:type eigenvectors: np.ndarray
:param num_factors: Number of factors for the correlation matrix
:type num_factors: int
:return: Denoised correlation matrix
:rtype: np.ndarray

#### `function denoised_corr2`

```python
def denoised_corr2eigenvalues: np.ndarray, eigenvectors: np.ndarray, num_factors: int, alpha: float=0:
```

> Remove noise from the correlation matrix through targeted shrinkage.

:param eigenvalues: Eigenvalues of the correlation matrix
:type eigenvalues: np.ndarray
:param eigenvectors: Eigenvectors of the correlation matrix
:type eigenvectors: np.ndarray
:param num_factors: Number of factors for the correlation matrix
:type num_factors: int
:param alpha: Shrinkage parameter
:type alpha: float
:return: Denoised correlation matrix
:rtype: np.ndarray

#### `function form_block_matrix`

```python
def form_block_matrixn_blocks: int, block_size: int, block_correlation: float:
```

> Forms a block diagonal correlation matrix.

:param n_blocks: Number of blocks
:type n_blocks: int
:param block_size: Size of each block
:type block_size: int
:param block_correlation: Correlation within each block
:type block_correlation: float
:return: Block diagonal correlation matrix
:rtype: np.ndarray

#### `function form_true_matrix`

```python
def form_true_matrixn_blocks: int, block_size: int, block_correlation: float:
```

> Forms a shuffled block diagonal correlation matrix and the corresponding covariance matrix.

:param n_blocks: Number of blocks
:type n_blocks: int
:param block_size: Size of each block
:type block_size: int
:param block_correlation: Correlation within each block
:type block_correlation: float
:return: Mean and covariance matrix
:rtype: Tuple[np.ndarray, np.ndarray]

#### `function simulates_cov_mu`

```python
def simulates_cov_mumu0: np.ndarray, cov0: np.ndarray, n_obs: int, shrink: bool=False:
```

> Simulates multivariate normal observations and computes the sample mean and covariance.

:param mu0: True mean
:type mu0: np.ndarray
:param cov0: True covariance matrix
:type cov0: np.ndarray
:param n_obs: Number of observations
:type n_obs: int
:param shrink: Whether to use Ledoit-Wolf shrinkage
:type shrink: bool
:return: Sample mean and covariance matrix
:rtype: Tuple[np.ndarray, np.ndarray]

#### `function corr_to_cov`

```python
def corr_to_covcorr: np.ndarray, std: np.ndarray:
```

> Converts a correlation matrix to a covariance matrix.

:param corr: Correlation matrix
:type corr: np.ndarray
:param std: Standard deviations
:type std: np.ndarray
:return: Covariance matrix
:rtype: np.ndarray

#### `function denoise_cov`

```python
def denoise_covcov0: np.ndarray, q: float, bandwidth: float:
```

> De-noises the covariance matrix.

:param cov0: Covariance matrix
:type cov0: np.ndarray
:param q: Ratio of number of observations to number of variables
:type q: float
:param bandwidth: Bandwidth parameter
:type bandwidth: float
:return: De-noised covariance matrix
:rtype: np.ndarray

#### `function optimal_portfolio`

```python
def optimal_portfoliocov: np.ndarray, mu: np.ndarray=None:
```

> Computes the optimal portfolio weights.

:param cov: Covariance matrix
:type cov: np.ndarray
:param mu: Expected returns
:type mu: np.ndarray
:return: Optimal portfolio weights
:rtype: np.ndarray


### 📄 `RiskLabAI\data\differentiation\differentiation.py`

#### `function calculate_weights`

```python
def calculate_weightsdegree: float, size: int:
```

> Compute the weights for fractionally differentiated series.

:param degree: Degree of the binomial series.
:param size: Length of the time series.
:return: Array of weights.

Formula:
    .. math::
        w(k) = -w(k-1) / k * (degree - k + 1)

#### `function plot_weights`

```python
def plot_weightsdegree_range: tuple[float, float], number_degrees: int, size: int:
```

> Plot the weights of fractionally differentiated series.

:param degree_range: Tuple containing the minimum and maximum degree values.
:param number_degrees: Number of degrees to plot.
:param size: Length of the time series.

#### `function fractional_difference`

```python
def fractional_differenceseries: pd.DataFrame, degree: float, threshold: float=0.01:
```

> Compute the standard fractionally differentiated series.

:param series: Dataframe of dates and prices.
:param degree: Degree of the binomial series.
:param threshold: Threshold for weight-loss.
:return: Dataframe of fractionally differentiated series.

Methodology reference:
    De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons, p. 82.

#### `function calculate_weights_ffd`

```python
def calculate_weights_ffddegree: float, threshold: float:
```

> Compute the weights for fixed-width window fractionally differentiated method.

:param degree: Degree of the binomial series.
:param threshold: Threshold for weight-loss.
:return: Array of weights.

Methodology reference:
    De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons, p. 83.

#### `function fractional_difference_fixed`

```python
def fractional_difference_fixedseries: pd.DataFrame, degree: float, threshold: float=1e-05:
```

> Compute the fixed-width window fractionally differentiated series.

:param series: Dataframe of dates and prices.
:param degree: Degree of the binomial series.
:param threshold: Threshold for weight-loss.
:return: Dataframe of fractionally differentiated series.

Methodology reference:
    De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons, p. 83.

#### `function fractional_difference_fixed_single`

```python
def fractional_difference_fixed_singleseries: pd.Series, degree: float, threshold: float=1e-05:
```

> Compute the fixed-width window fractionally differentiated series.

:param series: Series of dates and prices.
:param degree: Degree of the binomial series.
:param threshold: Threshold for weight-loss.
:return: Fractionally differentiated series.

Methodology reference:
    De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons, p. 83.

#### `function minimum_ffd`

```python
def minimum_ffdinput_series: pd.DataFrame:
```

> Find the minimum degree value that passes the ADF test.

:param input_series: Dataframe of input data.
:return: Dataframe of ADF test results.

Methodology reference:
    De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons, p. 85.

#### `function get_weights`

```python
def get_weightsdegree: float, length: int:
```

> Calculate the weights for the fractional differentiation method.

:param degree: Degree of binomial series.
:param length: Length of the series.
:return: Array of calculated weights.

Related mathematical formula:
.. math::
    w_i = -w_{i-1}/i*(degree - i + 1)

#### `function fractional_difference`

```python
def fractional_differenceseries: pd.DataFrame, degree: float, threshold: float=0.01:
```

> Calculate the fractionally differentiated series using the fixed-width window method.

:param series: DataFrame of dates and prices.
:param degree: Degree of binomial series.
:param threshold: Threshold for weight-loss.
:return: DataFrame of fractionally differentiated series.

Related mathematical formula:
.. math::
    F_t^{(d)} = \sum_{i=0}^{t} w_i F_{t-i}

#### `function minimum_adf_degree`

```python
def minimum_adf_degreeinput_series: pd.DataFrame:
```

> Find the minimum degree value that passes the ADF test.

:param input_series: DataFrame of input series.
:return: DataFrame of output results with ADF statistics.

Related mathematical formula:
.. math::
    F_t^{(d)} = \sum_{i=0}^{t} w_i F_{t-i}

#### `function fractionally_differentiated_log_price`

```python
def fractionally_differentiated_log_priceinput_series: pd.Series, threshold=0.01, step=0.1, base_p_value=0.05:
```

> Calculate the fractionally differentiated log price with the minimum degree differentiation
that passes the Augmented Dickey-Fuller (ADF) test.

:param input_series: Time series of input data.
:param threshold: The threshold for fractionally differentiating the log price.
:param step: The increment step for adjusting the differentiation degree.
:param base_p_value: The significance level for the ADF test.
:return: Fractionally differentiated log price series.

Methodology reference:
    De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons, p. 85.


### 📄 `RiskLabAI\data\distance\distance_metric.py`

#### `function calculate_variation_of_information`

```python
def calculate_variation_of_informationx: np.ndarray, y: np.ndarray, bins: int, norm: bool=False:
```

> Calculates Variation of Information.

:param x: First data array.
:param y: Second data array.
:param bins: Number of bins for the histogram.
:param norm: If True, the result will be normalized.

:return: Variation of Information.

Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
Methodology: Snippet 3.2, Page 44

#### `function calculate_number_of_bins`

```python
def calculate_number_of_binsnum_observations: int, correlation: float=None:
```

> Calculates the optimal number of bins for discretization.

:param num_observations: Number of observations.
:param correlation: Correlation value. If None, the function will use the univariate case.

:return: Optimal number of bins.

Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
Methodology: Snippet 3.3, Page 46

#### `function calculate_variation_of_information_extended`

```python
def calculate_variation_of_information_extendedx: np.ndarray, y: np.ndarray, norm: bool=False:
```

> Calculates Variation of Information with calculating number of bins.

:param x: First data array.
:param y: Second data array.
:param norm: If True, the result will be normalized.

:return: Variation of Information.

Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
Methodology: Snippet 3.3, Page 46

#### `function calculate_mutual_information`

```python
def calculate_mutual_informationx: np.ndarray, y: np.ndarray, norm: bool=False:
```

> Calculates Mutual Information with calculating number of bins.

:param x: First data array.
:param y: Second data array.
:param norm: If True, the result will be normalized.

:return: Mutual Information.

Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
Methodology: Snippet 3.4, Page 48

#### `function calculate_distance`

```python
def calculate_distancedependence: np.ndarray, metric: str='angular':
```

> Calculates distance from a dependence matrix.

:param dependence: Dependence matrix.
:param metric: Metric used to calculate distance. Available options are "angular" and "absolute_angular".

:return: Distance matrix.

Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.

#### `function calculate_kullback_leibler_divergence`

```python
def calculate_kullback_leibler_divergencep: np.ndarray, q: np.ndarray:
```

> Calculates Kullback-Leibler divergence from two discrete probability distributions defined on the same probability space.

:param p: First distribution.
:param q: Second distribution.

:return: Kullback-Leibler divergence.

Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.

#### `function calculate_cross_entropy`

```python
def calculate_cross_entropyp: np.ndarray, q: np.ndarray:
```

> Calculates cross-entropy from two discrete probability distributions defined on the same probability space.

:param p: First distribution.
:param q: Second distribution.

:return: Cross-entropy.

Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.


### 📄 `RiskLabAI\data\labeling\financial_labels.py`

#### `function calculate_t_value_linear_regression`

```python
def calculate_t_value_linear_regressionprice: pd.Series:
```

> Calculate the t-value of a linear trend.

This function computes the t-value of a linear trend in a time series of prices. 
The t-value is calculated as the ratio of the slope to the standard error of the regression.

:param price: Time series of prices as a Pandas Series.
:return: Calculated t-value as a float.

#### `function find_trend_using_trend_scanning`

```python
def find_trend_using_trend_scanningmolecule: pd.Index, close: pd.Series, span: Tuple[int, int]:
```

> Implement the trend scanning method to find trends.

This function identifies trends in a time series of prices using the trend scanning method. 
It calculates the t-value for linear regression of price over a range of spans and 
identifies the span with the maximum absolute t-value as the trend. The sign of the t-value 
indicates the direction of the trend.

:param molecule: Index of observations to label as a Pandas Index.
:param close: Time series of prices as a Pandas Series.
:param span: Range of span lengths to evaluate for the maximum absolute t-value as a tuple.
:return: DataFrame containing trend information with columns ['End Time', 't-Value', 'Trend'].


### 📄 `RiskLabAI\data\labeling\labeling.py`

#### `function cusum_filter_events_dynamic_threshold`

```python
def cusum_filter_events_dynamic_thresholdprices: pd.Series, threshold: pd.Series:
```

> Detect events using the Symmetric Cumulative Sum (CUSUM) filter.

The Symmetric CUSUM filter is a change-point detection algorithm used to identify events where the price difference
exceeds a predefined threshold.

:param prices: A pandas Series of prices.
:param threshold: A pandas Series containing the predefined threshold values for event detection.
:return: A pandas DatetimeIndex containing timestamps of detected events.

References:
- De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons. (Methodology: 39)

#### `function symmetric_cusum_filter`

```python
def symmetric_cusum_filterprices: pd.Series, threshold: float:
```

> Implements the symmetric CUSUM filter.

The symmetric CUSUM filter is a change-point detection algorithm used to identify events where the price difference exceeds a predefined threshold.

:param prices: A pandas Series of prices.
:param threshold: The predefined threshold for detecting events.
:return: A pandas DatetimeIndex of event timestamps.

References:
- De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons. (Methodology: 39)

#### `function aggregate_ohlcv`

```python
def aggregate_ohlcvtick_data_grouped:
```

> Aggregates tick data into OHLCV bars.

:param tick_data_grouped: A pandas GroupBy object of tick data.
:return: A pandas DataFrame with OHLCV bars.

#### `function generate_time_bars`

```python
def generate_time_barstick_data: pd.DataFrame, frequency: str='5Min':
```

> Generates time bars from tick data.

:param tick_data: A pandas DataFrame of tick data.
:param frequency: The frequency for time bar aggregation.
:return: A pandas DataFrame with time bars.

#### `function compute_daily_volatility`

```python
def compute_daily_volatilityclose: pd.Series, span: int=63:
```

> Computes the daily volatility at intraday estimation points.

:param close: A pandas Series of close prices.
:param span: The span parameter for the EWMA.
:return: A pandas DataFrame with returns and volatilities.

References:
- De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons. (Methodology: Page 44)

#### `function daily_volatility_with_log_returns`

```python
def daily_volatility_with_log_returnsclose: pd.Series, span: int=100:
```

> Calculate the daily volatility at intraday estimation points using Exponentially Weighted Moving Average (EWMA).

:param close: A pandas Series of daily close prices.
:param span: The span parameter for the Exponentially Weighted Moving Average (EWMA).
:return: A pandas Series containing daily volatilities.

References:
- De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons. (Methodology: Page 44)

#### `function triple_barrier`

```python
def triple_barrierclose: pd.Series, events: pd.DataFrame, profit_taking_stop_loss: list[float, float], molecule: list:
```
#### `function get_barrier_touch_time`

```python
def get_barrier_touch_timeclose: pd.Series, time_events: pd.DatetimeIndex, ptsl: float, target: pd.Series, return_min: float, num_threads: int, timestamp: pd.Series=False:
```

> Finds the time of the first barrier touch.

:param close: A dataframe of dates and close prices.
:param time_events: A pandas time index containing the timestamps that will seed every triple barrier.
:param ptsl: A non-negative float that sets the width of the two barriers.
:param target: A pandas series of targets, expressed in terms of absolute returns.
:param return_min: The minimum target return required for running a triple barrier search.
:param num_threads: The number of threads.
:param timestamp: A pandas series with the timestamps of the vertical barriers (False when disabled).
:return: A dataframe with timestamp of the vertical barrier and unit width of the horizontal barriers.

#### `function vertical_barrier`

```python
def vertical_barrierclose: pd.Series, time_events: pd.DatetimeIndex, number_days: int:
```

> Shows one way to define a vertical barrier.

:param close: A dataframe of prices and dates.
:param time_events: A vector of timestamps.
:param number_days: A number of days for the vertical barrier.
:return: A pandas series with the timestamps of the vertical barriers.

#### `function get_labels`

```python
def get_labelsevents: pd.DataFrame, close: pd.Series:
```

> Label the observations.

:param events: A dataframe with timestamp of the vertical barrier and unit width of the horizontal barriers.
:param close: A dataframe of dates and close prices.
:return: A dataframe with the return realized at the time of the first touched barrier and the label.

#### `function meta_events`

```python
def meta_eventsclose: pd.Series, time_events: pd.DatetimeIndex, ptsl: List[float], target: pd.Series, return_min: float, num_threads: int, timestamp: pd.Series=False, side: pd.Series=None:
```
#### `function meta_labeling`

```python
def meta_labelingevents: pd.DataFrame, close: pd.Series:
```

> Expands label to incorporate meta-labeling.

:param events: DataFrame with timestamp of vertical barrier and unit width of the horizontal barriers.
:param close: Series of close prices with date indices.
:return: DataFrame containing the return and binary labels for each event.

Reference:
De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
Methodology: 51

#### `function drop_label`

```python
def drop_labelevents: pd.DataFrame, percent_min: float=0.05:
```

> Presents a procedure that recursively drops observations associated with extremely rare labels.

:param events: DataFrame with columns: Dates, ret, and bin.
:param percent_min: Minimum percentage.
:return: DataFrame with the updated events.

Reference:
De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
Methodology: 54

#### `function lin_parts`

```python
def lin_partsnum_atoms: int, num_threads: int:
```

> Partition of atoms with a single loop.

:param num_atoms: Total number of atoms.
:param num_threads: Number of threads for parallel processing.
:return: Numpy array with partition indices.

#### `function nested_parts`

```python
def nested_partsnum_atoms: int, num_threads: int, upper_triang: bool=False:
```

> Partition of atoms with an inner loop.

:param num_atoms: Total number of atoms.
:param num_threads: Number of threads for parallel processing.
:param upper_triang: Whether the first rows are the heaviest.
:return: Numpy array with partition indices.

#### `function mp_pandas_obj`

```python
def mp_pandas_objfunc, pd_obj, num_threads: int=24, mp_batches: int=1, lin_mols: bool=True, **kargs:
```

> Parallelize jobs, return a DataFrame or Series.

:param func: Function to be parallelized.
:param pd_obj: Tuple with argument name for the molecule and list of atoms grouped into molecules.
:param num_threads: Number of threads for parallel processing.
:param mp_batches: Number of multi-processing batches.
:param lin_mols: Whether to use linear molecule partitioning.
:param kargs: Any other arguments needed by func.
:return: DataFrame with the results of the parallelized function.

Example:
df1 = mp_pandas_obj(func, ('molecule', df0.index), 24, **kargs)

#### `function process_jobs_`

```python
def process_jobs_jobs: list:
```

> Run jobs sequentially, for debugging.

:param jobs: List of jobs to be processed.
:return: List of job results.

#### `function report_progress`

```python
def report_progressjob_num: int, num_jobs: int, time0: float, task: str:
```

> Report progress as asynchronous jobs are completed.

:param job_num: Current job number.
:param num_jobs: Total number of jobs.
:param time0: Start time.
:param task: Task name.
:return: None

#### `function process_jobs`

```python
def process_jobsjobs: list, task: str=None, num_threads: int=24:
```

> Run jobs in parallel in multiple threads.

:param jobs: List of jobs to be processed.
:param task: Task name for progress reporting.
:param num_threads: Number of threads for parallel processing.
:return: List of job results.

#### `function expand_call`

```python
def expand_callkargs:
```

> Expand the arguments of a callback function, kargs['func'].

:param kargs: Dictionary with the function to call and the arguments to pass.
:return: Result of the function call.


### 📄 `RiskLabAI\data\structures\abstract_bars.py`

> A base class for the various bar types. Includes the logic shared between classes, to minimise the amount of
duplicated code.

#### `class AbstractBars`

> Abstract class that contains the base properties which are shared between the subtypes.
This class subtypes are as follows:
    1- AbstractImbalanceBars
    2- AbstractRunBars
    3- StandardBars
    4- TimeBars

##### `method __init__`

```python
def __init__self, bar_type: str:
```

> AbstractBars constructor function
:param bar_type: type of bar. e.g. time_bars, expected_dollar_imbalance_bars, fixed_tick_run_bars, volume_standard_bars etc.

##### `method construct_bars_from_data`

```python
def construct_bars_from_dataself, data: Union[list, tuple, np.ndarray]:
```

> This function are implemented by all concrete or abstract subtypes. The function is used to construct bars from
input ticks data.
:param data: tabular data that contains date_time, price, and volume columns
:return: constructed bars

##### `method update_base_fields`

```python
def update_base_fieldsself, price: float, tick_rule: int, volume: float:
```

> Update the base fields (that all bars have them.) with price, tick rule and volume of current tick
:param price: price of current tick
:param tick_rule: tick rule of current tick computed before
:param volume: volume of current tick
:return:

##### `method _bar_construction_condition`

```python
def _bar_construction_conditionself, threshold:
```

> Compute the condition of whether next bar should sample with current and previous tick datas or not.
:return: whether next bar should form with current and previous tick datas or not.

##### `method _reset_cached_fields`

```python
def _reset_cached_fieldsself:
```

> This function are used (directly or override) by all concrete or abstract subtypes. The function is used to reset cached fields in bars construction process when next bar is sampled.
:return:

##### `method _tick_rule`

```python
def _tick_ruleself, price: float=0:
```

> Compute the tick rule term as explained on page 29 of Advances in Financial Machine Learning
:param price: price of current tick
:return: tick rule

##### `method _high_and_low_price_update`

```python
def _high_and_low_price_updateself, price: float:
```

> Update the high and low prices using the current tick price.
:param price: price of current tick
:return: updated high and low prices

##### `method _construct_next_bar`

```python
def _construct_next_barself, date_time: str, tick_index: int, price: float, high_price: float, low_price: float, threshold: float:
```

> sample next bar, given ticks data. the bar's fields are as follows:
    1- date_time
    2- open
    3- high
    4- low
    5- close
    6- cumulative_volume: total cumulative volume of to be constructed bar ticks
    7- cumulative_buy_volume: total cumulative buy volume of to be constructed bar ticks
    8- cumulative_ticks total cumulative ticks number of to be constructed bar ticks
    9- cumulative_dollar_value total cumulative dollar value (price * volume) of to be constructed bar ticks

the bar will have appended to the total list of sampled bars.

:param date_time: timestamp of the to be constructed bar
:param tick_index:
:param price: price of last tick of to be constructed bar (used as close price)
:param high_price: highest price of ticks in the period of bar sampling process
:param low_price: lowest price of ticks in the period of bar sampling process
:return: sampled bar


### 📄 `RiskLabAI\data\structures\abstract_imbalance_bars.py`

#### `class AbstractImbalanceBars`

> Abstract class that contains the imbalance properties which are shared between the subtypes.
This class subtypes are as follows:
    1- ExpectedImbalanceBars
    2- FixedImbalanceBars

The class implements imbalance bars sampling logic as explained on page 29,30 of Advances in Financial Machine Learning.

##### `method __init__`

```python
def __init__self, bar_type: str, window_size_for_expected_n_ticks_estimation: int, window_size_for_expected_imbalance_estimation: int, initial_estimate_of_expected_n_ticks_in_bar: int, analyse_thresholds: bool:
```

> AbstractImbalanceBars constructor function
:param bar_type: type of bar. e.g. expected_dollar_imbalance_bars, fixed_tick_imbalance_bars etc.
:param window_size_for_expected_n_ticks_estimation: window size used to estimate number of ticks expectation
:param initial_estimate_of_expected_n_ticks_in_bar: initial estimate of number of ticks expectation window size
:param window_size_for_expected_imbalance_estimation: window size used to estimate imbalance expectation
:param analyse_thresholds: whether return thresholds values (θ, number of ticks expectation, imbalance expectation) in a tabular format

##### `method construct_bars_from_data`

```python
def construct_bars_from_dataself, data: Union[list, tuple, np.ndarray]:
```

> The function is used to construct bars from input ticks data.
:param data: tabular data that contains date_time, price, and volume columns
:return: constructed bars

##### `method _bar_construction_condition`

```python
def _bar_construction_conditionself, threshold:
```

> Compute the condition of whether next bar should sample with current and previous tick datas or not.
:return: whether next bar should form with current and previous tick datas or not.

##### `method _reset_cached_fields`

```python
def _reset_cached_fieldsself:
```

> This function are used (directly or override) by all concrete or abstract subtypes. The function is used to reset cached fields in bars construction process when next bar is sampled.
:return:

##### `method _expected_number_of_ticks`

```python
def _expected_number_of_ticksself:
```

> Calculate number of ticks expectation when new imbalance bar is sampled.

:return: number of ticks expectation.


### 📄 `RiskLabAI\data\structures\abstract_information_driven_bars.py`

> A base class for the various bar types. Includes the logic shared between classes, to minimise the amount of
duplicated code.

#### `class AbstractInformationDrivenBars`

> Abstract class that contains the information driven properties which are shared between the subtypes.
This class subtypes are as follows:
    1- AbstractImbalanceBars
    2- AbstractRunBars

The class implements imbalance bars sampling logic as explained on page 29,30,31,32 of Advances in Financial Machine Learning.

##### `method __init__`

```python
def __init__self, bar_type: str, window_size_for_expected_n_ticks_estimation: int, initial_estimate_of_expected_n_ticks_in_bar: int, window_size_for_expected_imbalance_estimation: int:
```

> AbstractInformationDrivenBars constructor function
:param bar_type: type of bar. e.g. expected_dollar_imbalance_bars, fixed_tick_run_bars etc.
:param window_size_for_expected_n_ticks_estimation: window size used to estimate number of ticks expectation
:param initial_estimate_of_expected_n_ticks_in_bar: initial estimate of number of ticks expectation window size
:param window_size_for_expected_imbalance_estimation: window size used to estimate imbalance expectation

##### `method _ewma_expected_imbalance`

```python
def _ewma_expected_imbalanceself, array: list, window: int, warm_up: bool=False:
```

> Calculates expected imbalance (2P[b_t=1]-1) using EWMA as defined on page 29 of Advances in Financial Machine Learning.
:param array: imbalances list
:param window: EWMA window for expectation calculation
:param warm_up: whether warm up period passed or not
:return: expected_imbalance: 2P[b_t=1]-1 which approximated using EWMA expectation

##### `method _imbalance_at_tick`

```python
def _imbalance_at_tickself, price: float, signed_tick: int, volume: float:
```

> Calculate the imbalance at tick t (current tick) (θ_t) using tick data as defined on page 29 of Advances in Financial Machine Learning
:param price: price of tick
:param signed_tick: tick rule of current tick computed before
:param volume: volume of current tick
:return: imbalance: imbalance of current tick

##### `method _expected_number_of_ticks`

```python
def _expected_number_of_ticksself:
```

> Calculate number of ticks expectation when new imbalance bar is sampled.


### 📄 `RiskLabAI\data\structures\abstract_run_bars.py`

#### `class AbstractRunBars`

> Abstract class that contains the run properties which are shared between the subtypes.
This class subtypes are as follows:
    1- ExpectedRunBars
    2- FixedRunBars

The class implements run bars sampling logic as explained on page 31,32 of Advances in Financial Machine Learning.

##### `method __init__`

```python
def __init__self, bar_type: str, window_size_for_expected_n_ticks_estimation: int, window_size_for_expected_imbalance_estimation: int, initial_estimate_of_expected_n_ticks_in_bar: int, analyse_thresholds: bool:
```

> AbstractRunBars constructor function
:param bar_type: type of bar. e.g. expected_dollar_run_bars, fixed_tick_run_bars etc.
:param window_size_for_expected_n_ticks_estimation: window size used to estimate number of ticks expectation
:param initial_estimate_of_expected_n_ticks_in_bar: initial estimate of number of ticks expectation window size
:param window_size_for_expected_imbalance_estimation: window size used to estimate imbalance expectation
:param analyse_thresholds: whether return thresholds values (θ, number of ticks expectation, imbalance expectation) in a tabular format

##### `method construct_bars_from_data`

```python
def construct_bars_from_dataself, data: Union[list, tuple, np.ndarray]:
```

> The function is used to construct bars from input ticks data.
:param data: tabular data that contains date_time, price, and volume columns
:return: constructed bars

##### `method _bar_construction_condition`

```python
def _bar_construction_conditionself, threshold:
```

> Compute the condition of whether next bar should sample with current and previous tick datas or not.
:return: whether next bar should form with current and previous tick datas or not.

##### `method _reset_cached_fields`

```python
def _reset_cached_fieldsself:
```

> This function are used (directly or override) by all concrete or abstract subtypes. The function is used to reset cached fields in bars construction process when next bar is sampled.
:return:

##### `method _expected_number_of_ticks`

```python
def _expected_number_of_ticksself:
```

> Calculate number of ticks expectation when new imbalance bar is sampled.


### 📄 `RiskLabAI\data\structures\data_structures_lopez.py`

#### `function progress_bar`

```python
def progress_barvalue: int, end_value: int, start_time: float, bar_length: int=20:
```

> Display a progress bar in the console.

:param value: Current progress value.
:param end_value: The end value indicating 100% progress.
:param start_time: Time when the event started.
:param bar_length: The length of the progress bar in characters. Default is 20.

#### `function ewma`

```python
def ewmainput_array: np.ndarray, window_length: int:
```

> Computes the Exponentially Weighted Moving Average (EWMA).

:param input_array: The input time series array.
:param window_length: Window length for the EWMA.
:return: The EWMA values.

#### `function compute_grouping`

```python
def compute_groupingtarget_col: pd.Series, initial_expected_ticks: int, bar_size: float:
```

> Group a DataFrame based on a feature and calculates thresholds.

:param target_col: Target column of tick dataframe.
:param initial_expected_ticks: Initial expected ticks.
:param bar_size: Initial expected size in each tick.
:return: Arrays of times_delta, thetas_absolute, thresholds, times, thetas, grouping_id.

#### `function generate_information_driven_bars`

```python
def generate_information_driven_barstick_data: pd.DataFrame, bar_type: str='volume', tick_expected_initial: int=2000:
```

> Implements Information-Driven Bars as per the methodology described in
"Advances in financial machine learning" by De Prado (2018).

:param tick_data: DataFrame of tick data.
:param bar_type: Type of the bars, options: "tick", "volume", "dollar".
:param tick_expected_initial: Initial expected ticks value.
:return: A tuple containing the OHLCV DataFrame, thetas absolute array,
    and thresholds array.

#### `function ohlcv`

```python
def ohlcvtick_data_grouped: pd.core.groupby.generic.DataFrameGroupBy:
```

> Computes various statistics for the grouped tick data.

Takes a grouped dataframe, combines the data, and creates a new one with
information about prices, volume, and other statistics. This is typically
used in the context of financial tick data to generate OHLCV data
(Open, High, Low, Close, Volume).

:param tick_data_grouped: Grouped DataFrame containing tick data.
:return: A DataFrame containing OHLCV data and other derived statistics.

#### `function generate_time_bar`

```python
def generate_time_bartick_data: pd.DataFrame, frequency: str='5Min':
```

> Generates time bars for tick data.

This function groups tick data by a specified time frequency and then
computes OHLCV (Open, High, Low, Close, Volume) statistics.

:param tick_data: DataFrame containing tick data.
:param frequency: Time frequency for rounding datetime.
:return: A DataFrame containing OHLCV data grouped by time.

#### `function generate_tick_bar`

```python
def generate_tick_bartick_data: pd.DataFrame, ticks_per_bar: int=10, number_bars: int=None:
```

> Generates tick bars for tick data.

This function groups tick data by a specified number of ticks and then
computes OHLCV statistics.

:param tick_data: DataFrame containing tick data.
:param ticks_per_bar: Number of ticks in each bar.
:param number_bars: Number of bars to generate.
:return: A DataFrame containing OHLCV data grouped by tick count.

#### `function generate_volume_bar`

```python
def generate_volume_bartick_data: pd.DataFrame, volume_per_bar: int=10000, number_bars: int=None:
```

> Generates volume bars for tick data.

This function groups tick data by a specified volume size and then computes OHLCV statistics.

:param tick_data: DataFrame containing tick data.
:param volume_per_bar: Volume size for each bar.
:param number_bars: Number of bars to generate.

:return: A DataFrame containing OHLCV data grouped by volume.

#### `function generate_dollar_bar`

```python
def generate_dollar_bartick_data: pd.DataFrame, dollar_per_bar: float=100000, number_bars: int=None:
```

> Generates dollar bars for tick data.

This function groups tick data by a specified dollar amount and then computes OHLCV statistics.

:param tick_data: DataFrame containing tick data.
:param dollar_per_bar: Dollar amount for each bar.
:param number_bars: Number of bars to generate.

:return: A DataFrame containing OHLCV data grouped by dollar amount.

#### `function calculate_pca_weights`

```python
def calculate_pca_weightscovariance_matrix: np.ndarray, risk_distribution: np.ndarray=None, risk_target: float=1.0:
```

> Calculates hedging weights using the covariance matrix, risk distribution, and risk target.

:param covariance_matrix: Covariance matrix.
:param risk_distribution: Risk distribution vector.
:param risk_target: Risk target value.

:return: Weights.

#### `function events`

```python
def eventsinput_data: pd.DataFrame, threshold: float:
```

> Implementation of the symmetric CUSUM filter.

This function computes time events when certain price change thresholds are met.

:param input_data: DataFrame of prices and dates.
:param threshold: Threshold for price change.

:return: DatetimeIndex containing events.


### 📄 `RiskLabAI\data\structures\filtering_lopez.py`

#### `function symmetric_cusum_filter`

```python
def symmetric_cusum_filterinput_data: pd.DataFrame, threshold: float:
```

> Implementation of the symmetric CUSUM filter.

This method is used to detect changes in a time series data.

:param input_data: DataFrame containing price data.
:param threshold: Threshold value for the CUSUM filter.

:return: Datetime index of events based on the symmetric CUSUM filter.

.. note:: 
   Reference:
   De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
   Methodology 39.

.. math::
   S_t^+ = max(0, S_{t-1}^+ + \Delta p_t)
   S_t^- = min(0, S_{t-1}^- + \Delta p_t)

   where:
   - :math:`S_t^+` is the positive CUSUM at time :math:`t`
   - :math:`S_t^-` is the negative CUSUM at time :math:`t`
   - :math:`\Delta p_t` is the price change at time :math:`t`


### 📄 `RiskLabAI\data\structures\hedging.py`

#### `function pca_weights`

```python
def pca_weightscov: np.ndarray, risk_distribution: Optional[np.ndarray]=None, risk_target: float=1.0:
```

> Calculates hedging weights using covariance, risk distribution, and risk target.

The function uses Principal Component Analysis (PCA) to determine the weights.
If the risk distribution is not provided, all risk is allocated to the principal
component with the smallest eigenvalue.

:param cov: Covariance matrix.
:param risk_distribution: Risk distribution, defaults to None.
:param risk_target: Risk target, defaults to 1.0.

:return: Weights calculated based on PCA.

.. note::
   Reference:
   De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
   Methodology 36.

.. math::
   w = EV . \sqrt{\frac{\rho T}{\lambda}}

   where:
   - :math:`w` are the weights.
   - :math:`EV` is the matrix of eigenvectors.
   - :math:`\rho` is the risk distribution.
   - :math:`T` is the risk target.
   - :math:`\lambda` is the eigenvalues.


### 📄 `RiskLabAI\data\structures\imbalance_bars.py`

#### `class ExpectedImbalanceBars`

> Concrete class that contains the properties which are shared between all various type of ewma imbalance bars (dollar, volume, tick).

##### `method __init__`

```python
def __init__self, bar_type: str, window_size_for_expected_n_ticks_estimation: int, initial_estimate_of_expected_n_ticks_in_bar: int, window_size_for_expected_imbalance_estimation: int, expected_ticks_number_bounds: Tuple[float, float], analyse_thresholds: bool:
```

> ExpectedImbalanceBars constructor function
:param bar_type: type of bar. e.g. expected_dollar_imbalance_bars, fixed_tick_imbalance_bars etc.
:param window_size_for_expected_n_ticks_estimation: window size used to estimate number of ticks expectation
:param initial_estimate_of_expected_n_ticks_in_bar: initial estimate of number of ticks expectation window size
:param window_size_for_expected_imbalance_estimation: window size used to estimate imbalance expectation
:param expected_ticks_number_bounds lower and upper bound of possible number of expected ticks that used to force bars sampling convergence.
:param analyse_thresholds: whether return thresholds values (θ, number of ticks expectation, imbalance expectation) in a tabular format

##### `method _expected_number_of_ticks`

```python
def _expected_number_of_ticksself:
```

> Calculate number of ticks expectation when new imbalance bar is sampled.

:return: number of ticks expectation.

#### `class FixedImbalanceBars`

> Concrete class that contains the properties which are shared between all various type of const imbalance bars (dollar, volume, tick).

##### `method __init__`

```python
def __init__self, bar_type: str, window_size_for_expected_n_ticks_estimation: int, initial_estimate_of_expected_n_ticks_in_bar: int, window_size_for_expected_imbalance_estimation: int, analyse_thresholds: bool:
```

> FixedImbalanceBars constructor function
:param bar_type: type of bar. e.g. expected_dollar_imbalance_bars, fixed_tick_imbalance_bars etc.
:param window_size_for_expected_n_ticks_estimation: window size used to estimate number of ticks expectation
:param initial_estimate_of_expected_n_ticks_in_bar: initial estimate of number of ticks expectation window size
:param window_size_for_expected_imbalance_estimation: window size used to estimate imbalance expectation
:param analyse_thresholds: whether return thresholds values (θ, number of ticks expectation, imbalance expectation) in a tabular format

##### `method _expected_number_of_ticks`

```python
def _expected_number_of_ticksself:
```

> Calculate number of ticks expectation when new imbalance bar is sampled.

:return: number of ticks expectation.


### 📄 `RiskLabAI\data\structures\infomation_driven_bars.py`

#### `function generate_information_driven_bars`

```python
def generate_information_driven_barstick_data: pd.DataFrame, bar_type: str='volume', initial_expected_ticks: int=2000:
```

> Implements Information-Driven Bars.

This function computes the Information-Driven Bars based on tick data and the chosen bar type.

:param tick_data: DataFrame of tick data.
:type tick_data: pd.DataFrame
:param bar_type: Type of the bar. Can be "tick", "volume", or "dollar".
:type bar_type: str, default "volume"
:param initial_expected_ticks: The initial value of expected ticks.
:type initial_expected_ticks: int, default 2000

:return: Tuple containing the OHLCV DataFrame, absolute thetas, and thresholds.
:rtype: Tuple[pd.DataFrame, np.ndarray, np.ndarray]

.. note:: 
   Reference:
   De Prado, M. (2018) Advances in Financial Machine Learning. John Wiley & Sons.

.. math::
   E_b = |ar{x}|

   where:
   - :math:`E_b` is the expected value of the bars.
   - :math:`ar{x}` is the mean of the input data.

The compute_thresholds function is called to compute times_delta, thetas_absolute, thresholds,
times, thetas, and grouping_id.


### 📄 `RiskLabAI\data\structures\run_bars.py`

#### `class ExpectedRunBars`

> Concrete class that contains the properties which are shared between all various type of ewma run bars (dollar, volume, tick).

##### `method __init__`

```python
def __init__self, bar_type: str, window_size_for_expected_n_ticks_estimation: int, initial_estimate_of_expected_n_ticks_in_bar: int, window_size_for_expected_imbalance_estimation: int, expected_ticks_number_bounds: Tuple[float], analyse_thresholds: bool:
```

> ExpectedRunBars constructor function
:param bar_type: type of bar. e.g. expected_dollar_imbalance_bars, fixed_tick_imbalance_bars etc.
:param window_size_for_expected_n_ticks_estimation: window size used to estimate number of ticks expectation
:param initial_estimate_of_expected_n_ticks_in_bar: initial estimate of number of ticks expectation window size
:param window_size_for_expected_imbalance_estimation: window size used to estimate imbalance expectation
:param expected_ticks_number_bounds lower and upper bound of possible number of expected ticks that used to force bars sampling convergence.
:param analyse_thresholds: whether return thresholds values (θ, number of ticks expectation, imbalance expectation) in a tabular format

##### `method _expected_number_of_ticks`

```python
def _expected_number_of_ticksself:
```

> Calculate number of ticks expectation when new imbalance bar is sampled.

:return: number of ticks expectation.

#### `class FixedRunBars`

> Concrete class that contains the properties which are shared between all various type of const run bars (dollar, volume, tick).

##### `method __init__`

```python
def __init__self, bar_type: str, window_size_for_expected_n_ticks_estimation: int, window_size_for_expected_imbalance_estimation: int, initial_estimate_of_expected_n_ticks_in_bar: int, analyse_thresholds: bool:
```

> Constructor.

:param bar_type: (str) Type of run bar to create. Example: "dollar_run".
:param window_size_for_expected_n_ticks_estimation: (int) Window size for E[T]s (number of previous bars to use for expected number of ticks estimation).
:param window_size_for_expected_imbalance_estimation: (int) Expected window used to estimate expected run.
:param initial_estimate_of_expected_n_ticks_in_bar: (int) Initial number of expected ticks.
:param batch_size: (int) Number of rows to read in from the csv, per batch.
:param analyse_thresholds: (bool) Flag to save  and return thresholds used to sample run bars.

##### `method _expected_number_of_ticks`

```python
def _expected_number_of_ticksself:
```

> Calculate number of ticks expectation when new imbalance bar is sampled.

:return: number of ticks expectation.


### 📄 `RiskLabAI\data\structures\standard_bars.py`

#### `class StandardBars`

> Concrete class that contains the properties which are shared between all various type of standard bars (dollar, volume, tick).

##### `method __init__`

```python
def __init__self, bar_type: str, threshold: float=50000:
```

> StandardBars constructor function
:param bar_type: type of bar. e.g. dollar_standard_bars, tick_standard_bars etc.
:param threshold: threshold that used to sampling process

##### `method construct_bars_from_data`

```python
def construct_bars_from_dataself, data: Union[list, tuple, np.ndarray]:
```

> The function is used to construct bars from input ticks data.
:param data: tabular data that contains date_time, price, and volume columns
:return: constructed bars

##### `method _bar_construction_condition`

```python
def _bar_construction_conditionself, threshold:
```

> Compute the condition of whether next bar should sample with current and previous tick datas or not.
:return: whether next bar should form with current and previous tick datas or not.


### 📄 `RiskLabAI\data\structures\standard_bars_lopez.py`

#### `function generate_dollar_bar_dataframe`

```python
def generate_dollar_bar_dataframetick_data: pd.DataFrame, dollar_per_bar: int=100000, number_bars: Optional[int]=None:
```

> Generates a dollar bar dataframe.

:param tick_data: DataFrame of tick data.
:type tick_data: pd.DataFrame
:param dollar_per_bar: Dollars in each bar, defaults to 100000.
:type dollar_per_bar: int, optional
:param number_bars: Number of bars, defaults to None.
:type number_bars: Optional[int], optional
:return: A dataframe containing OHLCV data and other relevant information based on dollar bars.
:rtype: pd.DataFrame

#### `function generate_tick_bar_dataframe`

```python
def generate_tick_bar_dataframetick_data: pd.DataFrame, tick_per_bar: int=10, number_bars: Optional[int]=None:
```

> Generates a tick bar dataframe.

:param tick_data: DataFrame of tick data.
:type tick_data: pd.DataFrame
:param tick_per_bar: Number of ticks in each bar, defaults to 10.
:type tick_per_bar: int, optional
:param number_bars: Number of bars, defaults to None.
:type number_bars: Optional[int], optional
:return: A dataframe containing OHLCV data and other relevant information based on tick bars.
:rtype: pd.DataFrame

#### `function generate_time_bar_dataframe`

```python
def generate_time_bar_dataframetick_data: pd.DataFrame, frequency: str='5Min':
```

> Generates a time bar dataframe.

:param tick_data: DataFrame of tick data.
:type tick_data: pd.DataFrame
:param frequency: Frequency for rounding date time, defaults to "5Min".
:type frequency: str, optional
:return: A dataframe containing OHLCV data and other relevant information based on time bars with the specified frequency.
:rtype: pd.DataFrame

#### `function generate_volume_bar_dataframe`

```python
def generate_volume_bar_dataframetick_data: pd.DataFrame, volume_per_bar: int=10000, number_bars: Optional[int]=None:
```

> Generates a volume bar dataframe.

:param tick_data: DataFrame of tick data.
:type tick_data: pd.DataFrame
:param volume_per_bar: Volumes in each bar, defaults to 10000.
:type volume_per_bar: int, optional
:param number_bars: Number of bars, defaults to None.
:type number_bars: Optional[int], optional
:return: A dataframe containing OHLCV data and other relevant information based on volume bars.
:rtype: pd.DataFrame


### 📄 `RiskLabAI\data\structures\time_bars.py`

#### `class TimeBars`

> Concrete class of TimeBars logic

##### `method __init__`

```python
def __init__self, resolution_type: str, resolution_units: int:
```

> TimeBars constructor function

:param resolution_type: (str) Type of bar resolution: ['D', 'H', 'MIN', 'S'].
:param resolution_units: (int) Number of days, minutes, etc.

##### `method construct_bars_from_data`

```python
def construct_bars_from_dataself, data: Union[list, tuple, np.ndarray]:
```

> The function is used to construct bars from input ticks data.
:param data: tabular data that contains date_time, price, and volume columns
:return: constructed bars

##### `method _bar_construction_condition`

```python
def _bar_construction_conditionself, threshold:
```

> Compute the condition of whether next bar should sample with current and previous tick datas or not.
:return: whether next bar should form with current and previous tick datas or not.


### 📄 `RiskLabAI\data\structures\utilities_lopez.py`

#### `function compute_thresholds`

```python
def compute_thresholdstarget_column: np.ndarray, initial_expected_ticks: int, initial_bar_size: float:
```

> Groups the target_column DataFrame based on a feature and calculates thresholds.

This function groups the target_column DataFrame based on a feature
and calculates the thresholds, which can be used in financial machine learning
applications such as dynamic time warping.

:param target_column: Target column of the DataFrame.
:type target_column: np.ndarray
:param initial_expected_ticks: Initial expected number of ticks.
:type initial_expected_ticks: int
:param initial_bar_size: Initial expected size of each tick.
:type initial_bar_size: float
:return: A tuple containing the time deltas, absolute theta values, thresholds,
    times, theta values, and grouping IDs.
:rtype: Tuple[List[float], np.ndarray, np.ndarray, List[int], np.ndarray, np.ndarray]

#### `function create_ohlcv_dataframe`

```python
def create_ohlcv_dataframetick_data_grouped: pd.core.groupby.DataFrameGroupBy:
```

> Takes a grouped DataFrame and creates a new one with OHLCV data and other relevant information.

:param tick_data_grouped: Grouped DataFrame based on some criteria (e.g., time).
:type tick_data_grouped: pd.core.groupby.DataFrameGroupBy
:return: A DataFrame containing OHLCV data and other relevant information.
:rtype: pd.DataFrame


### 📄 `RiskLabAI\data\synthetic_data\drift_burst_hypothesis.py`

#### `function drift_volatility_burst`

```python
def drift_volatility_burstbubble_length: int, a_before: float, a_after: float, b_before: float, b_after: float, alpha: float, beta: float, explosion_filter_width: float=0.1:
```

> Compute the drift and volatility for a burst scenario.

The drift and volatility are calculated based on:
.. math::
    drift = rac{a_{value}}{denominator^lpha}
    volatility = rac{b_{value}}{denominator^eta}

where:
.. math::
    denominator = |step - 0.5|

:param bubble_length: The length of the bubble.
:param a_before: 'a' value before the mid-point.
:param a_after: 'a' value after the mid-point.
:param b_before: 'b' value before the mid-point.
:param b_after: 'b' value after the mid-point.
:param alpha: Exponent for the drift calculation.
:param beta: Exponent for the volatility calculation.
:param explosion_filter_width: Width of the area around the explosion that denominators won't exceed. 
:return: A tuple containing the drift and volatility arrays.


### 📄 `RiskLabAI\data\synthetic_data\synthetic_controlled_environment.py`

#### `function compute_log_returns`

```python
def compute_log_returnsN: int, mu_vector: np.ndarray, kappa_vector: np.ndarray, theta_vector: np.ndarray, xi_vector: np.ndarray, dwS: np.ndarray, dwV: np.ndarray, Y: np.ndarray, n: np.ndarray, dt: float, sqrt_dt: float, lambda_vector: np.ndarray, m_vector: np.ndarray, v_vector: np.ndarray, regime_change: np.ndarray:
```

> Computes the log returns based on the Heston-Merton model.

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

#### `function heston_merton_log_returns`

```python
def heston_merton_log_returnsT: float, N: int, mu_vector: np.ndarray, kappa_vector: np.ndarray, theta_vector: np.ndarray, xi_vector: np.ndarray, rho_vector: np.ndarray, lambda_vector: np.ndarray, m_vector: np.ndarray, v_vector: np.ndarray, regime_change: np.ndarray, random_state=None:
```

> Computes the log returns based on the Heston-Merton model using Gaussian random numbers.

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

#### `function align_params_length`

```python
def align_params_lengthregime_params: Dict[str, Union[float, List[float]]]:
```

> Align the parameters' length within the provided regime parameters.

:param regime_params: Dictionary of regime parameters. Values can be floats or lists.
:return: A tuple containing the regime parameters with aligned lengths and the max length.

#### `function generate_prices_from_regimes`

```python
def generate_prices_from_regimesregimes: Dict[str, Dict[str, Union[float, List[float]]]], transition_matrix: np.ndarray, total_time: float, n_steps: int, random_state: int=None:
```

> Generate prices based on provided regimes and a Markov Chain.

:param regimes: Dictionary containing regime names and their respective parameters.
:param transition_matrix: Markov Chain transition matrix.
:param total_time: Total time for the simulation.
:param n_steps: Number of discrete steps in the simulation.
:param random_state: Seed for random number generation.
:return: A tuple containing the generated prices as a pandas Series and the simulated regimes.

#### `function parallel_generate_prices`

```python
def parallel_generate_pricesnumber_of_paths: int, regimes: Dict[str, Dict[str, Union[float, List[float]]]], transition_matrix: np.ndarray, total_time: float, number_of_steps: int, random_state: Union[int, None]=None, n_jobs: int=1:
```

> Parallel generation of prices using the provided regimes.

:param number_of_paths: The number of paths to generate.
:param regimes: Dictionary containing regime names and their respective parameters.
:param transition_matrix: Markov Chain transition matrix.
:param total_time: Total time for the simulation.
:param number_of_steps: Number of discrete steps in the simulation.
:param random_state: Seed for random number generation.
:param n_jobs: Number of parallel jobs to run.
:return: A tuple containing the generated prices and simulated regimes as pandas DataFrames.


### 📄 `RiskLabAI\data\weights\sample_weights.py`

#### `function expand_label_for_meta_labeling`

```python
def expand_label_for_meta_labelingclose_index: pd.Index, timestamp: pd.Series, molecule: pd.Index:
```

> Expand labels for meta-labeling.

This function expands labels to incorporate meta-labeling by taking
an event Index, a Series with the return and label of each period,
and an Index specifying the molecules to apply the function to. It then returns a Series with the count
of events spanning a bar for each molecule.

:param event_index: Index of events.
:param return_label_dataframe: Series containing returns and labels of each period.
:param molecule_index: Index specifying molecules to apply the function on.
:return: Series with the count of events spanning a bar for each molecule.

#### `function calculate_sample_weight`

```python
def calculate_sample_weighttimestamp: pd.DataFrame, concurrency_events: pd.DataFrame, molecule: pd.Index:
```

> Calculate sample weight using triple barrier method.

:param timestamp: DataFrame of events start and end for labelling.
:param concurrency_events: Data frame of concurrent events for each event.
:param molecule: Index that function must apply on it.
:return: Series of sample weights.

#### `function create_index_matgrix`

```python
def create_index_matgrixbar_index: pd.Index, timestamp: pd.DataFrame:
```

> Create an indicator matrix.

:param bar_index: Index of all data.
:param timestamp: DataFrame with starting and ending times of events.
:return: Indicator matrix.

#### `function calculate_average_uniqueness`

```python
def calculate_average_uniquenessindex_matrix: pd.DataFrame:
```

> Calculate average uniqueness from indicator matrix.

:param index_matrix: Indicator matrix.
:return: Series of average uniqueness values.

#### `function perform_sequential_bootstrap`

```python
def perform_sequential_bootstrapindex_matrix: pd.DataFrame, sample_length: int:
```

> Perform sequential bootstrap to generate a sample.

:param index_matrix: Matrix of indicators for events.
:param sample_length: Number of samples.
:return: List of indices representing the sample.

#### `function calculate_sample_weight_absolute_return`

```python
def calculate_sample_weight_absolute_returntimestamp: pd.DataFrame, concurrency_events: pd.DataFrame, returns: pd.DataFrame, molecule: pd.Index:
```

> Calculate sample weight using absolute returns.

:param timestamp: DataFrame for events.
:param concurrency_events: DataFrame that contains number of concurrent events for each event.
:param returns: DataFrame that contains returns.
:param molecule: Index for the calculation.
:return: Series of sample weights.

#### `function sample_weight_absolute_return_meta_labeling`

```python
def sample_weight_absolute_return_meta_labelingtimestamp: pd.Series, price: pd.Series, molecule: pd.Index:
```

> Calculate sample weights using absolute returns.

:param event_timestamps: Series containing event timestamps.
:param price_series: Series containing prices.
:param molecule_index: Index for the calculation.
:return: Series of sample weights.

#### `function calculate_time_decay`

```python
def calculate_time_decayweight: pd.Series, clf_last_weight: float=1.0:
```

> Calculate time decay on weight.

:param weight: Weight computed for each event.
:param clf_last_weight: Weight of oldest observation.
:return: Series of weights after applying time decay.


### 📄 `RiskLabAI\ensemble\bagging_classifier_accuracy.py`

#### `function bagging_classifier_accuracy`

```python
def bagging_classifier_accuracyN: int, p: float, k: int=2:
```

> Calculate the accuracy of a bagging classifier.

The function calculates the accuracy of a bagging classifier based on the given
parameters and according to the formula:

.. math::
    1 - \sum_{i=0}^{N/k} \binom{N}{i} p^i (1-p)^{N-i}

:param N: Number of independent classifiers.
:param p: Probability of a classifier labeling a prediction as 1.
:param k: Number of classes (default is 2).
:return: Bagging classifier accuracy.

Reference:
    De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    Methodology: page 96, "Improved Accuracy" section.


### 📄 `RiskLabAI\features\entropy_features\entropy.py`


### 📄 `RiskLabAI\features\entropy_features\kontoyiannis.py`

#### `function longest_match_length`

```python
def longest_match_lengthmessage: str, i: int, n: int:
```

> Calculate the length of the longest match.

:param message: Input encoded message
:type message: str
:param i: Index value
:type i: int
:param n: Length parameter
:type n: int
:return: Tuple containing matched length and substring
:rtype: tuple

#### `function kontoyiannis_entropy`

```python
def kontoyiannis_entropymessage: str, window: int=None:
```

> Calculate Kontoyiannis Entropy.

:param message: Input encoded message
:type message: str
:param window: Length of expanding window, default is None
:type window: int or None
:return: Calculated Kontoyiannis Entropy
:rtype: float


### 📄 `RiskLabAI\features\entropy_features\lempel_ziv.py`

#### `function lempel_ziv_entropy`

```python
def lempel_ziv_entropymessage: str:
```

> Calculate Lempel-Ziv Entropy.

:param message: Input encoded message
:type message: str
:return: Calculated Lempel-Ziv Entropy
:rtype: float


### 📄 `RiskLabAI\features\entropy_features\plug_in.py`

#### `function plug_in_entropy_estimator`

```python
def plug_in_entropy_estimatormessage: str, approximate_word_length: int=1:
```

> Calculate Plug-in Entropy Estimator.

:param message: Input encoded message
:type message: str
:param approximate_word_length: Approximation of word length, default is 1
:type approximate_word_length: int
:return: Calculated Plug-in Entropy Estimator
:rtype: float


### 📄 `RiskLabAI\features\entropy_features\pmf.py`

#### `function probability_mass_function`

```python
def probability_mass_functionmessage: str, approximate_word_length: int:
```

> Calculate Probability Mass Function.

:param message: Input encoded message
:type message: str
:param approximate_word_length: Approximation of word length
:type approximate_word_length: int
:return: Probability Mass Function
:rtype: dict


### 📄 `RiskLabAI\features\entropy_features\shannon.py`

#### `function shannon_entropy`

```python
def shannon_entropymessage: str:
```

> Calculate Shannon Entropy.

:param message: Input encoded message
:type message: str
:return: Calculated Shannon Entropy
:rtype: float


### 📄 `RiskLabAI\features\feature_importance\clustered_feature_importance_mda.py`

#### `class ClusteredFeatureImportanceMDA`

##### `method __init__`

```python
def __init__:
```
##### `method compute`

```python
def computeself, classifier: RandomForestClassifier, x: pd.DataFrame, y: pd.Series, clusters: Dict[str, List[str]], n_splits: int=10, score_sample_weights: List[float]=None, train_sample_weights: List[float]=None:
```

> Compute clustered feature importance using MDA.

The feature importance is computed by comparing the performance
(log loss) of a trained classifier on shuffled data to its 
performance on non-shuffled data.

:param classifier: The Random Forest classifier to be trained.
:param x: The features DataFrame.
:param y: The target Series.
:param clusters: A dictionary where the keys are the cluster names
                 and the values are lists of features in each cluster.
:param n_splits: The number of splits for KFold cross-validation.
:param score_sample_weights: Sample weights to be used when computing the score.
:param train_sample_weights: Sample weights to be used during training.

:return: A DataFrame with feature importances and their standard deviations.

The related mathematical formulae:

.. math::

    \text{{importance}} = \frac{{-1 \times \text{{score with shuffled data}}}}
                          {{\text{{score without shuffled data}}}}

Using Central Limit Theorem for calculating the standard deviation:

.. math::

    \text{{StandardDeviation}} = \text{{std}} \times n^{-0.5}


### 📄 `RiskLabAI\features\feature_importance\clustered_feature_importance_mdi.py`

#### `class ClusteredFeatureImportanceMDI`

##### `method __init__`

```python
def __init__self, classifier: RandomForestClassifier, clusters: Dict[str, List[str]], x: pd.DataFrame, y: pd.Series:
```

> Initialize the ClusteredFeatureImportanceMDI class.

:param classifier: The Random Forest classifier.
:param clusters: A dictionary where the keys are the cluster names 
                 and the values are lists of features in each cluster.
:param x: The features DataFrame.
:param y: The target Series.

##### `method group_mean_std`

```python
def group_mean_stdself, dataframe: pd.DataFrame, clusters: Dict[str, List[str]]:
```

> Calculate the mean and standard deviation for clusters.

:param dataframe: A DataFrame of importances.
:param clusters: A dictionary of cluster definitions.

:return: A DataFrame with mean and standard deviation for each cluster.

Using Central Limit Theorem for standard deviation:

.. math::

    \text{{StandardDeviation}} = \text{{std}} \times n^{-0.5}

##### `method compute`

```python
def computeself:
```

> Compute aggregated feature importances for clusters.

:return: A DataFrame with aggregated importances for clusters.


### 📄 `RiskLabAI\features\feature_importance\clustering.py`

#### `function covariance_to_correlation`

```python
def covariance_to_correlationcovariance: np.ndarray:
```

> Derive the correlation matrix from a covariance matrix.

:param covariance: numpy ndarray
    The covariance matrix to convert to a correlation matrix.
:return: numpy ndarray
    The correlation matrix derived from the covariance matrix.

The conversion is done based on the following mathematical formula:
correlation = covariance / (std_i * std_j)
where std_i and std_j are the standard deviations of the i-th and j-th elements.

#### `function cluster_kmeans_base`

```python
def cluster_kmeans_basecorrelation: pd.DataFrame, number_clusters: int=10, iterations: int=10:
```

> Apply the K-means clustering algorithm.

:param correlation: pandas DataFrame
    The correlation matrix.
:param number_clusters: int, optional
    The maximum number of clusters. Default is 10.
:param iterations: int, optional
    The number of iterations to run the clustering. Default is 10.
:return: tuple
    A tuple containing the sorted correlation matrix, cluster membership, and silhouette scores.

This function is based on Snippet 4.1 from De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.

#### `function make_new_outputs`

```python
def make_new_outputscorrelation: pd.DataFrame, clusters: dict, clusters2: dict:
```

> Merge two sets of clusters and derive new outputs.

:param correlation: pandas DataFrame
    The correlation matrix.
:param clusters: dict
    The first set of clusters.
:param clusters2: dict
    The second set of clusters.
:return: tuple
    A tuple containing the new correlation matrix, new cluster membership, and new silhouette scores.

#### `function cluster_kmeans_top`

```python
def cluster_kmeans_topcorrelation: pd.DataFrame, number_clusters: int=None, iterations: int=10:
```

> Apply the K-means clustering algorithm with hierarchical re-clustering.

:param correlation: pandas DataFrame
    The correlation matrix.
:param number_clusters: int, optional
    The maximum number of clusters. Default is None.
:param iterations: int, optional
    The number of iterations to run the clustering. Default is 10.
:return: tuple
    A tuple containing the sorted correlation matrix, cluster membership, and silhouette scores.

This function is based on Snippet 4.2 from De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.

#### `function random_covariance_sub`

```python
def random_covariance_subnumber_observations: int, number_columns: int, sigma: float, random_state=None:
```

> Compute a sub covariance matrix.

Generates a covariance matrix based on random data.

:param number_observations: Number of observations.
:param number_columns: Number of columns.
:param sigma: Sigma for normal distribution.
:param random_state: Random state for reproducibility.
:return: Sub covariance matrix.

.. note:: Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
   Methodology: Snipet 4.3, Page 61.

#### `function random_block_covariance`

```python
def random_block_covariancenumber_columns: int, number_blocks: int, block_size_min: int=1, sigma: float=1.0, random_state=None:
```

> Compute a random block covariance matrix.

Generates a block random covariance matrix by combining multiple sub covariance matrices.

:param number_columns: Number of columns.
:param number_blocks: Number of blocks.
:param block_size_min: Minimum size of block.
:param sigma: Sigma for normal distribution.
:param random_state: Random state for reproducibility.
:return: Block random covariance matrix.

.. note:: Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
   Methodology: Snipet 4.3, Page 61.

#### `function random_block_correlation`

```python
def random_block_correlationnumber_columns: int, number_blocks: int, random_state=None, block_size_min: int=1:
```

> Compute a random block correlation matrix.

Generates a block random correlation matrix by adding two block random covariance matrices
and converting them to a correlation matrix.

:param number_columns: Number of columns.
:param number_blocks: Number of blocks.
:param random_state: Random state for reproducibility.
:param block_size_min: Minimum size of block.
:return: Block random correlation matrix.

.. note:: Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
   Methodology: Snipet 4.3, Page 61.

#### `function cov_to_corr`

```python
def cov_to_corrcovariance: np.ndarray:
```

> Convert a covariance matrix to a correlation matrix.

:param covariance: Covariance matrix.
:return: Correlation matrix.

.. math::
    correlation_{ij} = \\frac{covariance_{ij}}{\\sqrt{covariance_{ii} \cdot covariance_{jj}}}

#### `function cluster_kmeans_base`

```python
def cluster_kmeans_basecorrelation: pd.DataFrame, number_clusters: int=10, iterations: int=10:
```

> Perform KMeans clustering on a correlation matrix.

:param correlation: Correlation matrix.
:param number_clusters: Number of clusters, default is 10.
:param iterations: Number of iterations, default is 10.
:return: Sorted correlation matrix, clusters, silhouette scores.

.. note::
    The silhouette score is a measure of how similar an object is to its own cluster compared to other clusters.


### 📄 `RiskLabAI\features\feature_importance\feature_importance_controller.py`

#### `class FeatureImportanceController`

> Controller class to manage various feature importance strategies.

To use this controller class:

1. Initialize it with the type of feature importance strategy 
   you want to use, along with any required parameters for that strategy.
2. Call the `calculate_importance` method to perform the 
   feature importance calculation.

For example:

.. code-block:: python

   # Initialize the controller with a 'ClusteredMDA' strategy
   controller = FeatureImportanceController('ClusteredMDA', 
                                            classifier=my_classifier, 
                                            clusters=my_clusters)

   # Calculate feature importance
   result = controller.calculate_importance(my_x, my_y)

##### `method __init__`

```python
def __init__self, strategy_type: str, **kwargs:
```

> Initialize the controller with a specific feature importance strategy.

:param strategy_type: The type of feature importance strategy to use.
:param kwargs: Additional arguments to pass to the strategy class.

##### `method calculate_importance`

```python
def calculate_importanceself, x, y, **kwargs:
```

> Calculate feature importance based on the initialized strategy.

:param x: Feature data.
:param y: Target data.
:param kwargs: Additional arguments to pass to the calculation method.

:return: Feature importance results.


### 📄 `RiskLabAI\features\feature_importance\feature_importance_factory.py`

#### `class FeatureImportanceFactory`

> Factory class for building and fetching feature importance computation results.

Usage:

.. code-block:: python

    factory = FeatureImportanceFactory()
    factory.build(my_feature_importance_strategy_instance)
    results = factory.get_results()

##### `method __init__`

```python
def __init__self:
```

> Initialize the FeatureImportanceFactory class.

##### `method build`

```python
def buildself, feature_importance_strategy: FeatureImportanceStrategy:
```

> Build the feature importance based on the provided strategy.

:param feature_importance_strategy: An instance of a strategy 
    inheriting from FeatureImportanceStrategy.

:return: Current instance of the FeatureImportanceFactory.

##### `method get_results`

```python
def get_resultsself:
```

> Fetch the computed feature importance results.

:return: Dataframe containing the feature importance results.


### 📄 `RiskLabAI\features\feature_importance\feature_importance_mda.py`

#### `class FeatureImportanceMDA`

> Computes the feature importance using the Mean Decrease Accuracy (MDA) method.

The method shuffles each feature one by one and measures how much the performance 
(log loss in this context) decreases due to the shuffling.

.. math::

    \text{importance}_{j} = \frac{\text{score without shuffling} - \text{score with shuffling}_{j}}
    {\text{score without shuffling}}

##### `method __init__`

```python
def __init__self, classifier: object, x: pd.DataFrame, y: pd.Series, n_splits: int=10, score_sample_weights: Optional[List[float]]=None, train_sample_weights: Optional[List[float]]=None:
```

> Initialize the class with parameters.

:param classifier: The classifier object.
:param x: The feature data.
:param y: The target data.
:param n_splits: Number of splits for cross-validation.
:param score_sample_weights: Weights for scoring samples.
:param train_sample_weights: Weights for training samples.

##### `method compute`

```python
def computeself:
```

> Compute the feature importances.

:return: Feature importances as a dataframe with "Mean" and "StandardDeviation" columns.


### 📄 `RiskLabAI\features\feature_importance\feature_importance_mdi.py`

#### `class FeatureImportanceMDI`

> Computes the feature importance using the Mean Decrease Impurity (MDI) method.

The method calculates the importance of a feature by measuring the average impurity
decrease across all the trees in the forest, where impurity is calculated 
using metrics like Gini impurity or entropy.

.. math::

    \text{importance}_{j} = \frac{\text{average impurity decrease for feature j}}{\text{total impurity decrease}}

##### `method __init__`

```python
def __init__self, classifier: object, x: pd.DataFrame, y: Union[pd.Series, List[Optional[float]]]:
```

> Initialize the class with parameters.

:param classifier: The classifier object.
:param x: The feature data.
:param y: The target data.

##### `method compute`

```python
def computeself:
```

> Compute the feature importances.

:return: Feature importances as a dataframe with "Mean" and "StandardDeviation" columns.


### 📄 `RiskLabAI\features\feature_importance\feature_importance_sfi.py`

#### `class FeatureImportanceSFI`

> Computes the Single Feature Importance (SFI).

The method calculates the importance of each feature by evaluating its performance 
individually in the classifier.

##### `method __init__`

```python
def __init__self, classifier: object, x: pd.DataFrame, y: Union[pd.Series, List[Optional[float]]], n_splits: int=10, score_sample_weights: Optional[List[float]]=None, train_sample_weights: Optional[List[float]]=None, scoring: str='log_loss':
```

> Initialize the class with parameters.

:param classifier: The classifier object.
:param x: The feature data.
:param y: The target data.
:param n_splits: The number of splits for cross-validation.
:param score_sample_weights: Sample weights for scoring.
:param train_sample_weights: Sample weights for training.
:param scoring: Scoring method ("log_loss" or "accuracy").

##### `method compute`

```python
def computeself:
```

> Compute the Single Feature Importance.

:return: Feature importances as a dataframe with "FeatureName", "Mean", and "StandardDeviation" columns.


### 📄 `RiskLabAI\features\feature_importance\feature_importance_strategy.py`

#### `class FeatureImportanceStrategy`

> Abstract Base Class for computing feature importance.

Derived classes must implement the `compute` method to 
provide their own logic for computing feature importance.

##### `method compute`

```python
def computeself, *args, **kwargs:
```

> Abstract method to compute feature importance.

:param args: Positional arguments.
:param kwargs: Keyword arguments.
:return: A pandas DataFrame containing feature importances.

Note: Derived classes should provide a concrete implementation 
of this method with specific parameters and docstrings relevant 
to their implementation.


### 📄 `RiskLabAI\features\feature_importance\generate_synthetic_data.py`

#### `function get_test_dataset`

```python
def get_test_datasetn_features: int=100, n_informative: int=25, n_redundant: int=25, n_samples: int=10000, random_state: int=0, sigma_std: float=0.0:
```

> Generate a synthetic dataset with informative, redundant, and explanatory variables.

:param n_features: Total number of features
:type n_features: int
:param n_informative: Number of informative features
:type n_informative: int
:param n_redundant: Number of redundant features
:type n_redundant: int
:param n_samples: Number of samples to generate
:type n_samples: int
:param random_state: Random state for reproducibility
:type random_state: int
:param sigma_std: Standard deviation for generating redundant features, default is 0.0
:type sigma_std: float
:return: Tuple containing generated X (features) and y (labels)
:rtype: tuple


### 📄 `RiskLabAI\features\feature_importance\orthogonal_features.py`

#### `function compute_eigenvectors`

```python
def compute_eigenvectorsdot_product: np.ndarray, explained_variance_threshold: float:
```

> Compute eigenvalues and eigenvectors for orthogonal features.

:param dot_product: Input dot product matrix.
:type dot_product: np.ndarray
:param explained_variance_threshold: Threshold for cumulative explained variance.
:type explained_variance_threshold: float
:return: DataFrame containing eigenvalues, eigenvectors, and cumulative explained variance.
:rtype: pd.DataFrame

#### `function orthogonal_features`

```python
def orthogonal_featuresfeatures: np.ndarray, variance_threshold: float=0.95:
```

> Compute orthogonal features using eigenvalues and eigenvectors.

:param features: Features matrix.
:type features: np.ndarray
:param variance_threshold: Threshold for cumulative explained variance, default is 0.95.
:type variance_threshold: float
:return: Tuple containing orthogonal features and eigenvalues information.
:rtype: tuple


### 📄 `RiskLabAI\features\feature_importance\weighted_tau.py`

#### `function calculate_weighted_tau`

```python
def calculate_weighted_taufeature_importances: np.ndarray, principal_component_ranks: np.ndarray:
```

> Calculate the weighted Kendall's tau (τ) using feature importances and principal component ranks.

Kendall's tau is a measure of correlation between two rankings. The weighted version of
Kendall's tau takes into account the weights of the rankings. In this case, the weights
are the inverse of the principal component ranks.

:param feature_importances: Vector of feature importances.
:type feature_importances: np.ndarray
:param principal_component_ranks: Vector of principal component ranks.
:type principal_component_ranks: np.ndarray
:return: Weighted τ value.
:rtype: float

.. math::

    \\tau_B = \\frac{(P - Q)}{\\sqrt{(P + Q + T) (P + Q + U)}}

where:
    - P is the number of concordant pairs
    - Q is the number of discordant pairs
    - T is the number of ties only in the first ranking
    - U is the number of ties only in the second ranking


### 📄 `RiskLabAI\features\microstructural_features\bekker_parkinson_volatility_estimator.py`

#### `function sigma_estimates`

```python
def sigma_estimatesbeta: pd.Series, gamma: pd.Series:
```

> Compute Bekker-Parkinson volatility σ estimates.

This function calculates the Bekker-Parkinson volatility estimates based on the provided
beta and gamma values. The mathematical formula used is:

.. math::
    \sigma = \frac{(2^{0.5} - 1) \cdot (\beta ^ {0.5})}{3 - 2 \cdot (2^{0.5})}
            + \left(\frac{\gamma}{\left(\frac{8}{\pi}\right)^{0.5} \cdot (3 - 2 \cdot (2^{0.5}))}\right)^{0.5}

Negative resulting values are set to 0.

:param beta: β Estimates vector.
:param gamma: γ Estimates vector.
:return: Bekker-Parkinson volatility σ estimates.

Reference:
    De Prado, M. (2018) Advances in Financial Machine Learning, page 286, snippet 19.2.

#### `function bekker_parkinson_volatility_estimates`

```python
def bekker_parkinson_volatility_estimateshigh_prices: pd.Series, low_prices: pd.Series, window_span: int=20:
```

> Compute Bekker-Parkinson volatility estimates based on high and low prices.

Utilizes Corwin and Schultz estimation techniques to calculate the Bekker-Parkinson
volatility. The function first determines the beta and gamma values and then
uses them to compute the volatility estimates.

:param high_prices: High prices vector.
:param low_prices: Low prices vector.
:param window_span: Rolling window span for beta estimation.
:return: Bekker-Parkinson volatility estimates.

Reference:
    De Prado, M. (2018) Advances in Financial Machine Learning, page 286, "Corwin and Schultz" section.


### 📄 `RiskLabAI\features\microstructural_features\corwin_schultz.py`

#### `function beta_estimates`

```python
def beta_estimateshigh_prices: pd.Series, low_prices: pd.Series, window_span: int:
```

> Estimate β using Corwin and Schultz methodology.

:param high_prices: High prices vector.
:param low_prices: Low prices vector.
:param window_span: Rolling window span.
:return: Estimated β vector.

.. note:: Reference: Corwin, S. A., & Schultz, P. (2012). A simple way to estimate bid-ask spreads from daily high and low prices. The Journal of Finance, 67(2), 719-760.

#### `function gamma_estimates`

```python
def gamma_estimateshigh_prices: pd.Series, low_prices: pd.Series:
```

> Estimate γ using Corwin and Schultz methodology.

:param high_prices: High prices vector.
:param low_prices: Low prices vector.
:return: Estimated γ vector.

.. note:: Reference: Corwin, S. A., & Schultz, P. (2012). A simple way to estimate bid-ask spreads from daily high and low prices. The Journal of Finance, 67(2), 719-760.

#### `function alpha_estimates`

```python
def alpha_estimatesbeta: pd.Series, gamma: pd.Series:
```

> Estimate α using Corwin and Schultz methodology.

:param beta: β Estimates vector.
:param gamma: γ Estimates vector.
:return: Estimated α vector.

.. note:: Reference: Corwin, S. A., & Schultz, P. (2012). A simple way to estimate bid-ask spreads from daily high and low prices. The Journal of Finance, 67(2), 719-760.

#### `function corwin_schultz_estimator`

```python
def corwin_schultz_estimatorhigh_prices: pd.Series, low_prices: pd.Series, window_span: int=20:
```

> Estimate spread using Corwin and Schultz methodology.

:param high_prices: High prices vector.
:param low_prices: Low prices vector.
:param window_span: Rolling window span, default is 20.
:return: Estimated spread vector.

.. note:: Reference: Corwin, S. A., & Schultz, P. (2012). A simple way to estimate bid-ask spreads from daily high and low prices. The Journal of Finance, 67(2), 719-760.


### 📄 `RiskLabAI\features\structural_breaks\structural_breaks.py`

#### `function lag_dataframe`

```python
def lag_dataframemarket_data: pd.DataFrame, lags: int:
```

> Apply lags to DataFrame.

Reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
Methodology: Snippet 17.3

:param market_data: Data of price or log price.
:param lags: Arrays of lag or integer that shows number of lags.
:return: DataFrame with lagged data.

#### `function prepare_data`

```python
def prepare_dataseries: pd.DataFrame, constant: str, lags: int:
```

> Prepare the datasets.

Reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
Methodology: Snippet 17.2

:param series: Data of price or log price.
:param constant: String that must be "nc" or "ct" or "ctt".
:param lags: Arrays of lag or integer that shows number of lags.
:return: Tuple of y and x arrays.

#### `function compute_beta`

```python
def compute_betay: np.ndarray, x: np.ndarray:
```

> Fit the ADF specification.

Reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
Methodology: Snippet 17.4

:param y: Dependent variable.
:param x: Matrix of independent variable.
:return: Tuple of beta_mean and beta_variance.

#### `function adf`

```python
def adflog_price: pd.DataFrame, min_sample_length: int, constant: str, lags: int:
```

> SADF's inner loop.

Reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
Methodology: Snippet 17.1

:param log_price: Pandas DataFrame of log price.
:param min_sample_length: Minimum sample length.
:param constant: String that must be "nc" or "ct" or "ctt".
:param lags: Arrays of lag or integer that shows number of lags.
:return: Dictionary with Time and gsadf values.


### 📄 `RiskLabAI\hpc\hpc.py`

#### `function report_progress`

```python
def report_progressjob_number: int, total_jobs: int, start_time: float, task: str:
```

> Report the progress of a computing task.

:param job_number: The current job number.
:type job_number: int
:param total_jobs: The total number of jobs.
:type total_jobs: int
:param start_time: The start time of the computation.
:type start_time: float
:param task: The task being performed.
:type task: str
:return: None

#### `function process_jobs`

```python
def process_jobsjobs: list, task: str=None, num_threads: int=24:
```

> Process multiple jobs in parallel.

:param jobs: A list of jobs to be processed.
:type jobs: list
:param task: The task being performed.
:type task: str
:param num_threads: Number of threads to be used.
:type num_threads: int
:return: Outputs of the jobs.
:rtype: list

#### `function expand_call`

```python
def expand_callkargs: dict:
```

> Expand the arguments of a callback function, kargs['func'].

:param kargs: Arguments for the callback function.
:type kargs: dict
:return: Output of the callback function.

#### `function process_jobs_sequential`

```python
def process_jobs_sequentialjobs: list:
```

> Single-thread execution, for debugging.

:param jobs: A list of jobs to be processed.
:type jobs: list
:return: Outputs of the jobs.
:rtype: list

#### `function linear_partitions`

```python
def linear_partitionsnum_atoms: int, num_threads: int:
```

> Generate linear partitions for parallel computation.

:param num_atoms: Number of atoms.
:type num_atoms: int
:param num_threads: Number of threads.
:type num_threads: int
:return: The partitions.
:rtype: list

#### `function nested_partitions`

```python
def nested_partitionsnum_atoms: int, num_threads: int, upper_triangle: bool=False:
```

> Generate nested partitions for parallel computation.

:param num_atoms: Number of atoms.
:type num_atoms: int
:param num_threads: Number of threads.
:type num_threads: int
:param upper_triangle: Whether to generate partitions for the upper triangle.
:type upper_triangle: bool
:return: The partitions.
:rtype: list

The formula for partition size is given by:

.. math::

   partitions = \frac{-1 + \sqrt{1 + 4 \cdot (partitions[-1]^2 + partitions[-1] + \frac{n_atoms \cdot (n_atoms + 1)}{n_threads})}}{2}

#### `function mp_pandas_obj`

```python
def mp_pandas_objfunction, pandas_object: tuple, num_threads: int=2, mp_batches: int=1, linear_partition: bool=True, **kwargs:
```

> Parallelize jobs and return a DataFrame or Series.

:param function: The function to be parallelized.
:param pandas_object: A tuple containing the name of the argument used to pass the molecule and a list of atoms
                      that will be grouped into molecules.
:type pandas_object: tuple
:param num_threads: Number of threads to be used.
:type num_threads: int
:param mp_batches: Number of batches for multiprocessing.
:type mp_batches: int
:param linear_partition: Whether to use linear partitioning or nested partitioning.
:type linear_partition: bool
:param kwargs: Other arguments needed by the function.
:return: The result of the function parallelized.
:rtype: DataFrame or Series


### 📄 `RiskLabAI\optimization\hrp.py`

#### `function inverse_variance_weights`

```python
def inverse_variance_weightscovariance_matrix: pd.DataFrame:
```

> Compute the inverse-variance portfolio weights.

:param covariance_matrix: Covariance matrix of asset returns.
:type covariance_matrix: pd.DataFrame
:return: Array of portfolio weights.
:rtype: np.ndarray

#### `function cluster_variance`

```python
def cluster_variancecovariance_matrix: pd.DataFrame, clustered_items: list:
```

> Compute the variance of a cluster.

:param covariance_matrix: Covariance matrix of asset returns.
:type covariance_matrix: pd.DataFrame
:param clustered_items: List of indices of assets in the cluster.
:type clustered_items: list
:return: Variance of the cluster.
:rtype: float

#### `function quasi_diagonal`

```python
def quasi_diagonallinkage_matrix: np.ndarray:
```

> Return a sorted list of original items to reshape the correlation matrix.

:param linkage_matrix: Linkage matrix obtained from hierarchical clustering.
:type linkage_matrix: np.ndarray
:return: Sorted list of original items.
:rtype: list

#### `function recursive_bisection`

```python
def recursive_bisectioncovariance_matrix: pd.DataFrame, sorted_items: list:
```

> Compute the Hierarchical Risk Parity (HRP) weights.

:param covariance_matrix: Covariance matrix of asset returns.
:type covariance_matrix: pd.DataFrame
:param sorted_items: Sorted list of original items.
:type sorted_items: list
:return: DataFrame of asset weights.
:rtype: pd.Series

#### `function distance_corr`

```python
def distance_corrcorr_matrix: np.ndarray:
```

> Compute the distance matrix based on correlation.

:param corr_matrix: Correlation matrix.
:type corr_matrix: np.ndarray
:return: Distance matrix based on correlation.
:rtype: np.ndarray

#### `function plot_corr_matrix`

```python
def plot_corr_matrixpath: str, corr_matrix: np.ndarray, labels: list=None:
```

> Plot a heatmap of the correlation matrix.

:param path: Path to save the plot.
:type path: str
:param corr_matrix: Correlation matrix.
:type corr_matrix: np.ndarray
:param labels: List of labels for the assets (optional).
:type labels: list, optional

#### `function random_data`

```python
def random_datanum_observations: int, size_uncorr: int, size_corr: int, sigma_corr: float:
```

> Generate random data.

:param num_observations: Number of observations.
:type num_observations: int
:param size_uncorr: Size for uncorrelated data.
:type size_uncorr: int
:param size_corr: Size for correlated data.
:type size_corr: int
:param sigma_corr: Standard deviation for correlated data.
:type sigma_corr: float
:return: DataFrame of randomly generated data and list of column indices for correlated data.
:rtype: pd.DataFrame, list

#### `function random_data2`

```python
def random_data2number_observations: int, length_sample: int, size_uncorrelated: int, size_correlated: int, mu_uncorrelated: float, sigma_uncorrelated: float, sigma_correlated: float:
```

> Generate random data for Monte Carlo simulation.

:param number_observations: Number of observations.
:type number_observations: int
:param length_sample: Starting point for selecting random observations.
:type length_sample: int
:param size_uncorrelated: Size of uncorrelated data.
:type size_uncorrelated: int
:param size_correlated: Size of correlated data.
:type size_correlated: int
:param mu_uncorrelated: mu for uncorrelated data.
:type mu_uncorrelated: float
:param sigma_uncorrelated: sigma for uncorrelated data.
:type sigma_uncorrelated: float
:param sigma_correlated: sigma for correlated data.
:type sigma_correlated: float
:return: A tuple containing the generated data and the selected columns.
:rtype: np.ndarray, list

#### `function hrp`

```python
def hrpcov: np.ndarray, corr: np.ndarray:
```

> HRP method for constructing a hierarchical portfolio.

:param cov: Covariance matrix.
:type cov: np.ndarray
:param corr: Correlation matrix.
:type corr: np.ndarray
:return: Pandas series containing weights of the hierarchical portfolio.
:rtype: pd.Series

#### `function hrp_mc`

```python
def hrp_mcnumber_iterations: int=5000, number_observations: int=520, size_uncorrelated: int=5, size_correlated: int=5, mu_uncorrelated: float=0, sigma_uncorrelated: float=0.01, sigma_correlated: float=0.25, length_sample: int=260, test_size: int=22:
```

> Monte Carlo simulation for out of sample comparison of HRP method.

:param number_iterations: Number of iterations.
:type number_iterations: int
:param number_observations: Number of observations.
:type number_observations: int
:param size_uncorrelated: Size of uncorrelated data.
:type size_uncorrelated: int
:param size_correlated: Size of correlated data.
:type size_correlated: int
:param mu_uncorrelated: mu for uncorrelated data.
:type mu_uncorrelated: float
:param sigma_uncorrelated: sigma for uncorrelated data.
:type sigma_uncorrelated: float
:param sigma_correlated: sigma for correlated data.
:type sigma_correlated: float
:param length_sample: Length for in sample.
:type length_sample: int
:param test_size: Observation for test set.
:type test_size: int
:return: None


### 📄 `RiskLabAI\optimization\hyper_parameter_tuning.py`

#### `class MyPipeline`

> Custom pipeline class to include sample_weight in fit_params.

##### `method fit`

```python
def fitself, X: pd.DataFrame, y: pd.DataFrame, sample_weight: list=None, **fit_params:
```

> Fit the pipeline while considering sample weights.

:param X: Feature data.
:param y: Labels of data.
:param sample_weight: Sample weights for fit, defaults to None.
:param **fit_params: Additional fit parameters.
:return: Fitted pipeline.

#### `function clf_hyper_fit`

```python
def clf_hyper_fitfeature_data: pd.DataFrame, label: pd.DataFrame, times: pd.Series, pipe_clf: Pipeline, param_grid: dict, validator_type: str='purgedkfold', validator_params: dict=None, bagging: list=[0, -1, 1.0], rnd_search_iter: int=0, n_jobs: int=-1, **fit_params:
```

> Perform hyperparameter tuning and model fitting.

:param feature_data: Data of features.
:param label: Labels of data.
:param times: The timestamp series associated with the labels.
:param pipe_clf: Our estimator.
:param param_grid: Parameter space.
:param validator_type: Type of cross-validator to create.
:param validator_params: Additional keyword arguments to be passed to the cross-validator's constructor.
:param bagging: Bagging type, defaults to [0, -1, 1.].
:param rnd_search_iter: Number of iterations for randomized search, defaults to 0.
:param n_jobs: Number of jobs for parallel processing, defaults to -1.
:param **fit_params: Additional fit parameters.
:return: Fitted pipeline.


### 📄 `RiskLabAI\optimization\nco.py`

#### `function covariance_to_correlation_matrix`

```python
def covariance_to_correlation_matrixcovariance: np.ndarray:
```

> Derive the correlation matrix from a covariance matrix.

:param covariance: Covariance matrix.
:type covariance: numpy.ndarray
:return: Correlation matrix.
:rtype: numpy.ndarray

#### `function get_optimal_portfolio_weights`

```python
def get_optimal_portfolio_weightscovariance: np.ndarray, mu: np.ndarray=None:
```

> Compute the optimal portfolio weights.

:param covariance: Covariance matrix.
:type covariance: numpy.ndarray
:param mu: Mean vector, defaults to None.
:type mu: numpy.ndarray, optional
:return: Portfolio weights.
:rtype: numpy.ndarray

#### `function get_optimal_portfolio_weights_nco`

```python
def get_optimal_portfolio_weights_ncocovariance: np.ndarray, mu: np.ndarray=None, number_clusters: int=None:
```

> Compute the optimal portfolio weights using the NCO algorithm.

:param covariance: Covariance matrix.
:type covariance: numpy.ndarray
:param mu: Mean vector, defaults to None.
:type mu: numpy.ndarray, optional
:param number_clusters: Maximum number of clusters, defaults to None.
:type number_clusters: int, optional
:return: Optimal portfolio weights using NCO algorithm.
:rtype: numpy.ndarray

#### `function cluster_k_means_base`

```python
def cluster_k_means_basecorrelation: pd.DataFrame, number_clusters: int=10, iterations: int=10:
```

> Perform clustering using the K-means algorithm.

:param correlation: Correlation matrix.
:type correlation: pd.DataFrame
:param number_clusters: Maximum number of clusters, defaults to 10.
:type number_clusters: int, optional
:param iterations: Number of iterations, defaults to 10.
:type iterations: int, optional
:return: Updated correlation matrix, cluster members, silhouette scores.
:rtype: tuple


### 📄 `RiskLabAI\pde\equation.py`

#### `class Equation`

> Base class for defining PDE related function.

Args:
eqn_config (dict): dictionary containing PDE configuration parameters

Attributes:
dim (int): dimensionality of the problem
total_time (float): total time horizon
num_time_interval (int): number of time steps
delta_t (float): time step size
sqrt_delta_t (float): square root of time step size
y_init (None): initial value of the function

##### `method __init__`

```python
def __init__self, eqn_config: dict:
```
##### `method sample`

```python
def sampleself, num_sample: int:
```

> Sample forward SDE.

Args:
num_sample (int): number of samples to generate

Returns:
Tensor: tensor of size [num_sample, dim+1] containing samples

##### `method r_u`

```python
def r_uself, t: float, x: Tensor, y: Tensor, z: Tensor:
```

> Interest rate in the PDE.

Args:
t (float): current time
x (Tensor): tensor of size [batch_size, dim] containing space coordinates
y (Tensor): tensor of size [batch_size, 1] containing function values
z (Tensor): tensor of size [batch_size, dim] containing gradients

Returns:
Tensor: tensor of size [batch_size, 1] containing generator values

##### `method h_z`

```python
def h_zself, t, x, y, z: Tensor:
```

> Function to compute H(z) in the PDE.

Args:
h (float): value of H function
z (Tensor): tensor of size [batch_size, dim] containing gradients

Returns:
Tensor: tensor of size [batch_size, dim] containing H(z)

##### `method terminal`

```python
def terminalself, t: float, x: Tensor:
```

> Terminal condition of the PDE.

Args:
t (float): current time
x (Tensor): tensor of size [batch_size, dim] containing space coordinates

Returns:
Tensor: tensor of size [batch_size, 1] containing terminal values

#### `class PricingDefaultRisk`

> Args:
eqn_config (dict): dictionary containing PDE configuration parameters

##### `method __init__`

```python
def __init__self, eqn_config:
```
##### `method sample`

```python
def sampleself, num_sample:
```

> Sample forward SDE.

Args:
num_sample (int): number of samples to generate

Returns:
tuple: tuple of two tensors: dw_sample of size [num_sample, dim, num_time_interval] and
x_sample of size [num_sample, dim, num_time_interval+1]

##### `method r_u`

```python
def r_uself, t, x, y, z:
```

> Interest rate in the PDE.

Args:
t (float): current time
x (torch.Tensor): tensor of size [batch_size, dim] containing space coordinates
y (torch.Tensor): tensor of size [batch_size, 1] containing function values
z (torch.Tensor): tensor of size [batch_size, dim] containing gradients

Returns:
torch.Tensor: tensor of size [batch_size, 1] containing generator values

##### `method h_z`

```python
def h_zself, t, x, y, z:
```

> Function to compute $h^T Z$ in the PDE.

Args:
t (float): current time
x (torch.Tensor): tensor of size [batch_size, dim] containing space coordinates
y (torch.Tensor): tensor of size [batch_size, 1] containing function value
z (torch.Tensor): tensor of size [batch_size, dim] containing gradients

Returns:
torch.Tensor: tensor of size [batch_size, 1] containing H(z)

##### `method sigma_matrix`

```python
def sigma_matrixself, x:
```
##### `method terminal`

```python
def terminalself, t, x:
```

> Terminal condition of the PDE.

Args:
t (float): current time
x (torch.Tensor): tensor of size [batch_size, dim] containing space coordinates

Returns:
torch.Tensor: tensor of size [batch_size, 1] containing terminal values

##### `method terminal_for_sample`

```python
def terminal_for_sampleself, x:
```

> Terminal condition of the PDE.

Args:
x (torch.Tensor): tensor of size [num_sample,batch_size, dim] containing space coordinates

Returns:
torch.Tensor: tensor of size [num_sample ,batch_size, 1] containing terminal values

#### `class HJBLQ`

> Args:
eqn_config (dict): dictionary containing PDE configuration parameters

##### `method __init__`

```python
def __init__self, eqn_config: dict:
```
##### `method sample`

```python
def sampleself, num_sample: int:
```

> Sample forward SDE.

Args:
num_sample (int): number of samples to generate

Returns:
tuple: tuple of two tensors: dw_sample of size [num_sample, dim, num_time_interval] and
x_sample of size [num_sample, dim, num_time_interval+1]

##### `method r_u`

```python
def r_uself, t: float, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor:
```

> Interest rate in the PDE.

Args:
t (float): current time
x (torch.Tensor): tensor of size [batch_size, dim] containing space coordinates
y (torch.Tensor): tensor of size [batch_size, 1] containing function values
z (torch.Tensor): tensor of size [batch_size, dim] containing gradients

Returns:
torch.Tensor: tensor of size [batch_size, 1] containing generator values

##### `method h_z`

```python
def h_zself, t: float, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor:
```

> Function to compute <h,z> in the PDE.

Args:
t (float): current time
x (torch.Tensor): tensor of size [batch_size, dim] containing space coordinates
y (torch.Tensor): tensor of size [batch_size, 1] containing function value
z (torch.Tensor): tensor of size [batch_size, dim] containing gradients

Returns:
torch.Tensor: tensor of size [batch_size, 1] containing H(z)

##### `method terminal`

```python
def terminalself, t: float, x: torch.Tensor:
```

> Terminal condition of the PDE.

Args:
t (float): current time
x (torch.Tensor): tensor of size [batch_size, dim] containing space coordinates

Returns:
torch.Tensor: tensor of size [batch_size, 1] containing terminal values

##### `method sigma_matrix`

```python
def sigma_matrixself, x:
```
#### `function terminal_for_sample`

```python
def terminal_for_sampleself, x:
```

> Terminal condition of the PDE.

Args:
x (torch.Tensor): tensor of size [num_sample,batch_size, dim] containing space coordinates

Returns:
torch.Tensor: tensor of size [num_sample ,batch_size, 1] containing terminal values

#### `class BlackScholesBarenblatt`

> Args:
eqn_config (dict): dictionary containing PDE configuration parameters

##### `method __init__`

```python
def __init__self, eqn_config: dict:
```
##### `method sample`

```python
def sampleself, num_sample: int:
```

> Sample forward SDE.

Args:
num_sample (int): number of samples to generate

Returns:
tuple: tuple of two tensors: dw_sample of size [num_sample, dim, num_time_interval] and
x_sample of size [num_sample, dim, num_time_interval+1]

##### `method r_u`

```python
def r_uself, t: float, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor:
```

> Interest rate in the PDE.

Args:
t (float): current time
x (torch.Tensor): tensor of size [batch_size, dim] containing space coordinates
y (torch.Tensor): tensor of size [batch_size, 1] containing function values
z (torch.Tensor): tensor of size [batch_size, dim] containing gradients

Returns:
torch.Tensor: tensor of size [batch_size, 1] containing generator values

##### `method h_z`

```python
def h_zself, t: float, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor:
```

> Function to compute <h,z> in the PDE.

Args:
t (float): current time
x (torch.Tensor): tensor of size [batch_size, dim] containing space coordinates
y (torch.Tensor): tensor of size [batch_size, 1] containing function value
z (torch.Tensor): tensor of size [batch_size, dim] containing gradients

Returns:
torch.Tensor: tensor of size [batch_size, 1] containing H(z)

##### `method terminal`

```python
def terminalself, t: float, x: torch.Tensor:
```

> Terminal condition of the PDE.

Args:
t (float): current time
x (torch.Tensor): tensor of size [batch_size, dim] containing space coordinates

Returns:
torch.Tensor: tensor of size [batch_size, 1] containing terminal values

##### `method sigma_matrix`

```python
def sigma_matrixself, x:
```
##### `method terminal_for_sample`

```python
def terminal_for_sampleself, x:
```

> Terminal condition of the PDE.

Args:
x (torch.Tensor): tensor of size [num_sample,batch_size, dim] containing space coordinates

Returns:
torch.Tensor: tensor of size [num_sample ,batch_size, 1] containing terminal values

#### `class PricingDiffRate`

> Nonlinear Black-Scholes equation with different interest rates for borrowing and lending
in Section 4.4 of Comm. Math. Stat. paper doi.org/10.1007/s40304-017-0117-6

##### `method __init__`

```python
def __init__self, eqn_config:
```
##### `method sample`

```python
def sampleself, num_sample:
```
##### `method r_u`

```python
def r_uself, t: float, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor:
```
##### `method h_z`

```python
def h_zself, t: float, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor:
```

> Function to compute <h,z> in the PDE.

Args:
t (float): current time
x (torch.Tensor): tensor of size [batch_size, dim] containing space coordinates
y (torch.Tensor): tensor of size [batch_size, 1] containing function value
z (torch.Tensor): tensor of size [batch_size, dim] containing gradients

Returns:
torch.Tensor: tensor of size [batch_size, 1] containing H(z)

##### `method terminal`

```python
def terminalself, t, x:
```
##### `method sigma_matrix`

```python
def sigma_matrixself, x:
```
##### `method terminal_for_sample`

```python
def terminal_for_sampleself, x:
```

> Terminal condition of the PDE.

Args:
x (torch.Tensor): tensor of size [num_sample,batch_size, dim] containing space coordinates

Returns:
torch.Tensor: tensor of size [num_sample ,batch_size, 1] containing terminal values


### 📄 `RiskLabAI\pde\model.py`

#### `class TimeNet`

> Neural network model for time dimension

##### `method __init__`

```python
def __init__self, output_dim: int:
```

> Initialize the neural network model with layers

:param output_dim: The output dimension of the neural network
:type output_dim: int

##### `method forward`

```python
def forwardself, x: torch.Tensor:
```

> Forward propagation through the network.

:param x: Input tensor
:type x: torch.Tensor
:return: Output tensor
:rtype: torch.Tensor

#### `class Net1`

> A class for defining a neural network with a single linear layer.

##### `method __init__`

```python
def __init__self, input_dim: int, output_dim: int:
```

> Initialize the network with a single linear layer.

:param input_dim: Number of input features
:type input_dim: int
:param output_dim: Number of output features
:type output_dim: int

##### `method forward`

```python
def forwardself, x: torch.Tensor:
```

> Forward propagation through the network.

:param x: Input tensor of shape (batch_size, input_dim)
:type x: torch.Tensor
:return: Output tensor of shape (batch_size, output_dim)
:rtype: torch.Tensor

#### `class MAB`

##### `method __init__`

```python
def __init__self, dim_q: int, dim_k: int, dim_v: int, num_heads: int, ln: bool=False:
```

> Multi-Head Self Attention Block.

:param dim_q: Dimension of query
:param dim_k: Dimension of key
:param dim_v: Dimension of value
:param num_heads: Number of attention heads
:param ln: Whether to use Layer Normalization

##### `method forward`

```python
def forwardself, q: torch.Tensor, k: torch.Tensor:
```

> Forward propagation.

:param q: Query tensor
:param k: Key tensor
:return: Output tensor

#### `class SAB`

##### `method __init__`

```python
def __init__self, dim_in: int, dim_out: int, num_heads: int, ln: bool=False:
```

> Self Attention Block.

:param dim_in: Input dimension
:param dim_out: Output dimension
:param num_heads: Number of attention heads
:param ln: Whether to use Layer Normalization

##### `method forward`

```python
def forwardself, x: torch.Tensor:
```

> Forward propagation.

:param x: Input tensor
:return: Output tensor

#### `class ISAB`

##### `method __init__`

```python
def __init__self, dim_in: int, dim_out: int, num_heads: int, num_inds: int, ln: bool=False:
```

> Induced Self Attention Block.

:param dim_in: Input dimension
:param dim_out: Output dimension
:param num_heads: Number of attention heads
:param num_inds: Number of inducing points
:param ln: Whether to use Layer Normalization

##### `method forward`

```python
def forwardself, x: torch.Tensor:
```

> Forward propagation.

:param x: Input tensor
:return: Output tensor

#### `class PMA`

##### `method __init__`

```python
def __init__self, dim: int, num_heads: int, num_seeds: int, ln: bool=False:
```

> Pooling Multihead Attention.

:param dim: Dimension of input and output
:param num_heads: Number of attention heads
:param num_seeds: Number of seed vectors
:param ln: Whether to use Layer Normalization

##### `method forward`

```python
def forwardself, x: torch.Tensor:
```

> Forward propagation.

:param x: Input tensor
:return: Output tensor

#### `class TimeNetForSet`

> Neural network model for time dimension.

Args:
    in_features (int): The input features dimension. Default is 1.
    out_features (int): The output features dimension. Default is 64.

##### `method __init__`

```python
def __init__self, in_features: int=1, out_features: int=64:
```
##### `method forward`

```python
def forwardself, t: torch.Tensor, x: torch.Tensor:
```

> Forward pass of the network.

Args:
    t (torch.Tensor): Input tensor for time dimension.
    x (torch.Tensor): Input tensor for features.

Returns:
    torch.Tensor: Output tensor.

##### `method freeze`

```python
def freezeself:
```

> Freezes the feature parameters.

#### `class DeepTimeSetTransformer`

##### `method __init__`

```python
def __init__self, input_dim: int:
```
##### `method forward`

```python
def forwardself, t: torch.Tensor, x: torch.Tensor:
```

> Forward pass of the network.

Args:
    t (torch.Tensor): Input tensor for time dimension.
    x (torch.Tensor): Input tensor for features.

Returns:
    torch.Tensor: Output tensor.

#### `class FBSNNNetwork`

##### `method __init__`

```python
def __init__self, layersize: list[int]:
```

> Initializes a neural network with multiple blocks.

Args:
- indim (int): input dimension
- layersize (List[int]): list of sizes of hidden layers
- outdim (int): output dimension

##### `method forward`

```python
def forwardself, x: torch.Tensor:
```

> Passes the input through the neural network.

Args:
- x (torch.Tensor): input tensor

Returns:
- torch.Tensor: output tensor

#### `class DeepBSDE`

##### `method __init__`

```python
def __init__self, layersize: list[int]:
```

> Initializes a neural network with multiple blocks.

Args:
- indim (int): input dimension
- layersize (List[int]): list of sizes of hidden layers
- outdim (int): output dimension

##### `method forward`

```python
def forwardself, x: torch.Tensor:
```

> Passes the input through the neural network.

Args:
- x (torch.Tensor): input tensor

Returns:
- torch.Tensor: output tensor

#### `class TimeDependentNetwork`

##### `method __init__`

```python
def __init__self, indim: int, layersize: list[int], outdim: int:
```

> Initializes a neural network with multiple blocks.

Args:
- indim (int): input dimension
- layersize (List[int]): list of sizes of hidden layers
- outdim (int): output dimension

##### `method forward`

```python
def forwardself, t: torch.Tensor, x: torch.Tensor:
```

> Passes the input through the neural network.

Args:
- t (torch.Tensor): tensor containing time information
- x (torch.Tensor): input tensor

Returns:
- torch.Tensor: output tensor

#### `class TimeDependentNetworkMonteCarlo`

##### `method __init__`

```python
def __init__self, indim: int, layersize: list[int], outdim: int, sigma: float:
```

> Initializes a neural network with multiple blocks.

Args:
- indim (int): input dimension
- layersize (List[int]): list of sizes of hidden layers
- outdim (int): output dimension
- sigma (float) : volatility

##### `method forward`

```python
def forwardself, t: torch.Tensor, x: torch.Tensor, y:
```

> Passes the input through the neural network.

Args:
- t (torch.Tensor): tensor containing time information
- x (torch.Tensor): input tensor

Returns:
- torch.Tensor: output tensor


### 📄 `RiskLabAI\pde\solver.py`

#### `function initialize_weights`

```python
def initialize_weightsm: nn.Module:
```

> Initializes the weights of the given module.

Args:
- m (nn.Module): the module to initialize weights of

Returns:
- None

#### `class FBSDESolver`

##### `method __init__`

```python
def __init__self, pde, layer_sizes, learning_rate, solving_method, device:
```

> Initializes the FBSDESolver.

Args:
- pde : the partial differential equation to solve
- layer_sizes (list[int]): list of sizes of hidden layers
- learning_rate (float): learning rate for optimization
- solving_method (str): method to solve the PDE ('Monte-Carlo', 'Deep-Time-SetTransformer', 'Basic')

##### `method compute_loss`

```python
def compute_lossself, y, dw, t, init, init_grad:
```
##### `method solve`

```python
def solveself, num_iterations, batch_size, init, sample_size=None:
```

> Solves the PDE.

Args:
- num_iterations (int): number of iterations for optimization
- batch_size (int): batch size for training
- init (torch.Tensor): initial value
- device (torch.device): device to perform calculations on ('cpu', 'cuda')
- sample_size (int, optional): sample size for Monte-Carlo method

Returns:
- list[torch.Tensor]: list of losses during optimization
- list[torch.Tensor]: list of initial values during optimization

#### `class FBSNNolver`

##### `method __init__`

```python
def __init__self, pde, layer_sizes, learning_rate, device:
```

> Initializes the FBSDESolver.

Args:
- pde : the partial differential equation to solve
- layer_sizes (list[int]): list of sizes of hidden layers
- learning_rate (float): learning rate for optimization
- solving_method (str): method to solve the PDE ('Monte-Carlo', 'Deep-Time-SetTransformer', 'Basic')

##### `method compute_loss`

```python
def compute_lossself, y, dw, t, init:
```
##### `method solve`

```python
def solveself, num_iterations, batch_size, init, sample_size=None:
```

> Solves the PDE.

Args:
- num_iterations (int): number of iterations for optimization
- batch_size (int): batch size for training
- init (torch.Tensor): initial value
- device (torch.device): device to perform calculations on ('cpu', 'cuda')
- sample_size (int, optional): sample size for Monte-Carlo method

Returns:
- list[torch.Tensor]: list of losses during optimization
- list[torch.Tensor]: list of initial values during optimization


### 📄 `RiskLabAI\utils\constants.py`


### 📄 `RiskLabAI\utils\ewma.py`

#### `function ewma`

```python
def ewmaarray, window:
```

> This function calculate Exponential Weighted Moving Average of array
:param array: input array
:param window: window size
:return: ewma array


### 📄 `RiskLabAI\utils\momentum_mean_reverting_strategy_sides.py`

#### `function determine_strategy_side`

```python
def determine_strategy_sideprices: pd.Series, fast_window: int=20, slow_window: int=50, exponential: bool=False, mean_reversion: bool=False:
```

> Determines the trading side (long or short) based on moving average crossovers and 
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


### 📄 `RiskLabAI\utils\progress.py`

#### `function progress_bar`

```python
def progress_barcurrent_progress: int, total_progress: int, start_time: float, bar_length: int=20:
```

> Display a terminal-style progress bar with completion percentage and estimated remaining time.

:param current_progress: Current value indicating the progress made.
:param total_progress: Total value representing the completion of the task.
:param start_time: The time at which the task started, typically acquired via time.time().
:param bar_length: Length of the progress bar in terminal characters, default is 20.

The displayed progress bar uses the formula:

.. math::
    \text{percentage} = \frac{\text{current\_progress}}{\text{total\_progress}}

The estimated remaining time is calculated based on elapsed time and progress made:

.. math::
    \text{remaining\_time} = \frac{\text{elapsed\_time} \times (\text{total\_progress} - \text{current\_progress})}{\text{current\_progress}}

:return: None


### 📄 `RiskLabAI\utils\smoothing_average.py`

#### `function compute_exponential_weighted_moving_average`

```python
def compute_exponential_weighted_moving_averageinput_series: np.ndarray, window_length: int:
```

> Compute the exponential weighted moving average (EWMA) of a time series array.

The EWMA is calculated using the formula:

.. math::
    EWMA_t = \\frac{x_t + (1 - \\alpha) x_{t-1} + (1 - \\alpha)^2 x_{t-2} + \\ldots}{\\omega_t}

where:

.. math::
    \\omega_t = 1 + (1 - \\alpha) + (1 - \\alpha)^2 + \\ldots + (1 - \\alpha)^t,
    \\alpha = \\frac{2}{{window\_length + 1}}

:param input_series: Input time series array.
:type input_series: np.ndarray
:param window_length: Window length for the exponential weighted moving average.
:type window_length: int
:return: An array containing the computed EWMA values.
:rtype: np.ndarray


### 📄 `RiskLabAI\utils\update_figure_layout.py`

#### `function update_figure_layout`

```python
def update_figure_layoutfig, title, xaxis_title, yaxis_title, legend_x=1, legend_y=1:
```
