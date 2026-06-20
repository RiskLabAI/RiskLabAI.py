# Changelog

All notable changes to RiskLabAI.py are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning: [SemVer](https://semver.org/).

## [2.0.1]

### Fixed
- `optimization.hrp.hrp` no longer raises `ValueError: Distance matrix must be
  symmetric` when the correlation matrix is only symmetric to floating-point
  tolerance (as produced by `cov_to_corr` or by denoising). The correlation
  distance is now symmetrised (`(d + dᵀ)/2`, zero diagonal) before
  `squareform`. Regression test added (`test_hrp_asymmetric_correlation`).
  Mirrors the same fix in RiskLabAI.jl v0.5.1.

## [2.0.0]

A **breaking** release that standardises the public API on PEP 8 names and makes
the core component registry the single way to construct cross-validators and
feature-importance strategies. **Every** renamed name keeps working with a
`DeprecationWarning` until 2.1.0, so existing code does not break on upgrade —
see `NAMING_CANON_2.0.0.md`.

### Changed (BREAKING)
- `backtest.bet_sizing` functions renamed to snake_case:
  `avgActiveSignals`→`avg_active_signals`,
  `mpAvgActiveSignals`→`mp_avg_active_signals`,
  `discreteSignal`→`discrete_signal`, `Signal`→`generate_signal`,
  `betSize`→`bet_size_sigmoid`, `getW`→`compute_sigmoid_width`,
  `TPos`→`target_position`, `inversePrice`→`inverse_price`,
  `limitPrice`→`limit_price`. Their parameters are snake_case too (e.g.
  `nThreads`→`n_threads`, `stepSize`→`step_size`, `acctualPrice`→`actual_price`,
  `maximumPositionSize`→`maximum_position_size`). Keyword-argument callers must
  update parameter names; positional calls are unaffected.
- Classes renamed: `pde.FBSNNolver`→`FBSNNSolver` (typo fix);
  `optimization.MyPipeline`→`SampleWeightedPipeline`.
- Constants given ASCII identifiers — the string *values* are unchanged, so
  stored data and internal dict keys are unaffected: `CUMULATIVE_θ`→
  `CUMULATIVE_THETA`, `CUMULATIVE_BUY_θ`→`CUMULATIVE_BUY_THETA`,
  `CUMULATIVE_SELL_θ`→`CUMULATIVE_SELL_THETA`.
- The core registries (`RiskLabAI.core.CROSS_VALIDATORS`, `FEATURE_IMPORTANCE`)
  are now the single way to construct those components. The bars stack imports
  its column-name constants explicitly (no more `from ...constants import *`).

### Deprecated
The following keep working until **2.1.0**, emitting a `DeprecationWarning`
that names the replacement:
- The old `bet_sizing` camelCase function names, `FBSNNolver`, `MyPipeline`,
  and the `CUMULATIVE_*θ` constant identifiers (accessed via `RiskLabAI.utils`).
- `CrossValidatorFactory`, `CrossValidatorController`,
  `FeatureImportanceFactory`, and `FeatureImportanceController` — use
  `RiskLabAI.core.CROSS_VALIDATORS.create(...)` /
  `RiskLabAI.core.FEATURE_IMPORTANCE.create(...)` instead.

### Added
- `RiskLabAI.core`: a non-breaking extension layer that makes the library easier
  to grow with new models.
  - `core.registry.Registry`: a reusable, case-insensitive component registry
    with lazy (import-path) registration, aliases, duplicate protection, optional
    kwarg filtering, and helpful "not found" errors. Generalises the three
    hand-written `name -> class` factories (`bars_initializer`,
    `cross_validator_factory`, `feature_importance_factory`).
  - `core.base`: base interfaces per model family — re-exports the existing
    `AbstractBars`, `CrossValidator`, `FeatureImportanceStrategy` (lazily), an
    `Estimator` structural `Protocol`, and new optional contracts
    `BaseLabeler` / `BaseBetSizer` / `BasePortfolioOptimizer` for the
    free-function families.
  - Per-family registries (`BARS`, `CROSS_VALIDATORS`, `FEATURE_IMPORTANCE`,
    `LABELERS`, `BET_SIZERS`, `PORTFOLIO_OPTIMIZERS`) pre-populated with the
    built-in catalogue, plus `list_components()` / `get_registry()` for
    discovery.
  - `EXTENDING.md`: a step-by-step guide with worked examples for adding a new
    model. Existing factories and public APIs are unchanged; importing
    `RiskLabAI.core` pulls in no heavy dependencies.

### Changed
- Removed duplicated code by single-sourcing two helpers (no public API change):
  - `data.labeling` no longer defines its own copies of the parallel-processing
    helpers (`lin_parts`, `process_jobs`, `expand_call`, `report_progress`).
    They are re-exported from the canonical `RiskLabAI.hpc` (`lin_parts` maps to
    `hpc.linear_partitions`); the names remain importable from `data.labeling`.
    The dead duplicate's `num_threads=24` default is gone (the canonical default
    is `-1` = all cores); the copies were unused inside the package.
  - `cluster.covariance_to_correlation` now delegates to the canonical
    `data.denoise.cov_to_corr`. Output is identical to floating-point precision
    for valid covariance matrices (verified over random inputs; the canonical
    version adds zero-std and exact-diagonal safeguards).
- Note: the two `sharpe_ratio` definitions were deliberately left separate.
  `backtest_statistics.sharpe_ratio` is numba-jitted and array-only (ddof=0),
  while `backtest_overfitting_simulation`'s relies on pandas `Series.std()`
  (ddof=1) in its rank-correlation path — unifying them would change results.
- Replaced library `print()` calls with the standard `logging` module across
  `controller.data_structure_controller`, `hpc`, `backtest.strategy_risk`,
  `features.feature_importance` (MDA/clustered MDA), `pde.solver`, and
  `utils.publication_plots` (errors -> `logger.error`, recoverable issues ->
  `logger.warning`, progress -> `logger.info`/`debug`). The `RiskLabAI` logger
  gets a `NullHandler`, so the library is silent by default; configure logging
  (e.g. `logging.basicConfig(level=logging.INFO)`) to see output. Return values
  are unchanged.
- `pde`: importing `RiskLabAI.pde` without PyTorch now raises a clear, actionable
  `ImportError` pointing to `pip install 'RiskLabAI[pde]'`, instead of a bare
  `ModuleNotFoundError: No module named 'torch'`. The base install remains
  torch-free (`import RiskLabAI` never imports the sub-package).

### Performance
- Vectorized three O(n^2) hot paths; outputs are unchanged (locked by
  `test/test_performance.py`, which checks each against a brute-force reference):
  - `data.differentiation.fractional_difference_std`: the per-row expanding
    weighted sum is now a single causal convolution (`np.convolve`) per column.
  - `backtest.bet_sizing.mpAvgActiveSignals`: the per-timepoint active-signal
    average is now prefix sums + `searchsorted` (interval-stabbing sweep),
    turning an O(n*m) double scan into O((n+m) log n) — ~1500x on a 3k-signal
    benchmark.
  - `data.labeling.triple_barrier`: per-event pandas label slicing replaced
    with positional numpy indexing (~70x on 50k closes / 2k events).

### Removed
- Deleted dead modules `utils/utilities_lopez.py` (unreferenced) and
  `utils/smoothing_average.py` (a duplicate of `utils.ewma`). The public
  `compute_exponential_weighted_moving_average` name is unaffected — it remains
  available from `RiskLabAI.utils` as an alias of the canonical `ewma`.

### Fixed
- `features.feature_importance` (MDA): feature shuffling mutated a column view
  in place, which raises `ValueError: array is read-only` under pandas
  copy-on-write (default since pandas 3.0). Now shuffles a copy and assigns back.

## [1.0.9]

### Fixed
- `utils.determine_strategy_side`: signal dtype was platform-dependent (int32 on
  Windows); now always int64.
- `data.labeling.calculate_t_value_linear_regression`: constant series now returns
  NaN (0/0 is undefined) as documented, instead of 0.0.
- `data.differentiation.fractional_difference_std`: result had object dtype; now float.
- Raw-string docstrings in `denoising`/`bekker_parkinson` (LaTeX `\(` raised
  SyntaxWarning on Python 3.12).
- `backtest.bet_sizing`: `avgActiveSignals`/`Signal` silently returned empty results — the module
  imported `mpPandasObj` while `RiskLabAI.hpc` exports `mp_pandas_obj`, so a placeholder returning
  an empty DataFrame always took over. The silent placeholder is removed; the real import is used.
- `data.structures` (`run_bars`, `imbalance_bars`, `abstract_run_bars`,
  `abstract_information_driven_bars`): removed broken `ewma` import fallbacks (one referenced
  `pd` without importing pandas — a latent `NameError`); `RiskLabAI.utils.ewma` is imported
  directly.
- `optimization.nco`: removed silent dummy-function fallback for the cluster-module import.
- Version split-brain: `__version__` (was hardcoded `0.0.93`) now reads the installed package
  version; `pyproject.toml` (1.0.8) is the single source of truth.
- Removed broken console-script entry point `RiskLabAI.app:entry_point` (module never existed;
  installed a crashing executable).
- Renamed `test/backtest/teste_backtest_overfitting_simulation.py` → `test_…` so pytest collects
  it; added `[tool.pytest.ini_options]` with `testpaths`.

### Added
- Test/lint CI (`ci.yml`): pytest matrix on Python 3.9-3.12 with all extras, a
  minimal-install job guaranteeing `pip install RiskLabAI` stays importable, and an
  advisory ruff/black job (enforced once the formatting pass lands).
- `black` + `ruff` configuration in `pyproject.toml`.

### Changed
- Sub-packages are now imported lazily (PEP 562): `import RiskLabAI` no longer pulls
  optional heavy dependencies; `RiskLabAI.pde` is reachable again (was disabled).
- Consolidated three overlapping release workflows into one `publish.yml` using PyPI
  trusted publishing (the old trio triple-fired per release with mixed auth).
- Added version bounds to core dependencies (notably `numpy<3`, `numba>=0.57`).
- Declared missing runtime dependencies `tqdm` and `sympy`; added `pytest` to the `dev` extra.
- Removed `.pypirc` from version control; extended `.gitignore` (`dist/`, `build/`,
  `*.egg-info/`, `.pypirc`, caches).
