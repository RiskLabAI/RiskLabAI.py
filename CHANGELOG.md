# Changelog

All notable changes to RiskLabAI.py are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning: [SemVer](https://semver.org/).

## [Unreleased]

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
