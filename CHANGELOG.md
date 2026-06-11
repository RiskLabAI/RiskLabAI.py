# Changelog

All notable changes to RiskLabAI.py are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning: [SemVer](https://semver.org/).

## [Unreleased]

### Fixed
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

### Changed
- Declared missing runtime dependencies `tqdm` and `sympy`; added `pytest` to the `dev` extra.
- Removed `.pypirc` from version control; extended `.gitignore` (`dist/`, `build/`,
  `*.egg-info/`, `.pypirc`, caches).
