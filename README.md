# RiskLabAI.py

[![PyPI version](https://badge.fury.io/py/RiskLabAI.svg)](https://badge.fury.io/py/RiskLabAI)
[![CI](https://github.com/RiskLabAI/RiskLabAI.py/actions/workflows/ci.yml/badge.svg)](https://github.com/RiskLabAI/RiskLabAI.py/actions/workflows/ci.yml)

A Python library for quantitative finance and financial machine learning,
implementing core methods from Marcos López de Prado's *Advances in Financial
Machine Learning* and *Machine Learning for Asset Managers*.

The library provides implementations for:

- **Financial data structures** — tick, volume, dollar, imbalance, and run bars
- **Labeling** — the triple-barrier method, meta-labeling, trend-scanning
- **Fractional differentiation** — standard and fixed-width window (FFD)
- **Sample weights**, **denoising** (Marčenko–Pastur), **distance metrics**
- **Cross-validation** — Purged K-Fold, Combinatorial Purged CV (+ adaptive/bagged), walk-forward
- **Feature importance** — MDI, MDA, SFI, and clustered variants
- **Portfolio optimization** — HRP, NCO, hedging
- **Backtest statistics** — PSR/DSR, PBO, strategy risk
- **Microstructure & entropy features**, **structural breaks**, and a Deep-BSDE PDE solver

There is a companion Julia package,
[RiskLabAI.jl](https://github.com/RiskLabAI/RiskLabAI.jl), which mirrors this
API.

## Installation

```bash
pip install RiskLabAI
```

The base install is lightweight. Heavier, optional capabilities are available as
extras:

| Extra | Installs | Enables |
|---|---|---|
| `RiskLabAI[pde]` | `torch` | the Deep-BSDE PDE solver (`RiskLabAI.pde`) |
| `RiskLabAI[plot]` | `matplotlib`, `seaborn`, `plotly` | plotting helpers |
| `RiskLabAI[synth]` | `quantecon` | synthetic-data utilities |
| `RiskLabAI[all]` | all of the above | everything |

```bash
pip install "RiskLabAI[all]"
```

For development (editable install + tests), see
[`INSTALLATION.md`](INSTALLATION.md).

## Quickstart

Sample dollar/volume/tick bars from raw ticks:

```python
from RiskLabAI.data.structures.standard_bars import StandardBars
from RiskLabAI.utils.constants import CUMULATIVE_DOLLAR

# ticks: an iterable of (datetime, price, volume)
ticks = [
    ("2020-01-01 10:00:00", 100.0, 10),
    ("2020-01-01 10:00:01", 101.0, 5),
    ("2020-01-01 10:00:02", 100.0, 20),
]

bars = StandardBars(bar_type=CUMULATIVE_DOLLAR, threshold=3000)
bar_list = bars.construct_bars_from_data(ticks)
# each bar: [date_time, idx, open, high, low, close, volume,
#            buy_volume, sell_volume, ticks, dollar, threshold]
```

Discover and construct components by name through the extension registry:

```python
import RiskLabAI.core as core

core.list_components()                       # {family: [available keys]}
cv = core.CROSS_VALIDATORS.create("purgedkfold", n_splits=5, times=event_times)
```

## Logging

RiskLabAI logs under the `"RiskLabAI"` logger and is silent by default. To see
progress and diagnostics, configure logging in your application:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Extending the library

RiskLabAI is built to be extended with new models. The `RiskLabAI.core` layer
provides a component registry and base interfaces so a new bar type, labeler,
cross-validator, etc. can be registered and discovered without editing central
code. See [`EXTENDING.md`](EXTENDING.md) for a step-by-step guide with worked
examples.

## Contributing

Contributions are welcome. The project uses `pytest` for tests and
`black` + `ruff` for formatting/linting (run before opening a PR):

```bash
pip install -e ".[all]" pytest black ruff
pytest -q --ignore=test/pde
black RiskLabAI test
ruff check RiskLabAI test
```

Please branch from `main`, keep changes focused, and update `CHANGELOG.md`.

## License

See [`LICENSE.txt`](LICENSE.txt).
