# RiskLabAI

[![PyPI version](https://badge.fury.io/py/RiskLabAI.svg)](https://badge.fury.io/py/RiskLabAI)
[![CI](https://github.com/RiskLabAI/RiskLabAI.py/actions/workflows/ci.yml/badge.svg)](https://github.com/RiskLabAI/RiskLabAI.py/actions/workflows/ci.yml)

RiskLabAI is a Python library for quantitative finance, financial machine
learning, and causal factor analysis. It provides research-oriented
implementations of methods associated with Marcos López de Prado's *Advances
in Financial Machine Learning*, *Machine Learning for Asset Managers*, and
*Causal Factor Investing*.

Version 3.1.0 preserved the clean 57-name causal-factor-analysis namespace
released in 3.0.0 and added 30 paper-derived names, producing an 87-name causal
API. The companion
[RiskLabAI.jl](https://github.com/RiskLabAI/RiskLabAI.jl) package independently
implements the same concepts. This parity statement applies to the causal API,
not to every Python module.

The 3.2.0 candidate adds Python methods for sampling, optimization, market
features, causal evaluation, synthetic data, and neural financial models.
See [the release notes](RELEASE_NOTES_NEXT_MAJOR.md) for scope and limitations.
Julia implementations of these new additions are deferred.

## What is included

- **Causal factor analysis** - constrained minimum-variance allocation,
  factor-mirage and allocation-misspecification diagnostics, graphical
  identification and factor roles, deterministic structural-model evaluation,
  treatment-effect formulas, search-adjusted false discovery, specification
  experiments, and evidence records for the seven-stage causal-factor protocol
- **Financial data structures** - tick, volume, dollar, imbalance, run, and
  time bars
- **Labeling and sample weights** - triple-barrier, meta-labeling,
  trend-scanning, uniqueness, and time-decay methods
- **Fractional differentiation and denoising** - fixed-width fractional
  differentiation, Marcenko-Pastur denoising, and targeted shrinkage
- **Validation and feature importance** - purged and combinatorial
  cross-validation, walk-forward validation, MDI, MDA, SFI, and clustered
  variants
- **Portfolio and backtest analytics** - HRP, NCO, hedging, PSR/DSR, PBO, and
  strategy-risk tools
- **Market features** - microstructure, entropy, structural-break, and
  technical-indicator utilities
- **Optional capabilities** - plotting, optimization, simulation, symbolic
  analysis, changepoint detection, acceleration, synthetic-control tools, and
  a Deep-BSDE PDE solver

## Compatibility

RiskLabAI 3.2.0 supports CPython 3.12,
3.13, and 3.14 with NumPy `>=2.2,<3`. The complete tested policy and
feature-specific limitations are in
[`docs/compatibility.md`](https://github.com/RiskLabAI/RiskLabAI.py/blob/main/docs/compatibility.md).

## Installation

Install the base package from PyPI:

```bash
python -m pip install RiskLabAI
```

Optional dependency groups can be installed individually or together:

| Extra | Enables |
|---|---|
| `speed` | Numba acceleration |
| `pde` | PyTorch support for Deep-BSDE, neural SDEs, TimeGAN, and deep hedging |
| `synth` | synthetic-control utilities using QuantEcon |
| `hpo` | hyperparameter tuning using Optuna |
| `plot` | Matplotlib, Seaborn, and Plotly helpers |
| `symbolic` | symbolic analysis using SymPy |
| `profile` | memory profiling |
| `simulation` | simulation progress support |
| `changepoints` | changepoint detection on Python 3.12-3.13 |
| `test` | the supported pytest test runner |

For example:

```bash
python -m pip install "RiskLabAI[plot,speed]"
```

There is intentionally no `all` extra. See
[`INSTALLATION.md`](https://github.com/RiskLabAI/RiskLabAI.py/blob/main/INSTALLATION.md) for environment setup, all tested optional
groups, and development instructions.

## Causal-factor quick start

```python
import numpy as np

from RiskLabAI.causal_factor_analysis import (
    CausalDAG,
    average_treatment_effect,
    check_backdoor_adjustment_set,
    minimum_variance_factor_weights,
)

covariance = np.diag([1.0, 2.0, 4.0])
factor_exposures = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
target_exposures = np.array([0.0, 1.0])

weights = minimum_variance_factor_weights(
    covariance,
    factor_exposures,
    target_exposures,
)
np.testing.assert_allclose(weights, [-2.0 / 7.0, 5.0 / 7.0, 2.0 / 7.0])

effect = average_treatment_effect(3.5, 1.25)
assert effect == 2.25

dag = CausalDAG(
    nodes=("T", "U", "Y"),
    directed_edges=(("U", "T"), ("U", "Y"), ("T", "Y")),
    observed_nodes=("T", "U", "Y"),
)
assert check_backdoor_adjustment_set(dag, "T", "Y", ("U",)).admissible
```

The complete deterministic example is
[`examples/causal_factor_analysis_quickstart.py`](https://github.com/RiskLabAI/RiskLabAI.py/blob/main/examples/causal_factor_analysis_quickstart.py).
The causal API and its limits are documented in
[`docs/causal_factor_analysis.md`](https://github.com/RiskLabAI/RiskLabAI.py/blob/main/docs/causal_factor_analysis.md).

## Financial-data quick start

```python
from RiskLabAI.data.structures.standard_bars import StandardBars
from RiskLabAI.utils.constants import CUMULATIVE_DOLLAR

ticks = [
    ("2020-01-01 10:00:00", 100.0, 10),
    ("2020-01-01 10:00:01", 101.0, 5),
    ("2020-01-01 10:00:02", 100.0, 20),
]

bars = StandardBars(bar_type=CUMULATIVE_DOLLAR, threshold=3000)
bar_list = bars.construct_bars_from_data(ticks)
```

## Extending the library

The `RiskLabAI.core` registries provide discovery and construction points for
bars, cross-validators, feature-importance models, labelers, bet sizers, and
portfolio optimizers. For example:

```python
import pandas as pd

from RiskLabAI.core import CROSS_VALIDATORS, list_components

catalogue = list_components()
start_times = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
event_times = pd.Series(
    pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05"]),
    index=start_times,
)
cross_validator = CROSS_VALIDATORS.create(
    "purgedkfold",
    n_splits=5,
    times=event_times,
)
```

See [`EXTENDING.md`](https://github.com/RiskLabAI/RiskLabAI.py/blob/main/EXTENDING.md)
for the extension interfaces and examples.

## Development

```bash
python -m pip install -e ".[test]" "black==26.5.1" "ruff==0.15.17"
python -m pytest -q
black --check src/RiskLabAI/causal_factor_analysis test/causal_factor_analysis
ruff check src/RiskLabAI/causal_factor_analysis test/causal_factor_analysis
```

Please branch from `main`, keep changes focused, and include tests for behavior
changes.

## Scope

RiskLabAI is research software, not investment advice. Graph routines evaluate
criteria on a caller-supplied directed acyclic graph; they do not discover or
certify that graph. Protocol records validate declared evidence structures;
they do not prove that empirical assumptions are true.

## License

RiskLabAI is distributed under the
[BSD 3-Clause License](https://github.com/RiskLabAI/RiskLabAI.py/blob/main/LICENSE).
