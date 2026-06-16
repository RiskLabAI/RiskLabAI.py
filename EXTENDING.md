# Extending RiskLabAI

This guide explains how to add a **new model** to RiskLabAI — for example, an
extension, alternative, or improvement of a method from López de Prado's books,
or an entirely novel idea — so that it plugs into the library cleanly and is
discoverable like the built-in components.

The machinery lives in `RiskLabAI.core` and is designed around three ideas:

1. **Base interfaces** (`RiskLabAI.core.base`) define the contract for each
   model family.
2. **A component registry** (`RiskLabAI.core.registry.Registry`) maps a name to
   a component so it can be discovered and constructed by string key.
3. **Per-family registries** (`RiskLabAI.core`) hold the built-in catalogue and
   are the place new models register themselves.

Nothing here is required to *use* the library — the existing functions, classes,
and factories are unchanged. This is the recommended path for *growing* it.

---

## The model families

| Family | Registry | Base interface | Built-ins today |
|---|---|---|---|
| Bars | `core.BARS` | `core.base.AbstractBars` | standard / time / imbalance / run bars |
| Cross-validation | `core.CROSS_VALIDATORS` | `core.base.CrossValidator` | KFold, WalkForward, PurgedKFold, CPCV (+ bagged, adaptive) |
| Feature importance | `core.FEATURE_IMPORTANCE` | `core.base.FeatureImportanceStrategy` | MDI, ClusteredMDI, MDA, ClusteredMDA, SFI |
| Labeling | `core.LABELERS` | `core.base.BaseLabeler` | *(extension point — free functions today)* |
| Bet sizing | `core.BET_SIZERS` | `core.base.BaseBetSizer` | *(extension point)* |
| Portfolio optimization | `core.PORTFOLIO_OPTIMIZERS` | `core.base.BasePortfolioOptimizer` | *(extension point)* |

Discover everything at runtime:

```python
import RiskLabAI.core as core

core.list_components()              # {family: [keys...]}
core.CROSS_VALIDATORS.available()   # ['adaptivecombinatorialpurged', 'kfold', ...]
"purgedkfold" in core.CROSS_VALIDATORS
```

Construction is by name (case-insensitive, aliases supported):

```python
cv = core.CROSS_VALIDATORS.create("kfold", n_splits=5)
```

Importing `RiskLabAI.core` is cheap: every built-in is registered *lazily*, so
heavy dependencies (pandas, numba, scikit-learn, torch) are only imported when a
component is actually created.

---

## Worked example 1 — a new bar type

A new bar type is a subclass of `AbstractBars`. Register it and it becomes
constructible by name alongside the built-ins.

```python
from typing import Any, Iterable, List

from RiskLabAI.core import BARS
from RiskLabAI.core.base import AbstractBars


@BARS.register("range_bars", aliases=("ohlc_range",),
               metadata={"family": "bars", "author": "your_paper_2026"})
class RangeBars(AbstractBars):
    """Sample a new bar whenever (high - low) exceeds a fixed range."""

    def __init__(self, threshold: float = 1.0):
        super().__init__(bar_type="range")
        self.threshold = threshold

    def _bar_construction_condition(self, threshold: float) -> bool:
        return (self.high_price - self.low_price) >= self.threshold

    def construct_bars_from_data(self, data: Iterable) -> List[List[Any]]:
        bars = []
        for date_time, price, volume in data:
            tick_rule = self._tick_rule(price)
            self.update_base_fields(price, tick_rule, volume)
            if self._bar_construction_condition(self.threshold):
                bars.append(self._construct_next_bar(
                    date_time, self.tick_counter, price,
                    self.high_price, self.low_price, self.threshold,
                ))
                self._reset_cached_fields()
            self.tick_counter += 1
        return bars


bars = BARS.create("range_bars", threshold=2.5)
```

---

## Worked example 2 — a new labeler

Labeling is currently exposed as free functions, which keep working. A *new*
class-based labeler should implement `BaseLabeler` and register in `LABELERS`:

```python
import pandas as pd

from RiskLabAI.core import LABELERS
from RiskLabAI.core.base import BaseLabeler


@LABELERS.register("fixed_horizon", metadata={"family": "labeling"})
class FixedHorizonLabeler(BaseLabeler):
    """Label the sign of the return over a fixed number of bars."""

    def __init__(self, horizon: int = 5):
        self.horizon = horizon

    def label(self, prices: pd.Series, events=None, **kwargs) -> pd.DataFrame:
        forward_return = prices.shift(-self.horizon) / prices - 1.0
        return pd.DataFrame({"label": forward_return.apply(_sign)})


def _sign(x: float) -> int:
    if pd.isna(x):
        return 0
    return int(x > 0) - int(x < 0)


labeler = LABELERS.create("fixed_horizon", horizon=10)
labels = labeler.label(my_price_series)
```

The same pattern applies to bet sizers (`BaseBetSizer` + `BET_SIZERS`) and
portfolio optimizers (`BasePortfolioOptimizer` + `PORTFOLIO_OPTIMIZERS`).

---

## Conventions and checklist

When you add a new model:

- [ ] **Subclass the family's base interface** (or, for a sklearn-style model,
      satisfy the `core.base.Estimator` protocol — `fit`/`predict`).
- [ ] **Register it** in the matching registry with a clear, snake_case key.
      Use `metadata={...}` to record the family, an AFML page, or the paper it
      comes from.
- [ ] **Keep the public API stable.** Adding new names is always fine; renaming
      or removing existing public names is a breaking change and needs a major
      version bump (see `CHANGELOG.md` and `CLAUDE.md`).
- [ ] **Write tests** with hand-computed numeric assertions (the existing suite
      sets the bar — bar OHLC values, exact HRP weights, purge boundaries).
- [ ] **Cross-language parity.** If this model also belongs in `RiskLabAI.jl`,
      mirror the public name and note any deliberate divergence (see the
      `PARITY.md` discipline in the improvement plan).

## Why a registry instead of editing a central `dict`?

The library previously hand-wrote a `name -> class` mapping in three places
(`controller/bars_initializer.py`,
`backtest/validation/cross_validator_factory.py`,
`features/feature_importance/feature_importance_factory.py`). A registry lets a
new model register itself *without editing a central file* — the open/closed
principle — and makes the full catalogue discoverable and testable. The existing
factories still work; `test/core/test_builtin_parity.py` asserts the registries
and factories stay in sync so they cannot silently drift apart.
```
