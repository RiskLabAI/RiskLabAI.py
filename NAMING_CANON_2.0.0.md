# RiskLabAI.py 2.0.0 — Naming Canon (proposal)

Status: **approved and implemented in 2.0.0.** All §2 renames were approved
(remove aliases in 2.1.0; keep the `θ` string values; scope folded in the
star-import cleanup and the controller/factory→registry refactor). This is the
breaking API cleanup from `IMPROVEMENT_PLAN.md` Phase 3, scoped to Python and
written so that **no existing user breaks on upgrade** — every renamed name
keeps working (with a `DeprecationWarning`) for one minor cycle before removal.

Governance (`CLAUDE.md`): "never break the public API without a major bump and a
deprecation note." This document is that note. It needs sign-off before any
code lands, because it changes public names.

---

## 1. The convention

After 2.0.0, the public API follows standard Python naming (PEP 8):

- **Functions and methods**: `snake_case`.
- **Classes**: `CapWords` (already true across the package — no class renames
  except two typo/clarity fixes below).
- **Constants**: `UPPER_SNAKE_CASE`, ASCII identifiers only.
- **Parameters**: `snake_case`.

The package is already ~95% compliant. The work is a small, concentrated set of
legacy names — chiefly the `bet_sizing` "camelCase island" — plus two misnamed
classes and three non-ASCII constants.

---

## 2. What changes

### 2.1 `backtest.bet_sizing` — functions (the camelCase island)

| Current (public) | New (2.0.0) | Notes |
|---|---|---|
| `avgActiveSignals` | `avg_active_signals` | |
| `mpAvgActiveSignals` | `mp_avg_active_signals` | worker for the above |
| `discreteSignal` | `discrete_signal` | |
| `Signal` | `generate_signal` | AFML Snippet 10.1; verb-form name |
| `betSize` | `bet_size_sigmoid` | the sigmoid bet-size `x / sqrt(w + x^2)` |
| `getW` | `compute_sigmoid_width` | implied width `w` from divergence/size |
| `TPos` | `target_position` | |
| `inversePrice` | `inverse_price` | |
| `limitPrice` | `limit_price` | |

These are exported at package level via `RiskLabAI.backtest`, so the aliases
(Section 3) live there too. `probability_bet_size`, `average_bet_sizes`, and
`strategy_bet_sizing` are already snake_case and unchanged.

### 2.2 Parameters (same functions)

Renamed to snake_case as part of the function change. Callers using these as
**keyword arguments** must update; positional callers are unaffected.

| Function | Current param | New param |
|---|---|---|
| `target_position` (`TPos`) | `acctualPrice` | `actual_price` (also fixes the typo) |
| `target_position` | `maximumPositionSize` | `maximum_position_size` |
| `limit_price` (`limitPrice`) | `targetPositionSize`, `cPosition`, `maximumPositionSize` | `target_position_size`, `current_position`, `maximum_position_size` |
| `discrete_signal` (`discreteSignal`) | `stepSize` | `step_size` |
| `generate_signal` (`Signal`) | `stepSize`, `nClasses`, `nThreads` | `step_size`, `n_classes`, `n_threads` |

(Parameter renames cannot carry a runtime deprecation shim cleanly, so they are
documented here and called out prominently in the 2.0.0 release notes. This is
the main reason the change is a *major* bump rather than a minor one.)

### 2.3 Classes — two fixes

| Current | New | Reason |
|---|---|---|
| `pde.FBSNNolver` | `FBSNNSolver` | typo (missing `S`) |
| `optimization.MyPipeline` | `SampleWeightedPipeline` | placeholder name; describes what it is |

### 2.4 Constants — ASCII identifiers

In `utils.constants` (re-exported from `RiskLabAI.utils`):

| Current identifier | New identifier |
|---|---|
| `CUMULATIVE_θ` | `CUMULATIVE_THETA` |
| `CUMULATIVE_BUY_θ` | `CUMULATIVE_BUY_THETA` |
| `CUMULATIVE_SELL_θ` | `CUMULATIVE_SELL_THETA` |

**Only the Python identifier changes.** The string *values* (`"Cumulative θ"`,
etc.) are used as internal dict keys in the bar-statistics machinery; leaving
the values untouched means no stored data, serialized output, or internal
lookup changes — purely the name you import. (If we ever want ASCII values too,
that is a separate, data-affecting decision.)

---

## 3. How nothing breaks: the deprecation shims

Every renamed name stays importable and callable throughout the 2.0.x series,
emitting a `DeprecationWarning` that names the replacement. Removal happens no
earlier than **2.1.0** (or 3.0.0), announced in the changelog.

Three mechanisms, by kind:

**Functions** — a tiny decorator keeps the old name as a thin wrapper:

```python
# RiskLabAI/_deprecation.py  (new, ~20 lines)
import functools, warnings

def deprecated_alias(new_func, old_name, *, removed_in):
    @functools.wraps(new_func)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"{old_name}() is deprecated and will be removed in "
            f"{removed_in}; use {new_func.__name__}() instead.",
            DeprecationWarning, stacklevel=2,
        )
        return new_func(*args, **kwargs)
    wrapper.__name__ = old_name
    return wrapper
```

```python
# in bet_sizing.py, after the renamed definitions
avgActiveSignals = deprecated_alias(avg_active_signals, "avgActiveSignals", removed_in="2.1.0")
Signal           = deprecated_alias(generate_signal,   "Signal",           removed_in="2.1.0")
# ... one line per renamed function
```

**Classes** — subclass with a warning in `__init__`:

```python
class FBSNNolver(FBSNNSolver):
    def __init__(self, *args, **kwargs):
        warnings.warn("FBSNNolver is deprecated; use FBSNNSolver.",
                      DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)
```

**Constants** — module-level `__getattr__` (PEP 562) on `utils/__init__.py`
warns on access and returns the new value:

```python
_DEPRECATED_CONSTANTS = {"CUMULATIVE_θ": "CUMULATIVE_THETA", ...}
def __getattr__(name):
    if name in _DEPRECATED_CONSTANTS:
        new = _DEPRECATED_CONSTANTS[name]
        warnings.warn(f"{name} is deprecated; use {new}.", DeprecationWarning, stacklevel=2)
        return globals()[new]
    # ... (existing lazy-plotting __getattr__ logic merges in here)
```

All old names also stay listed in the relevant `__all__` during 2.0.x so
`from ... import *` keeps working.

---

## 4. Tests, notebooks, parity

- **Tests**: update the suite to call the new names. Add one test that asserts
  each deprecated alias still works **and** raises `DeprecationWarning`
  (`pytest.warns(DeprecationWarning)`), so the shims are guaranteed intact until
  we intentionally remove them.
- **Notebooks** (`Notebooks.py`): update usages to new names; the aliases mean
  old notebooks still run (with warnings) until re-pinned.
- **`core` registry / `EXTENDING.md`**: unaffected (those already use
  snake_case and the registry keys are independent strings).
- **`PARITY.md`** (to be created with the Julia work): the shared py/jl name
  table should use these canonical names as the Python column.
- **`style_guide.md`**: update so it matches the code it governs (the audit
  noted it currently contradicts itself).

## 5. Related, optional, non-breaking hygiene (can ride along or stay separate)

- Replace `from RiskLabAI.utils.constants import *` in the bars stack with
  explicit imports (removes star-imports without changing the public API).
- The pass-through `*Controller` classes vs. factories question (architecture,
  not naming) — leave to a separate decision.

These are not required for the naming canon and carry no deprecation burden.

---

## 6. Release plan

1. Land the current non-breaking work as **1.1.0** first (extension
   architecture, dedup, logging, pde/dead-code, performance). The canon builds
   on a clean, green main.
2. Branch `feat/api-canon-2.0`. Implement renames + `_deprecation.py` + shims +
   updated tests in one PR. CI must be green (including the new
   `pytest.warns(DeprecationWarning)` tests).
3. Bump `pyproject.toml` to `2.0.0`; CHANGELOG gets a **`### Changed (BREAKING)`**
   section listing every rename (old → new) and the parameter changes, plus a
   **`### Deprecated`** section noting the aliases and their `removed_in`
   target.
4. Update `Notebooks.py` to the new names (separate PR is fine; aliases keep
   them working meanwhile).
5. Tag `v2.0.0`; release via the existing trusted-publishing flow.
6. One minor cycle later (2.1.0): drop the aliases and the `θ` identifiers;
   CHANGELOG `### Removed`.

## 7. Decisions needed from Prof. Arian

1. **New names**: approve the table in §2 (especially the judgment calls:
   `Signal`→`generate_signal`, `betSize`→`bet_size_sigmoid`,
   `getW`→`compute_sigmoid_width`, `MyPipeline`→`SampleWeightedPipeline`).
2. **Deprecation window**: remove aliases in **2.1.0** (recommended) or keep
   them through 3.0.0 (longer grace, more clutter)?
3. **Constant values**: keep the `"Cumulative θ"` string values as-is
   (recommended — no data impact) or also ASCII-ify them later?
4. **Scope**: rename-only in 2.0.0 (recommended), or fold in the §5 hygiene
   items too?
