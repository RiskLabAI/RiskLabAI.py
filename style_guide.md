# Style Guide (Python & Julia)

This guide governs the RiskLabAI **Python** (`RiskLabAI.py`) and **Julia**
(`RiskLabAI.jl`) packages. The two packages aim to mirror each other's public
API and structure where it makes sense (cross-language parity is a first-class
goal), so the conventions below are shared in spirit and differ only where the
languages do.

Where an automated tool enforces a rule, the tool is the source of truth and
this document does not restate the mechanics.

## Python

### Formatting and linting — enforced by tooling

Formatting and import order are owned by **black** and **ruff**; do not format
by hand. Both are pinned in `pyproject.toml` (`[tool.black]`, `[tool.ruff]`) and
in the `dev` extra, and both run in CI on every push and pull request. Before
committing:

```
python -m ruff check --fix RiskLabAI test
python -m black RiskLabAI test
```

This covers indentation (4 spaces), spacing around operators, argument
wrapping/alignment, line length, and import sorting — so none of those are
specified here.

### Naming — PEP 8 (the 2.0.0 canon)

Since 2.0.0 the public API follows standard Python naming (see
`NAMING_CANON_2.0.0.md`):

- Functions, methods, variables, and parameters: `snake_case`.
- Classes: `CapWords`.
- Constants: `UPPER_SNAKE_CASE`, ASCII identifiers only.
- "Number of X" is `n_x` (e.g. `n_threads`, `n_splits`), not `numX`.
- First argument is `self` for instance methods, `cls` for class methods. If a
  name clashes with a keyword, append a trailing underscore (`class_`), never a
  misspelling.
- Prefer descriptive names over `get`-prefixes and over abbreviations
  (`compute_sigmoid_width`, not `getW`). Accepted short forms: `cov`, `corr`,
  `vol`, `min`, `max`.

Public renames must follow the deprecation policy in `CLAUDE.md`: never break
the public API without a major version bump and a deprecation shim that keeps
the old name working (with a `DeprecationWarning`) for one minor cycle.

### Documentation

Public functions, classes, and methods carry NumPy-style docstrings (Parameters,
Returns, Raises, References). Comments explain *why*, not *what*; avoid
line-by-line narration.

## Julia

- Formatting: **JuliaFormatter** (4-space indent).
- File and folder names, types, modules, structs: `UpperCamelCase`.
- Functions, variables, macros: `lower_snake_case` (mirroring the Python API for
  parity; document any deliberate divergence).
- Mutating functions that write to their arguments end in `!`.
- Constants: `ALL_CAPS_AND_UNDERSCORES`.
- "Number of X" is `n_x`.
- DataFrames/time series: prefer `DataFrames`/`TimeArray`; do not use the pandas
  Julia wrapper. Convert to a `DataFrame` inside a code block when convenient and
  back to `TimeArray` afterward.

## Notebooks (both languages)

Each notebook has one top-level section (`#`) and nested subsections
(`##`, `###`, …). Each section is followed by a short descriptive paragraph
(roughly one to ten sentences). Cells need not map one-to-one to sections; a
series of cells may share a section. Notebooks must run top-to-bottom against a
pinned environment with seeded randomness.

## Figures

Figures should have transparent backgrounds.
