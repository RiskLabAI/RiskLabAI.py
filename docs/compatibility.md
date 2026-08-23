# Python runtime and dependency support

Status: **matrix verified; release blocked**.

This policy applies to the blocked RiskLabAI `3.0.0` release candidate. The
assigned version does not authorize a build, package, upload, publication, or
release.

## Supported interpreter and NumPy policy

RiskLabAI supports standard GIL builds of CPython 3.12, 3.13, and 3.14. The
base numerical policy is `numpy>=2.2,<3`. A matrix lane uses only versions for
which compatible distributions actually exist; it never forces an older
NumPy source release onto a newer interpreter.

| Lane | CPython | NumPy | Required-dependency position |
|---|---:|---:|---|
| 3.12 minimum | 3.12.13 | 2.2.0 | approved floors |
| 3.12 current | 3.12.13 | 2.5.2 | current compatible releases |
| 3.13 minimum | 3.13.9 | 2.2.0 | approved floors |
| 3.13 current | 3.13.9 | 2.5.2 | current compatible releases |
| 3.14 minimum | 3.14.3 | 2.3.2 | lowest compatible binary set |
| 3.14 current | 3.14.3 | 2.5.2 | current compatible releases |

The NumPy lower bound was reconsidered at metadata freeze on 2026-08-21. NumPy
2.2 remains inside the two-year Scientific Python SPEC 0 support window through
2026-12-08 and has passed the approved Python 3.12 and 3.13 minimum lanes, so
`numpy>=2.2,<3` is retained for RiskLabAI 3.0.0. Python 3.14 resolves NumPy
2.3.2 or newer because that is its first tested compatible binary lane. A
future RiskLabAI release must reassess the floor rather than silently carrying
it forward. See https://scientific-python.org/specs/spec-0000/.

The required dependency policy is:

- pandas `>=2.3,<4`;
- SciPy `>=1.15,<2`;
- scikit-learn `>=1.6,<2`;
- statsmodels `>=0.14.6,<0.15`;
- Joblib `>=1.4,<2` on Python below 3.14 and `>=1.5.2,<2` on Python 3.14;
- PyWavelets `>=1.9,<2`.

The Python 3.14 minimum lane resolves newer compatible SciPy and
scikit-learn releases where older releases do not publish compatible wheels.
This is an interpreter-specific resolution fact, not a reason to raise the
floor for Python 3.12 or 3.13.

## Optional capabilities

Optional dependencies do not constrain the lean base:

- `speed`: Numba `>=0.67,<0.68`, with NumPy `<2.6` for this group only;
- `pde`: PyTorch `>=2.10,<3`;
- `synth`: QuantEcon `>=0.11.4,<0.12`;
- `hpo`: Optuna `>=4.9,<5`;
- `plot`: matplotlib `>=3.10,<4`, seaborn `>=0.13.2,<0.14`, and Plotly `>=6,<7`;
- `symbolic`: SymPy `>=1.14,<2`;
- `profile`: memory-profiler `==0.61.0` provisionally;
- `simulation`: tqdm `>=4.70,<5`;
- `changepoints`: ruptures `==1.1.10` on Python 3.12 and 3.13 only.

Python 3.14 remains supported when ruptures is unavailable. Only
`pelt_change_points` is unavailable in that environment, and calling it raises
a direct dependency error. The rest of RiskLabAI remains importable.

The financial-feature simulation no longer relies on the stale `ta` package.
ADX, RSI, CCI, stochastic momentum, ROC, ATR, and Ichimoku lines are computed
internally from standard definitions and have hand-calculated oracle tests.

## Deprecation policy

Public compatibility aliases retained in RiskLabAI 3.0.0 remain callable and
emit `DeprecationWarning`. Their removal target is 4.0.0. Package-scoped tests
bind both continued availability and the stated target; no preserved alias is
deleted in this release.

## Matrix evidence

Every lane passed all 341 frozen causal-factor tests. Across six lanes this is
2,046 causal assertions with no failures.

The final self-contained package test tree collects exactly 692 tests: 343
preserved-library tests, 341 causal-factor tests, and 8 independent technical-
indicator oracles. Its complete Python 3.12 current-dependency run succeeded
with 691 passes and one declared Windows platform skip. The package inventory
records both the collected suite and the 680-test applicable base matrix used
on every supported interpreter/dependency lane.

The complete preserved base suite passed 331 applicable tests per lane. Five
platform or optional-capability cases were skipped by their declared guards;
the two QuantEcon-dependent cases were exercised separately in the `synth`
lanes. The new indicator-oracle suite passed 8 tests in every lane. Numba's
compiled and pure-Python paths agreed in every lane. PyTorch 2.10 and 2.13 both
passed the preserved PDE tests on all three interpreter lines. Plotting,
symbolic, profiling, HPO, and synthetic-data smoke tests passed in their
feature-specific environments.

One real conditional floor was discovered: Joblib releases before 1.5.2 are
not reliable for the preserved Windows parallel-HPO path on Python 3.14.
Joblib 1.5.2 and 1.5.3 pass; older Python lines continue to pass with Joblib
1.4.x.

The causal public design remains exactly 57 concepts. Compatibility repairs
did not import or reintroduce the excluded legacy causal implementation.

## Remaining release gates

The metadata, public surface, tests, and source allowlists are complete. No
package has been built or installed. Temporary artifact inspection and separate
human authorization for version-control, publication, and release remain
outstanding.
