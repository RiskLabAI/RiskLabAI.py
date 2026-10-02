# Python runtime and dependency support

Status: **3.2.0 candidate; not yet released**.

RiskLabAI 3.2.0 retains the Python and dependency support policy of 3.1.0.
The new Python methods preserve the existing optional dependency boundaries.
CVXPY, River, and PySensemakr are additional optional suppliers installed
separately for selected methods; see `INSTALLATION.md`. Package metadata is
prepared for local artifact inspection. No version shown in this document authorizes a
version-control action, upload, publication, or release.

## Supported interpreter and NumPy policy

RiskLabAI supports standard GIL builds of CPython 3.12, 3.13, and 3.14. The
base numerical policy is `numpy>=2.2,<3`. A matrix lane uses only versions for
which compatible distributions actually exist; it never forces an older
NumPy source release onto a newer interpreter.

The following exact versions describe the historical 3.1.0 validation matrix.
They are not a claim that these versions were rerun for 3.2.0.

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
`numpy>=2.2,<3` is retained for RiskLabAI 3.1.0. Python 3.14 resolves NumPy
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

## Continuous integration contract

The governed CI workflow installs the distribution non-editably before every
runtime test. Import-origin probes run outside the repository checkout and
reject any `RiskLabAI` module resolved from the preserved repository root.
Tests that launch child interpreters also use clean temporary working
directories, so a subprocess cannot silently resolve the preserved root
package in place of the installed `src` distribution.

The base matrix contains six valid interpreter/NumPy combinations: the lowest
supported NumPy binary and the latest supported NumPy 2.x release on each of
CPython 3.12, 3.13, and 3.14. Three additional optional-feature lanes install
the feature-specific groups on those interpreter lines. The `speed` group
keeps its narrower NumPy `<2.6` constraint, and the Python 3.14 lane verifies
that only the unavailable changepoint backend is skipped.

Pinned Black and Ruff checks cover the independently maintained clean causal
source and tests. Preserved source and regression tests remain governed by the
complete installed-package runtime matrix and are not rewritten solely to
satisfy a new style tool.

## Deprecation policy

Public compatibility aliases retained in RiskLabAI 3.1.0 remain callable and
emit `DeprecationWarning`. Their removal target is 4.0.0. Package-scoped tests
bind both continued availability and the stated target; no preserved alias is
deleted in this release.

## Historical 3.1.0 matrix evidence

The causal tree collects exactly 507 tests: the released 57-name suite, the 30
additive-name suite, independent analytical and graph oracles, boundary and
stability checks, and 12 shared numerical-fixture cases. All 507 passed on each
supported interpreter line. The fixture also passed at every minimum and
current NumPy endpoint.

Before the fixture was added, every required base endpoint passed the complete
preserved library with 845 passes and seven declared optional or platform
skips. The 12-case fixture then passed independently on all six unchanged
environments, closing each base endpoint at 857 passes and seven declared
skips. Across those endpoints this is 5,142 passes and 42 expected skips.

The maximally applicable Python 3.12 optional environment collects 869 cases;
its final complete run passed 868 and skipped only the Windows `lscpu` case.
The corresponding Python 3.13 evidence is 868 passes and one Windows skip.
Python 3.14 records 863 passes and two declared skips: Windows `lscpu` and the
unavailable changepoint backend. Numba's compiled and pure-Python paths agree,
the PyTorch PDE tests pass, and the preserved optional-feature smoke tests pass
within their declared groups.

One real conditional floor was discovered: Joblib releases before 1.5.2 are
not reliable for the preserved Windows parallel-HPO path on Python 3.14.
Joblib 1.5.2 and 1.5.3 pass; older Python lines continue to pass with Joblib
1.4.x.

The causal public design is exactly 87 names: the released 57-name contract as
an unchanged prefix plus 30 documented additions. The candidate did not import, modify, or
reintroduce the excluded legacy causal implementation.

## 3.2.0 validation scope

The existing Python and NumPy floors remain unchanged for this additive
candidate. The implementation commit passed the six base and three optional
GitHub CI lanes and the static checks. The version update requires a new CI
run before merging. Historical pass counts above are not 3.2.0 results.
The 3.2.0 candidate passed 1,178 tests locally on Python 3.13, with one
Windows platform skip and 73 existing warnings.
Supplemental CVXPY, River, and PySensemakr checks were run locally on Python
3.13; the existing optional CI groups do not install those three suppliers.
The causal namespace now contains 92 names, retaining the previous 87 names.
The five new causal exports do not imply new Julia parity.

## Remaining release gates

The public surface, analytical matrix, exact source and test inventories, and
final hygiene controls are complete. Temporary local wheel and
source-distribution artifacts are permitted only for inspection and must remain
under the isolated integration root. Every version-control, publication,
upload, and release action remains separately human-controlled.
