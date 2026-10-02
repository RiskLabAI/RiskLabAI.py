# RiskLabAI Python 3.2.0 release notes

Version **3.2.0** is prepared locally and remains unreleased.

## [3.2.0] - Unreleased

### Added

- Sequential bootstrap sampling, majorization-based optimization, financial
  network clearing, and optimal execution utilities.
- Ambiguity calculations for supplied priors, order-flow features, regime and
  drift adaptation, and tail-risk estimation and backtesting.
- Constrained and robust portfolio optimization and decision-focused losses.
- Causal bounds, finite-stratum transport, design-based interference, and
  inverse-probability and doubly robust policy evaluation.
- PyTorch implementations of TimeGAN, deep hedging, and finite-grid neural SDEs.
- Analytical examples, independent numerical checks, boundary validation, and
  regression tests for the new methods.

### Fixed

- Sample-concurrency counting for events that begin before the selected window.

### Scope and dependencies

- The neural-SDE implementation uses an explicitly corrected fixed-coordinate
  diffusion operator rather than the literal published nearest-frame
  construction. Its guarantees are bounded to the implemented finite grid;
  general dynamic no-arbitrage and empirical market performance are not claimed.
- Required dependencies and supported Python versions are unchanged. PyTorch
  is available through the existing `pde` extra. Selected new methods require
  separately installed CVXPY, River, or PySensemakr; these remain optional.
- The additions are Python implementations. Corresponding Julia additions and
  book notebooks are deferred.

## Historical 3.1.0 additive causal release notes

Version: **3.1.0**. The version is frozen for local integration and artifact
inspection. Version-control, publication, upload, and release actions remain
human-controlled and are not authorized by this document.

The candidate starts from RiskLabAI 3.0.0 and preserves its complete public
library and released 57-name `RiskLabAI.causal_factor_analysis` contract. It
adds 30 paper-derived names, producing an 87-name causal namespace with a
matching Julia implementation. No released causal name, signature, default,
or behavior is removed or silently changed.

The support policy remains CPython 3.12-3.14 with NumPy `>=2.2,<3`. The
additions require no new Python dependency and retain all required and
feature-specific optional dependency boundaries established in 3.0.0.

The additive analytical families cover general-variance factor-mirage
coefficients, allocation-misspecification evidence, accepted-DAG factor roles,
deterministic structural-model evaluation, and family- and selection-level
false-discovery calculations for searched trials. The two false-discovery
estimands remain explicitly separate.

Every implemented method is linked to an authoritative public source and checked
against direct analytical examples, independent mathematical or graph oracles,
validation boundaries, stability cases, and—in addition—shared numerical
Python-Julia fixtures. Cross-language agreement is not used as the sole
correctness authority.

Thirteen method units remain source-blocked because the papers conflict or omit
necessary mathematics, data, trained assets, or reproducibility settings.
Those omissions are documented rather than guessed. Excluded legacy causal
implementations are neither modified nor used as correctness authorities.

The complete preserved library, additive causal tests, formatting and static
checks, six approved base runtime/NumPy endpoints, and three optional-feature
runtime lines are verified. Python 3.14 remains supported when the optional
changepoint backend is unavailable.

Package metadata is frozen at 3.1.0 for the exact additive integration
candidate. The exact inventories, complete compatibility matrix, and temporary
wheel and source-distribution inspection must all pass before any human Git
handoff. Version control, publication, upload, and release remain human-only
and separately authorized.
