# RiskLabAI Python next-major release notes — blocked draft

Target version: **3.0.0**. The owner approved this exact version, but assigning
it does not authorize a build, artifact, publication, upload, or release.

The next major Python release preserves the prior public repository source
modules and adds the verified 57-concept `RiskLabAI.causal_factor_analysis`
namespace. It intentionally drops support for Python 3.9-3.11 and NumPy 1.x;
the supported policy is CPython 3.12-3.14 with NumPy `>=2.2,<3`.

Compatibility repairs include NumPy 2 trapezoidal integration, public
scikit-learn bootstrap logic, optional Numba acceleration, lazy optional
dependencies, a Python-3.14-specific Joblib floor, and self-contained standard
technical indicators. Python 3.14 remains supported when the optional
changepoint backend is unavailable.

Compatibility aliases that remain present in 3.0.0 now consistently warn of
removal in 4.0.0. No alias is silently deleted, and the warning contract is
locked by package-scoped tests.

The release metadata uses the PEP 639 `BSD-3-Clause` expression and license
file without the superseded license classifier. The governed CI workflow now
installs the distribution, rejects imports shadowed by the preserved repository
root in both parent and child interpreters, tests the six approved base
runtime/NumPy combinations, exercises feature-specific optional lanes, and
isolates the Python 3.14 changepoint limitation. Pinned static checks cover the
clean causal source and tests.

No source-conflicted causal result is admitted. The metadata contract is
complete; artifacts, publication, upload, and release remain blocked.

The static distributable metadata, exact 124-module runtime inventory,
78-file package-scoped test inventory, public API inventory, documentation,
examples, dependency groups, governed CI workflow, and 215-file intended
source-distribution list are now frozen. No retained package artifact has been
created. The remaining gates are future artifact inspection and separate human
authorization for version-control, publication, and release actions.
