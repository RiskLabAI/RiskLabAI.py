# RiskLabAI 3.0.0 pre-package source

Status: **local-only and blocked**.

This directory is a temporary clean release nucleus for the next major
RiskLabAI release across Python and Julia. It is not a separate product. The
Python distribution and import namespace remain `RiskLabAI`; the Julia package
remains `RiskLabAI` with UUID `a72881da-fdaa-49c1-8962-99caf4ccfee8`.

The distributable metadata contract is complete, but no wheel or source
distribution exists for this corrected source state, and nothing has been
published, registered, or released. The approved target version is `3.0.0`.
The source remains blocked until a separately authorized temporary artifact is
inspected and the owner separately authorizes version-control and release
actions.

## Final Python source scope

The `src` tree preserves all 117 runtime-source paths captured from the
current public Python `main` baseline and adds seven reviewed
`causal_factor_analysis` source files. The prior modules remain organized under
`backtest`, `cluster`, `controller`, `core`, `data`, `ensemble`, `features`,
`hpc`, `optimization`, `pde`, and `utils`. Their captured bytes are retained as
a continuity baseline; this does not clear later local changes or make the full
surface release-ready.

The completed book-derived causal-factor namespace exposes 57 public concepts:

- minimum-variance target-exposure allocation;
- published confounder and collider analytical diagnostics;
- immutable graphical-identification evidence and checks; and
- immutable evidence records for the seven-stage causal-factor protocol;
- randomized, stratified, difference-in-differences, and instrumental-variable
  treatment-effect identities; and
- exact fork, collider, and confounded-mediator population diagnostics with
  deterministic specification experiments.

The complete preserved source library has passed its approved compatibility
matrix on CPython 3.12.13, 3.13.9, and 3.14.3. The base policy is NumPy
`>=2.2,<3`, tested at the lowest genuinely compatible and current NumPy 2.x
release for each interpreter. Feature-specific dependencies remain optional;
Numba does not constrain the non-Numba base, and the unavailable Python 3.14
changepoint backend disables only that feature.

## Scope and parity hold

No path from the current public repository baseline has been deleted. The
completed 57-concept causal namespace is integrated here, and the Julia
candidate independently verifies the same 57 concepts. The full-library
runtime and dependency matrix, exact public API inventory, 78-file
package-scoped test inventory, distributable metadata, documentation, and
intended package file list are complete. The exact Python `3.0.0` and Julia
`1.0.0` targets are assigned. Artifact inspection and human-controlled
version-control, publication, and release decisions remain blocked.

The graph routines report implications of a caller-supplied DAG; they do not
discover or certify that graph, prove positivity or instrument strength, or
estimate an effect. The protocol validator checks structural evidence records;
it does not perform the empirical stages or prove that caller declarations are
true. The analytical functions implement only the source-consistent public
results recorded by the publication audit; source-conflicted results remain
excluded.

## Confirmed identity and stewardship

- Product and package identity: `RiskLabAI`
- Intended license: BSD-3-Clause
- Rights holder and public maintainer: Hamid Arian
- Copyright: 2022-2026
- Contact: arian@risklab.ai
- Python repository: https://github.com/RiskLabAI/RiskLabAI.py
- Python issues: https://github.com/RiskLabAI/RiskLabAI.py/issues
- Julia repository: https://github.com/RiskLabAI/RiskLabAI.jl
- Documentation: https://github.com/RiskLabAI/RiskLabAI.py#readme

The rights holder has confirmed permission to publish and license every file
admitted to the clean public package. This confirmation does not remove the
artifact-inspection or human-authorization blocks.

See `PACKAGE_IDENTITY.json` for the fail-closed machine-readable state.

The complete Python contract is recorded in `pyproject.toml`,
`PUBLIC_API.json`, `TEST_INVENTORY.json`, and `PACKAGE_FILES.json`. The future
artifact checks are specified in `PACKAGE_INSPECTION.md`; that document is an
inspection plan, not build or release authorization.

The 57-concept contract is documented in `docs/causal_factor_analysis.md`, and
runtime details are in `docs/compatibility.md`. A small deterministic example
is in `examples/causal_factor_analysis_quickstart.py`, and the blocked
next-major summary is in `RELEASE_NOTES_NEXT_MAJOR.md`.
