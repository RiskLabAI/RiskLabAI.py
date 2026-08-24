# Future Python package inspection specification

This specification applies only after a separate human authorization creates a
temporary RiskLabAI 3.0.0 wheel and source distribution. It does not authorize
building, installing, uploading, publishing, tagging, or releasing anything.

## Inputs

The inspector must use the final, hash-locked source tree and compare it with
`PACKAGE_FILES.json`, `PUBLIC_API.json`, `TEST_INVENTORY.json`, and
`pyproject.toml`. It must reject a changed input, an undecodable file, an
unresolved link, or a concurrent mutation before considering an artifact.

## Archive safety and membership

For both archive formats, reject absolute paths, drive-qualified paths,
backslashes, traversal components, duplicate paths, case-fold or Unicode
normalization collisions, control characters, trailing dots or spaces,
alternate data streams, links, device files, and unexpected native binaries.

The wheel may contain only the 124 declared Python runtime modules plus the
standard metadata directory created for RiskLabAI 3.0.0. Tests, examples,
local readiness controls, caches, bytecode, credentials, and undeclared data
must not be present in the wheel. The source distribution must contain exactly
the 215 release-source members declared in `PACKAGE_FILES.json`, including the
governed `.github/workflows/ci.yml`, plus only the
backend-generated metadata members that the inspection record enumerates and
hashes explicitly.

## Metadata

The normalized package metadata must agree exactly with `pyproject.toml`:
name RiskLabAI, version 3.0.0, Python 3.12 through 3.14, BSD-3-Clause, the
declared maintainer and contact, repository, issue, and documentation links,
all required dependency clauses, and every optional group. No dynamic version,
entry point, plugin, executable script, or undeclared dependency is permitted.

Every wheel record must have a valid cryptographic digest and size, except its
own permitted empty digest fields. The license and long description must match
the reviewed source bytes.

## Runtime checks

In fresh, isolated environments for every supported Python and valid NumPy
lane, verify the exact module origins, `RiskLabAI.__version__ == "3.0.0"`, the
13 root exports, all module and package exports in `PUBLIC_API.json`, and the
complete 57-feature causal API. Run the exact package-scoped test inventory and
the valid optional-feature lanes. Confirm that optional groups do not become
base dependencies and that changepoints alone is unavailable on Python 3.14.
Run installed-package tests outside the source checkout and reject any import
whose resolved path remains inside the repository.

## Final review

Scan every archive member name and every decoded public text member against the
release hygiene denylist. Record archive hashes, member hashes, metadata,
runtime origins, test results, supported environments, and every deviation.
Any deviation keeps the artifact rejected. Human release, publication, and
version-control authority remain separate decisions after inspection.
