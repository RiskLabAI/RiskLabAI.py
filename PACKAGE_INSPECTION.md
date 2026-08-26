# Future Python package inspection specification

This specification governs temporary local artifacts built from the exact
RiskLabAI 3.1.0 additive integration candidate. It does not authorize using
version control, uploading, publishing, tagging, or releasing anything.

## Inputs

The inspector must use the final, hash-locked source tree and compare it with
`PACKAGE_FILES.json`, `PUBLIC_API.json`, `TEST_INVENTORY.json`, and
`pyproject.toml`. It must reject a changed input, an undecodable file, an
unresolved link, or a concurrent mutation before considering an artifact.

`PACKAGE_FILES.json` separately records active release-source files and
already-public repository-only files. Human repository integration preserves
both classes without deletion; only the active class may enter the source
distribution, and only its declared runtime subset may enter the wheel.

## Archive safety and membership

For both archive formats, reject absolute paths, drive-qualified paths,
backslashes, traversal components, duplicate paths, case-fold or Unicode
normalization collisions, control characters, trailing dots or spaces,
alternate data streams, links, device files, and unexpected native binaries.

The wheel may contain only the Python runtime modules declared in
`PACKAGE_FILES.json`, plus the normalized standard metadata directory for the
approved version. Tests, examples, analytical control records, caches, bytecode,
credentials, and undeclared data must not be present in the wheel. The source
distribution must contain exactly the release-source members declared in
`PACKAGE_FILES.json`, including the governed workflow, plus only the
backend-generated metadata members that the inspection record enumerates and
hashes explicitly.

## Metadata

The normalized package metadata must agree exactly with the human-approved
`pyproject.toml`: name RiskLabAI, the frozen additive version, Python 3.12
through 3.14, BSD-3-Clause, the declared maintainer and contact, repository,
issue, and documentation links, all required dependency clauses, and every
optional group. No dynamic version, entry point, plugin, executable script, or
undeclared dependency is permitted.

Every wheel record must have a valid cryptographic digest and size, except its
own permitted empty digest fields. The license and long description must match
the reviewed source bytes.

## Runtime checks

In fresh, isolated environments for every supported Python and valid NumPy
lane, verify the exact module origins, the approved version binding, the root
exports, all module and package exports in `PUBLIC_API.json`, and the complete
87-name causal API. Require the released 57-name causal prefix to remain exact
and the 30 additions to match the parity contract. For every module that
declares `__all__`, require every listed name to be bound on that installed
module and resolvable by an explicit import; an advertised but unbound name
rejects the artifact. Run the exact package-scoped test inventory and the valid
optional-feature lanes.
Confirm that optional groups do not become base dependencies and that
changepoints alone is unavailable on Python 3.14. Run installed-package tests
outside the source checkout and reject any import whose resolved path remains
inside the repository. Tests that launch child interpreters must give those
processes clean temporary working directories and apply the same origin
rejection.

## Final review

Scan every archive member name and every decoded public text member against the
release hygiene denylist. Record archive hashes, member hashes, metadata,
runtime origins, test results, supported environments, and every deviation.
Any deviation keeps the artifact rejected. Human release, publication, and
version-control authority remain separate decisions after inspection.
