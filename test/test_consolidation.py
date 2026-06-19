"""
Tests for the duplication-consolidation pass (non-breaking).

Verifies that (a) the parallel-processing helpers historically exposed from
``RiskLabAI.data.labeling`` still import and are now the single-source
implementations from ``RiskLabAI.hpc``; and (b) the clustering covariance->
correlation converter delegates to the canonical ``denoise.cov_to_corr`` and
produces identical output for valid covariance matrices.
"""

import numpy as np


# --------------------------------------------------------------------------- #
# labeling helpers are now single-sourced from RiskLabAI.hpc
# --------------------------------------------------------------------------- #
def test_labeling_helpers_are_reexported_from_hpc():
    from RiskLabAI import hpc
    from RiskLabAI.data import labeling

    # Public names still import (backward compatibility).
    assert labeling.process_jobs is hpc.process_jobs
    assert labeling.expand_call is hpc.expand_call
    assert labeling.report_progress is hpc.report_progress
    # `lin_parts` was the historical name for hpc's `linear_partitions`.
    assert labeling.lin_parts is hpc.linear_partitions


def test_labeling_package_exports_still_resolve():
    # The names listed in RiskLabAI.data.labeling.__all__ must all be importable.
    from RiskLabAI.data import labeling

    for name in ("lin_parts", "process_jobs", "expand_call", "report_progress"):
        assert hasattr(labeling, name), name


def test_lin_parts_matches_linear_partitions_numerically():
    from RiskLabAI.data.labeling import lin_parts

    # Same partition boundaries for representative inputs.
    np.testing.assert_array_equal(lin_parts(100, 4), np.array([0, 25, 50, 75, 100]))
    np.testing.assert_array_equal(
        lin_parts(10, 3), np.ceil(np.linspace(0, 10, min(3, 10) + 1)).astype(int)
    )


# --------------------------------------------------------------------------- #
# cov -> corr is single-sourced (clustering delegates to denoise.cov_to_corr)
# --------------------------------------------------------------------------- #
def _reference_cov_to_corr(cov):
    std = np.sqrt(np.diag(cov))
    std[std == 0] = 1.0
    corr = cov / np.outer(std, std)
    corr[corr < -1] = -1.0
    corr[corr > 1] = 1.0
    np.fill_diagonal(corr, 1.0)
    return corr


def test_clustering_cov_to_corr_delegates_and_matches():
    from RiskLabAI.cluster.clustering import covariance_to_correlation
    from RiskLabAI.data.denoise.denoising import cov_to_corr

    rng = np.random.default_rng(7)
    for _ in range(50):
        n = int(rng.integers(2, 10))
        a = rng.standard_normal((n, n))
        cov = a @ a.T + np.diag(rng.uniform(0.01, 1.0, n))  # valid PSD, +diag

        out = covariance_to_correlation(cov)
        # Delegates to the canonical implementation.
        np.testing.assert_allclose(out, cov_to_corr(cov), rtol=0, atol=1e-12)
        # And matches an independent reference.
        np.testing.assert_allclose(out, _reference_cov_to_corr(cov), rtol=0, atol=1e-12)
        # Correlation diagonal is exactly 1.
        np.testing.assert_allclose(np.diag(out), np.ones(n), rtol=0, atol=1e-12)


def test_clustering_cov_to_corr_does_not_mutate_input():
    from RiskLabAI.cluster.clustering import covariance_to_correlation

    cov = np.array([[4.0, 2.0], [2.0, 9.0]])
    cov_copy = cov.copy()
    _ = covariance_to_correlation(cov)
    np.testing.assert_array_equal(cov, cov_copy)
