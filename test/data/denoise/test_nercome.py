"""
Tests for data/denoise/nercome.py (NERCOME sample-splitting covariance denoiser).
"""

import numpy as np
import pytest

from RiskLabAI.data.denoise.denoising import corr_to_cov, cov_to_corr, denoise_cov
from RiskLabAI.data.denoise.nercome import nercome_denoised_covariance


def _no_gap_population(p, rng, alpha=0.9):
    """A slowly-decaying (no clean gap) population covariance Q diag(i^-alpha) Q'."""
    eigenvalues = (np.arange(1, p + 1, dtype=float)) ** (-alpha)
    eigenvalues = eigenvalues / eigenvalues.mean()  # unit average variance
    q, _ = np.linalg.qr(rng.standard_normal((p, p)))
    return (q * eigenvalues) @ q.T


def _clean_gap_population(p, rng, n_factors=3, gap=9.0):
    """A clean-gap factor-plus-noise population covariance (a few large eigenvalues + a tight bulk)."""
    loadings = rng.standard_normal((p, n_factors)) * np.sqrt(gap)
    cov = loadings @ loadings.T + np.eye(p)
    return cov_to_corr(cov)


def _mp_clip_covariance(returns):
    """de Prado MP eigenvalue clipping of the sample covariance (the baseline)."""
    n, p = returns.shape
    std = returns.std(axis=0, ddof=1)
    std = np.where(std <= 0, 1.0, std)
    corr0 = cov_to_corr(np.cov(returns, rowvar=False))
    clipped_corr = cov_to_corr(denoise_cov(corr0, n / p, 0.01))
    return corr_to_cov(clipped_corr, std)


def _rel_frobenius(estimate, true):
    return float(np.linalg.norm(estimate - true) / np.linalg.norm(true))


def test_nercome_lower_covariance_error_on_no_gap_spectrum():
    """
    Replication of the Appraisal 24 mechanism: on a no-gap (slowly-decaying) spectrum NERCOME recovers
    the covariance with lower relative Frobenius error than MP eigenvalue clipping.
    """
    p = 20
    nercome_err, mp_err = [], []
    for s in range(10):
        rng = np.random.default_rng(s)
        sigma = _no_gap_population(p, rng)
        chol = np.linalg.cholesky(sigma + 1e-10 * np.eye(p))
        returns = rng.standard_normal((40, p)) @ chol.T  # T=40, p/n=0.5
        nercome_err.append(
            _rel_frobenius(
                nercome_denoised_covariance(returns, n_splits=30, random_state=s), sigma
            )
        )
        mp_err.append(_rel_frobenius(_mp_clip_covariance(returns), sigma))
    assert np.mean(nercome_err) < np.mean(
        mp_err
    )  # NERCOME more accurate on the no-gap spectrum


def test_nercome_better_conditioned_than_sample():
    rng = np.random.default_rng(1)
    sigma = _no_gap_population(24, rng)
    chol = np.linalg.cholesky(sigma + 1e-10 * np.eye(24))
    returns = rng.standard_normal((48, 24)) @ chol.T
    cov_nercome = nercome_denoised_covariance(returns, n_splits=40, random_state=1)
    assert np.linalg.cond(cov_nercome) < np.linalg.cond(np.cov(returns, rowvar=False))


def test_nercome_does_not_blow_up_on_clean_gap():
    """On a clean-gap stationary spectrum NERCOME stays competitive with clipping (converges, no free lunch)."""
    p = 20
    nercome_err, mp_err = [], []
    for s in range(8):
        rng = np.random.default_rng(100 + s)
        corr = _clean_gap_population(p, rng)
        chol = np.linalg.cholesky(corr + 1e-10 * np.eye(p))
        returns = rng.standard_normal((120, p)) @ chol.T
        nercome_err.append(
            _rel_frobenius(
                cov_to_corr(
                    nercome_denoised_covariance(returns, n_splits=30, random_state=s)
                ),
                corr,
            )
        )
        mp_err.append(_rel_frobenius(cov_to_corr(_mp_clip_covariance(returns)), corr))
    # NERCOME is within a small factor of clipping on a clean gap (no catastrophic loss)
    assert np.mean(nercome_err) < 1.25 * np.mean(mp_err)


def test_nercome_output_is_symmetric_pd():
    rng = np.random.default_rng(2)
    cov = nercome_denoised_covariance(
        rng.standard_normal((100, 10)), n_splits=20, random_state=2
    )
    assert cov.shape == (10, 10)
    assert np.allclose(cov, cov.T)
    assert np.all(np.linalg.eigvalsh(cov) > 0)


def test_nercome_edge_cases():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError):
        nercome_denoised_covariance(rng.standard_normal(50))  # 1-D
    with pytest.raises(ValueError):
        nercome_denoised_covariance(rng.standard_normal((3, 5)))  # too few observations
