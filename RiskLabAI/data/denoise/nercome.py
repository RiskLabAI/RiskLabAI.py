r"""
NERCOME: nonparametric eigenvalue-regularized covariance estimation (Lam 2016).

de Prado denoises the covariance by Marcenko-Pastur eigenvalue clipping (`denoising.denoise_cov`), which
assumes a clean noise bulk separated from the signal eigenvalues by a gap. When the eigenvalue spectrum
has no clean gap (a slowly-decaying bulk) or is non-stationary, that assumption breaks and clipping
degenerates toward the raw sample covariance. NERCOME (Lam 2016) instead regularizes the eigenvalues by
sample-splitting: it estimates the eigenvectors on one split of the data and the oracle eigenvalues by
projecting the held-out split's sample covariance onto those eigenvectors, averaging over many random
splits. This recovers the covariance more accurately and with better conditioning on no-gap /
non-stationary spectra, and converges to clipping where the gap assumption holds.

Admitted in Appraisal 24 (CONTRIBUTIONS_LEDGER 2026-06-27) - portfolio's first admitted extension. Regime
tag, verbatim from the verdict:

    prefer NERCOME over MP clipping for covariance estimation when the eigenvalue spectrum has no clean
    gap or is non-stationary (better accuracy and conditioning, and lower OOS volatility via NCO /
    min-variance); it converges to clipping on clean-gap stationary spectra. It costs more turnover and
    concentration than clipping, gives no risk-adjusted-return edge (none does - 1/N stands), and HRP
    does not benefit (use it through NCO / min-variance).

Held-out confirmed (appraisals/24_results, HELDOUT.md): on the sealed no-gap x non-stationary x T=30
corner NERCOME has distinguishably lower covariance error (relative Frobenius 0.563 vs MP clipping 0.684)
and better conditioning (29.6 vs 46.3), with lower NCO realized risk (1.315 vs 1.553); HRP is insensitive.
Unlike `denoise_cov`, which cleans a covariance matrix, NERCOME is a data-driven (sample-splitting)
estimator, so it takes the return matrix. Evidence and caveats: appraisals/24_verdict.md.

References
----------
Lam, C. (2016) Nonparametric eigenvalue-regularized precision or covariance matrix estimator.
    The Annals of Statistics, 44(3), 928-953.
Lopez de Prado, M. (2018) Advances in Financial Machine Learning, ch. 2 (the MP-clipping baseline).
"""

from __future__ import annotations

import numpy as np

from .denoising import corr_to_cov, cov_to_corr


def _ensure_positive_definite(matrix: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Project a symmetric matrix to the nearest PD one by flooring its eigenvalues."""
    matrix = (matrix + matrix.T) / 2.0
    values, vectors = np.linalg.eigh(matrix)
    floor = (
        max(eps, eps * float(values.max())) if values.size and values.max() > 0 else eps
    )
    values = np.clip(values, floor, None)
    out = (vectors * values) @ vectors.T
    return (out + out.T) / 2.0


def nercome_denoised_covariance(
    returns: np.ndarray,
    n_splits: int = 50,
    split_fraction: float = 2.0 / 3.0,
    random_state: int | None = None,
) -> np.ndarray:
    r"""
    NERCOME denoised covariance (Lam 2016), estimated by sample-splitting from a return matrix.

    For each of ``n_splits`` random row permutations the ``T`` observations are split into two halves; the
    eigenvectors ``P`` come from the first half's sample covariance, and the regularized eigenvalues are
    the oracle projection ``d_i = p_i' S2 p_i`` of the second half's sample covariance ``S2`` onto those
    eigenvectors. The estimator ``P diag(d) P'`` is averaged over the splits (each term is PSD, so the
    average is PSD). The returns are standardized to unit sample variance, NERCOME-cleaned in correlation
    space, then scaled back to a covariance by the sample column standard deviations.

    Prefer NERCOME over MP eigenvalue clipping when the eigenvalue spectrum has no clean gap or is
    non-stationary (better accuracy and conditioning, and lower out-of-sample volatility via NCO /
    min-variance); it converges to clipping on clean-gap stationary spectra, costs more turnover and
    concentration, gives no risk-adjusted-return edge, and HRP does not benefit (use it through NCO /
    min-variance). See the module docstring for the full verbatim regime tag and appraisals/24_verdict.md.
    Unlike :func:`denoising.denoise_cov`, this takes the return matrix, not a covariance.

    Parameters
    ----------
    returns : np.ndarray
        The ``T x p`` matrix of asset returns (rows = observations, columns = assets).
    n_splits : int, default 50
        Number of random sample-splits averaged.
    split_fraction : float, default 2/3
        Fraction ``m / T`` of rows used for the eigenvector (first) half (Lam's ~2T/3 default).
    random_state : int, optional
        Seed for the split permutations.

    Returns
    -------
    np.ndarray
        The ``p x p`` NERCOME denoised covariance (symmetric positive-definite).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> returns = rng.standard_normal((120, 8))
    >>> cov = nercome_denoised_covariance(returns, n_splits=20, random_state=0)
    >>> cov.shape == (8, 8) and np.allclose(cov, cov.T)
    True
    """
    returns = np.asarray(returns, dtype=float)
    if returns.ndim != 2:
        raise ValueError("returns must be a 2-D (T x p) array")
    n, p = returns.shape
    if n < 4:
        raise ValueError("NERCOME needs at least 4 observations")
    std = returns.std(axis=0, ddof=1)
    std = np.where(std <= 0, 1.0, std)
    z = returns / std

    m = max(p + 1, int(round(split_fraction * n)))
    m = min(m, n - 2)
    rng = np.random.default_rng(random_state)
    accumulated = np.zeros((p, p))
    used = 0
    for _ in range(n_splits):
        order = rng.permutation(n)
        first, second = order[:m], order[m:]
        if len(second) < 2:
            continue
        cov_first = np.cov(z[first], rowvar=False)
        cov_second = np.cov(z[second], rowvar=False)
        _, vectors = np.linalg.eigh(cov_first)
        oracle = np.einsum(
            "ij,jk,ik->i", vectors.T, cov_second, vectors.T
        )  # p_i' S2 p_i
        oracle = np.clip(oracle, 0.0, None)
        accumulated += (vectors * oracle) @ vectors.T
        used += 1
    estimate = accumulated / max(used, 1)
    correlation = cov_to_corr(_ensure_positive_definite(estimate))
    return corr_to_cov(correlation, std)
