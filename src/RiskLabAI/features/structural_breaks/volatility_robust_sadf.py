r"""
Volatility-robust SADF / GSADF via wild-bootstrap critical values (Harvey-Leybourne-Sollis-Taylor 2016).

The admitted SADF / GSADF bubble detector (`structural_breaks.get_sadf_sequence` /
`get_gsadf_statistic`) uses a sup-ADF statistic whose null distribution assumes homoskedastic errors.
Under non-stationary volatility the test over-rejects, flagging spurious bubbles (the mild oversizing
noted in Appraisal 05). The volatility-robust variant computes the SAME sup-ADF statistics but calibrates
the critical values by a wild bootstrap (Rademacher sign-flip of the first-difference residuals), which
preserves the series' own volatility pattern while destroying any explosive autocorrelation, restoring
correct size under non-stationary volatility.

Admitted in Appraisal 26 (CONTRIBUTIONS_LEDGER 2026-06-27). Regime tag, verbatim from the verdict:

    prefer it over plain SADF/GSADF when the series' volatility may be non-stationary: it holds nominal
    size where plain SADF over-rejects ~9x, at a modest power cost, and converges to plain SADF under
    constant volatility. Pairs with the admitted GSADF/BSADF.

Held-out confirmed (appraisals/26_results, HELDOUT.md): on the sealed variance-break-6x bubble null the
volatility-robust SADF holds nominal size (~0.067) where plain SADF over-rejects (~0.667, the homoskedastic
critical value flags spurious bubbles), and it matches plain SADF under constant volatility at a modest
power cost. The plain SADF / GSADF baseline is unchanged. Evidence: appraisals/26_verdict.md.

References
----------
Harvey, D. I., Leybourne, S. J., Sollis, R. and Taylor, A. M. R. (2016) Tests for explosive financial
    bubbles in the presence of non-stationary volatility. Journal of Empirical Finance, 38, 548-574.
Phillips, P. C. B., Shi, S. and Yu, J. (2015) Testing for multiple bubbles. International Economic Review.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .structural_breaks import _psy_sadf_bsadf_sequences, psy_minimum_window


def _sup_adf_statistics(log_price: np.ndarray, min_length: int) -> tuple:
    sadf_seq, bsadf_seq = _psy_sadf_bsadf_sequences(
        np.asarray(log_price, float), min_length
    )
    sadf = float(np.nanmax(sadf_seq)) if np.isfinite(sadf_seq).any() else np.nan
    gsadf = float(np.nanmax(bsadf_seq)) if np.isfinite(bsadf_seq).any() else np.nan
    return sadf, gsadf


def volatility_robust_sadf(
    log_price: np.ndarray | pd.Series,
    min_length: int | None = None,
    n_bootstrap: int = 99,
    random_state: int | None = None,
) -> dict:
    r"""
    Volatility-robust SADF / GSADF wild-bootstrap test for an explosive (bubble) episode.

    Computes the observed SADF and GSADF sup-ADF statistics, then calibrates their p-values by a wild
    bootstrap: each bootstrap path multiplies the demeaned first-difference residuals by an independent
    Rademacher (+/- 1) sign and re-cumulates, preserving the series' (possibly non-stationary) volatility
    while removing any explosive autocorrelation; the sup-ADF statistics are recomputed on each path to
    form the null distribution. The p-value is the share of bootstrap statistics at least as large as the
    observed one.

    Prefer this over plain SADF / GSADF when the series' volatility may be non-stationary: it holds nominal
    size where the plain test over-rejects, at a modest power cost, and converges to the plain test under
    constant volatility. It pairs with the admitted GSADF / BSADF (the same sup-ADF statistics). See the
    module docstring for the verbatim regime tag and appraisals/26_verdict.md.

    Parameters
    ----------
    log_price : np.ndarray or pd.Series
        The log-price level series.
    min_length : int, optional
        Minimum (PSY) window for the sup-ADF recursion; defaults to ``psy_minimum_window(len)``.
    n_bootstrap : int, default 99
        Wild-bootstrap replications.
    random_state : int, optional
        Seed for the Rademacher signs.

    Returns
    -------
    dict
        ``sadf``, ``gsadf`` (observed statistics), ``sadf_pvalue``, ``gsadf_pvalue`` (wild-bootstrap),
        and ``reject_sadf`` / ``reject_gsadf`` (at the 5% level).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> y = np.cumsum(rng.standard_normal(200))            # a random walk (no bubble)
    >>> res = volatility_robust_sadf(y, n_bootstrap=49, random_state=0)
    >>> set(res) == {"sadf", "gsadf", "sadf_pvalue", "gsadf_pvalue", "reject_sadf", "reject_gsadf"}
    True
    """
    y = np.asarray(
        log_price.values if isinstance(log_price, pd.Series) else log_price, dtype=float
    )
    if y.ndim != 1 or y.size < 8:
        raise ValueError("log_price must be a 1-D series of length >= 8")
    nmin = int(min_length) if min_length is not None else psy_minimum_window(y.size)
    sadf_obs, gsadf_obs = _sup_adf_statistics(y, nmin)

    residual = np.diff(y)
    residual = residual - residual.mean()
    y0 = y[0]
    rng = np.random.default_rng(random_state)
    ge_sadf = ge_gsadf = 0
    for _ in range(n_bootstrap):
        signs = rng.integers(0, 2, size=residual.shape[0]) * 2 - 1
        path = np.empty_like(y)
        path[0] = y0
        path[1:] = y0 + np.cumsum(signs * residual)
        sadf_b, gsadf_b = _sup_adf_statistics(path, nmin)
        if np.isfinite(sadf_b) and np.isfinite(sadf_obs) and sadf_b >= sadf_obs:
            ge_sadf += 1
        if np.isfinite(gsadf_b) and np.isfinite(gsadf_obs) and gsadf_b >= gsadf_obs:
            ge_gsadf += 1
    p_sadf = (1 + ge_sadf) / (n_bootstrap + 1)
    p_gsadf = (1 + ge_gsadf) / (n_bootstrap + 1)
    return {
        "sadf": sadf_obs,
        "gsadf": gsadf_obs,
        "sadf_pvalue": float(p_sadf),
        "gsadf_pvalue": float(p_gsadf),
        "reject_sadf": bool(p_sadf < 0.05),
        "reject_gsadf": bool(p_gsadf < 0.05),
    }
