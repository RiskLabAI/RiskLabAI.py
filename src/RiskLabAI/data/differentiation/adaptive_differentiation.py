r"""
Adaptive Fractional Differencing (AFD), a bias-corrected differencing-order estimate.

de Prado's fixed-width fractional differentiation (`differentiation.py`) finds the minimum order d that
makes a series stationary via an ADF search. On finite samples that min-d-via-ADF is biased and
under-differences: the under-powered ADF test passes while the series still carries spurious residual
long-memory weights, so the recovered order is too small. AFD instead estimates the differencing order
directly from the data's long-memory structure, with a finite-sample bias correction, and so recovers
the genuine order far more accurately when memory is strong and the sample is finite.

Admitted in Appraisal 13 (CONTRIBUTIONS_LEDGER 2026-06-27). Regime tag, verbatim from the verdict:

    prefer AFD over fixed-width FFD when the differencing order itself must be right - strong long memory
    and finite samples, where min-d-via-ADF under-differences; on weak memory the gap narrows. The
    implemented AFD is a tractable clean-room approximation of the published wavelet-Hurst + ridge +
    CV-truncation method (no public code).

The real-data predictive-lift confirmation is a tracked obligation (``REAL_DATA_FOLLOWUPS.md``): the
order accuracy is established against ground truth, but the downstream predictive value of the more
accurate order is not yet decisive. Evidence and caveats: appraisals/13_verdict.md (and 13b if logged).

This is a tractable clean-room approximation of the published AFD (IEEE Access 2025; no public code):
the differencing order is anchored at ``0.5 + d_hat`` where ``d_hat`` is a finite-sample-bias-corrected
Hurst estimate of the increments (a ridge-style equal-weight blend of a wavelet-variance and an R/S
Hurst estimate, which lowers the variance/bias of either alone), the fixed-width-FFD truncation is
chosen by cross-validation over weight thresholds, and the order is raised to the ADF stationarity
boundary if not yet stationary. ``pywt`` (PyWavelets) is an optional dependency: when present AFD uses
the wavelet + R/S Hurst blend, otherwise the R/S estimate alone (the blend is the validated default).

References
----------
Adaptive Fractional Differencing (2025) IEEE Access. (Clean-room approximation; no public source.)
Lopez de Prado, M. (2018) Advances in Financial Machine Learning, ch. 5 (the FFD baseline).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from .differentiation import fractional_difference_fixed_single

try:  # optional wavelet dependency (enables the wavelet-variance Hurst component of the blend)
    import pywt

    _HAS_PYWT = True
except (
    ImportError
):  # pragma: no cover - exercised only in environments without PyWavelets
    _HAS_PYWT = False

DEFAULT_DELTA_GRID = [round(i * 0.05, 3) for i in range(0, 25)]  # 0.00 .. 1.20
DEFAULT_CV_THRESHOLDS = (1e-3, 1e-4, 1e-5)
DEFAULT_ADF_SIGNIFICANCE = 0.05
DEFAULT_ADF_MAXLAG = 1


def wavelet_variance_hurst(
    series: np.ndarray, wavelet: str = "db2", skip_fine: int = 1
) -> float:
    r"""
    Wavelet-variance Hurst estimate: :math:`\log_2 \mathrm{Var}(\text{detail}_j) \approx (2H-1)\,j`.

    Returns ``nan`` if ``pywt`` is not installed or the series is too short to form enough octaves.
    """
    if not _HAS_PYWT:
        return float("nan")
    x = np.asarray(series, dtype=float)
    x = x - x.mean()
    n = x.size
    dec_len = pywt.Wavelet(wavelet).dec_len
    max_level = pywt.dwt_max_level(n, dec_len)
    level = max(3, min(max_level, int(np.log2(n)) - 2))
    if level < 3:
        return float("nan")
    coefficients = pywt.wavedec(x, wavelet, level=level)
    details = coefficients[1:]  # coarsest .. finest
    octaves = np.arange(level, 0, -1)
    variances = np.array([np.mean(d**2) for d in details])
    mask = octaves >= (1 + skip_fine)
    j = octaves[mask].astype(float)
    v = np.where(variances[mask] <= 0, 1e-12, variances[mask])
    if j.size < 2:
        return float("nan")
    slope = np.polyfit(j, np.log2(v), 1)[0]
    return (slope + 1.0) / 2.0


def rescaled_range_hurst(series: np.ndarray) -> float:
    """Classical rescaled-range (R/S) Hurst estimate (the log-log slope of R/S against window size)."""
    x = np.asarray(series, dtype=float)
    n = x.size
    if n < 32:
        return float("nan")
    sizes = np.unique(
        np.floor(np.logspace(np.log10(8), np.log10(n // 2), 8)).astype(int)
    )
    rs_means, used = [], []
    for s in sizes:
        k = n // s
        if k < 1:
            continue
        values = []
        for i in range(k):
            segment = x[i * s : (i + 1) * s]
            deviations = np.cumsum(segment - segment.mean())
            spread = deviations.max() - deviations.min()
            scale = segment.std()
            if scale > 0:
                values.append(spread / scale)
        if values:
            rs_means.append(np.mean(values))
            used.append(s)
    if len(used) < 2:
        return float("nan")
    return float(np.polyfit(np.log(used), np.log(rs_means), 1)[0])


def adaptive_differencing_order(increments: np.ndarray) -> float:
    r"""
    Bias-corrected memory order :math:`\hat{d}` of a series of increments.

    A ridge-style equal-weight blend of the wavelet-variance and R/S Hurst estimates,
    :math:`\hat{d} = \bar{H} - 0.5`, clipped to ``[0, 0.99]``. Falls back to the R/S estimate alone if
    ``pywt`` is unavailable. Clean-room approximation of the published wavelet-Hurst + ridge procedure.
    """
    h_wavelet = wavelet_variance_hurst(increments)
    h_rs = rescaled_range_hurst(increments)
    estimates = [h for h in (h_wavelet, h_rs) if np.isfinite(h)]
    if not estimates:
        return float("nan")
    h_blend = float(np.mean(estimates))
    return float(np.clip(h_blend - 0.5, 0.0, 0.99))


def _adf_pvalue(values: np.ndarray, maxlag: int) -> float:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size < 20 or np.allclose(v.std(), 0.0):
        return 1.0
    try:
        return float(adfuller(v, maxlag=maxlag, regression="c", autolag=None)[1])
    except Exception:  # pragma: no cover
        return 1.0


def _memory_retained(differenced: pd.Series, level: pd.Series) -> float:
    common = differenced.dropna()
    if len(common) < 5:
        return float("nan")
    level_aligned = level.reindex(common.index)
    ok = (~common.isna()) & (~level_aligned.isna())
    if ok.sum() < 5 or common[ok].std() == 0 or level_aligned[ok].std() == 0:
        return float("nan")
    return float(
        abs(np.corrcoef(common[ok].to_numpy(), level_aligned[ok].to_numpy())[0, 1])
    )


def adaptive_fractional_difference(
    series: pd.Series,
    delta_grid: list | None = None,
    cv_thresholds: tuple = DEFAULT_CV_THRESHOLDS,
    adf_significance: float = DEFAULT_ADF_SIGNIFICANCE,
    adf_maxlag: int = DEFAULT_ADF_MAXLAG,
) -> dict:
    r"""
    Adaptive Fractional Differencing of a (price/level) series.

    Estimates the memory order from the series increments with a bias-corrected Hurst blend
    (:func:`adaptive_differencing_order`), anchors the differencing order at that estimate plus one
    (the integration order of a price), chooses the fixed-width-FFD weight-truncation threshold by
    cross-validation (the threshold that retains the most memory subject to ADF stationarity), and
    raises the order along ``delta_grid`` only as far as the ADF stationarity boundary if needed.

    Prefer AFD over fixed-width FFD when the differencing order itself must be right (strong long memory,
    finite samples), where de Prado's min-d-via-ADF under-differences; on weak memory the gap narrows.
    See the module docstring for the full regime tag, the clean-room-approximation note, the tracked
    real-data follow-up, and appraisals/13_verdict.md.

    Parameters
    ----------
    series : pd.Series
        The price / level series to difference.
    delta_grid : list, optional
        Ascending grid of candidate orders. Defaults to 0.00 .. 1.20 in steps of 0.05.
    cv_thresholds : tuple, default (1e-3, 1e-4, 1e-5)
        The FFD weight-truncation thresholds searched by cross-validation.
    adf_significance : float, default 0.05
        The ADF p-value below which the series is judged stationary.
    adf_maxlag : int, default 1
        The ADF lag (matches the repo's fixed-width FFD search).

    Returns
    -------
    dict
        ``order`` (the differencing order applied), ``d_hat`` (the bias-corrected memory order of the
        increments), ``adf_pvalue``, ``memory_retained`` (abs correlation of the differenced series with
        the level), ``threshold`` (the chosen truncation), and ``series`` (the differenced pd.Series).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> price = pd.Series(np.cumsum(rng.standard_normal(500)))     # an integrated series
    >>> result = adaptive_fractional_difference(price)
    >>> 0.0 <= result["order"] <= 2.0 and result["adf_pvalue"] <= 1.0
    True
    """
    if delta_grid is None:
        delta_grid = DEFAULT_DELTA_GRID
    level = series.dropna()
    increments = level.diff().dropna().to_numpy()
    d_hat_increment = adaptive_differencing_order(increments)
    if not np.isfinite(d_hat_increment):
        return {
            "order": float("nan"),
            "d_hat": float("nan"),
            "adf_pvalue": 1.0,
            "memory_retained": float("nan"),
            "threshold": float("nan"),
            "series": pd.Series(dtype=float),
        }
    order_anchor = float(np.clip(0.5 + d_hat_increment, 0.0, max(delta_grid)))

    # CV truncation: the threshold with max retained memory among stationary diffs at the anchor order.
    best_threshold, best = cv_thresholds[1], None
    for threshold in cv_thresholds:
        diff = fractional_difference_fixed_single(
            series, order_anchor, threshold
        ).dropna()
        if len(diff) < 20:
            continue
        pvalue = _adf_pvalue(diff.to_numpy(), adf_maxlag)
        memory = _memory_retained(diff, series)
        candidate = (
            pvalue < adf_significance,
            memory if np.isfinite(memory) else -1.0,
            threshold,
            pvalue,
            diff,
        )
        if best is None or (candidate[0], candidate[1]) > (best[0], best[1]):
            best = candidate
    if best is not None:
        best_threshold = best[2]

    # Anchor at the order estimate; raise it upward only to reach the stationarity boundary.
    order = order_anchor
    diff = fractional_difference_fixed_single(series, order, best_threshold).dropna()
    pvalue = _adf_pvalue(diff.to_numpy(), adf_maxlag) if len(diff) >= 20 else 1.0
    grid_up = [g for g in delta_grid if g > order_anchor]
    while pvalue >= adf_significance and grid_up:
        order = grid_up.pop(0)
        diff = fractional_difference_fixed_single(
            series, order, best_threshold
        ).dropna()
        pvalue = _adf_pvalue(diff.to_numpy(), adf_maxlag) if len(diff) >= 20 else 1.0

    return {
        "order": float(order),
        "d_hat": float(d_hat_increment),
        "adf_pvalue": float(pvalue),
        "memory_retained": _memory_retained(diff, series),
        "threshold": float(best_threshold),
        "series": diff,
    }
