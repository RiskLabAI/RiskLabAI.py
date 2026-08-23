r"""
Sharpe-ratio inference under non-normal, autocorrelated returns (Lopez de Prado-Lipton-Zoonekynd 2025).

The sampling variance of the Sharpe estimator depends on the higher moments AND the serial correlation
of returns. de Prado's Probabilistic Sharpe Ratio (`probabilistic_sharpe_ratio`) corrects the higher
moments but assumes serial independence, so under autocorrelation its confidence interval is too narrow
and its test over-rejects. The LPLZ (2025) method takes the heteroskedasticity-and-autocorrelation-
consistent (HAC, Newey-West) long-run variance of the Sharpe influence function, correcting for both at
once.

The Sharpe influence function is

.. math::

    \mathrm{IF}_t = z_t - \tfrac{1}{2}\,\widehat{SR}\,(z_t^2 - 1), \qquad z_t = (r_t - \mu)/\sigma,

and the asymptotic variance of :math:`\sqrt{T}(\widehat{SR} - SR)` is the long-run variance of
:math:`\mathrm{IF}_t`. Under i.i.d. returns this long-run variance equals
:math:`1 - S\,\widehat{SR} + \tfrac{K-1}{4}\widehat{SR}^2` (exactly the PSR denominator), so the method
converges to the PSR when there is no serial correlation.

Admitted in Appraisal 08 (CONTRIBUTIONS_LEDGER 2026-06-27). Regime tag, verbatim from the verdict:

    Prefer the LPLZ (2025) Sharpe inference - a HAC of the Sharpe influence function - when returns
    show material serial correlation and/or non-normality (estimable from the sample): it restores
    near-nominal CI coverage and test size where the PSR under-covers and over-rejects (PSR size ~ 0.20
    vs nominal 0.05 under AR(1)). It converges to the PSR on near-normal i.i.d. returns with no
    over-coverage. Caveats: mild small-sample over-rejection at short tracks, coverage restored toward
    (not fully to) nominal at strong autocorrelation, and wider CIs - the honest power cost. Lo (2002)
    is the autocorrelation-only intermediate, dominated by LPLZ under non-normality.

Evidence and caveats: appraisals/08_verdict.md.

References
----------
Lopez de Prado, M., Lipton, A. and Zoonekynd, V. (2025) Sharpe-ratio inference under non-normal,
    serially-correlated returns. (Open code at github.com/zoonek/2025-sharpe-ratio - NOT consulted or
    vendored; this is reimplemented clean-room from the influence-function / HAC math, GOVERNANCE 3.)
Newey, W. K. and West, K. D. (1987) A simple, positive semi-definite, heteroskedasticity and
    autocorrelation consistent covariance matrix. Econometrica, 55(3), 703-708.
Lo, A. W. (2002) The statistics of Sharpe ratios. Financial Analysts Journal, 58(4), 36-52.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import stats as ss


def _sharpe(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else 0.0


def sharpe_ratio_influence_function(returns: np.ndarray) -> np.ndarray:
    r"""
    The Sharpe-ratio influence function :math:`\mathrm{IF}_t = z_t - \tfrac{1}{2}SR(z_t^2 - 1)`.

    Parameters
    ----------
    returns : np.ndarray
        The return series.

    Returns
    -------
    np.ndarray
        The per-observation influence function (mean ~ 0).
    """
    r = np.asarray(returns, dtype=float)
    mu = r.mean()
    sigma = r.std(ddof=1)
    if sigma == 0:
        return np.zeros_like(r)
    z = (r - mu) / sigma
    sr = mu / sigma
    return z - 0.5 * sr * (z**2 - 1.0)


def newey_west_long_run_variance(series: np.ndarray, lag: int) -> float:
    r"""
    Newey-West (Bartlett-kernel) long-run variance of a series.

    .. math::
        \widehat{\Omega} = \gamma_0 + 2 \sum_{k=1}^{L} \left(1 - \frac{k}{L+1}\right) \gamma_k,

    where :math:`\gamma_k` is the lag-k autocovariance. With ``lag == 0`` this is the sample variance.

    Parameters
    ----------
    series : np.ndarray
        The series whose long-run variance is estimated.
    lag : int
        The HAC bandwidth L (number of Bartlett lags).

    Returns
    -------
    float
        The estimated long-run variance (floored at a small positive number).
    """
    x = np.asarray(series, dtype=float)
    x = x - x.mean()
    t = x.size
    if t == 0:
        return 1e-12
    total = float(x @ x) / t
    for k in range(1, lag + 1):
        weight = 1.0 - k / (lag + 1.0)
        total += 2.0 * weight * float(x[k:] @ x[:-k]) / t
    return max(total, 1e-12)


def newey_west_automatic_lag(number_of_returns: int) -> int:
    r"""Newey-West automatic bandwidth :math:`\lfloor 4 (T/100)^{2/9} \rfloor` (at least 1)."""
    return max(int(np.floor(4.0 * (number_of_returns / 100.0) ** (2.0 / 9.0))), 1)


def lplz_sharpe_inference(
    returns: np.ndarray,
    confidence_level: float = 0.95,
    lag: Optional[int] = None,
    null_sharpe_ratio: float = 0.0,
) -> dict[str, float]:
    r"""
    Lopez de Prado-Lipton-Zoonekynd (2025) Sharpe-ratio inference (HAC of the influence function).

    Returns the per-period Sharpe ratio, its autocorrelation-and-non-normality-robust standard error,
    a confidence interval, and the test of ``null_sharpe_ratio``. The standard error is
    :math:`\sqrt{\widehat{\Omega}/T}` where :math:`\widehat{\Omega}` is the Newey-West long-run variance
    of the Sharpe influence function.

    Prefer this over the PSR when returns show material serial correlation and/or non-normality
    (estimable from the sample): it restores near-nominal CI coverage and test size where the PSR
    under-covers and over-rejects, and converges to the PSR on near-normal i.i.d. returns with no
    over-coverage. The honest cost is a wider CI under autocorrelation. See the module docstring for the
    full regime tag and appraisals/08_verdict.md.

    Parameters
    ----------
    returns : np.ndarray
        The strategy return series.
    confidence_level : float, default 0.95
        The CI confidence level.
    lag : int, optional
        The HAC (Newey-West) bandwidth. Defaults to :func:`newey_west_automatic_lag`.
    null_sharpe_ratio : float, default 0.0
        The Sharpe under the null hypothesis tested.

    Returns
    -------
    dict
        ``sharpe_ratio``, ``standard_error``, ``confidence_interval`` (a (lo, hi) tuple),
        ``test_statistic``, ``p_value`` (two-sided), ``significant`` (reject the null at the level),
        and ``lag`` (the bandwidth used).

    Examples
    --------
    >>> import numpy as np
    >>> from RiskLabAI.backtest import lplz_sharpe_inference
    >>> rng = np.random.default_rng(0)
    >>> result = lplz_sharpe_inference(0.1 + rng.standard_normal(240))
    >>> result["standard_error"] > 0
    True
    """
    r = np.asarray(returns, dtype=float)
    t = r.size
    sr = _sharpe(r)
    if t < 3 or r.std(ddof=1) == 0:
        return {
            "sharpe_ratio": sr,
            "standard_error": float("nan"),
            "confidence_interval": (float("nan"), float("nan")),
            "test_statistic": float("nan"),
            "p_value": float("nan"),
            "significant": False,
            "lag": 0,
        }
    if lag is None:
        lag = newey_west_automatic_lag(t)
    lrv = newey_west_long_run_variance(sharpe_ratio_influence_function(r), lag)
    standard_error = float(np.sqrt(lrv / t))
    z = ss.norm.ppf(0.5 + confidence_level / 2.0)
    ci = (sr - z * standard_error, sr + z * standard_error)
    test_statistic = (
        (sr - null_sharpe_ratio) / standard_error
        if standard_error > 0
        else float("nan")
    )
    p_value = float(2.0 * ss.norm.sf(abs(test_statistic)))
    return {
        "sharpe_ratio": sr,
        "standard_error": standard_error,
        "confidence_interval": ci,
        "test_statistic": float(test_statistic),
        "p_value": p_value,
        "significant": bool(p_value < (1.0 - confidence_level)),
        "lag": int(lag),
    }
