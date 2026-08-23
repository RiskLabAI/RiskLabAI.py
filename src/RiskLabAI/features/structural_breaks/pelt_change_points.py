r"""
PELT exact multiple change-point detection (Killick, Fearnhead and Eckley 2012).

de Prado's CUSUM (Chu-Stinchcombe-White) structural-break test targets a single mean shift and is, by
construction, blind to a pure variance change. PELT (Pruned Exact Linear Time) finds the exact set of
multiple change-points that minimizes a segment cost plus a per-change penalty, so with a Gaussian
(mean-and-variance) cost it detects and dates multiple and variance change-points that CUSUM misses,
without over-segmenting when penalized appropriately.

Admitted in Appraisal 26 (CONTRIBUTIONS_LEDGER 2026-06-27). Regime tag, verbatim from the verdict:

    prefer it over CUSUM for detecting and dating multiple and/or variance change-points (which CUSUM
    misses), without over-segmenting; CUSUM remains adequate for a single mean shift.

Held-out confirmed (appraisals/26_results, HELDOUT.md): on the sealed mixed mean+variance corner (3 true
change-points) PELT recovers 2.65 of 3 on average with a 0.000 over-segmentation rate (it never exceeds
the true count), where CUSUM recovers far fewer of these mean+variance change-points. The CUSUM baseline
is unchanged. Evidence: appraisals/26_verdict.md.

PELT is provided through the ``ruptures`` package (BSD-2-Clause), an optional dependency imported lazily;
the GPL-3 ``exuber`` reference was not used. Install it with ``pip install ruptures``.

References
----------
Killick, R., Fearnhead, P. and Eckley, I. A. (2012) Optimal detection of changepoints with a linear
    computational cost. Journal of the American Statistical Association, 107(500), 1590-1598.
"""

from __future__ import annotations

import numpy as np


def pelt_change_points(
    series,
    model: str = "normal",
    penalty_multiplier: float = 3.0,
    min_size: int = 10,
    jump: int = 5,
) -> list:
    r"""
    Detect and date change-points with PELT (exact multiple change-point detection).

    Fits a PELT model with a BIC-style penalty ``penalty_multiplier * log(T)`` and returns the interior
    change indices (the segment boundaries, excluding the series endpoints). With ``model="normal"`` the
    Gaussian cost captures both mean and variance changes, so PELT recovers multiple and variance
    change-points that CUSUM misses, without over-segmenting under an adequate penalty.

    Prefer PELT over CUSUM for detecting and dating multiple and/or variance change-points; CUSUM remains
    adequate for a single mean shift. See the module docstring for the verbatim regime tag and
    appraisals/26_verdict.md.

    Parameters
    ----------
    series : array-like
        The 1-D series to segment.
    model : str, default "normal"
        ``ruptures`` cost model (``"normal"`` for mean+variance, ``"l2"`` for mean only, ``"rbf"`` ...).
    penalty_multiplier : float, default 3.0
        Penalty ``= penalty_multiplier * log(T)`` (higher penalizes more change-points).
    min_size : int, default 10
        Minimum segment length.
    jump : int, default 5
        Grid stride for candidate change-points (``ruptures`` speed knob).

    Returns
    -------
    list of int
        The interior change-point indices, ascending (empty if none).

    Raises
    ------
    ImportError
        If ``ruptures`` is not installed (it is an optional dependency).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> x = np.concatenate([rng.standard_normal(120), rng.standard_normal(120) + 3.0])
    >>> cps = pelt_change_points(x)                       # doctest: +SKIP
    >>> any(100 < c < 140 for c in cps)                   # doctest: +SKIP
    True
    """
    try:
        import ruptures as rpt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "pelt_change_points requires ruptures (pip install ruptures)"
        ) from exc

    x = np.asarray(series, dtype=float).reshape(-1, 1)
    n = x.shape[0]
    if n < 2 * int(min_size):
        return []
    algorithm = rpt.Pelt(model=model, min_size=int(min_size), jump=int(jump)).fit(x)
    breakpoints = algorithm.predict(pen=float(penalty_multiplier) * np.log(n))
    return [int(b) for b in breakpoints if 0 < b < n]
