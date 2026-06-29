"""
Tests for data/differentiation/adaptive_differentiation.py (Adaptive Fractional Differencing).
"""

import numpy as np
import pandas as pd
import pytest
from scipy.signal import fftconvolve
from statsmodels.tsa.stattools import adfuller

from RiskLabAI.data.differentiation.adaptive_differentiation import (
    DEFAULT_DELTA_GRID,
    _HAS_PYWT,
    adaptive_differencing_order,
    adaptive_fractional_difference,
    rescaled_range_hurst,
    wavelet_variance_hurst,
)
from RiskLabAI.data.differentiation.differentiation import (
    fractional_difference_fixed_single,
)


def _ffd_min_d(price, alpha=0.05, threshold=1e-4):
    """de Prado baseline: the minimum order on the grid whose fixed-width FFD is ADF-stationary."""
    last = DEFAULT_DELTA_GRID[-1]
    for d in DEFAULT_DELTA_GRID:
        diff = fractional_difference_fixed_single(price, d, threshold).dropna()
        if len(diff) < 20:
            continue
        last = d
        if (
            float(adfuller(diff.to_numpy(), maxlag=1, regression="c", autolag=None)[1])
            < alpha
        ):
            return d
    return last


def _arfima_increments(d, n, rng, burn=512):
    """ARFIMA(0,d,0) increments x = (1-L)^{-d} eps, with known memory order d (Hurst = d + 0.5)."""
    m = n + burn
    eps = rng.standard_normal(m)
    psi = np.empty(m)
    psi[0] = 1.0
    for k in range(1, m):
        psi[k] = psi[k - 1] * (k - 1 + d) / k
    return fftconvolve(eps, psi)[:m][burn:]


def test_hurst_sanity_on_known_d():
    """Validation: ARFIMA(0,d,0) increments with d=0.3 give a Hurst estimate near d + 0.5 = 0.8."""
    if not _HAS_PYWT:
        pytest.skip("PyWavelets not installed")
    rng = np.random.default_rng(0)
    increments = _arfima_increments(0.3, 4000, rng)
    h_rs = rescaled_range_hurst(increments)
    h_wave = wavelet_variance_hurst(increments)
    assert np.isfinite(h_wave)
    assert abs(0.5 * (h_rs + h_wave) - 0.8) < 0.12
    assert abs(adaptive_differencing_order(increments) - 0.3) < 0.15


def test_afd_recovers_order_better_than_ffd_strong_memory():
    """
    Replication of the Appraisal 13 held-out mechanism: on strong-memory short samples (d=0.45, N=250),
    posed on the integrated price (true boundary order tau = 1 + d = 1.45), AFD recovers the order with
    a smaller mean error than de Prado's min-d-via-ADF baseline.
    """
    d, n = 0.45, 250
    tau = 1.0 + d
    afd_err, ffd_err = [], []
    for s in range(20):
        rng = np.random.default_rng(100 + s)
        price = pd.Series(np.cumsum(_arfima_increments(d, n, rng)))
        afd = adaptive_fractional_difference(price)
        afd_err.append(abs(afd["order"] - tau))
        ffd_err.append(abs(_ffd_min_d(price) - tau))
    assert np.mean(afd_err) < np.mean(ffd_err)


def test_afd_returns_stationary_and_finite():
    """AFD returns a finite order, a differenced series, and reaches ADF stationarity on an integrated series."""
    rng = np.random.default_rng(7)
    price = pd.Series(np.cumsum(rng.standard_normal(500)))
    result = adaptive_fractional_difference(price)
    assert np.isfinite(result["order"]) and 0.0 <= result["order"] <= 2.0
    assert len(result["series"].dropna()) > 0
    assert result["adf_pvalue"] <= 0.10  # differenced to (near) stationarity


def test_afd_edge_cases():
    """Short/degenerate series are handled without error."""
    short = adaptive_fractional_difference(pd.Series(np.arange(10.0)))
    assert "order" in short
    flat = adaptive_fractional_difference(pd.Series(np.ones(300)))
    assert np.isfinite(flat["order"]) or np.isnan(flat["order"])


def test_rs_hurst_random_walk_half():
    """R/S Hurst of i.i.d. noise (the increments of a random walk) is near 0.5."""
    rng = np.random.default_rng(9)
    h = rescaled_range_hurst(rng.standard_normal(4000))
    assert abs(h - 0.5) < 0.15
