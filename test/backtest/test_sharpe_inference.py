"""
Tests for backtest/sharpe_inference.py (LPLZ HAC Sharpe inference).
"""

import numpy as np

from RiskLabAI.backtest.sharpe_inference import (
    lplz_sharpe_inference,
    newey_west_automatic_lag,
    newey_west_long_run_variance,
    sharpe_ratio_influence_function,
)


def test_newey_west_lag0_is_sample_variance():
    """With lag 0 the Newey-West long-run variance is the sample variance."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    assert np.isclose(newey_west_long_run_variance(x, 0), np.var(x), atol=1e-12)


def test_influence_function_iid_variance_matches_psr_denominator():
    """The i.i.d. variance of the influence function equals 1 - S*SR + (K-1)/4 * SR^2 (PSR denom)."""
    from scipy import stats as ss

    rng = np.random.default_rng(1)
    r = 0.15 + rng.standard_normal(20000)  # near-normal, SR ~ 0.15
    sr = r.mean() / r.std(ddof=1)
    skew = ss.skew(r)
    kurt = ss.kurtosis(r, fisher=False)
    psr_denom = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    ifv = np.var(sharpe_ratio_influence_function(r))
    assert np.isclose(ifv, psr_denom, atol=0.02)


def test_lplz_converges_to_iid_se_when_no_autocorrelation():
    """On i.i.d. near-normal returns the LPLZ SE ~ the textbook sqrt((1 + SR^2/2)/T)."""
    rng = np.random.default_rng(2)
    r = 0.1 + rng.standard_normal(4000)
    res = lplz_sharpe_inference(r)
    sr = res["sharpe_ratio"]
    iid_se = np.sqrt((1 + 0.5 * sr**2) / r.size)
    assert np.isclose(res["standard_error"], iid_se, rtol=0.15)


def test_positive_autocorrelation_inflates_se():
    """Positive AR(1) autocorrelation widens the LPLZ SE relative to the i.i.d. SE."""
    rng = np.random.default_rng(3)
    n = 1000
    innov = rng.standard_normal(n)
    ar = np.zeros(n)
    for t in range(1, n):
        ar[t] = 0.5 * ar[t - 1] + innov[t]
    ar = 0.1 + (ar - ar.mean()) / ar.std()
    se_ar = lplz_sharpe_inference(ar, lag=10)["standard_error"]
    se_iid = lplz_sharpe_inference(ar, lag=0)["standard_error"]
    assert se_ar > se_iid


def test_lplz_coverage_better_than_iid_under_ar1():
    """LPLZ CI coverage of the true Sharpe is closer to nominal than the i.i.d. (lag 0) CI under AR(1)."""
    phi, true_sr, n, z = 0.4, 0.15, 120, 1.959963985
    cov_lplz = cov_iid = 0
    n_sims = 600
    scale = np.sqrt(1 - phi**2)
    for s in range(n_sims):
        rng = np.random.default_rng(100 + s)
        innov = rng.standard_normal(n + 200)
        x = np.zeros(n + 200)
        for t in range(1, n + 200):
            x[t] = phi * x[t - 1] + scale * innov[t]
        r = true_sr + x[200:]
        lplz = lplz_sharpe_inference(r)
        iid = lplz_sharpe_inference(r, lag=0)
        cov_lplz += (
            lplz["confidence_interval"][0] <= true_sr <= lplz["confidence_interval"][1]
        )
        cov_iid += (
            iid["confidence_interval"][0] <= true_sr <= iid["confidence_interval"][1]
        )
    cov_lplz /= n_sims
    cov_iid /= n_sims
    assert abs(cov_lplz - 0.95) < abs(cov_iid - 0.95)  # LPLZ nearer nominal
    assert cov_iid < 0.95  # the iid CI under-covers under autocorrelation


def test_edge_cases():
    """Short and degenerate series return NaN standard errors without error."""
    short = lplz_sharpe_inference(np.array([0.1, 0.2]))
    assert np.isnan(short["standard_error"])
    constant = lplz_sharpe_inference(np.zeros(50))
    assert np.isnan(constant["standard_error"])
    assert newey_west_automatic_lag(120) >= 1
