"""
Tests for strategy_risk.py
"""

import numpy as np
import pytest
from RiskLabAI.backtest.strategy_risk import (
    sharpe_ratio_trials,
    implied_precision,
    bin_frequency,
    binomial_sharpe_ratio,
    mix_gaussians,
    failure_probability,
)

def test_sharpe_ratio_trials():
    """Test sharpe_ratio_trials."""
    # With p=0.5, mean should be ~0
    mean, std, sr = sharpe_ratio_trials(p=0.5, n_run=100000)
    assert np.isclose(mean, 0.0, atol=0.01)
    assert np.isclose(std, 1.0, atol=0.01)
    assert np.isclose(sr, 0.0, atol=0.01)
    
    # With p=1.0, mean=1, std=0, sr=0 (by implementation)
    mean, std, sr = sharpe_ratio_trials(p=1.0, n_run=100)
    assert np.isclose(mean, 1.0)
    assert np.isclose(std, 0.0)
    assert np.isclose(sr, 0.0)

def test_binomial_sharpe_ratio():
    """Test binomial_sharpe_ratio."""
    # 50/50 win/loss
    sr = binomial_sharpe_ratio(
        stop_loss=-0.01,
        profit_taking=0.01,
        frequency=252,
        probability=0.5
    )
    # E[R] = 0.5*0.01 + 0.5*(-0.01) = 0
    # Stdev = (0.01 - (-0.01)) * sqrt(0.5*0.5) = 0.02 * 0.5 = 0.01
    # SR_trade = 0 / 0.01 = 0
    assert np.isclose(sr, 0.0)
    
    # High precision
    sr_high_p = binomial_sharpe_ratio(
        stop_loss=-0.01,
        profit_taking=0.01,
        frequency=252,
        probability=0.6
    )
    # E[R] = 0.6*0.01 + 0.4*(-0.01) = 0.006 - 0.004 = 0.002
    # Stdev = (0.01 - (-0.01)) * sqrt(0.6*0.4) = 0.02 * sqrt(0.24) = 0.009798
    # SR_trade = 0.002 / 0.009798 = 0.2041
    # SR_annual = 0.2041 * sqrt(252) = 3.24037...
    assert np.isclose(sr_high_p, 3.240370349, atol=1e-5) # <-- CORRECTED VALUE


def test_implied_precision_and_bin_frequency():
    """Test that implied_precision and bin_frequency are inverses."""
    # Note: implied_precision assumes sl is positive, bin_frequency
    # uses the binomial_sharpe_ratio formula which assumes sl is negative.
    # We must be consistent.

    freq = bin_frequency(
        stop_loss=-0.01,
        profit_taking=0.01,
        precision=0.6,
        target_sharpe_ratio=3.2403703492039306 # <-- USE CORRECT SR
    )
    assert np.isclose(freq, 252, atol=0.01) # <-- TIGHTENED TOLERANCE

    # Test implied_precision with sl=0.01 (positive)
    prec = implied_precision(
        stop_loss=0.01, # Positive
        profit_taking=0.01,
        frequency=252,
        target_sharpe_ratio=3.2396
    )
    # This won't match, the formulas are inconsistent.
    # Let's re-derive implied_precision from binomial_sharpe_ratio
    # S = ( (pt-sl)*p + sl ) / ( (pt-sl)*sqrt(p(1-p)) ) * sqrt(f)
    # S^2 = [ (pt-sl)p + sl ]^2 / [ (pt-sl)^2 * p(1-p) ] * f
    # S^2 * (pt-sl)^2 * (p - p^2) = f * [ (pt-sl)^2 p^2 + 2*sl*(pt-sl)p + sl^2 ]
    # [ S^2(pt-sl)^2 + f(pt-sl)^2 ] p^2 
    #   + [ -S^2(pt-sl)^2 + 2*f*sl*(pt-sl) ] p
    #   + [ f*sl^2 ] = 0
    # a = (S^2 + f)(pt-sl)^2
    # b = (2*f*sl - S^2(pt-sl))(pt-sl)
    # c = f*sl^2
    # This matches the user's `implied_precision` function,
    # but it assumes `sl` is *negative*.
    
    prec_recalc = implied_precision(
        stop_loss=-0.01, # Pass negative
        profit_taking=0.01,
        frequency=252,
        target_sharpe_ratio=3.2396
    )
    assert np.isclose(prec_recalc, 0.6, atol=1e-4)

def test_mix_gaussians():
    """Test the mix_gaussians function."""
    n_obs = 1000
    p = 0.5
    mix = mix_gaussians(
        mu1=10, mu2=-10, sigma1=1, sigma2=1, probability=p, n_obs=n_obs
    )
    assert len(mix) == n_obs
    # Mean should be (10*0.5) + (-10*0.5) = 0
    assert np.isclose(np.mean(mix), 0.0, atol=0.5)
    
    p = 0.8
    mix_biased = mix_gaussians(
        mu1=10, mu2=-10, sigma1=1, sigma2=1, probability=p, n_obs=n_obs
    )
    # Mean should be (10*0.8) + (-10*0.2) = 8 - 2 = 6
    assert np.isclose(np.mean(mix_biased), 6.0, atol=0.5)