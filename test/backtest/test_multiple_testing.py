"""
Tests for backtest/multiple_testing.py (Holm FWER + BHY FDR Sharpe haircuts).
"""

import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests

from RiskLabAI.backtest.multiple_testing import (
    benjamini_hochberg_yekutieli_adjusted_p_values,
    haircut_sharpe_ratios,
    holm_adjusted_p_values,
    sharpe_ratio_p_values,
)


@pytest.fixture
def p_values():
    return np.array([0.001, 0.02, 0.04, 0.3, 0.5, 0.008, 0.15])


def test_holm_matches_statsmodels(p_values):
    """Replication: Holm-adjusted p-values match statsmodels' 'holm' reference."""
    ours = holm_adjusted_p_values(p_values)
    ref = multipletests(p_values, method="holm")[1]
    assert np.allclose(ours, ref, atol=1e-12)


def test_bhy_matches_statsmodels(p_values):
    """Replication: BHY-adjusted p-values match statsmodels' 'fdr_by' (Benjamini-Yekutieli)."""
    ours = benjamini_hochberg_yekutieli_adjusted_p_values(p_values)
    ref = multipletests(p_values, method="fdr_by")[1]
    assert np.allclose(ours, ref, atol=1e-12)


def test_sharpe_to_pvalue_roundtrip():
    """A Sharpe of 0 maps to p=0.5; larger Sharpe maps to a smaller one-sided p-value."""
    p = sharpe_ratio_p_values(np.array([0.0, 0.2, 0.4]), number_of_returns=100)
    assert np.isclose(p[0], 0.5)
    assert p[0] > p[1] > p[2]


def test_holm_controls_fwer_under_global_null():
    """Under the global null, Holm flags a false discovery at ~the nominal rate (<= 0.05)."""
    rng = np.random.default_rng(0)
    false_positive = 0
    n_sims = 800
    for _ in range(n_sims):
        sharpes = rng.standard_normal(40) / np.sqrt(120)  # null Sharpes, T=120
        res = haircut_sharpe_ratios(
            sharpes, 120, method="holm", significance_level=0.05
        )
        false_positive += bool(res["significant"].any())
    assert false_positive / n_sims <= 0.08  # bounded near nominal (MC noise allowance)


def test_holm_dominates_bonferroni(p_values):
    """Holm is uniformly at least as powerful as Bonferroni (adjusted p never larger)."""
    holm = holm_adjusted_p_values(p_values)
    bonferroni = np.clip(p_values * p_values.size, 0, 1)
    assert np.all(holm <= bonferroni + 1e-12)


def test_haircut_reduces_sharpe():
    """The haircut Sharpe is never above the raw Sharpe (selection discounts it)."""
    sharpes = np.array([0.3, 0.22, 0.1, 0.05, 0.01])
    res = haircut_sharpe_ratios(sharpes, number_of_returns=150, method="bhy")
    assert np.all(res["haircut_sharpe_ratios"] <= sharpes + 1e-9)
    assert res["adjusted_p_values"].shape == sharpes.shape


def test_edge_cases():
    """Empty input, a single strategy, and an unknown method are handled."""
    assert holm_adjusted_p_values(np.array([])).size == 0
    assert benjamini_hochberg_yekutieli_adjusted_p_values(np.array([])).size == 0
    single = haircut_sharpe_ratios(np.array([0.3]), 120, method="holm")
    assert single["adjusted_p_values"].size == 1
    with pytest.raises(ValueError):
        haircut_sharpe_ratios(np.array([0.1, 0.2]), 120, method="nope")
