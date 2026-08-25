r"""
Multiple-testing haircuts for the Sharpe ratio (Holm FWER, Benjamini-Hochberg-Yekutieli FDR).

When a researcher screens many strategies or factors and reports the significant ones, the selection
inflates the apparent significance. These corrections discount it by controlling, across the whole
family of trials, either the family-wise error rate (the chance of *any* false discovery) or the
false-discovery rate (the expected *fraction* of false discoveries). They complement the Deflated
Sharpe Ratio (`probabilistic_sharpe_ratio` / `expected_max_sharpe_ratio`): the DSR asks whether the
single best strategy survives its trial count, while these screen a whole family with a stated
error-control target.

Admitted in Appraisal 07 (CONTRIBUTIONS_LEDGER 2026-06-27). Regime tag, verbatim from the verdict:

    To judge a family of screened strategies or factors (not just the single best), prefer Holm when
    you must control the chance of any false positive across the family (FWER), and BHY when you want
    to bound the expected fraction of false discoveries (FDR) and can accept lower power. The Deflated
    Sharpe remains the tool when the question is whether one specific best strategy survives its trial
    count. Bonferroni is dominated by Holm; the t>3 hurdle over-rejects; the double-bootstrap does not
    control its stated error as implemented and is not admitted.

Evidence and caveats: appraisals/07_verdict.md.

References
----------
Holm, S. (1979) A simple sequentially rejective multiple test procedure.
    Scandinavian Journal of Statistics, 6(2), 65-70.
Benjamini, Y. and Yekutieli, D. (2001) The control of the false discovery rate in multiple testing
    under dependency. Annals of Statistics, 29(4), 1165-1188.
Harvey, C. R. and Liu, Y. (2015) Backtesting. Journal of Portfolio Management, 42(1), 13-28.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as ss


def sharpe_ratio_p_values(
    sharpe_ratios: np.ndarray, number_of_returns: int
) -> np.ndarray:
    r"""
    One-sided p-values for a positive edge (H1: SR > 0) from per-period Sharpe ratios.

    Under the i.i.d. null the Sharpe t-statistic is :math:`t = \hat{SR}\sqrt{T}`, so the one-sided
    p-value is :math:`\Phi(-t)`.

    Parameters
    ----------
    sharpe_ratios : np.ndarray
        Per-period Sharpe ratios of the screened strategies.
    number_of_returns : int
        The number of return observations T each Sharpe is estimated from.

    Returns
    -------
    np.ndarray
        One-sided p-values, one per strategy.
    """
    t_stats = np.asarray(sharpe_ratios, dtype=float) * np.sqrt(number_of_returns)
    return ss.norm.sf(t_stats)


def holm_adjusted_p_values(p_values: np.ndarray) -> np.ndarray:
    r"""
    Holm (1979) step-down family-wise-error-rate adjusted p-values.

    Sorting the p-values ascending, the adjusted value at rank i is
    :math:`\max_{j \le i} (M - j + 1) p_{(j)}`, clipped to 1, then mapped back to the input order. A
    strategy is a discovery at level alpha when its adjusted p-value is below alpha. Holm controls the
    FWER under any dependence and uniformly dominates Bonferroni (same size, at least as powerful).

    Parameters
    ----------
    p_values : np.ndarray
        Raw per-strategy p-values.

    Returns
    -------
    np.ndarray
        Holm-adjusted p-values, in the original order.
    """
    p = np.asarray(p_values, dtype=float)
    m = p.size
    if m == 0:
        return p.copy()
    order = np.argsort(p)
    sorted_p = p[order]
    adjusted_sorted = np.maximum.accumulate((m - np.arange(m)) * sorted_p)
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted
    return adjusted


def benjamini_hochberg_yekutieli_adjusted_p_values(p_values: np.ndarray) -> np.ndarray:
    r"""
    Benjamini-Hochberg-Yekutieli (BHY, 2001) step-up false-discovery-rate adjusted p-values.

    Uses the Yekutieli dependence constant :math:`c(M) = \sum_{i=1}^{M} 1/i`, valid under arbitrary
    dependence. Sorting the p-values ascending, the adjusted value at rank i is the running minimum
    (from the largest rank down) of :math:`c(M)\, M\, p_{(i)} / i`, clipped to 1. A strategy is a
    discovery at level alpha when its adjusted p-value is below alpha. BHY controls the FDR (the
    expected fraction of false discoveries) under any dependence.

    Parameters
    ----------
    p_values : np.ndarray
        Raw per-strategy p-values.

    Returns
    -------
    np.ndarray
        BHY-adjusted p-values, in the original order.
    """
    p = np.asarray(p_values, dtype=float)
    m = p.size
    if m == 0:
        return p.copy()
    order = np.argsort(p)
    sorted_p = p[order]
    c_m = np.sum(1.0 / np.arange(1, m + 1))
    ranks = np.arange(1, m + 1)
    factor = c_m * m / ranks
    # Step-up: enforce monotonicity from the largest p-value downward.
    adjusted_sorted = np.minimum.accumulate((factor * sorted_p)[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted
    return adjusted


def haircut_sharpe_ratios(
    sharpe_ratios: np.ndarray,
    number_of_returns: int,
    method: str = "holm",
    significance_level: float = 0.05,
) -> dict[str, np.ndarray]:
    r"""
    Apply a multiple-testing haircut to a family of screened Sharpe ratios.

    Converts each Sharpe to its one-sided p-value, adjusts the p-values for multiple testing (Holm for
    FWER or BHY for FDR), flags the surviving discoveries at ``significance_level``, and reports the
    haircut Sharpe - the Sharpe implied by the adjusted p-value, :math:`\hat{SR}_{adj} =
    \Phi^{-1}(1 - p_{adj}) / \sqrt{T}` - so the discount from selection is explicit.

    Prefer Holm when you must control the chance of any false positive across the family (FWER), and
    BHY when you want to bound the expected fraction of false discoveries (FDR) and can accept lower
    power. The Deflated Sharpe remains the tool when the question is whether one specific best strategy
    survives its trial count. See the module docstring for the full regime tag and
    appraisals/07_verdict.md.

    Parameters
    ----------
    sharpe_ratios : np.ndarray
        Per-period Sharpe ratios of the screened strategies (the trial set).
    number_of_returns : int
        The number of return observations T.
    method : str, default "holm"
        ``"holm"`` (FWER) or ``"bhy"`` (FDR).
    significance_level : float, default 0.05
        The error-control level alpha.

    Returns
    -------
    dict
        ``p_values`` (raw), ``adjusted_p_values``, ``significant`` (boolean discoveries at alpha), and
        ``haircut_sharpe_ratios`` (the Sharpe implied by each adjusted p-value).

    Examples
    --------
    >>> import numpy as np
    >>> from RiskLabAI.backtest import haircut_sharpe_ratios
    >>> sharpes = np.array([0.25, 0.18, 0.05, 0.02])  # per-period Sharpes from a 4-strategy search
    >>> result = haircut_sharpe_ratios(sharpes, number_of_returns=120, method="holm")
    >>> bool(result["significant"][0])
    True
    """
    p_values = sharpe_ratio_p_values(sharpe_ratios, number_of_returns)
    if method == "holm":
        adjusted = holm_adjusted_p_values(p_values)
    elif method == "bhy":
        adjusted = benjamini_hochberg_yekutieli_adjusted_p_values(p_values)
    else:
        raise ValueError(f"unknown method {method!r}; use 'holm' or 'bhy'.")
    significant = adjusted < significance_level
    haircut_t = ss.norm.isf(np.clip(adjusted, 1e-300, 1.0))
    haircut_sr = haircut_t / np.sqrt(number_of_returns)
    return {
        "p_values": p_values,
        "adjusted_p_values": adjusted,
        "significant": significant,
        "haircut_sharpe_ratios": haircut_sr,
    }
