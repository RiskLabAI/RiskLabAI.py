r"""
Advanced bet sizing: probability calibration before sizing, and distributionally-robust Kelly.

de Prado sizes a bet from a predicted probability via a sigmoid (AFML ch.10, in `bet_sizing.py`). Two
admitted extensions sit alongside that baseline:

- **Platt calibration before sizing** (Meyer-Barziy-Joubert 2023). A classifier's raw probability is
  usually miscalibrated, so sizing straight from it over- or under-bets. Fitting a Platt (logistic)
  calibrator on a held-out block and sizing from the calibrated probability recovers near-optimal sizing
  when the input is miscalibrated, and is a safe no-op when it is already calibrated.
- **Distributionally-robust Kelly** (Sun-Boyd 2018). Point-estimate Kelly is fragile to estimation
  error. Sizing at the pessimistic edge of a confidence box on the win probability gives near-optimal
  growth with materially lower drawdown under estimation uncertainty, and converges to full Kelly as the
  box collapses with abundant data.

Admitted in Appraisal 18 (CONTRIBUTIONS_LEDGER 2026-06-27). Scope tags, verbatim from the verdict:

    Platt calibration: calibrate the meta-label probability before sizing when it is miscalibrated
    (measurable by ECE); use Platt, not isotonic, which overfits well-calibrated inputs; the benefit is
    largest under strong miscalibration and converges to no-op when already calibrated.

    DR-Kelly: prefer DR-Kelly over de Prado sigmoid / full Kelly when estimation uncertainty is material
    (small samples, noisy edge): near-optimal growth with lower drawdown; converges to Kelly with
    abundant data.

Held-out confirmed (appraisals/18_results, HELDOUT.md): on the sealed high-miscalibration corner Platt
closes the growth gap to the Kelly optimum distinguishably more than the de Prado sigmoid (paired
0.0032 [0.0030, 0.0034]); on the sealed tight-drawdown / small-sample corner DR-Kelly has lower
from-initial drawdown than full Kelly (0.144 vs 0.409) with near-optimal growth. Real-data net-of-cost
confirmation for both is a tracked obligation (`REAL_DATA_FOLLOWUPS.md`). Isotonic calibration and
risk-constrained Kelly were NOT admitted (shelved). Evidence and caveats: appraisals/18_verdict.md.

The DR-Kelly box-ambiguity worst case has a closed form (Kelly at the pessimistic probability, because a
long favorable bet's log-growth is monotone in the win probability), so no convex solver is required here;
``cvxpy`` (Apache-2.0) would only be needed for a general, non-box ambiguity set.

References
----------
Platt, J. (1999) Probabilistic outputs for support vector machines and comparisons to regularized
    likelihood methods. Advances in Large Margin Classifiers.
Meyer, B., Barziy, I. and Joubert, J. F. (2023) Meta-labeling: calibration and the bet-sizing of
    meta-label probabilities.
Sun, Q. and Boyd, S. (2018) Distributional robustness in Kelly gambling. (Distributionally-robust Kelly.)
Lopez de Prado, M. (2018) Advances in Financial Machine Learning, ch. 10 (the de Prado sigmoid baseline).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def kelly_bet_fraction(
    probability,
    payout_ratio: float = 1.0,
    max_fraction: float = 1.0,
) -> np.ndarray:
    r"""
    Growth-optimal Kelly fraction for a binary bet, clipped to ``[0, max_fraction]`` (long only).

    For a favorable bet that pays ``+payout_ratio`` per unit staked on a win (probability ``p``) and
    ``-1`` on a loss, the Kelly fraction is :math:`f^* = (p(b+1) - 1)/b` with ``b = payout_ratio``.

    Parameters
    ----------
    probability : float or np.ndarray
        Win probability/probabilities.
    payout_ratio : float, default 1.0
        Net odds ``b`` (even money by default).
    max_fraction : float, default 1.0
        Upper clip on the fraction.

    Returns
    -------
    np.ndarray
        The Kelly fraction(s) in ``[0, max_fraction]``.

    Examples
    --------
    >>> float(kelly_bet_fraction(0.6))
    0.2
    """
    p = np.asarray(probability, dtype=float)
    b = float(payout_ratio)
    f = (p * (b + 1.0) - 1.0) / b
    return np.clip(f, 0.0, max_fraction)


def expected_calibration_error(probabilities, outcomes, n_bins: int = 10) -> float:
    r"""
    Expected Calibration Error: the bin-mass-weighted gap between confidence and accuracy.

    Use this to decide whether calibration is worthwhile: a high ECE indicates the probabilities are
    miscalibrated (the regime where :class:`PlattCalibrator` helps); a near-zero ECE means calibration
    should be a no-op.

    Parameters
    ----------
    probabilities : np.ndarray
        Predicted probabilities in ``[0, 1]``.
    outcomes : np.ndarray
        Binary realized outcomes ``{0, 1}``.
    n_bins : int, default 10
        Number of equal-width probability bins.

    Returns
    -------
    float
        The ECE in ``[0, 1]``.
    """
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = p.size
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


class PlattCalibrator:
    r"""
    Platt scaling: a logistic recalibration of predicted probabilities, fit on a held-out block.

    Fits a logistic regression of the binary outcome on ``logit(p_hat)``; ``transform`` maps a raw
    probability to the calibrated one. Prefer Platt over isotonic calibration: isotonic overfits the
    calibration map and hurts already-calibrated inputs, whereas Platt's two-parameter logistic form is
    a safe near-no-op when the input is already calibrated (the characterized failure mode in the
    verdict). Calibrate before sizing (feed the calibrated probability into the de Prado sigmoid or a
    Kelly sizer) when ECE indicates miscalibration. See the module docstring for the verbatim scope tag
    and appraisals/18_verdict.md.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> y = rng.integers(0, 2, 500)
    >>> p_hat = np.clip(0.5 + 0.3 * (2 * y - 1) + 0.1 * rng.standard_normal(500), 0.01, 0.99)
    >>> cal = PlattCalibrator().fit(p_hat, y)
    >>> p_cal = cal.transform(p_hat)
    >>> p_cal.shape == p_hat.shape
    True
    """

    def __init__(self) -> None:
        self._model: LogisticRegression | None = None

    @staticmethod
    def _logit(p_hat) -> np.ndarray:
        x = np.clip(np.asarray(p_hat, dtype=float), 1e-4, 1 - 1e-4)
        return np.log(x / (1.0 - x)).reshape(-1, 1)

    def fit(self, predicted_probabilities, outcomes) -> PlattCalibrator:
        """Fit the Platt logistic map on (predicted probability, binary outcome). Returns ``self``."""
        y = np.asarray(outcomes, dtype=int)
        if np.unique(y).size < 2:
            # Degenerate block (one class only): leave as an identity calibrator.
            self._model = None
            return self
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        model.fit(self._logit(predicted_probabilities), y)
        self._model = model
        return self

    def transform(self, predicted_probabilities) -> np.ndarray:
        """Map raw probabilities to calibrated probabilities (identity if fit on one class only)."""
        if self._model is None:
            return np.asarray(predicted_probabilities, dtype=float)
        return self._model.predict_proba(self._logit(predicted_probabilities))[:, 1]


def distributionally_robust_kelly_fraction(
    probability_estimate,
    n_observations: int,
    payout_ratio: float = 1.0,
    max_fraction: float = 1.0,
    confidence_z: float = 1.0,
) -> np.ndarray:
    r"""
    Distributionally-robust Kelly fraction over a confidence box on the win probability (Sun-Boyd 2018).

    Builds a one-sided confidence box ``p_lo = p_hat - z * sqrt(p_hat (1 - p_hat) / n)`` and sizes with
    growth-optimal Kelly at the pessimistic ``p_lo``. For a long favorable bet the worst-case log-growth
    over the box is attained at the lowest win probability, so this closed form is the box-ambiguity
    distributionally-robust optimum (no convex solver needed). As ``n`` grows the box collapses and the
    fraction converges to full Kelly at ``p_hat``.

    Prefer DR-Kelly over de Prado sigmoid / full Kelly when estimation uncertainty is material (small
    samples, noisy edge): near-optimal growth with lower drawdown; converges to Kelly with abundant data.
    See the module docstring for the verbatim scope tag and appraisals/18_verdict.md.

    Parameters
    ----------
    probability_estimate : float or np.ndarray
        The estimated win probability ``p_hat``.
    n_observations : int
        The number of observations behind the estimate (drives the box width).
    payout_ratio : float, default 1.0
        Net odds ``b``.
    max_fraction : float, default 1.0
        Upper clip on the fraction.
    confidence_z : float, default 1.0
        Box half-width in standard errors (1.0 ~ one-sigma pessimism).

    Returns
    -------
    np.ndarray
        The distributionally-robust Kelly fraction(s) in ``[0, max_fraction]``.

    Examples
    --------
    >>> import numpy as np
    >>> small = float(distributionally_robust_kelly_fraction(0.6, n_observations=40))
    >>> large = float(distributionally_robust_kelly_fraction(0.6, n_observations=100000))
    >>> small < large <= 0.2 + 1e-9   # converges up to full Kelly (0.2) as n grows
    True
    """
    p_hat = np.clip(np.asarray(probability_estimate, dtype=float), 1e-4, 1 - 1e-4)
    standard_error = np.sqrt(p_hat * (1.0 - p_hat) / max(int(n_observations), 1))
    p_lo = np.clip(p_hat - confidence_z * standard_error, 1e-4, 1 - 1e-4)
    return kelly_bet_fraction(
        p_lo, payout_ratio=payout_ratio, max_fraction=max_fraction
    )
