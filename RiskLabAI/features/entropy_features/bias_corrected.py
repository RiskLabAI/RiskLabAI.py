"""
Bias-corrected Shannon entropy estimators (Miller-Madow, Grassberger, NSB).

de Prado's plug-in / maximum-likelihood entropy (`plug_in_entropy_estimator`, AFML ch.18) is
systematically biased downward when the number of distinct symbols (the alphabet, here the number of
possible n-grams) is large relative to the sample, because unobserved symbols contribute zero. The
bias is approximately -(K - 1) / (2N), growing with the alphabet K and shrinking with the sample N.
These estimators add back the missing mass: Miller-Madow analytically to first order, Grassberger to
higher order, and NSB by integrating over a near-uniform-entropy Bayesian prior.

All three operate on the same n-gram counts as the plug-in (via
:func:`~RiskLabAI.features.entropy_features.pmf.ngram_counts`) and return entropy in bits, normalized
by the word length, so they are drop-in comparable with `plug_in_entropy_estimator`. The plug-in's
behaviour is unchanged.

Admitted in Appraisal 06 (CONTRIBUTIONS_LEDGER 2026-06-27). Regime tag (preferred-when / avoid-when),
verbatim from the ledger:

    Prefer a bias-corrected estimator over the plug-in whenever the symbol counts are undersampled (a
    large effective alphabet relative to the sample: long words, fine encodings, or short windows).
    NSB is most accurate in deep undersampling; Grassberger is close at lower cost (the practical
    default when undersampled); Miller-Madow is the cheap first-order fix. All converge to the plug-in
    when N is much larger than K with no over-correction, so a correction is a safe default and the
    plug-in remains fine when well-sampled. The gain is negligible for coarse encodings (e.g. binary),
    and a gap caused by non-stationarity (e.g. sigma encoding) is not an entropy-bias problem the
    correction can fix.

Evidence and caveats: appraisals/06_verdict.md.

References
----------
Miller, G. (1955) Note on the bias of information estimates. In Information Theory in Psychology.
Grassberger, P. (2008) Entropy estimates from insufficient samplings. arXiv:physics/0307138.
Nemenman, I., Shafee, F. and Bialek, W. (2002) Entropy and inference, revisited. NIPS 14.
Wolpert, D. H. and Wolf, D. R. (1995) Estimating functions of probability distributions from a
    finite set of samples. Physical Review E, 52(6), 6841.
"""

from functools import lru_cache
from typing import Optional

import numpy as np
from scipy import integrate, optimize
from scipy.special import gammaln, psi

from .pmf import ngram_counts

_LN2 = np.log(2.0)


def _counts_array(message: str, approximate_word_length: int) -> np.ndarray:
    """Positive n-gram counts of a message as a float array (empty if the message is too short)."""
    counts = ngram_counts(message, approximate_word_length)
    if not counts:
        return np.zeros(0)
    return np.asarray(list(counts.values()), dtype=np.float64)


def _plugin_bits(counts: np.ndarray) -> float:
    """Plug-in / maximum-likelihood entropy (bits) of a count vector: H = -sum p log2 p."""
    n = counts.sum()
    if n <= 0:
        return 0.0
    p = counts / n
    return float(-np.sum(p * np.log2(p)))


def miller_madow_entropy(message: str, approximate_word_length: int = 1) -> float:
    """
    Miller-Madow bias-corrected Shannon entropy, normalized by word length (bits).

    The plug-in entropy plus the first-order analytic correction (K_hat - 1) / (2N), where K_hat is
    the number of observed n-grams and N the number of n-grams counted. The cheapest correction; it
    captures most of the gain at moderate undersampling.

    Prefer a bias-corrected estimator over the plug-in whenever the symbol counts are undersampled (a
    large effective alphabet relative to the sample: long words, fine encodings, or short windows).
    Miller-Madow is the cheap first-order fix; it converges to the plug-in when N is much larger than
    K, with no over-correction. The gain is negligible for coarse encodings (e.g. binary). See the
    module docstring for the full regime tag and appraisals/06_verdict.md.

    Parameters
    ----------
    message : str
        Input string (e.g. a discretized time series "110100...").
    approximate_word_length : int, default 1
        The n-gram length; the result is normalized by it, matching `plug_in_entropy_estimator`.

    Returns
    -------
    float
        The Miller-Madow entropy in bits per symbol.

    Examples
    --------
    >>> from RiskLabAI.features.entropy_features import miller_madow_entropy
    >>> miller_madow_entropy("110100110101", approximate_word_length=1) >= 0
    True
    """
    counts = _counts_array(message, approximate_word_length)
    if counts.size == 0:
        return 0.0
    n = counts.sum()
    k_hat = counts.size
    corrected = _plugin_bits(counts) + (k_hat - 1.0) / (2.0 * n * _LN2)
    return corrected / approximate_word_length


def _grassberger_g(n: np.ndarray) -> np.ndarray:
    """G(n) = psi(n) + 0.5 (-1)^n [psi((n+1)/2) - psi(n/2)] (Grassberger 2008), in nats."""
    sign = np.where(n.astype(np.int64) % 2 == 0, 1.0, -1.0)
    return psi(n) + 0.5 * sign * (psi((n + 1.0) / 2.0) - psi(n / 2.0))


def grassberger_entropy(message: str, approximate_word_length: int = 1) -> float:
    """
    Grassberger (2008) bias-corrected Shannon entropy, normalized by word length (bits).

    Uses the digamma finite-sample correction H = ln N - (1/N) sum n_i G(n_i), converted to bits. A
    higher-order correction with no tuning, intermediate in cost between Miller-Madow and NSB.

    Prefer a bias-corrected estimator over the plug-in whenever the symbol counts are undersampled (a
    large effective alphabet relative to the sample: long words, fine encodings, or short windows).
    Grassberger is close to NSB at lower cost (the practical default when undersampled); it converges
    to the plug-in when N is much larger than K, with no over-correction. The gain is negligible for
    coarse encodings (e.g. binary). See the module docstring for the full regime tag and
    appraisals/06_verdict.md.

    Parameters
    ----------
    message : str
        Input string.
    approximate_word_length : int, default 1
        The n-gram length; the result is normalized by it.

    Returns
    -------
    float
        The Grassberger entropy in bits per symbol.

    Examples
    --------
    >>> from RiskLabAI.features.entropy_features import grassberger_entropy
    >>> grassberger_entropy("110100110101", approximate_word_length=1) >= 0
    True
    """
    counts = _counts_array(message, approximate_word_length)
    if counts.size == 0:
        return 0.0
    n = counts.sum()
    h_nats = np.log(n) - (1.0 / n) * np.sum(counts * _grassberger_g(counts))
    return float(h_nats / _LN2) / approximate_word_length


# --- NSB (Nemenman-Shafee-Bialek 2002) -------------------------------------------------------
#
# Posterior mean entropy at a fixed Dirichlet concentration beta (Wolpert-Wolf 1995), integrated over
# beta against the NSB prior P(beta) proportional to d/dbeta E[H | beta], which makes the a priori mean
# entropy xi(beta) = psi0(K beta + 1) - psi0(beta + 1) near-uniform on (0, ln K). K is the true
# alphabet size; integration is in the xi variable (uniform prior) for stability, and the beta(xi)
# inversion depends only on (K, n_points) so it is cached.


def _xi_of_beta(beta: float, k: int) -> float:
    """A priori mean entropy (nats) implied by concentration beta over K symbols."""
    return psi(k * beta + 1.0) - psi(beta + 1.0)


@lru_cache(maxsize=256)
def _nsb_beta_grid(
    k: int, n_points: int
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """
    Quadrature nodes for the NSB xi-integral over (0, ln K): cached ``(xi_nodes, beta_nodes)``.

    xi(beta) is monotone increasing from 0 (beta -> 0) to ln K (beta -> inf); each node is inverted
    once with Brent's method. Depends only on (K, n_points), so it is cached across calls (the result
    is returned as tuples to keep it hashable / immutable).
    """
    xi_max = np.log(k)
    xi_nodes = np.linspace(
        xi_max / (n_points + 1), xi_max * n_points / (n_points + 1), n_points
    )
    beta_nodes = np.empty(n_points)
    for i, xi in enumerate(xi_nodes):

        def _f(beta: float, target: float = xi) -> float:
            return _xi_of_beta(beta, k) - target

        beta_nodes[i] = optimize.brentq(_f, 1e-8, 1e8, xtol=1e-12)
    return tuple(xi_nodes.tolist()), tuple(beta_nodes.tolist())


def _log_evidence(beta: np.ndarray, k: int, n: float, counts: np.ndarray) -> np.ndarray:
    """Unnormalised log marginal likelihood at each concentration in ``beta`` (nats, vectorised)."""
    b = beta[:, None]
    obs_term = np.sum(gammaln(counts[None, :] + b) - gammaln(b), axis=1)
    return gammaln(k * beta) - gammaln(n + k * beta) + obs_term


def _mean_entropy_given_beta(
    beta: np.ndarray, k: int, n: float, counts: np.ndarray, k_obs: int
) -> np.ndarray:
    """
    Posterior mean entropy (nats) at each concentration in ``beta`` (Wolpert-Wolf 1995).

    As beta -> 0 this tends to psi(N + 1) - (1/N) sum n_i psi(n_i + 1), the maximum-likelihood-like
    Bayesian limit; as beta -> inf it tends to the entropy of the uniform distribution over K.
    """
    denom = n + k * beta
    b = beta[:, None]
    observed = np.sum((counts[None, :] + b) * psi(counts[None, :] + b + 1.0), axis=1)
    unobserved = (k - k_obs) * beta * psi(beta + 1.0)
    return psi(denom + 1.0) - (observed + unobserved) / denom


def _nsb_from_counts(counts: np.ndarray, k: int, n_points: int) -> float:
    """NSB posterior mean entropy (bits) for a count vector and true alphabet size K."""
    n = counts.sum()
    if n <= 0 or k <= 1:
        return 0.0
    k_obs = counts.size

    xi_tuple, beta_tuple = _nsb_beta_grid(k, n_points)
    xi_nodes = np.asarray(xi_tuple)
    beta = np.asarray(beta_tuple)

    log_ev = _log_evidence(beta, k, n, counts)
    mean_h = _mean_entropy_given_beta(beta, k, n, counts, k_obs)

    # Uniform prior in xi; stabilise the evidence by its maximum before exponentiating.
    log_ev -= np.max(log_ev)
    weight = np.exp(log_ev)
    numerator = integrate.trapezoid(weight * mean_h, xi_nodes)
    normaliser = integrate.trapezoid(weight, xi_nodes)
    if normaliser <= 0 or not np.isfinite(normaliser):
        return float("nan")
    return float((numerator / normaliser) / _LN2)


def nsb_entropy(
    message: str,
    approximate_word_length: int = 1,
    alphabet_size: Optional[int] = None,
    n_points: int = 80,
) -> float:
    """
    NSB (Nemenman-Shafee-Bialek 2002) Bayesian Shannon entropy, normalized by word length (bits).

    The Wolpert-Wolf posterior mean entropy integrated over the Dirichlet concentration against a
    prior chosen so the implied prior on entropy is near-uniform. NSB needs the true alphabet size; it
    is the most accurate estimator in deep undersampling, and the most expensive.

    Prefer a bias-corrected estimator over the plug-in whenever the symbol counts are undersampled (a
    large effective alphabet relative to the sample: long words, fine encodings, or short windows).
    NSB is most accurate in deep undersampling; it converges to the plug-in when N is much larger than
    K, with no over-correction. The gain is negligible for coarse encodings (e.g. binary), and a gap
    caused by non-stationarity (e.g. sigma encoding) is not an entropy-bias problem the correction can
    fix. See the module docstring for the full regime tag and appraisals/06_verdict.md.

    Parameters
    ----------
    message : str
        Input string.
    approximate_word_length : int, default 1
        The n-gram length; the result is normalized by it.
    alphabet_size : int, optional
        The number of distinct base symbols the encoding can produce. The effective alphabet of words
        is ``alphabet_size ** approximate_word_length``. Defaults to the number of distinct symbols
        observed in the message (a lower bound; pass the encoding's true alphabet when known).
    n_points : int, default 80
        Number of quadrature nodes for the Bayesian integral.

    Returns
    -------
    float
        The NSB entropy in bits per symbol.

    Examples
    --------
    >>> from RiskLabAI.features.entropy_features import nsb_entropy
    >>> nsb_entropy("110100110101", approximate_word_length=1, alphabet_size=2) >= 0
    True
    """
    counts = _counts_array(message, approximate_word_length)
    if counts.size == 0:
        return 0.0
    base = alphabet_size if alphabet_size is not None else len(set(message))
    base = max(int(base), 1)
    k_eff = base**approximate_word_length
    # The effective alphabet cannot be smaller than the number of distinct words observed.
    k_eff = max(k_eff, counts.size)
    return _nsb_from_counts(counts, k_eff, n_points) / approximate_word_length
