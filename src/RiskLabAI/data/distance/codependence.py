r"""
Nonparametric codependence kernels: the KSG mutual-information estimator and distance correlation.

de Prado's codependence kernel (`distance_metric.calculate_mutual_information`) estimates mutual
information from a binned (histogram) joint distribution, whose bias and variance grow as the sample
shrinks relative to the bin count and which depends on an arbitrary bin choice. These two estimators
are binning-free alternatives for the codependence-kernel role of the variation-of-information metric
and ONC clustering:

- the Kraskov-Stogbauer-Grassberger (KSG) k-nearest-neighbour mutual-information estimator, which uses
  adaptive kNN distances rather than a fixed grid, so it is far less biased on short, noisy, or
  nonlinear / heavy-tailed samples; and
- distance correlation, a tuning-free dependence index in [0, 1] that is zero only at independence and
  detects nonlinear dependence with no estimation parameter.

Admitted in Appraisal 11 (CONTRIBUTIONS_LEDGER 2026-06-27). Regime tags, verbatim from the verdict:

    KSG: prefer KSG over binned MI/VI on short, noisy, or nonlinear/heavy-tailed samples - it is far
    less biased there and is essentially unbiased on linear dependence; on large samples with simple
    near-linear dependence the binned estimator is at least as good, so KSG converges rather than
    dominating everywhere. Use raw KSG; the surrogate de-bias adds nothing.

    Distance correlation: a tuning-free nonlinear dependence screen / codependence kernel - preferred
    when a parameter-free nonlinear screen or maximally stable clustering is wanted (best real-data
    cluster stability). It is a dependence index, not a metric on partitions like VI (keep VI/KSG for
    the metric role).

Evidence and caveats: appraisals/11_verdict.md.

References
----------
Kraskov, A., Stogbauer, H. and Grassberger, P. (2004) Estimating mutual information.
    Physical Review E, 69(6), 066138.
Szekely, G. J., Rizzo, M. L. and Bakirov, N. K. (2007) Measuring and testing dependence by
    correlation of distances. Annals of Statistics, 35(6), 2769-2794.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma


def _jitter(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Break exact ties with negligible data-scaled noise so kNN distances are well defined."""
    values = np.asarray(values, dtype=float).ravel()
    scale = np.std(values)
    if scale == 0.0:
        scale = 1.0
    return values + rng.standard_normal(values.shape) * scale * 1e-10


def ksg_mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    k: int = 4,
    random_state: int | None = 0,
) -> float:
    r"""
    Kraskov-Stogbauer-Grassberger mutual information (algorithm 1), in nats.

    Uses the Chebyshev (max) norm in the joint space; for each point the distance to its k-th joint
    neighbour sets the marginal search radius, and the marginal neighbour counts ``n_x``, ``n_y`` enter
    the digamma estimator

    .. math::
        \hat{I}(X;Y) = \psi(k) + \psi(N) - \langle \psi(n_x+1) + \psi(n_y+1) \rangle .

    Being binning-free it is far less biased than histogram MI on short / nonlinear / heavy-tailed
    samples, and essentially unbiased on linear dependence; on large near-linear samples binned MI is at
    least as good, so prefer KSG in the former regime. It can return a slightly negative value for
    (near-)independent data, which is a characterized property of the estimator, not an error. See the
    module docstring for the full regime tag and appraisals/11_verdict.md.

    Parameters
    ----------
    x, y : np.ndarray
        The two 1-D samples (equal length).
    k : int, default 4
        Number of nearest neighbours.
    random_state : int, optional
        Seed for the tie-breaking jitter (deterministic by default).

    Returns
    -------
    float
        The estimated mutual information in nats.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> z = rng.standard_normal(2000)
    >>> x = z + 0.1 * rng.standard_normal(2000)
    >>> y = z + 0.1 * rng.standard_normal(2000)
    >>> ksg_mutual_information(x, y) > 0.5   # strong dependence
    True
    """
    rng = np.random.default_rng(random_state)
    x = _jitter(x, rng)
    y = _jitter(y, rng)
    n = x.size
    if n < 2:
        return 0.0
    k = min(k, n - 1)
    z = np.column_stack([x, y])
    joint = cKDTree(z)
    distances, _ = joint.query(z, k=k + 1, p=np.inf)  # k+1 to exclude self
    eps = distances[:, -1]
    tree_x = cKDTree(x.reshape(-1, 1))
    tree_y = cKDTree(y.reshape(-1, 1))
    radius = eps * (1.0 - 1e-10)
    n_x = (
        np.asarray(
            tree_x.query_ball_point(
                x.reshape(-1, 1), radius, p=np.inf, return_length=True
            )
        )
        - 1
    )
    n_y = (
        np.asarray(
            tree_y.query_ball_point(
                y.reshape(-1, 1), radius, p=np.inf, return_length=True
            )
        )
        - 1
    )
    n_x = np.maximum(n_x, 0)
    n_y = np.maximum(n_y, 0)
    mi = digamma(k) + digamma(n) - np.mean(digamma(n_x + 1) + digamma(n_y + 1))
    return float(mi)


def _double_center(distance_matrix: np.ndarray) -> np.ndarray:
    row = distance_matrix.mean(axis=1, keepdims=True)
    col = distance_matrix.mean(axis=0, keepdims=True)
    return distance_matrix - row - col + distance_matrix.mean()


def distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    r"""
    Distance correlation (Szekely-Rizzo-Bakirov 2007), a dependence index in [0, 1].

    Computed from the double-centered pairwise-distance matrices: with
    :math:`A = \mathrm{centered}(|x_i - x_j|)` and :math:`B = \mathrm{centered}(|y_i - y_j|)`,

    .. math::
        \mathrm{dCor} = \sqrt{\frac{\overline{A \cdot B}}{\sqrt{\overline{A \cdot A}\,\overline{B \cdot B}}}}.

    It is zero only at population independence and detects nonlinear dependence with no estimation
    parameter, so it is a tuning-free nonlinear dependence screen / codependence kernel. It is a
    dependence index, not a metric on partitions like the variation of information (keep VI / KSG for the
    metric role). See the module docstring for the full regime tag and appraisals/11_verdict.md.

    Parameters
    ----------
    x, y : np.ndarray
        The two 1-D samples (equal length).

    Returns
    -------
    float
        The distance correlation in [0, 1].

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(500)
    >>> y = x ** 2 + 0.1 * rng.standard_normal(500)   # nonlinear dependence
    >>> distance_correlation(x, y) > 0.3
    True
    """
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    centered_a = _double_center(a)
    centered_b = _double_center(b)
    dcov2 = (centered_a * centered_b).mean()
    dvar_x = (centered_a * centered_a).mean()
    dvar_y = (centered_b * centered_b).mean()
    denom = np.sqrt(dvar_x * dvar_y)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(max(dcov2, 0.0) / denom))
