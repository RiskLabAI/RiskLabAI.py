"""
Tests for data/distance/codependence.py (KSG mutual information, distance correlation).
"""

import numpy as np

from RiskLabAI.data.distance.codependence import (
    distance_correlation,
    ksg_mutual_information,
)
from RiskLabAI.data.distance.distance_metric import calculate_mutual_information


def _gaussian_pair(rho, n, rng):
    z = rng.standard_normal(n)
    x = z
    y = rho * z + np.sqrt(1.0 - rho**2) * rng.standard_normal(n)
    return x, y


def test_ksg_matches_gaussian_closed_form():
    """Replication: at large N the KSG estimate matches the closed-form Gaussian MI -0.5 ln(1-rho^2)."""
    rng = np.random.default_rng(0)
    rho = 0.6
    true_mi = -0.5 * np.log(1.0 - rho**2)
    x, y = _gaussian_pair(rho, 5000, rng)
    estimate = ksg_mutual_information(x, y)
    assert abs(estimate - true_mi) < 0.05  # essentially unbiased on linear dependence


def test_ksg_beats_binned_on_short_nonlinear_sample():
    """
    Replication of the Appraisal 11 mechanism: on short nonlinear samples KSG has lower error against
    the true MI than the binned baseline (averaged over seeds).
    """
    rho = 0.6
    true_mi = -0.5 * np.log(
        1.0 - rho**2
    )  # MI is invariant to the monotone margin transform
    ksg_err, binned_err = [], []
    for s in range(30):
        rng = np.random.default_rng(s)
        x, y = _gaussian_pair(rho, 100, rng)
        x_nl, y_nl = np.exp(x), y**3  # monotone nonlinear margins (MI preserved)
        ksg_err.append((ksg_mutual_information(x_nl, y_nl) - true_mi) ** 2)
        binned_err.append((calculate_mutual_information(x_nl, y_nl) - true_mi) ** 2)
    assert np.sqrt(np.mean(ksg_err)) < np.sqrt(np.mean(binned_err))


def test_ksg_near_zero_for_independence():
    """KSG is near zero (and may go slightly negative) for independent data."""
    rng = np.random.default_rng(1)
    mi = ksg_mutual_information(rng.standard_normal(1000), rng.standard_normal(1000))
    assert abs(mi) < 0.05


def test_ksg_deterministic_and_short_series():
    rng1, rng2 = np.random.default_rng(2), np.random.default_rng(2)
    x1, y1 = _gaussian_pair(0.5, 200, rng1)
    x2, y2 = _gaussian_pair(0.5, 200, rng2)
    assert ksg_mutual_information(x1, y1) == ksg_mutual_information(x2, y2)
    assert np.isfinite(ksg_mutual_information(np.arange(6.0), np.arange(6.0)[::-1]))


def test_distance_correlation_in_unit_interval_and_independence():
    """dCor is in [0, 1] and near zero for independent variables."""
    rng = np.random.default_rng(3)
    d = distance_correlation(rng.standard_normal(400), rng.standard_normal(400))
    assert 0.0 <= d <= 1.0
    assert d < 0.15


def test_distance_correlation_detects_nonlinear_where_pearson_fails():
    """dCor detects a quadratic dependence that Pearson correlation misses."""
    rng = np.random.default_rng(4)
    x = rng.standard_normal(600)
    y = x**2 + 0.1 * rng.standard_normal(600)
    pearson = abs(np.corrcoef(x, y)[0, 1])
    dcor = distance_correlation(x, y)
    assert pearson < 0.15  # linear correlation blind to the symmetric dependence
    assert dcor > 0.3  # distance correlation sees it


def test_distance_correlation_increases_with_dependence():
    rng = np.random.default_rng(5)
    weak = distance_correlation(*_gaussian_pair(0.2, 500, rng))
    strong = distance_correlation(*_gaussian_pair(0.9, 500, rng))
    assert strong > weak
