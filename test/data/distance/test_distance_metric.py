"""
Tests for data/distance/distance_metric.py
"""

import numpy as np
import pytest
from RiskLabAI.data.distance import (
    calculate_number_of_bins,
    calculate_variation_of_information,
    calculate_mutual_information,
    calculate_distance,
)

@pytest.fixture
def sample_arrays():
    """Fixture for sample arrays."""
    x = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    y = np.array([1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2])
    z = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]) # Independent
    return x, y, z

def test_calculate_number_of_bins():
    """Test optimal bin calculation."""
    # Univariate
    bins_uni = calculate_number_of_bins(1000)
    assert bins_uni > 0
    
    # Bivariate
    bins_bi_high_corr = calculate_number_of_bins(1000, correlation=0.99)
    bins_bi_low_corr = calculate_number_of_bins(1000, correlation=0.01)
    
    assert bins_bi_low_corr < bins_bi_high_corr

def test_calculate_variation_of_information(sample_arrays):
    """Test VI calculation."""
    x, y, z = sample_arrays
    bins = 3
    
    # VI(X, X) should be 0
    vi_self = calculate_variation_of_information(x, x, bins, norm=True)
    assert np.isclose(vi_self, 0.0)
    
    # VI(X, Z) where Z is independent should be H(X) + H(Z)
    # VI_norm should be 1
    vi_indep = calculate_variation_of_information(x, z, bins, norm=True)
    assert np.isclose(vi_indep, 1.0, atol=0.1)
    
    # VI(X, Y) should be between 0 and 1
    vi_partial = calculate_variation_of_information(x, y, bins, norm=True)
    assert 0 < vi_partial <= 1

def test_calculate_mutual_information(sample_arrays):
    """Test MI calculation."""
    x, y, z = sample_arrays
    
    # MI(X, X) = H(X). MI_norm = 1
    mi_self = calculate_mutual_information(x, x, norm=True)
    assert np.isclose(mi_self, 1.0)

    # MI(X, Z) where Z is independent should be 0
    mi_indep = calculate_mutual_information(x, z, norm=True)
    assert np.isclose(mi_indep, 0.0, atol=0.1)

def test_calculate_distance():
    """Test angular distance calculation."""
    corr = np.array([[1.0, 0.5, 0.0],
                     [0.5, 1.0, -0.5],
                     [0.0, -0.5, 1.0]])
    
    # Angular
    dist_ang = calculate_distance(corr, metric="angular")
    # d(0,0) = sqrt(0.5 * (1 - 1)) = 0
    # d(0,1) = sqrt(0.5 * (1 - 0.5)) = sqrt(0.25) = 0.5
    # d(1,2) = sqrt(0.5 * (1 - (-0.5))) = sqrt(0.75) = 0.866
    assert np.isclose(dist_ang[0, 0], 0.0)
    assert np.isclose(dist_ang[0, 1], 0.5)
    assert np.isclose(dist_ang[1, 2], np.sqrt(0.75))
    
    # Absolute Angular
    dist_abs_ang = calculate_distance(corr, metric="absolute_angular")
    # d(1,2) = sqrt(0.5 * (1 - |-0.5|)) = sqrt(0.5 * 0.5) = 0.5
    assert np.isclose(dist_abs_ang[0, 0], 0.0)
    assert np.isclose(dist_abs_ang[0, 1], 0.5)
    assert np.isclose(dist_abs_ang[1, 2], 0.5)