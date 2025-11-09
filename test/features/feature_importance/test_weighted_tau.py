"""
Tests for weighted_tau.py
"""

import pytest
import numpy as np
from .weighted_tau import calculate_weighted_tau

def test_weighted_tau():
    """Test the weighted tau calculation."""
    # Perfect positive correlation
    imp = np.array([0.5, 0.3, 0.1])
    ranks = np.array([1, 2, 3])
    tau_pos = calculate_weighted_tau(imp, ranks)
    assert np.isclose(tau_pos, 1.0)
    
    # Perfect negative correlation
    imp_neg = np.array([0.1, 0.3, 0.5])
    tau_neg = calculate_weighted_tau(imp_neg, ranks)
    assert np.isclose(tau_neg, -1.0)
    
    # Mixed correlation
    imp_mix = np.array([0.5, 0.1, 0.3])
    tau_mix = calculate_weighted_tau(imp_mix, ranks)
    assert -1.0 < tau_mix < 1.0