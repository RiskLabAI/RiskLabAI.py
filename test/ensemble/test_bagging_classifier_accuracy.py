"""
Tests for ensemble/bagging_classifier_accuracy.py
"""

import pytest
import numpy as np
from RiskLabAI.ensemble.bagging_classifier_accuracy import bagging_classifier_accuracy

def test_bagging_accuracy():
    """
    Test the bagging classifier accuracy.
    """
    # 1. If p = 0.5, accuracy should remain 0.5
    assert np.isclose(bagging_classifier_accuracy(N=101, p=0.5), 0.5)

    # 2. If p > 0.5, accuracy should improve (be > p)
    p_high = 0.6
    assert bagging_classifier_accuracy(N=101, p=p_high) > p_high

    # 3. If p < 0.5, accuracy should degrade (be < p)
    p_low = 0.4
    assert bagging_classifier_accuracy(N=101, p=p_low) < p_low

    # 4. If N=1, accuracy should be p
    assert np.isclose(bagging_classifier_accuracy(N=1, p=0.7), 0.7)
    
    # 5. Test with N=3, p=0.7
    # P(X=2) + P(X=3)
    # P(X=2) = comb(3, 2) * (0.7**2) * (0.3**1) = 3 * 0.49 * 0.3 = 0.441
    # P(X=3) = comb(3, 3) * (0.7**3) * (0.3**0) = 1 * 0.343 * 1 = 0.343
    # Total = 0.441 + 0.343 = 0.784
    assert np.isclose(bagging_classifier_accuracy(N=3, p=0.7), 0.784)