"""
RiskLabAI Ensemble Module

Implements ensemble-related methods, such as the theoretical
accuracy of bagging classifiers.
"""

# From the theoretical function file (bagging_classifier_accuracy.py)
from .bagging_classifier_accuracy import bagging_classifier_accuracy

# From the empirical class file (empirical_bagging_accuracy.py)
from .empirical_bagging_accuracy import (
    BaggingClassifierAccuracy,
    calculate_bootstrap_accuracy,
    plot_bootstrap_accuracy_distribution
)

# Expose all names for import
__all__ = [
    "bagging_classifier_accuracy",  # The function
    "BaggingClassifierAccuracy",  # The class
    "calculate_bootstrap_accuracy",
    "plot_bootstrap_accuracy_distribution"
]