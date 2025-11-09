"""
RiskLabAI Ensemble Module

Implements ensemble-related methods, such as the theoretical
accuracy of bagging classifiers.
"""

from .bagging_classifier_accuracy import bagging_classifier_accuracy

__all__ = [
    "bagging_classifier_accuracy"
]