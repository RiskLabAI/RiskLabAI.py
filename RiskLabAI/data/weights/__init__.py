"""
RiskLabAI Sample Weights Module

Implements sample weighting techniques for financial machine learning,
focusing on uniqueness (concurrency) and time decay.

Reference:
    De Prado, M. (2018) Advances in financial machine learning.
    John Wiley & Sons, Chapter 4.
"""

from .sample_weights import (
    expand_label_for_meta_labeling,
    calculate_average_uniqueness,
    sample_weight_absolute_return_meta_labeling,
    calculate_time_decay,
)

__all__ = [
    "expand_label_for_meta_labeling",
    "calculate_average_uniqueness",
    "sample_weight_absolute_return_meta_labeling",
    "calculate_time_decay",
]