"""
RiskLabAI Data Labeling Module

Implements the core logic for financial time series labeling, including:
- The Triple-Barrier Method
- Meta-Labeling
- Trend-Scanning Labels
- CUSUM Filtering

Reference:
    De Prado, M. (2018) Advances in financial machine learning.
    John Wiley & Sons, Chapters 3 & 4.
"""

from .labeling import (
    cusum_filter_events_dynamic_threshold,
    symmetric_cusum_filter,
    daily_volatility_with_log_returns,
    vertical_barrier,
    triple_barrier,
    meta_events,
    meta_labeling,
    lin_parts,
    process_jobs,
    expand_call,
    report_progress,
)
from .financial_labels import (
    calculate_t_value_linear_regression,
    find_trend_using_trend_scanning,
)

__all__ = [
    # from labeling.py
    "cusum_filter_events_dynamic_threshold",
    "symmetric_cusum_filter",
    "daily_volatility_with_log_returns",
    "vertical_barrier",
    "triple_barrier",
    "meta_events",
    "meta_labeling",
    "lin_parts",
    "process_jobs",
    "expand_call",
    "report_progress",
    
    # from financial_labels.py
    "calculate_t_value_linear_regression",
    "find_trend_using_trend_scanning",
]