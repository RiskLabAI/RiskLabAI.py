"""
RiskLabAI Structural Breaks Features Module

Implements methods for detecting structural breaks in time series,
including the Augmented Dickey-Fuller (ADF) and (G)SADF tests
as described by de Prado.
"""

from .structural_breaks import (
    lag_dataframe,
    prepare_data,
    compute_beta,
    get_expanding_window_adf,
    get_bsadf_statistic,
)

__all__ = [
    "lag_dataframe",
    "prepare_data",
    "compute_beta",
    "get_expanding_window_adf",
    "get_bsadf_statistic",
]