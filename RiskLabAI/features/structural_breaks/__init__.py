"""
RiskLabAI Structural Breaks Features Module

Implements methods for detecting structural breaks in time series,
including the Augmented Dickey-Fuller (ADF) and (G)SADF tests
as described by de Prado, plus the Phillips-Shi-Yu (2015) GSADF / BSADF
multiple-bubble detector and date-stamping.
"""

from .structural_breaks import (
    compute_beta,
    get_bsadf_sequence,
    get_bsadf_statistic,
    get_bubble_episodes,
    get_expanding_window_adf,
    get_gsadf_statistic,
    get_sadf_sequence,
    lag_dataframe,
    prepare_data,
    psy_minimum_window,
    simulate_psy_critical_values,
)

__all__ = [
    "lag_dataframe",
    "prepare_data",
    "compute_beta",
    "get_expanding_window_adf",
    "get_bsadf_statistic",
    "psy_minimum_window",
    "get_sadf_sequence",
    "get_bsadf_sequence",
    "get_gsadf_statistic",
    "get_bubble_episodes",
    "simulate_psy_critical_values",
]
