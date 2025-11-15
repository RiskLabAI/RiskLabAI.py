"""
RiskLabAI Utilities Package

This module provides common helper functions used across the library,
including:
- Constants
- EWMA calculation
- Progress bar
- Strategy side determination
- Plotly figure layout helpers
- Publication-quality Matplotlib plotting
"""

from .constants import *
from .ewma import ewma
from .progress import progress_bar
from .momentum_mean_reverting_strategy_sides import determine_strategy_side
from .update_figure_layout import update_figure_layout
from .publication_plots import (
    setup_publication_style,
    apply_plot_style,
    finalize_plot
)

# --- Alias for Backward Compatibility ---
# 'smoothing_average.py' is a duplicate of 'ewma.py'.
# We import 'ewma' and alias it to 'compute_exponential_weighted_moving_average'
# to maintain compatibility with modules that imported the old name.
# You can safely delete the 'smoothing_average.py' file.
compute_exponential_weighted_moving_average = ewma

__all__ = [
    # constants
    "DATE_TIME",
    "TIMESTAMP",
    "TICK_NUMBER",
    "OPEN_PRICE",
    "HIGH_PRICE",
    "LOW_PRICE",
    "CLOSE_PRICE",
    "CUMULATIVE_TICKS",
    "CUMULATIVE_DOLLAR",
    "THRESHOLD",
    "CUMULATIVE_VOLUME",
    "CUMULATIVE_BUY_VOLUME",
    "CUMULATIVE_SELL_VOLUME",
    "CUMULATIVE_θ",
    "CUMULATIVE_BUY_θ",
    "CUMULATIVE_SELL_θ",
    "EXPECTED_IMBALANCE",
    "EXPECTED_TICKS_NUMBER",
    "EXPECTED_BUY_IMBALANCE",
    "EXPECTED_SELL_IMBALANCE",
    "EXPECTED_BUY_TICKS_PROPORTION",
    "BUY_TICKS_NUMBER",
    "N_TICKS_ON_BAR_FORMATION",
    "PREVIOUS_TICK_RULE",
    "EXPECTED_IMBALANCE_WINDOW",
    "PREVIOUS_BARS_N_TICKS_LIST",
    "PREVIOUS_TICK_IMBALANCES_LIST",
    "PREVIOUS_TICK_IMBALANCES_BUY_LIST",
    "PREVIOUS_TICK_IMBALANCES_SELL_LIST",
    "PREVIOUS_BARS_BUY_TICKS_PROPORTIONS_LIST",
    "N_PREVIOUS_BARS_FOR_EXPECTED_N_TICKS_ESTIMATION",
    
    # ewma
    "ewma",
    "compute_exponential_weighted_moving_average",  # Alias
    
    # progress
    "progress_bar",
    
    # momentum_mean_reverting_strategy_sides
    "determine_strategy_side",
    
    # update_figure_layout
    "update_figure_layout",

    # publication_plots
    "setup_publication_style",
    "apply_plot_style",
    "finalize_plot",
]