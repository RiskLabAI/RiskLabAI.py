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
from .momentum_mean_reverting_strategy_sides import determine_strategy_side
from .progress import progress_bar

# Plotting helpers are imported lazily (PEP 562) so that the base install
# does not require matplotlib/seaborn/plotly (RiskLabAI[plot] extra).
_LAZY_PLOTTING = {
    "update_figure_layout": ("update_figure_layout", "update_figure_layout"),
    "setup_publication_style": ("publication_plots", "setup_publication_style"),
    "apply_plot_style": ("publication_plots", "apply_plot_style"),
    "finalize_plot": ("publication_plots", "finalize_plot"),
}


def __getattr__(name):
    if name in _LAZY_PLOTTING:
        from importlib import import_module

        module_name, attr = _LAZY_PLOTTING[name]
        value = getattr(import_module(f".{module_name}", __name__), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# --- Alias for backward compatibility ---
# The historical `compute_exponential_weighted_moving_average` name now maps to
# the canonical, numba-jitted `ewma` (the former `smoothing_average.py`
# duplicate has been removed).
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
