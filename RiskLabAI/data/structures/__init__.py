"""
RiskLabAI Data Structures Module

Implements various bar types (Standard, Time, Imbalance, Run) and
related financial data structures and algorithms from de Prado (2018).

Note: This module contains both a modern, class-based implementation
(e.g., StandardBars, ImbalanceBars) and an older, functional implementation
(e.g., generate_tick_bar_dataframe, infomation_driven_bars.py).
The class-based approach is recommended.
"""

from .abstract_bars import AbstractBars
from .abstract_information_driven_bars import AbstractInformationDrivenBars
from .abstract_imbalance_bars import AbstractImbalanceBars
from .abstract_run_bars import AbstractRunBars

# Concrete Class Implementations
from .standard_bars import StandardBars
from .time_bars import TimeBars
from .imbalance_bars import ExpectedImbalanceBars, FixedImbalanceBars
from .run_bars import ExpectedRunBars, FixedRunBars

# Hedging
from .hedging import pca_weights

# Filtering (Duplicated)
from .filtering_lopez import symmetric_cusum_filter

# Utilities & Deprecated Functional Bars
from .utilities_lopez import (
    compute_thresholds,
    create_ohlcv_dataframe
)
from .data_structures_lopez import (
    progress_bar,
    ewma,
    compute_grouping,
    generate_information_driven_bars as functional_info_bars, # Renamed
    generate_time_bar, # Duplicated
    generate_tick_bar, # Duplicated
    generate_volume_bar, # Duplicated
    generate_dollar_bar, # Duplicated
    calculate_pca_weights, # Duplicated
    events, # Duplicated
)
from .standard_bars_lopez import (
    generate_dollar_bar_dataframe,
    generate_tick_bar_dataframe,
    generate_time_bar_dataframe,
    generate_volume_bar_dataframe,
)
# Note: Filename has a typo
from .infomation_driven_bars import generate_information_driven_bars


__all__ = [
    # Abstract Classes
    "AbstractBars",
    "AbstractInformationDrivenBars",
    "AbstractImbalanceBars",
    "AbstractRunBars",
    
    # Concrete Bar Classes
    "StandardBars",
    "TimeBars",
    "ExpectedImbalanceBars",
    "FixedImbalanceBars",
    "ExpectedRunBars",
    "FixedRunBars",
    
    # Hedging
    "pca_weights",
    
    # Filtering
    "symmetric_cusum_filter",
    
    # Utilities
    "create_ohlcv_dataframe",
    "ewma",
    "progress_bar",

    # Deprecated/Duplicated Functional API
    "compute_thresholds",
    "compute_grouping",
    "generate_information_driven_bars",
    "functional_info_bars",
    "generate_time_bar",
    "generate_tick_bar",
    "generate_volume_bar",
    "generate_dollar_bar",
    "generate_dollar_bar_dataframe",
    "generate_tick_bar_dataframe",
    "generate_time_bar_dataframe",
    "generate_volume_bar_dataframe",
    "calculate_pca_weights",
    "events",
]