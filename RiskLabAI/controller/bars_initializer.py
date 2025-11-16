"""
Controller class to act as a factory for initializing
various bar types (Standard, Time, Imbalance, Run).
"""

from typing import Tuple, Union, Dict, Callable, Optional, Any
import pandas as pd

# Import bar types
from RiskLabAI.data.structures.imbalance_bars import (
    ExpectedImbalanceBars, FixedImbalanceBars
)
from RiskLabAI.data.structures.run_bars import ExpectedRunBars, FixedRunBars
from RiskLabAI.data.structures.standard_bars import StandardBars
from RiskLabAI.data.structures.time_bars import TimeBars
from RiskLabAI.data.structures.abstract_bars import AbstractBars

# Import constants
from RiskLabAI.utils.constants import (
    CUMULATIVE_DOLLAR, CUMULATIVE_VOLUME, CUMULATIVE_TICKS
)

class BarsInitializerController:
    """
    Controller for initializing various types of bars.

    This class acts as a factory, providing static methods to
    construct different bar objects with sensible default parameters.
    """

    def __init__(self):
        """
        Initializes the controller and maps method names to methods.
        """
        self.method_name_to_method: Dict[str, Callable[..., AbstractBars]] = {
            "expected_dollar_imbalance_bars": self.initialize_expected_dollar_imbalance_bars,
            "expected_volume_imbalance_bars": self.initialize_expected_volume_imbalance_bars,
            "expected_tick_imbalance_bars": self.initialize_expected_tick_imbalance_bars,
            "fixed_dollar_imbalance_bars": self.initialize_fixed_dollar_imbalance_bars,
            "fixed_volume_imbalance_bars": self.initialize_fixed_volume_imbalance_bars,
            "fixed_tick_imbalance_bars": self.initialize_fixed_tick_imbalance_bars,
            "expected_dollar_run_bars": self.initialize_expected_dollar_run_bars,
            "expected_volume_run_bars": self.initialize_expected_volume_run_bars,
            "expected_tick_run_bars": self.initialize_expected_tick_run_bars,
            "fixed_dollar_run_bars": self.initialize_fixed_dollar_run_bars,
            "fixed_volume_run_bars": self.initialize_fixed_volume_run_bars,
            "fixed_tick_run_bars": self.initialize_fixed_tick_run_bars,
            "dollar_standard_bars": self.initialize_dollar_standard_bars,
            "volume_standard_bars": self.initialize_volume_standard_bars,
            "tick_standard_bars": self.initialize_tick_standard_bars,
            "time_bars": self.initialize_time_bars,
        }

    @staticmethod
    def initialize_expected_dollar_imbalance_bars(
        window_size_for_expected_n_ticks_estimation: int = 10000,
        window_size_for_expected_imbalance_estimation: int = 10000,
        initial_estimate_of_expected_n_ticks_in_bar: int = 20000,
        expected_ticks_number_bounds: Optional[Tuple[float, float]] = None,
        analyse_thresholds: bool = False,
        **kwargs: Any, # Accept extra kwargs but don't use them
    ) -> ExpectedImbalanceBars:
        """
        Initialize expected dollar imbalance bars.
        """
        return ExpectedImbalanceBars(
            bar_type="dollar_imbalance",
            window_size_for_expected_n_ticks_estimation=window_size_for_expected_n_ticks_estimation,
            initial_estimate_of_expected_n_ticks_in_bar=initial_estimate_of_expected_n_ticks_in_bar,
            window_size_for_expected_imbalance_estimation=window_size_for_expected_imbalance_estimation,
            expected_ticks_number_bounds=expected_ticks_number_bounds,
            analyse_thresholds=analyse_thresholds,
        )

    @staticmethod
    def initialize_expected_volume_imbalance_bars(
        window_size_for_expected_n_ticks_estimation: int = 10000,
        initial_estimate_of_expected_n_ticks_in_bar: int = 20000,
        window_size_for_expected_imbalance_estimation: int = 10000,
        expected_ticks_number_bounds: Optional[Tuple[float, float]] = None,
        analyse_thresholds: bool = False,
        **kwargs: Any,
    ) -> ExpectedImbalanceBars:
        """Initialize expected volume imbalance bars."""
        return ExpectedImbalanceBars(
            bar_type="volume_imbalance",
            window_size_for_expected_n_ticks_estimation=window_size_for_expected_n_ticks_estimation,
            initial_estimate_of_expected_n_ticks_in_bar=initial_estimate_of_expected_n_ticks_in_bar,
            window_size_for_expected_imbalance_estimation=window_size_for_expected_imbalance_estimation,
            expected_ticks_number_bounds=expected_ticks_number_bounds,
            analyse_thresholds=analyse_thresholds,
        )

    @staticmethod
    def initialize_expected_tick_imbalance_bars(
        window_size_for_expected_n_ticks_estimation: int = 10000,
        initial_estimate_of_expected_n_ticks_in_bar: int = 20000,
        window_size_for_expected_imbalance_estimation: int = 10000,
        expected_ticks_number_bounds: Optional[Tuple[float, float]] = None,
        analyse_thresholds: bool = False,
        **kwargs: Any,
    ) -> ExpectedImbalanceBars:
        """Initialize expected tick imbalance bars."""
        return ExpectedImbalanceBars(
            bar_type="tick_imbalance",
            window_size_for_expected_n_ticks_estimation=window_size_for_expected_n_ticks_estimation,
            initial_estimate_of_expected_n_ticks_in_bar=initial_estimate_of_expected_n_ticks_in_bar,
            window_size_for_expected_imbalance_estimation=window_size_for_expected_imbalance_estimation,
            expected_ticks_number_bounds=expected_ticks_number_bounds,
            analyse_thresholds=analyse_thresholds,
        )

    @staticmethod
    def initialize_fixed_dollar_imbalance_bars(
        initial_estimate_of_expected_n_ticks_in_bar: int = 20000,
        window_size_for_expected_imbalance_estimation: int = 10000,
        analyse_thresholds: bool = False,
        **kwargs: Any,
    ) -> FixedImbalanceBars:
        """Initialize fixed dollar imbalance bars."""
        return FixedImbalanceBars(
            bar_type="dollar_imbalance",
            initial_estimate_of_expected_n_ticks_in_bar=initial_estimate_of_expected_n_ticks_in_bar,
            window_size_for_expected_imbalance_estimation=window_size_for_expected_imbalance_estimation,
            analyse_thresholds=analyse_thresholds,
        )

    @staticmethod
    def initialize_fixed_volume_imbalance_bars(
        initial_estimate_of_expected_n_ticks_in_bar: int = 20000,
        window_size_for_expected_imbalance_estimation: int = 10000,
        analyse_thresholds: bool = False,
        **kwargs: Any,
    ) -> FixedImbalanceBars:
        """Initialize fixed volume imbalance bars."""
        return FixedImbalanceBars(
            bar_type="volume_imbalance",
            initial_estimate_of_expected_n_ticks_in_bar=initial_estimate_of_expected_n_ticks_in_bar,
            window_size_for_expected_imbalance_estimation=window_size_for_expected_imbalance_estimation,
            analyse_thresholds=analyse_thresholds,
        )

    @staticmethod
    def initialize_fixed_tick_imbalance_bars(
        initial_estimate_of_expected_n_ticks_in_bar: int = 20000,
        window_size_for_expected_imbalance_estimation: int = 10000,
        analyse_thresholds: bool = False,
        **kwargs: Any,
    ) -> FixedImbalanceBars:
        """Initialize fixed tick imbalance bars."""
        return FixedImbalanceBars(
            bar_type="tick_imbalance",
            initial_estimate_of_expected_n_ticks_in_bar=initial_estimate_of_expected_n_ticks_in_bar,
            window_size_for_expected_imbalance_estimation=window_size_for_expected_imbalance_estimation,
            analyse_thresholds=analyse_thresholds,
        )

    @staticmethod
    def initialize_expected_dollar_run_bars(
        window_size_for_expected_n_ticks_estimation: int = 10000,
        initial_estimate_of_expected_n_ticks_in_bar: int = 20000,
        window_size_for_expected_imbalance_estimation: int = 10000,
        expected_ticks_number_bounds: Optional[Tuple[float, float]] = None,
        analyse_thresholds: bool = False,
        **kwargs: Any,
    ) -> ExpectedRunBars:
        """Initialize expected dollar run bars."""
        return ExpectedRunBars(
            bar_type="dollar_run",
            window_size_for_expected_n_ticks_estimation=window_size_for_expected_n_ticks_estimation,
            initial_estimate_of_expected_n_ticks_in_bar=initial_estimate_of_expected_n_ticks_in_bar,
            window_size_for_expected_imbalance_estimation=window_size_for_expected_imbalance_estimation,
            expected_ticks_number_bounds=expected_ticks_number_bounds,
            analyse_thresholds=analyse_thresholds,
        )

    @staticmethod
    def initialize_expected_volume_run_bars(
        window_size_for_expected_n_ticks_estimation: int = 10000,
        initial_estimate_of_expected_n_ticks_in_bar: int = 20000,
        window_size_for_expected_imbalance_estimation: int = 10000,
        expected_ticks_number_bounds: Optional[Tuple[float, float]] = None,
        analyse_thresholds: bool = False,
        **kwargs: Any,
    ) -> ExpectedRunBars:
        """Initialize expected volume run bars."""
        return ExpectedRunBars(
            bar_type="volume_run",
            window_size_for_expected_n_ticks_estimation=window_size_for_expected_n_ticks_estimation,
            initial_estimate_of_expected_n_ticks_in_bar=initial_estimate_of_expected_n_ticks_in_bar,
            window_size_for_expected_imbalance_estimation=window_size_for_expected_imbalance_estimation,
            expected_ticks_number_bounds=expected_ticks_number_bounds,
            analyse_thresholds=analyse_thresholds,
        )

    @staticmethod
    def initialize_expected_tick_run_bars(
        window_size_for_expected_n_ticks_estimation: int = 10000,
        initial_estimate_of_expected_n_ticks_in_bar: int = 20000,
        window_size_for_expected_imbalance_estimation: int = 10000,
        expected_ticks_number_bounds: Optional[Tuple[float, float]] = None,
        analyse_thresholds: bool = False,
        **kwargs: Any,
    ) -> ExpectedRunBars:
        """Initialize expected tick run bars."""
        return ExpectedRunBars(
            bar_type="tick_run",
            window_size_for_expected_n_ticks_estimation=window_size_for_expected_n_ticks_estimation,
            initial_estimate_of_expected_n_ticks_in_bar=initial_estimate_of_expected_n_ticks_in_bar,
            window_size_for_expected_imbalance_estimation=window_size_for_expected_imbalance_estimation,
            expected_ticks_number_bounds=expected_ticks_number_bounds,
            analyse_thresholds=analyse_thresholds,
        )

    @staticmethod
    def initialize_fixed_dollar_run_bars(
        initial_estimate_of_expected_n_ticks_in_bar: int = 20000,
        window_size_for_expected_imbalance_estimation: int = 10000,
        analyse_thresholds: bool = False,
        **kwargs: Any,
    ) -> FixedRunBars:
        """Initialize fixed dollar run bars."""
        return FixedRunBars(
            bar_type="dollar_run",
            initial_estimate_of_expected_n_ticks_in_bar=initial_estimate_of_expected_n_ticks_in_bar,
            window_size_for_expected_imbalance_estimation=window_size_for_expected_imbalance_estimation,
            analyse_thresholds=analyse_thresholds,
        )

    @staticmethod
    def initialize_fixed_volume_run_bars(
        initial_estimate_of_expected_n_ticks_in_bar: int = 20000,
        window_size_for_expected_imbalance_estimation: int = 10000,
        analyse_thresholds: bool = False,
        **kwargs: Any,
    ) -> FixedRunBars:
        """Initialize fixed volume run bars."""
        return FixedRunBars(
            bar_type="volume_run",
            initial_estimate_of_expected_n_ticks_in_bar=initial_estimate_of_expected_n_ticks_in_bar,
            window_size_for_expected_imbalance_estimation=window_size_for_expected_imbalance_estimation,
            analyse_thresholds=analyse_thresholds,
        )

    @staticmethod
    def initialize_fixed_tick_run_bars(
        initial_estimate_of_expected_n_ticks_in_bar: int = 20000,
        window_size_for_expected_imbalance_estimation: int = 10000,
        analyse_thresholds: bool = False,
        **kwargs: Any,
    ) -> FixedRunBars:
        """Initialize fixed tick run bars."""
        return FixedRunBars(
            bar_type="tick_run",
            initial_estimate_of_expected_n_ticks_in_bar=initial_estimate_of_expected_n_ticks_in_bar,
            window_size_for_expected_imbalance_estimation=window_size_for_expected_imbalance_estimation,
            analyse_thresholds=analyse_thresholds,
        )

    @staticmethod
    def initialize_dollar_standard_bars(
        threshold: Union[float, pd.Series] = 70000000,
        **kwargs: Any,
    ) -> StandardBars:
        """Initialize dollar standard bars."""
        return StandardBars(bar_type=CUMULATIVE_DOLLAR, threshold=threshold)

    @staticmethod
    def initialize_volume_standard_bars(
        threshold: Union[float, pd.Series] = 30000,
        **kwargs: Any,
    ) -> StandardBars:
        """Initialize volume standard bars."""
        return StandardBars(bar_type=CUMULATIVE_VOLUME, threshold=threshold)

    @staticmethod
    def initialize_tick_standard_bars(
        threshold: Union[float, pd.Series] = 6000,
        **kwargs: Any,
    ) -> StandardBars:
        """Initialize tick standard bars."""
        return StandardBars(bar_type=CUMULATIVE_TICKS, threshold=threshold)

    @staticmethod
    def initialize_time_bars(
        resolution_type: str = "D",
        resolution_units: int = 1,
        **kwargs: Any,
    ) -> TimeBars:
        """Initialize time bars."""
        return TimeBars(
            resolution_type=resolution_type, resolution_units=resolution_units
        )