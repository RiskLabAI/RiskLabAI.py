from typing import Tuple, Union

import pandas as pd

from RiskLabAI.data.structures.imbalance_bars import ExpectedImbalanceBars, FixedImbalanceBars
from RiskLabAI.data.structures.run_bars import ExpectedRunBars, FixedRunBars
from RiskLabAI.data.structures.standard_bars import StandardBars
from RiskLabAI.data.structures.time_bars import TimeBars

from RiskLabAI.utils.constants import CUMULATIVE_DOLLAR, CUMULATIVE_VOLUME, CUMULATIVE_TICKS


class BarsInitializerController:
    """
    Controller for initializing various types of bars.
    """

    def __init__(self):
        self.method_name_to_method = {
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
            "time_bars": self.initialize_time_bars
        }

    @staticmethod
    def initialize_expected_dollar_imbalance_bars(
            window_size_for_expected_n_ticks_estimation: int = 3,
            window_size_for_expected_imbalance_estimation: int = 10000,
            initial_estimate_of_expected_n_ticks_in_bar: int = 20000,
            expected_ticks_number_bounds: Tuple[float] = None,
            analyze_thresholds: bool = False
    ) -> ExpectedImbalanceBars:
        """
        Initialize expected dollar imbalance bars.

        :param window_size_for_expected_n_ticks_estimation: The window size for estimating the expected number of ticks.
        :param window_size_for_expected_imbalance_estimation: The window size for estimating the expected imbalance.
        :param initial_estimate_of_expected_n_ticks_in_bar: The initial estimate for the expected number of ticks in a bar.
        :param expected_ticks_number_bounds: Bounds for the expected number of ticks in a bar.
        :param analyze_thresholds: Flag indicating whether to analyze thresholds.

        :return: An instance of ExpectedImbalanceBars.
        """
        return ExpectedImbalanceBars(
            bar_type='dollar_imbalance',
            window_size_for_expected_n_ticks_estimation=window_size_for_expected_n_ticks_estimation,
            initial_estimate_of_expected_n_ticks_in_bar=initial_estimate_of_expected_n_ticks_in_bar,
            window_size_for_expected_imbalance_estimation=window_size_for_expected_imbalance_estimation,
            expected_ticks_number_bounds=expected_ticks_number_bounds,
            analyse_thresholds=analyze_thresholds
        )

    @staticmethod
    def initialize_expected_volume_imbalance_bars(
            window_size_for_expected_n_ticks_estimation: int = 3,
            window_size_for_expected_imbalance_estimation: int = 10000,
            initial_estimate_of_expected_n_ticks_in_bar: int = 20000,
            expected_ticks_number_bounds: Tuple[float] = None,
            analyse_thresholds: bool = False,
    ):

        bars = ExpectedImbalanceBars(bar_type='volume_imbalance',
                                     window_size_for_expected_n_ticks_estimation=window_size_for_expected_n_ticks_estimation,
                                     initial_estimate_of_expected_n_ticks_in_bar=initial_estimate_of_expected_n_ticks_in_bar,
                                     window_size_for_expected_imbalance_estimation=window_size_for_expected_imbalance_estimation,
                                     expected_ticks_number_bounds=expected_ticks_number_bounds,
                                     analyse_thresholds=analyse_thresholds)

        return bars

    @staticmethod
    def initialize_expected_tick_imbalance_bars(
            window_size_for_expected_n_ticks_estimation: int = 3,
            window_size_for_expected_imbalance_estimation: int = 10000,
            initial_estimate_of_expected_n_ticks_in_bar: int = 20000,
            expected_ticks_number_bounds: Tuple[float] = None,
            analyse_thresholds: bool = False,
    ):

        bars = ExpectedImbalanceBars(bar_type='tick_imbalance',
                                     window_size_for_expected_n_ticks_estimation=window_size_for_expected_n_ticks_estimation,
                                     initial_estimate_of_expected_n_ticks_in_bar=initial_estimate_of_expected_n_ticks_in_bar,
                                     window_size_for_expected_imbalance_estimation=window_size_for_expected_imbalance_estimation,
                                     expected_ticks_number_bounds=expected_ticks_number_bounds,
                                     analyse_thresholds=analyse_thresholds)

        return bars

    @staticmethod
    def initialize_fixed_dollar_imbalance_bars(
            window_size_for_expected_imbalance_estimation: int = 10000,
            initial_estimate_of_expected_n_ticks_in_bar: int = 20000,
            analyse_thresholds: bool = False,
    ):

        bars = FixedImbalanceBars(bar_type='dollar_imbalance', window_size_for_expected_n_ticks_estimation=None,
                                  initial_estimate_of_expected_n_ticks_in_bar=initial_estimate_of_expected_n_ticks_in_bar,
                                  window_size_for_expected_imbalance_estimation=window_size_for_expected_imbalance_estimation,
                                  analyse_thresholds=analyse_thresholds)

        return bars

    @staticmethod
    def initialize_fixed_volume_imbalance_bars(
            window_size_for_expected_imbalance_estimation: int = 10000,
            initial_estimate_of_expected_n_ticks_in_bar: int = 20000,
            analyse_thresholds: bool = False,
    ):

        bars = FixedImbalanceBars(bar_type='volume_imbalance', window_size_for_expected_n_ticks_estimation=None,
                                  initial_estimate_of_expected_n_ticks_in_bar=initial_estimate_of_expected_n_ticks_in_bar,
                                  window_size_for_expected_imbalance_estimation=window_size_for_expected_imbalance_estimation,
                                  analyse_thresholds=analyse_thresholds)

        return bars

    @staticmethod
    def initialize_fixed_tick_imbalance_bars(
            window_size_for_expected_imbalance_estimation: int = 10000,
            initial_estimate_of_expected_n_ticks_in_bar: int = 20000,
            analyse_thresholds: bool = False,
    ):

        bars = FixedImbalanceBars(bar_type='tick_imbalance', window_size_for_expected_n_ticks_estimation=None,
                                  initial_estimate_of_expected_n_ticks_in_bar=initial_estimate_of_expected_n_ticks_in_bar,
                                  window_size_for_expected_imbalance_estimation=window_size_for_expected_imbalance_estimation,
                                  analyse_thresholds=analyse_thresholds)

        return bars

    @staticmethod
    def initialize_expected_dollar_run_bars(
            window_size_for_expected_n_ticks_estimation: int = 3,
            window_size_for_expected_imbalance_estimation: int = 10000,
            initial_estimate_of_expected_n_ticks_in_bar: int = 20000,
            expected_ticks_number_bounds: Tuple[float] = None,
            analyse_thresholds: bool = False,
    ):

        bars = ExpectedRunBars(bar_type='dollar_run',
                               window_size_for_expected_n_ticks_estimation=window_size_for_expected_n_ticks_estimation,
                               initial_estimate_of_expected_n_ticks_in_bar=initial_estimate_of_expected_n_ticks_in_bar,
                               window_size_for_expected_imbalance_estimation=window_size_for_expected_imbalance_estimation,
                               expected_ticks_number_bounds=expected_ticks_number_bounds,
                               analyse_thresholds=analyse_thresholds)

        return bars

    @staticmethod
    def initialize_expected_volume_run_bars(
            window_size_for_expected_n_ticks_estimation: int = 3,
            window_size_for_expected_imbalance_estimation: int = 10000,
            initial_estimate_of_expected_n_ticks_in_bar: int = 20000,
            expected_ticks_number_bounds: Tuple[float] = None,
            analyse_thresholds: bool = False,
    ):

        bars = ExpectedRunBars(bar_type='volume_run',
                               window_size_for_expected_n_ticks_estimation=window_size_for_expected_n_ticks_estimation,
                               initial_estimate_of_expected_n_ticks_in_bar=initial_estimate_of_expected_n_ticks_in_bar,
                               window_size_for_expected_imbalance_estimation=window_size_for_expected_imbalance_estimation,
                               expected_ticks_number_bounds=expected_ticks_number_bounds,
                               analyse_thresholds=analyse_thresholds)

        return bars

    @staticmethod
    def initialize_expected_tick_run_bars(
            window_size_for_expected_n_ticks_estimation: int = 3,
            window_size_for_expected_imbalance_estimation: int = 10000,
            initial_estimate_of_expected_n_ticks_in_bar: int = 20000,
            expected_ticks_number_bounds: Tuple[float] = None,
            analyse_thresholds: bool = False,
    ):

        bars = ExpectedRunBars(bar_type='tick_run',
                               window_size_for_expected_n_ticks_estimation=window_size_for_expected_n_ticks_estimation,
                               initial_estimate_of_expected_n_ticks_in_bar=initial_estimate_of_expected_n_ticks_in_bar,
                               window_size_for_expected_imbalance_estimation=window_size_for_expected_imbalance_estimation,
                               expected_ticks_number_bounds=expected_ticks_number_bounds,
                               analyse_thresholds=analyse_thresholds)

        return bars

    @staticmethod
    def initialize_fixed_dollar_run_bars(
            window_size_for_expected_n_ticks_estimation: int = 3,
            window_size_for_expected_imbalance_estimation: int = 10000,
            initial_estimate_of_expected_n_ticks_in_bar: int = 20000,
            analyse_thresholds: bool = False,
    ):

        bars = FixedRunBars(
            bar_type='dollar_run',
            window_size_for_expected_n_ticks_estimation=window_size_for_expected_n_ticks_estimation,
            window_size_for_expected_imbalance_estimation=window_size_for_expected_imbalance_estimation,
            initial_estimate_of_expected_n_ticks_in_bar=initial_estimate_of_expected_n_ticks_in_bar,
            analyse_thresholds=analyse_thresholds)

        return bars

    @staticmethod
    def initialize_fixed_volume_run_bars(
            window_size_for_expected_n_ticks_estimation: int = 3,
            window_size_for_expected_imbalance_estimation: int = 10000,
            initial_estimate_of_expected_n_ticks_in_bar: int = 20000,
            analyse_thresholds: bool = False,
    ):

        bars = FixedRunBars(
            bar_type='volume_run',
            window_size_for_expected_n_ticks_estimation=window_size_for_expected_n_ticks_estimation,
            window_size_for_expected_imbalance_estimation=window_size_for_expected_imbalance_estimation,
            initial_estimate_of_expected_n_ticks_in_bar=initial_estimate_of_expected_n_ticks_in_bar,
            analyse_thresholds=analyse_thresholds
        )

        return bars

    @staticmethod
    def initialize_fixed_tick_run_bars(
            window_size_for_expected_n_ticks_estimation: int = 3,
            window_size_for_expected_imbalance_estimation: int = 10000,
            initial_estimate_of_expected_n_ticks_in_bar: int = 20000,
            analyse_thresholds: bool = False,
    ):

        bars = FixedRunBars(
            bar_type='tick_run',
            window_size_for_expected_n_ticks_estimation=window_size_for_expected_n_ticks_estimation,
            window_size_for_expected_imbalance_estimation=window_size_for_expected_imbalance_estimation,
            initial_estimate_of_expected_n_ticks_in_bar=initial_estimate_of_expected_n_ticks_in_bar,
            analyse_thresholds=analyse_thresholds
        )

        return bars

    @staticmethod
    def initialize_dollar_standard_bars(
            threshold: Union[float, pd.Series] = 70000000,
    ):

        bars = StandardBars(
            bar_type=CUMULATIVE_DOLLAR,
            threshold=threshold,
        )
        return bars

    @staticmethod
    def initialize_volume_standard_bars(
            threshold: Union[float, pd.Series] = 30000,
    ):

        bars = StandardBars(
            bar_type=CUMULATIVE_VOLUME,
            threshold=threshold
        )

        return bars

    @staticmethod
    def initialize_tick_standard_bars(
            threshold: Union[float, pd.Series] = 6000,
    ):

        bars = StandardBars(
            bar_type=CUMULATIVE_TICKS,
            threshold=threshold,
        )
        return bars

    @staticmethod
    def initialize_time_bars(
            resolution_type: str = 'D',
            resolution_units: int = 1,
    ):

        bars = TimeBars(
            resolution_type=resolution_type,
            resolution_units=resolution_units,
        )

        return bars
