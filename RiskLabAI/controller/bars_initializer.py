from typing import Tuple

from RiskLabAI.data.structures.imbalance_bar import ImbalanceBar
from RiskLabAI.data.structures.run_bar import RunBar
from RiskLabAI.data.structures.standard_bar import StandardBar
from RiskLabAI.data.structures.time_bar import TimeBar


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
            expected_ticks_number_bounds: Tuple[float, float] = None,
            analyze_thresholds: bool = False
    ) -> ImbalanceBar:
        """
        Initialize expected dollar imbalance bars.

        :param window_size_for_expected_n_ticks_estimation: The window size for estimating the expected number of ticks.
        :param window_size_for_expected_imbalance_estimation: The window size for estimating the expected imbalance.
        :param initial_estimate_of_expected_n_ticks_in_bar: The initial estimate for the expected number of ticks in a
         bar.
        :param expected_ticks_number_bounds: Bounds for the expected number of ticks in a bar.
        :param analyze_thresholds: Flag indicating whether to analyze thresholds.

        :return: An instance of ExpectedImbalanceBars.
        """
        return ImbalanceBar(
            bar_type='dollar_imbalance',
            num_ticks_type='expected',
            window_size_for_expected_n_ticks_estimation=window_size_for_expected_n_ticks_estimation,
            window_size_for_expected_imbalance_estimation=window_size_for_expected_imbalance_estimation,
            initial_estimate_of_expected_n_ticks_in_bar=initial_estimate_of_expected_n_ticks_in_bar,
            expected_ticks_number_bounds=expected_ticks_number_bounds,
            analyse_thresholds=analyze_thresholds
        )

    @staticmethod
    def initialize_expected_volume_imbalance_bars(
            window_size_for_expected_n_ticks_estimation: int = 3,
            window_size_for_expected_imbalance_estimation: int = 10000,
            initial_estimate_of_expected_n_ticks_in_bar: int = 20000,
            expected_ticks_number_bounds: Tuple[float, float] = None,
            analyse_thresholds: bool = False,
    ):
        bars = ImbalanceBar(bar_type='volume_imbalance',
                            num_ticks_type='expected',
                            window_size_for_expected_n_ticks_estimation=window_size_for_expected_n_ticks_estimation,
                            window_size_for_expected_imbalance_estimation=window_size_for_expected_imbalance_estimation,
                            initial_estimate_of_expected_n_ticks_in_bar=initial_estimate_of_expected_n_ticks_in_bar,
                            expected_ticks_number_bounds=expected_ticks_number_bounds,
                            analyse_thresholds=analyse_thresholds)

        return bars

    @staticmethod
    def initialize_expected_tick_imbalance_bars(
            window_size_for_expected_n_ticks_estimation: int = 3,
            window_size_for_expected_imbalance_estimation: int = 10000,
            initial_estimate_of_expected_n_ticks_in_bar: int = 20000,
            expected_ticks_number_bounds: Tuple[float, float] = None,
            analyse_thresholds: bool = False,
    ):
        bars = ImbalanceBar(bar_type='tick_imbalance',
                            num_ticks_type='expected',
                            window_size_for_expected_n_ticks_estimation=window_size_for_expected_n_ticks_estimation,
                            window_size_for_expected_imbalance_estimation=window_size_for_expected_imbalance_estimation,
                            initial_estimate_of_expected_n_ticks_in_bar=initial_estimate_of_expected_n_ticks_in_bar,
                            expected_ticks_number_bounds=expected_ticks_number_bounds,
                            analyse_thresholds=analyse_thresholds)

        return bars

    @staticmethod
    def initialize_fixed_dollar_imbalance_bars(
            window_size_for_expected_imbalance_estimation: int = 10000,
            initial_estimate_of_expected_n_ticks_in_bar: int = 20000,
            analyse_thresholds: bool = False,
    ):
        bars = ImbalanceBar(bar_type='dollar_imbalance',
                            num_ticks_type='fixed',
                            window_size_for_expected_n_ticks_estimation=0,
                            window_size_for_expected_imbalance_estimation=window_size_for_expected_imbalance_estimation,
                            initial_estimate_of_expected_n_ticks_in_bar=initial_estimate_of_expected_n_ticks_in_bar,
                            analyse_thresholds=analyse_thresholds)

        return bars

    @staticmethod
    def initialize_fixed_volume_imbalance_bars(
            window_size_for_expected_imbalance_estimation: int = 10000,
            initial_estimate_of_expected_n_ticks_in_bar: int = 20000,
            analyse_thresholds: bool = False,
    ):
        bars = ImbalanceBar(bar_type='volume_imbalance',
                            num_ticks_type='fixed',
                            window_size_for_expected_n_ticks_estimation=0,
                            window_size_for_expected_imbalance_estimation=window_size_for_expected_imbalance_estimation,
                            initial_estimate_of_expected_n_ticks_in_bar=initial_estimate_of_expected_n_ticks_in_bar,
                            analyse_thresholds=analyse_thresholds)

        return bars

    @staticmethod
    def initialize_fixed_tick_imbalance_bars(
            window_size_for_expected_imbalance_estimation: int = 10000,
            initial_estimate_of_expected_n_ticks_in_bar: int = 20000,
            analyse_thresholds: bool = False,
    ):
        bars = ImbalanceBar(bar_type='tick_imbalance',
                            num_ticks_type='fixed',
                            window_size_for_expected_n_ticks_estimation=0,
                            window_size_for_expected_imbalance_estimation=window_size_for_expected_imbalance_estimation,
                            initial_estimate_of_expected_n_ticks_in_bar=initial_estimate_of_expected_n_ticks_in_bar,
                            analyse_thresholds=analyse_thresholds)

        return bars

    @staticmethod
    def initialize_expected_dollar_run_bars(
            window_size_for_expected_n_ticks_estimation: int = 3,
            window_size_for_expected_imbalance_estimation: int = 10000,
            initial_estimate_of_expected_n_ticks_in_bar: int = 20000,
            expected_ticks_number_bounds: Tuple[float, float] = None,
            analyse_thresholds: bool = False,
    ):
        bars = RunBar(bar_type='dollar_run',
                      num_ticks_type='expected',
                      window_size_for_expected_n_ticks_estimation=window_size_for_expected_n_ticks_estimation,
                      window_size_for_expected_imbalance_estimation=window_size_for_expected_imbalance_estimation,
                      initial_estimate_of_expected_n_ticks_in_bar=initial_estimate_of_expected_n_ticks_in_bar,
                      expected_ticks_number_bounds=expected_ticks_number_bounds,
                      analyse_thresholds=analyse_thresholds)

        return bars

    @staticmethod
    def initialize_expected_volume_run_bars(
            window_size_for_expected_n_ticks_estimation: int = 3,
            window_size_for_expected_imbalance_estimation: int = 10000,
            initial_estimate_of_expected_n_ticks_in_bar: int = 20000,
            expected_ticks_number_bounds: Tuple[float, float] = None,
            analyse_thresholds: bool = False,
    ):
        bars = RunBar(bar_type='volume_run',
                      num_ticks_type='expected',
                      window_size_for_expected_n_ticks_estimation=window_size_for_expected_n_ticks_estimation,
                      window_size_for_expected_imbalance_estimation=window_size_for_expected_imbalance_estimation,
                      initial_estimate_of_expected_n_ticks_in_bar=initial_estimate_of_expected_n_ticks_in_bar,
                      expected_ticks_number_bounds=expected_ticks_number_bounds,
                      analyse_thresholds=analyse_thresholds)

        return bars

    @staticmethod
    def initialize_expected_tick_run_bars(
            window_size_for_expected_n_ticks_estimation: int = 3,
            window_size_for_expected_imbalance_estimation: int = 10000,
            initial_estimate_of_expected_n_ticks_in_bar: int = 20000,
            expected_ticks_number_bounds: Tuple[float, float] = None,
            analyse_thresholds: bool = False,
    ):
        bars = RunBar(bar_type='tick_run',
                      num_ticks_type='expected',
                      window_size_for_expected_n_ticks_estimation=window_size_for_expected_n_ticks_estimation,
                      window_size_for_expected_imbalance_estimation=window_size_for_expected_imbalance_estimation,
                      initial_estimate_of_expected_n_ticks_in_bar=initial_estimate_of_expected_n_ticks_in_bar,
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
        bars = RunBar(
            bar_type='dollar_run',
            num_ticks_type='fixed',
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
        bars = RunBar(
            bar_type='volume_run',
            num_ticks_type='fixed',
            window_size_for_expected_n_ticks_estimation=window_size_for_expected_n_ticks_estimation,
            window_size_for_expected_imbalance_estimation=window_size_for_expected_imbalance_estimation,
            initial_estimate_of_expected_n_ticks_in_bar=initial_estimate_of_expected_n_ticks_in_bar,
            analyse_thresholds=analyse_thresholds)

        return bars

    @staticmethod
    def initialize_fixed_tick_run_bars(
            window_size_for_expected_n_ticks_estimation: int = 3,
            window_size_for_expected_imbalance_estimation: int = 10000,
            initial_estimate_of_expected_n_ticks_in_bar: int = 20000,
            analyse_thresholds: bool = False,
    ):
        bars = RunBar(
            bar_type='tick_run',
            num_ticks_type='fixed',
            window_size_for_expected_n_ticks_estimation=window_size_for_expected_n_ticks_estimation,
            window_size_for_expected_imbalance_estimation=window_size_for_expected_imbalance_estimation,
            initial_estimate_of_expected_n_ticks_in_bar=initial_estimate_of_expected_n_ticks_in_bar,
            analyse_thresholds=analyse_thresholds)

        return bars

    @staticmethod
    def initialize_dollar_standard_bars(
            dollar_per_bar: int = 1000000,
            num_bars: int = 100,
    ):
        bars = StandardBar(
            bar_type="dollar",
            unit_per_bar=dollar_per_bar,
            number_bars=num_bars
        )
        return bars

    @staticmethod
    def initialize_volume_standard_bars(
            volume_per_bar: int = 1000000,
            num_bars: int = 100,
    ):
        bars = StandardBar(
            bar_type="volume",
            unit_per_bar=volume_per_bar,
            number_bars=num_bars
        )
        return bars

    @staticmethod
    def initialize_tick_standard_bars(
            tick_per_bar: int = 1000000,
            num_bars: int = 100,
    ):
        bars = StandardBar(
            bar_type="tick",
            unit_per_bar=tick_per_bar,
            number_bars=num_bars
        )
        return bars

    @staticmethod
    def initialize_time_bars(
            resolution_type: str = 'D',
            resolution_units: int = 1,
    ):
        bars = TimeBar(
            resolution_type=resolution_type,
            resolution_units=resolution_units,
        )

        return bars
