"""
A base class for the various bar types. Includes the logic shared between classes, to minimise the amount of
duplicated code.
"""
from abc import ABC
from typing import Union, Tuple
import numpy as np

from RiskLabAI.utils.ewma import ewma
from RiskLabAI.data.structures.base import AbstractBar
from RiskLabAI.utils.constants import *


class AbstractInformationDrivenBar(AbstractBar, ABC):
    """
    Abstract class that contains the information driven properties which are shared between the subtypes.
    This class subtypes are as follows:
        1- AbstractImbalanceBars
        2- AbstractRunBars

    The class implements imbalance bars sampling logic as explained on page 29,30,31,32 of Advances in Financial
    Machine Learning.
    """

    def __init__(
            self,
            bar_type: str,
            num_ticks_type: str,
            window_size_for_expected_n_ticks_estimation: int,
            window_size_for_expected_imbalance_estimation: int,
            initial_estimate_of_expected_n_ticks_in_bar: int,
            expected_ticks_number_bounds: Tuple[float, float],
            analyse_thresholds: bool
    ):
        """
        AbstractInformationDrivenBars constructor function
        :param bar_type: type of bar. e.g. expected_dollar_imbalance_bars, fixed_tick_run_bars etc.
        :param num_ticks_type: determines expected number of ticks calculation type. Accepted values are: expected,
                               fixed
        :param window_size_for_expected_n_ticks_estimation: window size used to estimate number of ticks expectation
        :param initial_estimate_of_expected_n_ticks_in_bar: initial estimate of number of ticks expectation window size
        :param window_size_for_expected_imbalance_estimation: The window size used to estimate imbalance expectation
        :param expected_ticks_number_bounds: lower and upper bound of possible number of expected ticks that used to
                force bars sampling convergence. It's been used when bar_type is expected_imbalance.
        :param analyse_thresholds: whether return thresholds values (θ, number of ticks expectation, imbalance
                expectation) in a tabular format
        """
        assert num_ticks_type in ["expected", "fixed"], ("Invalid num_ticks_type. "
                                                         "Accepted values are: expected, fixed")
        assert bar_type in ["volume_imbalance", "dollar_imbalance", "tick_imbalance"], ("Invalid bar_type. "
                                                                                        "Accepted values are: "
                                                                                        "volume_imbalance, "
                                                                                        "dollar_imbalance, "
                                                                                        "tick_imbalance")
        super().__init__(bar_type)

        self.information_driven_bars_statistics = {
            EXPECTED_TICKS_NUMBER: initial_estimate_of_expected_n_ticks_in_bar,
            EXPECTED_IMBALANCE_WINDOW: window_size_for_expected_imbalance_estimation,
            PREVIOUS_BARS_N_TICKS_LIST: [],
        }

        if analyse_thresholds:
            self.analyse_thresholds = []
        else:
            self.analyse_thresholds = None

        if expected_ticks_number_bounds is None:
            self.expected_ticks_number_lower_bound, self.expected_ticks_number_upper_bound = 0, np.inf
        else:
            assert expected_ticks_number_bounds[0] < expected_ticks_number_bounds[1], ("Invalid bounds for expected "
                                                                                       "ticks number. lower bound"
                                                                                       " should be less than upper"
                                                                                       " bound")
            self.expected_ticks_number_lower_bound, self.expected_ticks_number_upper_bound = (
                expected_ticks_number_bounds)

        self.num_ticks_type = num_ticks_type
        self.window_size_for_expected_n_ticks_estimation = window_size_for_expected_n_ticks_estimation

    def _ewma_expected_imbalance(self, array: list, window: int, warm_up: bool = False) -> np.ndarray:
        """
        Calculates expected imbalance (2P[b_t=1]-1) using EWMA as defined on page 29 of Advances in Financial Machine
         Learning.
        :param array: imbalances list
        :param window: EWMA window for expectation calculation
        :param warm_up: whether warm up period passed or not
        :return: expected_imbalance: 2P[b_t=1]-1 which approximated using EWMA expectation
        """

        if len(array) < self.information_driven_bars_statistics[EXPECTED_TICKS_NUMBER] and warm_up:
            ewma_window = np.nan
        else:
            ewma_window = int(min(len(array), window))

        if np.isnan(ewma_window):
            expected_imbalance = np.nan
        else:
            expected_imbalance = ewma(
                np.array(array[-ewma_window:], dtype=float),
                window=ewma_window
            )[-1]

        return expected_imbalance

    def _imbalance_at_tick(self, price: float, signed_tick: int, volume: float) -> float:
        """
        Calculate the imbalance at tick t (current tick) (θ_t) using tick data as defined on page 29 of Advances in
        Financial Machine Learning
        :param price: price of tick
        :param signed_tick: tick rule of current tick computed before
        :param volume: volume of current tick
        :return: imbalance: imbalance of current tick
        """
        if self.bar_type == 'tick_imbalance' or self.bar_type == 'tick_run':
            imbalance = signed_tick

        elif self.bar_type == 'dollar_imbalance' or self.bar_type == 'dollar_run':
            imbalance = signed_tick * volume * price

        elif self.bar_type == 'volume_imbalance' or self.bar_type == 'volume_run':
            imbalance = signed_tick * volume

        else:
            raise ValueError('Unknown imbalance metric, possible values are tick/dollar/volume imbalance/run')

        return imbalance

    def update_base_fields(self, price: float, tick_rule: int, volume: float):
        super().update_base_fields(price, tick_rule, volume)
        self.information_driven_bars_statistics[PREVIOUS_BARS_N_TICKS_LIST].append(
            self.base_statistics[CUMULATIVE_TICKS])

    def _expected_number_of_ticks(self) -> Union[float, int]:
        """
        Calculate number of ticks expectation when new imbalance bar is sampled.
        """
        if self.num_ticks_type == "expected":
            previous_bars_n_ticks_list = self.information_driven_bars_statistics[PREVIOUS_BARS_N_TICKS_LIST]
            expected_ticks_number = ewma(np.array(
                previous_bars_n_ticks_list[-self.window_size_for_expected_n_ticks_estimation:], dtype=float),
                self.window_size_for_expected_n_ticks_estimation)[-1]

            return min(max(expected_ticks_number, self.expected_ticks_number_lower_bound),
                       self.expected_ticks_number_upper_bound)
        else:
            return self.information_driven_bars_statistics[EXPECTED_TICKS_NUMBER]
