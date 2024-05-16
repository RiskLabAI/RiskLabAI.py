from typing import Union, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from RiskLabAI.data.structures.run_bar import RunBar
from RiskLabAI.utils.ewma import ewma

from RiskLabAI.utils.constants import *


class ExpectedRunBar(RunBar):
    """
    Concrete class that contains the properties which are shared between all various type of ewma run bars (dollar, volume, tick).
    """

    def __init__(
            self,
            bar_type: str,
            window_size_for_expected_n_ticks_estimation: int,
            initial_estimate_of_expected_n_ticks_in_bar: int,
            window_size_for_expected_imbalance_estimation: int,
            expected_ticks_number_bounds: Tuple[float],
            analyse_thresholds: bool
    ):
        """
        ExpectedRunBars constructor function
        :param bar_type: type of bar. e.g. expected_dollar_imbalance_bars, fixed_tick_imbalance_bars etc.
        :param window_size_for_expected_n_ticks_estimation: window size used to estimate number of ticks expectation
        :param initial_estimate_of_expected_n_ticks_in_bar: initial estimate of number of ticks expectation window size
        :param window_size_for_expected_imbalance_estimation: window size used to estimate imbalance expectation
        :param expected_ticks_number_bounds lower and upper bound of possible number of expected ticks that used to force bars sampling convergence.
        :param analyse_thresholds: whether return thresholds values (θ, number of ticks expectation, imbalance expectation) in a tabular format
        """

        RunBar.__init__(
            self,
            bar_type,
            window_size_for_expected_n_ticks_estimation,
            window_size_for_expected_imbalance_estimation,
            initial_estimate_of_expected_n_ticks_in_bar,
            analyse_thresholds
        )

        if expected_ticks_number_bounds is None:
            self.expected_ticks_number_lower_bound, self.expected_ticks_number_upper_bound = 0, np.inf
        else:
            self.expected_ticks_number_lower_bound, self.expected_ticks_number_upper_bound = expected_ticks_number_bounds

    def _expected_number_of_ticks(self) -> Union[float, int]:
        """
        Calculate number of ticks expectation when new imbalance bar is sampled.

        :return: number of ticks expectation.
        """

        previous_bars_n_ticks_list = self.run_bars_statistics[PREVIOUS_BARS_N_TICKS_LIST]
        expected_ticks_number = ewma(np.array(
            previous_bars_n_ticks_list[-self.window_size_for_expected_n_ticks_estimation:], dtype=float),
            self.window_size_for_expected_n_ticks_estimation
        )[-1]

        return min(max(expected_ticks_number, self.expected_ticks_number_lower_bound),
                   self.expected_ticks_number_upper_bound)


class FixedRunBar(RunBar):
    """
    Concrete class that contains the properties which are shared between all various type of const run bars (dollar, volume, tick).
    """

    def __init__(self, bar_type: str, window_size_for_expected_n_ticks_estimation: int,
                 window_size_for_expected_imbalance_estimation: int,
                 initial_estimate_of_expected_n_ticks_in_bar: int, analyse_thresholds: bool):
        """
        Constructor.

        :param bar_type: (str) Type of run bar to create. Example: "dollar_run".
        :param window_size_for_expected_n_ticks_estimation: (int) Window size for E[T]s (number of previous bars to use for expected number of ticks estimation).
        :param window_size_for_expected_imbalance_estimation: (int) Expected window used to estimate expected run.
        :param initial_estimate_of_expected_n_ticks_in_bar: (int) Initial number of expected ticks.
        :param batch_size: (int) Number of rows to read in from the csv, per batch.
        :param analyse_thresholds: (bool) Flag to save  and return thresholds used to sample run bars.
        """

        RunBar.__init__(
            self,
            bar_type,
            window_size_for_expected_n_ticks_estimation,
            window_size_for_expected_imbalance_estimation,
            initial_estimate_of_expected_n_ticks_in_bar,
            analyse_thresholds
        )

    def _expected_number_of_ticks(self) -> Union[float, int]:
        """
        Calculate number of ticks expectation when new imbalance bar is sampled.

        :return: number of ticks expectation.
        """

        return self.information_driven_bars_statistics[EXPECTED_TICKS_NUMBER]
