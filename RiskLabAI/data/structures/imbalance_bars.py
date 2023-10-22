from typing import Tuple, Union

import numpy as np

from RiskLabAI.data.structures.abstract_imbalance_bars import AbstractImbalanceBars
from RiskLabAI.utils.constants import *
from RiskLabAI.utils.ewma import ewma


class ExpectedImbalanceBars(AbstractImbalanceBars):
    """
    Concrete class that contains the properties which are shared between all various type of ewma imbalance bars (dollar, volume, tick).
    """

    def __init__(
            self,
            bar_type: str,
            window_size_for_expected_n_ticks_estimation: int,
            initial_estimate_of_expected_n_ticks_in_bar: int,
            window_size_for_expected_imbalance_estimation: int,
            expected_ticks_number_bounds: Tuple[float, float],
            analyse_thresholds: bool
    ):
        """
        ExpectedImbalanceBars constructor function
        :param bar_type: type of bar. e.g. expected_dollar_imbalance_bars, fixed_tick_imbalance_bars etc.
        :param window_size_for_expected_n_ticks_estimation: window size used to estimate number of ticks expectation
        :param initial_estimate_of_expected_n_ticks_in_bar: initial estimate of number of ticks expectation window size
        :param window_size_for_expected_imbalance_estimation: window size used to estimate imbalance expectation
        :param expected_ticks_number_bounds lower and upper bound of possible number of expected ticks that used to force bars sampling convergence.
        :param analyse_thresholds: whether return thresholds values (θ, number of ticks expectation, imbalance expectation) in a tabular format
        """

        AbstractImbalanceBars.__init__(
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

        previous_bars_n_ticks_list = self.imbalance_bars_statistics[PREVIOUS_BARS_N_TICKS_LIST]
        expected_ticks_number = ewma(np.array(
            previous_bars_n_ticks_list[-self.window_size_for_expected_n_ticks_estimation:], dtype=float),
            self.window_size_for_expected_n_ticks_estimation)[-1]

        return min(max(expected_ticks_number, self.expected_ticks_number_lower_bound),
                   self.expected_ticks_number_upper_bound)


class FixedImbalanceBars(AbstractImbalanceBars):
    """
    Concrete class that contains the properties which are shared between all various type of const imbalance bars (dollar, volume, tick).
    """

    def __init__(
            self,
            bar_type: str,
            window_size_for_expected_n_ticks_estimation: int,
            initial_estimate_of_expected_n_ticks_in_bar: int,
            window_size_for_expected_imbalance_estimation: int,
            analyse_thresholds: bool
    ):
        """
        FixedImbalanceBars constructor function
        :param bar_type: type of bar. e.g. expected_dollar_imbalance_bars, fixed_tick_imbalance_bars etc.
        :param window_size_for_expected_n_ticks_estimation: window size used to estimate number of ticks expectation
        :param initial_estimate_of_expected_n_ticks_in_bar: initial estimate of number of ticks expectation window size
        :param window_size_for_expected_imbalance_estimation: window size used to estimate imbalance expectation
        :param analyse_thresholds: whether return thresholds values (θ, number of ticks expectation, imbalance expectation) in a tabular format
        """

        AbstractImbalanceBars.__init__(self, bar_type, window_size_for_expected_n_ticks_estimation,
                                       window_size_for_expected_imbalance_estimation,
                                       initial_estimate_of_expected_n_ticks_in_bar, analyse_thresholds)

    def _expected_number_of_ticks(self) -> Union[float, int]:
        """
        Calculate number of ticks expectation when new imbalance bar is sampled.

        :return: number of ticks expectation.
        """

        return self.information_driven_bars_statistics[EXPECTED_TICKS_NUMBER]
