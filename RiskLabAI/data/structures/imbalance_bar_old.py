from typing import Tuple, Union

import numpy as np

from RiskLabAI.data.structures.imbalance_bar import ImbalanceBar
from RiskLabAI.utils.constants import *
from RiskLabAI.utils.ewma import ewma


class ExpectedImbalanceBar(ImbalanceBar):
    """
    Concrete class that contains the properties which are shared between all various type of ewma imbalance bars (dollar, volume, tick).
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
        ExpectedImbalanceBars constructor function
        :param bar_type: type of bar. e.g. expected_dollar_imbalance_bars, fixed_tick_imbalance_bars etc.
        :param window_size_for_expected_n_ticks_estimation: window size used to estimate number of ticks expectation
        :param initial_estimate_of_expected_n_ticks_in_bar: initial estimate of number of ticks expectation window size
        :param window_size_for_expected_imbalance_estimation: window size used to estimate imbalance expectation
        :param expected_ticks_number_bounds lower and upper bound of possible number of expected ticks that used to force bars sampling convergence.
        :param analyse_thresholds: whether return thresholds values (θ, number of ticks expectation, imbalance expectation) in a tabular format
        """

        ImbalanceBar.__init__(
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




class FixedImbalanceBar(ImbalanceBar):
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

        ImbalanceBar.__init__(self, bar_type, window_size_for_expected_n_ticks_estimation,
                              window_size_for_expected_imbalance_estimation,
                              initial_estimate_of_expected_n_ticks_in_bar, analyse_thresholds)

    def _expected_number_of_ticks(self) -> Union[float, int]:
        """
        Calculate number of ticks expectation when new imbalance bar is sampled.

        :return: number of ticks expectation.
        """

        return self.information_driven_bars_statistics[EXPECTED_TICKS_NUMBER]
