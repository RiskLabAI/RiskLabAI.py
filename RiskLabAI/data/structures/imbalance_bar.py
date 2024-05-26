from typing import Any

from RiskLabAI.data.structures.base_information_driven_bar import *
from RiskLabAI.utils.constants import *


class ImbalanceBar(AbstractInformationDrivenBar):
    """
    Concrete class that contains the properties which are shared between all various type of ewma imbalance bars
    (dollar, volume, tick).
    """

    def __init__(
            self,
            bar_type: str,
            num_ticks_type: str,
            window_size_for_expected_n_ticks_estimation: int,
            window_size_for_expected_imbalance_estimation: int,
            initial_estimate_of_expected_n_ticks_in_bar: int,
            expected_ticks_number_bounds: Tuple[float, float] = None,
            analyse_thresholds: bool = False
    ):
        """
        AbstractImbalanceBars constructor function
        :param bar_type: type of bar. Accepted values are: volume_imbalance, dollar_imbalance, tick_imbalance
        :param num_ticks_type: determines expected number of ticks calculation type. Accepted values are:
                               fixed, expected
        :param window_size_for_expected_n_ticks_estimation: window size used to estimate number of ticks expectation
        :param window_size_for_expected_imbalance_estimation: The window size used to estimate imbalance expectation
        :param initial_estimate_of_expected_n_ticks_in_bar: initial estimate of number of ticks expectation window size
        :param expected_ticks_number_bounds: lower and upper bound of possible number of expected ticks that used to
               force bars sampling convergence. It's been used when bar_type is expected_imbalance.
        :param analyse_thresholds: whether return thresholds values (θ, number of ticks expectation, imbalance
                expectation) in a tabular format
        """
        super().__init__(
            bar_type,
            num_ticks_type,
            window_size_for_expected_n_ticks_estimation,
            window_size_for_expected_imbalance_estimation,
            initial_estimate_of_expected_n_ticks_in_bar,
            expected_ticks_number_bounds,
            analyse_thresholds
        )

        self.imbalance_bars_statistics = {
            CUMULATIVE_THETA: 0,
            EXPECTED_IMBALANCE: np.nan,
            PREVIOUS_BARS_N_TICKS_LIST: [],
            PREVIOUS_TICK_IMBALANCES_LIST: [],
        }

    def _bar_construction_condition(self, threshold):
        if not np.isnan(self.imbalance_bars_statistics[EXPECTED_IMBALANCE]):
            cumulative_theta = self.imbalance_bars_statistics[CUMULATIVE_THETA]
            condition_is_met = np.abs(cumulative_theta) > threshold

            return condition_is_met

        else:
            return False

    def _compute_tick_data(self, date_time, price, volume, tick_rule) -> Tuple[float, Any]:
        # imbalance calculations
        imbalance = self._imbalance_at_tick(price, tick_rule, volume)
        self.imbalance_bars_statistics[PREVIOUS_TICK_IMBALANCES_LIST].append(imbalance)
        self.imbalance_bars_statistics[CUMULATIVE_THETA] += imbalance

        # initialize expected imbalance first time, where expected imbalance estimation is nan
        if np.isnan(self.imbalance_bars_statistics[EXPECTED_IMBALANCE]):
            self.imbalance_bars_statistics[EXPECTED_IMBALANCE] = self._ewma_expected_imbalance(
                self.imbalance_bars_statistics[PREVIOUS_TICK_IMBALANCES_LIST],
                self.information_driven_bars_statistics[EXPECTED_IMBALANCE_WINDOW],
                warm_up=True
            )

        if self.analyse_thresholds is not None:
            self.imbalance_bars_statistics[TIMESTAMP] = date_time
            self.analyse_thresholds.append(dict(self.imbalance_bars_statistics))

        # is construction condition met to construct next bar or not
        expected_ticks_number = self.information_driven_bars_statistics[EXPECTED_TICKS_NUMBER]
        expected_imbalance = 2 * self.imbalance_bars_statistics[EXPECTED_IMBALANCE] - 1
        threshold = expected_ticks_number * np.abs(expected_imbalance)

        return threshold, None

    def _after_construction_process(self, date_time, price, volume, tick_rule, other_data: Any) -> None:
        self.information_driven_bars_statistics[PREVIOUS_BARS_N_TICKS_LIST].append(
            self.base_statistics[CUMULATIVE_TICKS])

        # update expected number of ticks based on formed bars
        self.information_driven_bars_statistics[EXPECTED_TICKS_NUMBER] = self._expected_number_of_ticks()

        # update expected imbalance (ewma)
        self.imbalance_bars_statistics[EXPECTED_IMBALANCE] = self._ewma_expected_imbalance(
            self.imbalance_bars_statistics[PREVIOUS_TICK_IMBALANCES_LIST],
            self.information_driven_bars_statistics[EXPECTED_IMBALANCE_WINDOW],
            warm_up=True,  # this field just apply for run bars
        )

    def _reset_cached_fields(self):
        super()._reset_cached_fields()
        self.imbalance_bars_statistics[CUMULATIVE_THETA] = 0
