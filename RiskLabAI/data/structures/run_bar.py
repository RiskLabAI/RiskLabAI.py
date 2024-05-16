from typing import Any

from RiskLabAI.data.structures.base_information_driven_bar import *
from RiskLabAI.utils.constants import *


class RunBar(AbstractInformationDrivenBar):
    """
    Abstract class that contains the run properties which are shared between the subtypes.
    This class subtypes are as follows:
        1- ExpectedRunBars
        2- FixedRunBars

    The class implements run bars sampling logic as explained on page 31,32 of Advances in Financial Machine Learning.
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

        self.len_data = 0

        self.run_bars_statistics = {
            CUMULATIVE_BUY_THETA: 0,
            CUMULATIVE_SELL_THETA: 0,

            EXPECTED_BUY_IMBALANCE: np.ndarray([]),
            EXPECTED_SELL_IMBALANCE: np.ndarray([]),
            EXPECTED_BUY_TICKS_PROPORTION: np.ndarray([]),

            BUY_TICKS_NUMBER: 0,

            # Previous bars number of ticks and previous tick imbalances
            PREVIOUS_BARS_N_TICKS_LIST: [],
            PREVIOUS_TICK_IMBALANCES_BUY_LIST: [],
            PREVIOUS_TICK_IMBALANCES_SELL_LIST: [],
            PREVIOUS_BARS_BUY_TICKS_PROPORTIONS_LIST: []
        }

    def _pre_process_data(self, data: Union[list, tuple, np.ndarray]) -> None:
        self.len_data = len(data)

    def _bar_construction_condition(
            self,
            threshold
    ):
        max_proportion = max(
            self.information_driven_bars_statistics[EXPECTED_IMBALANCE_WINDOW] *
            self.run_bars_statistics[EXPECTED_BUY_TICKS_PROPORTION],

            self.run_bars_statistics[EXPECTED_SELL_IMBALANCE] *
            (1 - self.run_bars_statistics[EXPECTED_BUY_TICKS_PROPORTION])
        )

        if not np.isnan(max_proportion):
            max_theta = max(
                self.run_bars_statistics[CUMULATIVE_BUY_THETA],
                self.run_bars_statistics[CUMULATIVE_SELL_THETA]
            )

            condition_is_met = max_theta > threshold
            return condition_is_met
        else:
            return False

    def _compute_tick_data(self, date_time, price, volume, tick_rule) -> Tuple[float, Any]:
        imbalance = self._imbalance_at_tick(price, tick_rule, volume)

        if imbalance > 0:
            self.run_bars_statistics[PREVIOUS_TICK_IMBALANCES_BUY_LIST].append(imbalance)
            self.run_bars_statistics[CUMULATIVE_BUY_THETA] += imbalance
            self.run_bars_statistics[BUY_TICKS_NUMBER] += 1

        elif imbalance < 0:
            self.run_bars_statistics[PREVIOUS_TICK_IMBALANCES_SELL_LIST].append(-imbalance)
            self.run_bars_statistics[CUMULATIVE_SELL_THETA] += -imbalance

        warm_up = (self.run_bars_statistics[EXPECTED_BUY_IMBALANCE].size == 0 or
                   self.run_bars_statistics[EXPECTED_SELL_IMBALANCE].size == 0)

        # initialize expected imbalance first time, when initial_estimate_of_expected_n_ticks_in_bar passed
        if self.len_data == 0 and warm_up:
            self.run_bars_statistics[EXPECTED_BUY_IMBALANCE] = self._ewma_expected_imbalance(
                self.run_bars_statistics[PREVIOUS_TICK_IMBALANCES_BUY_LIST],
                self.information_driven_bars_statistics[EXPECTED_IMBALANCE_WINDOW],
                warm_up
            )
            self.run_bars_statistics[EXPECTED_SELL_IMBALANCE] = self._ewma_expected_imbalance(
                self.run_bars_statistics[PREVIOUS_TICK_IMBALANCES_SELL_LIST],
                self.information_driven_bars_statistics[EXPECTED_IMBALANCE_WINDOW],
                warm_up
            )

            if (self.run_bars_statistics[EXPECTED_BUY_IMBALANCE].size > 0 and
                    self.run_bars_statistics[EXPECTED_SELL_IMBALANCE].size > 0):
                self.run_bars_statistics[EXPECTED_BUY_TICKS_PROPORTION] = \
                    self.run_bars_statistics[BUY_TICKS_NUMBER] / self.base_statistics[CUMULATIVE_TICKS]

        if self.analyse_thresholds is not None:
            self.run_bars_statistics[TIMESTAMP] = date_time
            self.analyse_thresholds.append(dict(self.run_bars_statistics))

        # is construction condition met to construct next bar or not

        max_proportion = max(
            self.information_driven_bars_statistics[EXPECTED_IMBALANCE_WINDOW] *
            self.run_bars_statistics[EXPECTED_BUY_TICKS_PROPORTION],

            self.run_bars_statistics[EXPECTED_SELL_IMBALANCE] *
            (1 - self.run_bars_statistics[EXPECTED_BUY_TICKS_PROPORTION])
        )
        threshold = self.information_driven_bars_statistics[EXPECTED_TICKS_NUMBER] * max_proportion
        return threshold, None

    def _after_construction_process(self, date_time, price, volume, tick_rule, other_data: Any) -> None:
        self.run_bars_statistics[PREVIOUS_BARS_BUY_TICKS_PROPORTIONS_LIST].append(
            self.run_bars_statistics[BUY_TICKS_NUMBER] / self.base_statistics[CUMULATIVE_TICKS])

        # update expected number of ticks based on formed bars
        self.information_driven_bars_statistics[EXPECTED_TICKS_NUMBER] = self._expected_number_of_ticks()

        # update buy ticks proportions
        self.run_bars_statistics[EXPECTED_BUY_TICKS_PROPORTION] = ewma(
            np.array(self.run_bars_statistics[PREVIOUS_BARS_BUY_TICKS_PROPORTIONS_LIST][
                     -self.window_size_for_expected_n_ticks_estimation:],
                     dtype=float),
            self.window_size_for_expected_n_ticks_estimation)[-1]

        # update expected imbalance (ewma)
        self.run_bars_statistics[EXPECTED_BUY_IMBALANCE] = self._ewma_expected_imbalance(
            self.run_bars_statistics[PREVIOUS_TICK_IMBALANCES_BUY_LIST],
            self.information_driven_bars_statistics[EXPECTED_IMBALANCE_WINDOW]
        )
        self.run_bars_statistics[EXPECTED_SELL_IMBALANCE] = self._ewma_expected_imbalance(
            self.run_bars_statistics[PREVIOUS_TICK_IMBALANCES_SELL_LIST],
            self.information_driven_bars_statistics[EXPECTED_IMBALANCE_WINDOW]
        )

    def _reset_cached_fields(self):
        super()._reset_cached_fields()
        self.run_bars_statistics[CUMULATIVE_BUY_THETA] = 0
        self.run_bars_statistics[CUMULATIVE_SELL_THETA] = 0
        self.run_bars_statistics[BUY_TICKS_NUMBER] = 0
