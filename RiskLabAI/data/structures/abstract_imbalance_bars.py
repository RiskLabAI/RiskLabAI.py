from RiskLabAI.data.structures.abstract_information_driven_bars import *
from RiskLabAI.utils.constants import *


class AbstractImbalanceBars(AbstractInformationDrivenBars):
    """
    Abstract class that contains the imbalance properties which are shared between the subtypes.
    This class subtypes are as follows:
        1- ExpectedImbalanceBars
        2- FixedImbalanceBars

    The class implements imbalance bars sampling logic as explained on page 29,30 of Advances in Financial Machine Learning.
    """

    def __init__(
            self,
            bar_type: str,
            window_size_for_expected_n_ticks_estimation: int,
            window_size_for_expected_imbalance_estimation: int,
            initial_estimate_of_expected_n_ticks_in_bar: int,
            analyse_thresholds: bool
    ):
        """
        AbstractImbalanceBars constructor function
        :param bar_type: type of bar. e.g. expected_dollar_imbalance_bars, fixed_tick_imbalance_bars etc.
        :param window_size_for_expected_n_ticks_estimation: window size used to estimate number of ticks expectation
        :param initial_estimate_of_expected_n_ticks_in_bar: initial estimate of number of ticks expectation window size
        :param window_size_for_expected_imbalance_estimation: window size used to estimate imbalance expectation
        :param analyse_thresholds: whether return thresholds values (θ, number of ticks expectation, imbalance expectation) in a tabular format
        """

        super().__init__(
            bar_type,
            window_size_for_expected_n_ticks_estimation,
            initial_estimate_of_expected_n_ticks_in_bar,
            window_size_for_expected_imbalance_estimation
        )

        self.imbalance_bars_statistics = {
            CUMULATIVE_θ: 0,
            EXPECTED_IMBALANCE: np.nan,

            # previous bars number of ticks list and previous tick imbalances list
            PREVIOUS_BARS_N_TICKS_LIST: [],
            PREVIOUS_TICK_IMBALANCES_LIST: [],
        }

        if analyse_thresholds:
            self.analyse_thresholds = []
        else:
            self.analyse_thresholds = None

    def construct_bars_from_data(self, data: Union[list, tuple, np.ndarray]) -> list:
        """
        The function is used to construct bars from input ticks data.
        :param data: tabular data that contains date_time, price, and volume columns
        :return: constructed bars
        """

        bars_list = []
        for tick_data in data:
            self.tick_counter += 1

            date_time, price, volume = tuple(tick_data)
            tick_rule = self._tick_rule(price)
            self.update_base_fields(price, tick_rule, volume)

            # imbalance calculations
            imbalance = self._imbalance_at_tick(price, tick_rule, volume)
            self.imbalance_bars_statistics[PREVIOUS_TICK_IMBALANCES_LIST].append(imbalance)
            self.imbalance_bars_statistics[CUMULATIVE_θ] += imbalance

            # initialize expected imbalance first time, where expected imbalance estimation is nan
            if np.isnan(self.imbalance_bars_statistics[EXPECTED_IMBALANCE]):
                self.imbalance_bars_statistics[EXPECTED_IMBALANCE] = self._ewma_expected_imbalance(
                    self.imbalance_bars_statistics[PREVIOUS_TICK_IMBALANCES_LIST],
                    self.information_driven_bars_statistics[EXPECTED_IMBALANCE_WINDOW],
                    warm_up=True,  # this field just apply for run bars
                )

            if self.analyse_thresholds is not None:
                self.imbalance_bars_statistics['timestamp'] = date_time
                self.analyse_thresholds.append(dict(self.imbalance_bars_statistics))

            # is construction condition met to construct next bar or not
            expected_ticks_number = self.information_driven_bars_statistics[EXPECTED_TICKS_NUMBER]
            expected_imbalance = self.imbalance_bars_statistics[EXPECTED_IMBALANCE]
            threshold = expected_ticks_number * np.abs(expected_imbalance)
            is_construction_condition_met = self._bar_construction_condition(threshold)

            if is_construction_condition_met:
                next_bar = self._construct_next_bar(
                    date_time,
                    self.tick_counter,
                    price,
                    self.high_price,
                    self.low_price,
                    threshold
                )

                bars_list.append(next_bar)

                self.imbalance_bars_statistics[PREVIOUS_BARS_N_TICKS_LIST].append(
                    self.base_statistics[CUMULATIVE_TICKS])

                # update expected number of ticks based on formed bars
                self.information_driven_bars_statistics[EXPECTED_TICKS_NUMBER] = self._expected_number_of_ticks()

                # update expected imbalance (ewma)
                self.imbalance_bars_statistics[EXPECTED_IMBALANCE] = self._ewma_expected_imbalance(
                    self.imbalance_bars_statistics[PREVIOUS_TICK_IMBALANCES_LIST],
                    self.information_driven_bars_statistics[EXPECTED_IMBALANCE_WINDOW],
                    warm_up=True,  # this field just apply for run bars
                )

                # reset cached fields
                self._reset_cached_fields()

        return bars_list

    def _bar_construction_condition(
            self,
            threshold
    ):
        """
        Compute the condition of whether next bar should sample with current and previous tick datas or not.
        :return: whether next bar should form with current and previous tick datas or not.
        """

        if not np.isnan(self.imbalance_bars_statistics[EXPECTED_IMBALANCE]):
            cumulative_θ = self.imbalance_bars_statistics[CUMULATIVE_θ]
            condition_is_met = np.abs(cumulative_θ) > threshold

            return condition_is_met

        else:
            return False

    def _reset_cached_fields(
            self
    ):
        """
        This function are used (directly or override) by all concrete or abstract subtypes. The function is used to reset cached fields in bars construction process when next bar is sampled.
        :return:
        """
        super()._reset_cached_fields()
        self.imbalance_bars_statistics[CUMULATIVE_θ] = 0

    @abstractmethod
    def _expected_number_of_ticks(self) -> Union[float, int]:
        """
        Calculate number of ticks expectation when new imbalance bar is sampled.

        :return: number of ticks expectation.
        """
