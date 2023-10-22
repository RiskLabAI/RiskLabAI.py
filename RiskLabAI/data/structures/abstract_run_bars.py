from RiskLabAI.data.structures.abstract_information_driven_bars import *
from RiskLabAI.utils.constants import *


class AbstractRunBars(AbstractInformationDrivenBars):
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
        window_size_for_expected_n_ticks_estimation: int,
        window_size_for_expected_imbalance_estimation: int,
        initial_estimate_of_expected_n_ticks_in_bar: int,
        analyse_thresholds: bool
    ):
        """
        AbstractRunBars constructor function
        :param bar_type: type of bar. e.g. expected_dollar_run_bars, fixed_tick_run_bars etc.
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

        self.run_bars_statistics = {
            CUMULATIVE_BUY_θ: 0,
            CUMULATIVE_SELL_θ: 0,

            EXPECTED_BUY_IMBALANCE: np.nan,
            EXPECTED_SELL_IMBALANCE: np.nan,
            EXPECTED_BUY_TICKS_PROPORTION: np.nan,

            BUY_TICKS_NUMBER: 0,

            # Previous bars number of ticks and previous tick imbalances
            PREVIOUS_BARS_N_TICKS_LIST: [],
            PREVIOUS_TICK_IMBALANCES_BUY_LIST: [],
            PREVIOUS_TICK_IMBALANCES_SELL_LIST: [],
            PREVIOUS_BARS_BUY_TICKS_PROPORTIONS_LIST: []
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

            if imbalance > 0:
                self.run_bars_statistics[PREVIOUS_TICK_IMBALANCES_BUY_LIST].append(imbalance)
                self.run_bars_statistics[CUMULATIVE_BUY_θ] += imbalance
                self.run_bars_statistics[BUY_TICKS_NUMBER] += 1

            elif imbalance < 0:
                self.run_bars_statistics[PREVIOUS_TICK_IMBALANCES_SELL_LIST].append(-imbalance)
                self.run_bars_statistics[CUMULATIVE_SELL_θ] += -imbalance

            warm_up = np.isnan([self.run_bars_statistics[EXPECTED_BUY_IMBALANCE],
                                self.run_bars_statistics[EXPECTED_SELL_IMBALANCE]]).any()

            # initialize expected imbalance first time, when initial_estimate_of_expected_n_ticks_in_bar passed
            if len(bars_list) == 0 and warm_up:

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

                if not np.isnan([self.run_bars_statistics[EXPECTED_BUY_IMBALANCE],
                                 self.run_bars_statistics[EXPECTED_SELL_IMBALANCE]]).any():
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

                self.run_bars_statistics[PREVIOUS_BARS_N_TICKS_LIST].append(self.base_statistics[CUMULATIVE_TICKS])
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

        max_proportion = max(
            self.information_driven_bars_statistics[EXPECTED_IMBALANCE_WINDOW] *
            self.run_bars_statistics[EXPECTED_BUY_TICKS_PROPORTION],

            self.run_bars_statistics[EXPECTED_SELL_IMBALANCE] *
            (1 - self.run_bars_statistics[EXPECTED_BUY_TICKS_PROPORTION])
        )

        if not np.isnan(max_proportion):
            max_θ = max(
                self.run_bars_statistics[CUMULATIVE_BUY_θ],
                self.run_bars_statistics[CUMULATIVE_SELL_θ]
            )

            condition_is_met = max_θ > threshold
            return condition_is_met
        else:
            return False

    def _reset_cached_fields(self):
        """
        This function are used (directly or override) by all concrete or abstract subtypes. The function is used to reset cached fields in bars construction process when next bar is sampled.
        :return:
        """
        super()._reset_cached_fields()
        self.run_bars_statistics[CUMULATIVE_BUY_θ] = 0
        self.run_bars_statistics[CUMULATIVE_SELL_θ] = 0
        self.run_bars_statistics[BUY_TICKS_NUMBER] = 0

    @abstractmethod
    def _expected_number_of_ticks(self) -> Union[float, int]:
        """
        Calculate number of ticks expectation when new imbalance bar is sampled.
        """