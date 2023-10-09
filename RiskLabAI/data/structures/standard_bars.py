from typing import Union

import numpy as np

from RiskLabAI.data.structures.abstract_bars import AbstractBars


class StandardBars(AbstractBars):
    """
    Concrete class that contains the properties which are shared between all various type of standard bars (dollar, volume, tick).
    """

    def __init__(
            self,
            bar_type: str,
            threshold: float = 50000,
    ):
        """
        StandardBars constructor function
        :param bar_type: type of bar. e.g. dollar_standard_bars, tick_standard_bars etc.
        :param threshold: threshold that used to sampling process
        """

        AbstractBars.__init__(self, bar_type)
        self.threshold = threshold

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

            # is construction condition met to construct next bar or not
            threshold = self.threshold
            is_construction_condition_met = self._bar_construction_condition(threshold)
            if is_construction_condition_met:
                next_bar = self._construct_next_bar(
                    date_time,
                    self.tick_counter,
                    price,
                    self.high_price,
                    self.low_price,
                    threshold,
                )

                bars_list.append(next_bar)
                self._reset_cached_fields()

        return bars_list

    def _bar_construction_condition(self, threshold) -> bool:
        """
        Compute the condition of whether next bar should sample with current and previous tick datas or not.
        :return: whether next bar should form with current and previous tick datas or not.
        """

        return self.base_statistics[self.bar_type] >= threshold
