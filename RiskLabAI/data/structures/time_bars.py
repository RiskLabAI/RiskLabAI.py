
from typing import Union

import numpy as np

from RiskLabAI.data.structures.abstract_bars import AbstractBars


class TimeBars(AbstractBars):
    """
    Concrete class of TimeBars logic
    """

    def __init__(self, resolution_type: str, resolution_units: int):
        """
        TimeBars constructor function

        :param resolution_type: (str) Type of bar resolution: ['D', 'H', 'MIN', 'S'].
        :param resolution_units: (int) Number of days, minutes, etc.
        """

        AbstractBars.__init__(self, bar_type="time")

        self.resolution_to_n_seconds = {
            'S': 1,
            'MIN': 60,
            'H': 60 * 60,
            'D': 60 * 60 * 24,
            'W': 60 * 60 * 24 * 7,
        }  

        self.resolution_type = resolution_type
        self.resolution_units = resolution_units
        self.threshold_in_seconds = self.resolution_units * self.resolution_to_n_seconds[self.resolution_type]
        self.timestamp = None
        self.timestamp_threshold = np.nan

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

            self.timestamp_threshold = (int(float(date_time.timestamp())) // self.threshold_in_seconds + 1) * self.threshold_in_seconds

            # initialize self.timestamp first time
            threshold = self.timestamp_threshold

            if self.timestamp is None:
                self.timestamp = self.timestamp_threshold

            # is construction condition met to construct next bar or not
            elif self._bar_construction_condition(threshold):
                next_bar = self._construct_next_bar(
                    date_time,
                    self.tick_counter,
                    self.close_price,
                    self.high_price,
                    self.low_price,
                    threshold,
                )

                bars_list.append(next_bar)

                # reset cached fields
                self._reset_cached_fields()
                self.timestamp = self.timestamp_threshold  # Current bar timestamp update

            self.close_price = price

        return bars_list

    def _bar_construction_condition(self, threshold) -> bool:
        """
        Compute the condition of whether next bar should sample with current and previous tick datas or not.
        :return: whether next bar should form with current and previous tick datas or not.
        """

        return self.timestamp < threshold
